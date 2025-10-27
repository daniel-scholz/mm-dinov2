# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import json
import logging
import os
import sys
import time
from typing import List, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torchmetrics
from cuml.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics import MetricTracker

from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.distributed import get_global_rank, get_global_size
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import evaluate, extract_features
from dinov2.utils.dtype import as_torch_dtype

logger = logging.getLogger("dinov2")

DEFAULT_MAX_ITER = 1_000
C_POWER_RANGE = torch.linspace(-6, 5, 45)
_CPU_DEVICE = torch.device("cpu")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_dataset_str",
        type=str,
        help="Test dataset",
        required=False,
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_dataset_str",
        type=str,
        help="Test dataset",
        required=False,
    )
    parser.add_argument(
        "--finetune-dataset-str",
        dest="finetune_dataset_str",
        type=str,
        help="Fine-tuning dataset",
    )
    parser.add_argument(
        "--finetune-on-val",
        action="store_true",
        help="If there is no finetune dataset, whether to choose the "
        "hyperparameters on the val set instead of 10%% of the train dataset",
    )
    parser.add_argument(
        "--metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Metric type",
    )
    parser.add_argument(
        "--train-features-device",
        type=str,
        help="Device to gather train features (cpu, cuda, cuda:0, etc.), default: %(default)s",
    )
    parser.add_argument(
        "--train-dtype",
        type=str,
        help="Data type to convert the train features to (default: %(default)s)",
    )
    parser.add_argument(
        "--max-train-iters",
        type=int,
        help="Maximum number of train iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--backbone", type=str, help="Backbone model to use", default="dinov2"
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        finetune_dataset_str=None,
        metric_type=MetricType.MEAN_ACCURACY,
        train_features_device="cpu",
        train_dtype="float64",
        max_train_iters=DEFAULT_MAX_ITER,
        finetune_on_val=False,
    )
    return parser


class LogRegModule(nn.Module):
    def __init__(
        self,
        C,
        max_iter=DEFAULT_MAX_ITER,
        dtype=torch.float64,
        device=_CPU_DEVICE,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.estimator = LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            output_type="numpy",
            tol=1e-12,
            linesearch_max_iter=50,
        )

    def forward(self, samples, targets):
        samples_device = samples.device
        samples = samples.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            samples = samples.numpy()
        probas = self.estimator.predict_proba(samples)
        return {"preds": torch.from_numpy(probas).to(samples_device), "target": targets}

    def fit(self, train_features, train_labels):
        train_features = train_features.to(dtype=self.dtype, device=self.device)
        train_labels = train_labels.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            # both cuML and sklearn only work with numpy arrays on CPU
            train_features = train_features.numpy()
            train_labels = train_labels.numpy().astype(int)
        self.estimator.fit(train_features, train_labels)


def evaluate_model(
    *,
    logreg_model,
    logreg_metrics: torchmetrics.MetricCollection,
    test_data_loader,
    device,
):
    postprocessors = {"metrics": logreg_model}
    metrics = {"metrics": logreg_metrics}
    return evaluate(nn.Identity(), test_data_loader, postprocessors, metrics, device)


def train_for_C(
    *,
    C,
    max_iter,
    train_features,
    train_labels,
    dtype=torch.float64,
    device=_CPU_DEVICE,
):
    logreg_model = LogRegModule(C, max_iter=max_iter, dtype=dtype, device=device)
    logreg_model.fit(train_features, train_labels)
    return logreg_model


def train_and_evaluate(
    *,
    C,
    max_iter,
    train_features,
    train_labels,
    logreg_metrics: torchmetrics.MetricCollection,
    test_data_loader,
    train_dtype=torch.float64,
    train_features_device,
    eval_device,
    mri_sequences=None,
):
    logreg_model = train_for_C(
        C=C,
        max_iter=max_iter,
        train_features=train_features,
        train_labels=train_labels,
        dtype=train_dtype,
        device=train_features_device,
    )
    return evaluate_model(
        logreg_model=logreg_model,
        logreg_metrics=logreg_metrics,
        test_data_loader=test_data_loader,
        device=eval_device,
    )


def sweep_C_values(
    *,
    train_features,
    train_labels,
    test_data_loader,
    metric_type,
    num_classes,
    train_dtype=torch.float64,
    train_features_device=_CPU_DEVICE,
    max_train_iters=DEFAULT_MAX_ITER,
):
    if metric_type == MetricType.PER_CLASS_ACCURACY:
        # If we want to output per-class accuracy, we select the hyperparameters with mean per class
        metric_type = MetricType.MEAN_PER_CLASS_ACCURACY
    logreg_metric = build_metric(metric_type, num_classes=num_classes)
    metric_tracker = MetricTracker(logreg_metric, maximize=True)
    ALL_C = 10**C_POWER_RANGE
    logreg_models = {}

    train_features = train_features.to(dtype=train_dtype, device=train_features_device)
    train_labels = train_labels.to(device=train_features_device)

    for i in range(get_global_rank(), len(ALL_C), get_global_size()):
        C = ALL_C[i].item()
        logger.info(
            f"Training for C = {C:.5f}, dtype={train_dtype}, "
            f"features: {train_features.shape}, {train_features.dtype}, "
            f"labels: {train_labels.shape}, {train_labels.dtype}"
        )
        logreg_models[C] = train_for_C(
            C=C,
            max_iter=max_train_iters,
            train_features=train_features,
            train_labels=train_labels,
            dtype=train_dtype,
            device=train_features_device,
        )

    gather_list = [None for _ in range(get_global_size())]
    torch.distributed.all_gather_object(gather_list, logreg_models)

    logreg_models_gathered = {}
    for logreg_dict in gather_list:
        logreg_models_gathered.update(logreg_dict)

    for i in range(len(ALL_C)):
        metric_tracker.increment()
        C = ALL_C[i].item()
        evals = evaluate_model(
            logreg_model=logreg_models_gathered[C],
            logreg_metrics=metric_tracker,
            test_data_loader=test_data_loader,
            device=torch.cuda.current_device(),
        )
        logger.info(f"Trained for C = {C:.5f}, accuracies = {evals}")

        best_stats, which_epoch = metric_tracker.best_metric(return_step=True)
        best_stats_100 = {k: 100.0 * v for k, v in best_stats.items()}
        best_key = list(best_stats.keys())[0]
        if which_epoch[best_key] == i:
            best_C = C
    logger.info(f"Sweep best {best_stats_100}, best C = {best_C:.6f}")

    return best_stats, best_C


def eval_log_regression(
    *,
    model,
    train_dataset,
    val_dataset,
    finetune_dataset,
    train_features=None,
    train_labels=None,
    metric_types: List[MetricType],
    batch_size,
    num_workers,
    finetune_on_val=False,
    train_dtype=torch.float64,
    train_features_device=_CPU_DEVICE,
    max_train_iters=DEFAULT_MAX_ITER,
    mri_sequences=None,
):
    """
    Implements the "standard" process for log regression evaluation:
    The value of C is chosen by training on train_dataset and evaluating on
    finetune_dataset. Then, the final model is trained on a concatenation of
    train_dataset and finetune_dataset, and is evaluated on val_dataset.
    If there is no finetune_dataset, the value of C is the one that yields
    the best results on a random 10% subset of the train dataset
    """

    start = time.time()

    if train_features is None:
        train_features, train_labels = extract_features(
            model,
            train_dataset,
            batch_size,
            num_workers,
            gather_on_cpu=(train_features_device == _CPU_DEVICE),
            mri_sequences=mri_sequences,
        )
    val_features, val_labels = extract_features(
        model,
        val_dataset,
        batch_size,
        num_workers,
        gather_on_cpu=(train_features_device == _CPU_DEVICE),
        mri_sequences=mri_sequences,
    )
    val_features_cls = val_features[:, 0]
    val_data_loader = torch.utils.data.DataLoader(
        TensorDataset(val_features_cls, val_labels),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        persistent_workers=False,
    )

    if finetune_dataset is None and finetune_on_val:
        logger.info("Choosing hyperparameters on the val dataset")
        finetune_features, finetune_labels = val_features, val_labels
    elif finetune_dataset is None and not finetune_on_val:
        logger.info("Choosing hyperparameters on 10% of the train dataset")
        torch.manual_seed(0)

        indices = torch.randperm(len(train_features), device=train_features.device)
        finetune_index = indices[: len(train_features) // 10]
        train_index = indices[len(train_features) // 10 :]

        finetune_features, finetune_labels = (
            train_features[finetune_index],
            train_labels[finetune_index],
        )
        train_features, train_labels = (
            train_features[train_index],
            train_labels[train_index],
        )

    else:
        logger.info("Choosing hyperparameters on the finetune dataset")
        finetune_features, finetune_labels = extract_features(
            model,
            finetune_dataset,
            batch_size,
            num_workers,
            gather_on_cpu=(train_features_device == _CPU_DEVICE),
            mri_sequences=mri_sequences,
        )

    # release the model - free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_features_cls = train_features[:, 0]
    finetune_features_cls = finetune_features[:, 0]

    finetune_data_loader = torch.utils.data.DataLoader(
        TensorDataset(finetune_features_cls, finetune_labels),
        batch_size=batch_size,
        drop_last=False,
    )

    if len(train_labels.shape) > 1:
        num_classes = train_labels.shape[1]
    else:
        num_classes = train_labels.max() + 1
    num_classes = int(num_classes)

    logger.info("Using cuML for logistic regression")
    logger.info(
        f"Using metric {metric_types[0]} for determining the best hyperparameters"
    )

    best_stats, best_C = sweep_C_values(
        train_features=train_features_cls,
        train_labels=train_labels,
        test_data_loader=finetune_data_loader,
        metric_type=metric_types[0],
        num_classes=num_classes,
        train_dtype=train_dtype,
        train_features_device=train_features_device,
        max_train_iters=max_train_iters,
    )

    if not finetune_on_val:
        logger.info("Best parameter found, concatenating features")
        # undo index extraction, put finetune features back where they came from
        train_features_all = torch.zeros(
            (len(train_features) + len(finetune_features), *train_features[0].shape)
        )
        train_labels_all = torch.zeros(
            len(train_labels) + len(finetune_labels), dtype=train_labels.dtype
        )

        train_features_all[train_index] = train_features
        train_labels_all[train_index] = train_labels

        train_features_all[finetune_index] = finetune_features
        train_labels_all[finetune_index] = finetune_labels

        # train_features = torch.cat((train_features, finetune_features))
        # train_labels = torch.cat((train_labels, finetune_labels))
        train_features = train_features_all
        train_labels = train_labels_all

    logger.info("Training final model")
    logreg_metrics = [build_metric(m, num_classes=num_classes) for m in metric_types]
    logreg_metrics = torchmetrics.MetricCollection(
        {k: v for metric in logreg_metrics for k, v in metric.items()}
    )
    evals = train_and_evaluate(
        C=best_C,
        max_iter=max_train_iters,
        train_features=train_features_cls,
        train_labels=train_labels,
        logreg_metrics=logreg_metrics.clone(),
        test_data_loader=val_data_loader,
        eval_device=torch.cuda.current_device(),
        train_dtype=train_dtype,
        train_features_device=train_features_device,
        mri_sequences=mri_sequences,
    )

    best_stats = evals[1]["metrics"]

    best_stats["best_C"] = best_C

    logger.info(f"Log regression evaluation done in {int(time.time() - start)}s")
    return best_stats, train_features, train_labels, val_features, val_labels


def eval_log_regression_with_model(
    model,
    train_features=None,
    train_labels=None,
    train_dataset_str="ImageNet:split=TRAIN",
    val_dataset_str="ImageNet:split=VAL",
    finetune_dataset_str=None,
    autocast_dtype=torch.float,
    finetune_on_val=False,
    metric_types: List[MetricType] = [MetricType.MEAN_ACCURACY],
    train_dtype=torch.float64,
    train_features_device=_CPU_DEVICE,
    max_train_iters=DEFAULT_MAX_ITER,
    num_workers=0,
    mri_sequences=None,
):
    cudnn.benchmark = True

    transform = make_classification_eval_transform(
        resize_size=224, mean=(0.45), std=(0.225)
    )
    target_transform = None

    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=transform,
        target_transform=target_transform,
    )
    val_dataset = make_dataset(
        dataset_str=val_dataset_str,
        transform=transform,
        target_transform=target_transform,
    )
    if finetune_dataset_str is not None:
        finetune_dataset = make_dataset(
            dataset_str=finetune_dataset_str,
            transform=transform,
            target_transform=target_transform,
        )
    else:
        finetune_dataset = None

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_logreg, train_features, train_labels, val_features, val_labels = (
            eval_log_regression(
                train_features=train_features,
                train_labels=train_labels,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                finetune_dataset=finetune_dataset,
                metric_types=metric_types,
                batch_size=64,
                num_workers=num_workers,
                finetune_on_val=finetune_on_val,
                train_dtype=train_dtype,
                train_features_device=train_features_device,
                max_train_iters=max_train_iters,
                mri_sequences=mri_sequences,
            )
        )

    results_dict = format_results_logreg(results_dict_logreg)

    eval_split = next(
        part.split("=")[1] for part in val_dataset_str.split(":") if "split=" in part
    )
    logger.info(
        "\n".join(
            [
                "Training of the supervised logistic regression on frozen features completed.\n"
                f"{json.dumps(results_dict, indent=2)}",
                "obtained for C = {c:.6f}".format(c=results_dict["best_C"]),
                f"on the {eval_split} split.",
            ]
        )
    )

    torch.distributed.barrier()
    return (
        results_dict,
        train_features,
        train_labels,
        val_features,
        val_labels,
    )


def format_results_logreg(results_dict_logreg: dict) -> dict:
    results_dict = {"best_C": results_dict_logreg["best_C"]}
    for k, v in results_dict_logreg.items():
        if k != "best_C":
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    v = v.item()
                    results_dict[k] = v
                else:
                    _v = {f"{k}.{kk}": vv.item() for kk, vv in enumerate(v)}
                    results_dict[k] = v.mean().item()
                    results_dict.update(_v)

            elif isinstance(v, dict):
                # join the keys with a dot
                v = {f"{k}.{kk}": vv.item() for kk, vv in v.items()}
                results_dict.update(v)
    return results_dict


def main(args):
    model, dataset_paths, autocast_dtype = setup_and_build_model(args)
    train_str, val_str, test_str = dataset_paths

    val_results_dict = eval_log_regression_with_model(
        model=model,
        train_dataset_str=train_str,
        val_dataset_str=val_str,
        finetune_dataset_str=None,
        autocast_dtype=autocast_dtype,
        finetune_on_val=False,
        metric_types=[
            MetricType.MATTHEWS_CORRELATION_COEFFICIENT,
            MetricType.MULTICLASS_AUROC,
            args.metric_type,
        ],
        train_dtype=as_torch_dtype(args.train_dtype),
        train_features_device=torch.device(args.train_features_device),
        max_train_iters=args.max_train_iters,
    )
    test_results_dict = {}
    if test_str is not None:
        test_results_dict = eval_log_regression_with_model(
            model=model,
            train_dataset_str=train_str,
            val_dataset_str=test_str,
            finetune_dataset_str=args.finetune_dataset_str,
            autocast_dtype=autocast_dtype,
            finetune_on_val=False,
            metric_types=[
                MetricType.MATTHEWS_CORRELATION_COEFFICIENT,
                MetricType.MULTICLASS_AUROC,
                args.metric_type,
            ],
            train_dtype=as_torch_dtype(args.train_dtype),
            train_features_device=torch.device(args.train_features_device),
            max_train_iters=args.max_train_iters,
        )
    results_dict = {"val": val_results_dict, "test": test_results_dict}
    logger.info(f"Results at manual evaluation: {results_dict}")

    eval_dir = os.path.join(args.output_dir, "eval", "manual")
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    return 0


if __name__ == "__main__":
    description = "DINOv2 logistic regression evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
