# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import math
import os
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel

import dinov2.distributed as distributed
from dinov2.data import SamplerType
from dinov2.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    make_classification_eval_transform,
    make_glioma_classification_train_transform,
)
from dinov2.eval.classification.utils import (
    LinearPostprocessor,
    setup_linear_classifiers,
)
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import (
    ModelWithIntermediateLayers,
    apply_method_to_nested_values,
    bitfit,
    collate_fn_3d,
    evaluate,
    make_data_loaders,
    make_datasets,
    str2bool,
    trainable_parameters,
)
from dinov2.logging import MetricLogger

logger = logging.getLogger("dinov2")


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
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--val-epochs",
        type=int,
        help="Number of epochs for testing on validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-epochs",
        type=int,
        help="Number of epochs between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--optimizer", default="sgd", type=str, help="Optimizer to use [adam, sgd]"
    )
    parser.add_argument(
        "--optimizer", default="sgd", type=str, help="Optimizer to use [adam, sgd]"
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
    )
    parser.add_argument("--n-last-blocks", nargs="+", type=int)
    parser.add_argument("--avgpools", nargs="+", type=str2bool)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--metric-types",
        type=MetricType,
        nargs="+",
        choices=list(MetricType),
        help="Validation metrics",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--fine-tune",
        type=bool,
        help="Whether to finetune the backbone.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        help="The name of the backbone model to use [dinov2, vit-large-imagenet21k]",
    )
    parser.add_argument(
        "--peft",
        type=str,
        help="The name of the peft technique to use [lora]",
    )

    parser.add_argument("--image-size", type=int, help="The size of input image")

    parser.add_argument(
        "--grad-clip-val",
        type=float,
        default=0.0,
        help="Max norm of the gradients for clipping (0.0 to disable)",
    )
    parser.set_defaults(
        train_dataset_str="NIHChestXray:split=TRAIN",
        val_dataset_str=None,
        test_dataset_str="NIHChestXray:split=TEST",
        epochs=10,
        val_epochs=None,
        batch_size=128,
        num_workers=8,
        epoch_length=None,
        save_checkpoint_frequency=5,
        eval_period_epochs=5,
        learning_rates=[1e-3, 5e-3, 1e-2, 5e-2],
        backbone_learning_rate=1e-5,
        n_last_blocks=[1, 4],
        avgpools=[True, False],
        metric_types=[
            MetricType.MULTICLASS_AUROC,
        ],
        classifier_fpath=None,
        fine_tune=False,
        backbone="dinov2",
        peft=None,
        image_size=224,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metric_types,
    metrics_file_path,
    num_of_classes,
    iteration,
    prefixstring="",
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    labels = list(data_loader.dataset.class_names)
    metric = [
        build_metric(mt, num_classes=num_of_classes, labels=labels)
        for mt in metric_types
    ]
    metric = torchmetrics.MetricCollection({k: v for m in metric for k, v in m.items()})

    postprocessors = {
        k: LinearPostprocessor(v)
        for k, v in linear_classifiers.classifiers_dict.items()
    }
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    logger.info("")
    results_dict = {}
    max_score = -np.inf
    best_classifier = ""
    eval_metric = "mcc"  # str(list(metric)[0])

    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric[eval_metric].item() > max_score
        ) or classifier_string == best_classifier_on_val:
            max_score = metric[eval_metric].item()
            best_classifier = classifier_string

    results_dict["best_classifier"] = {
        "name": best_classifier,
        "results": apply_method_to_nested_values(
            results_dict_temp[best_classifier], method_name="item", nested_types=(dict)
        ),
    }

    logger.info(f"best classifier: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"{prefixstring}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


def eval_linear(
    *,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of epochs, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric_types,
    num_of_classes,
    resume=True,
    classifier_fpath=None,
    is_multilabel=True,
    grad_clip_val=0.0,  # New parameter for gradient clipping value
):
    if feature_model.fine_tune:
        checkpointer = Checkpointer(
            nn.Sequential(feature_model, linear_classifiers),
            output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        checkpointer = Checkpointer(
            linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler
        )
    start_iter = (
        checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get(
            "iteration", 0
        )
        + 1
    )

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, checkpoint_period, max_iter=max_iter
    )
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"
    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)

        # calculate loss
        if is_multilabel:
            losses = {}
            batch_size = labels.shape[0]
            for k, v in outputs.items():
                per_class_loss = torch.tensor([0.0], device=torch.cuda.current_device())
                for batch_index in range(batch_size):  # Loop through each batch
                    batch_predictions = v[batch_index]
                    batch_labels = labels[batch_index]
                    for index, class_ in enumerate(
                        batch_predictions
                    ):  # Loop through each class prediciton
                        per_class_loss += nn.BCEWithLogitsLoss()(
                            class_.float(), batch_labels[index].float()
                        )
                    losses[f"loss_{k}"] = per_class_loss / len(
                        batch_labels
                    )  # Take average of all binary classification losses
        else:
            # add weighting per class
            label_distribution = torch.bincount(
                labels, minlength=num_of_classes
            ).float() / labels.size(0)
            # inverse document frequency weighting
            class_weights = -torch.log(label_distribution + 1e-6)

            loss_fn = (
                nn.BCEWithLogitsLoss()
                if num_of_classes == 1
                else nn.CrossEntropyLoss(weight=class_weights)
            )
            losses = {f"loss_{k}": loss_fn(v, labels) for k, v in outputs.items()}

        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()

        if grad_clip_val > 0:
            params_to_clip = []
            for group in optimizer.param_groups:
                params_to_clip.extend(group["params"])
            if len(params_to_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_val)

        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            logger.info(f"lr {optimizer.param_groups[0]['lr']}")
            if len(optimizer.param_groups) > 1:
                metric_logger.update(lr_backbone=optimizer.param_groups[-1]["lr"])
                logger.info(f"lr_backbone {optimizer.param_groups[-1]['lr']}")

        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save(
                        "running_checkpoint_linear_eval", iteration=iteration
                    )
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if (
            eval_period > 0
            and (iteration % eval_period == 0 or iteration == 1)
            and iteration != max_iter
        ):
            _ = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
                metric_types=metric_types,
                num_of_classes=num_of_classes,
                iteration=iteration,
            )
            torch.cuda.synchronize()

        iteration = iteration + 1

    with open(metrics_file_path, "r") as f:
        lines = f.readlines()
    iterations = [int(line.strip().split(" ")[1]) for line in lines[::3]]
    results = [json.loads(line) for line in lines[1::3]]
    all_results = dict(zip(iterations, results))
    eval_metric = "mcc"

    best_iter = None
    max_score = -np.inf
    for iteration, result in all_results.items():
        score = result["best_classifier"]["results"][eval_metric]
        if score > max_score:
            max_score = score
            best_iter = iteration

    logger.info(f"Best classifier on val: {best_iter}")
    # restore checkpoint

    best_model_fp = os.path.join(output_dir, f"model_{str(best_iter - 1).zfill(7)}.pth")
    # print average weights of feature model to sanity check restoration
    avg_weight = torch.mean(
        torch.stack([torch.mean(torch.abs(p)) for p in feature_model.parameters()])
    )
    logger.info(f"Average weight of feature model: {avg_weight}")

    checkpointer.resume_or_load(best_model_fp, resume=False)
    logger.info(f"Restored best model from {best_model_fp}")

    avg_weight = torch.mean(
        torch.stack([torch.mean(torch.abs(p)) for p in feature_model.parameters()])
    )
    logger.info(f"Average weight of feature model after restoration: {avg_weight}")

    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
        metric_types=metric_types,
        num_of_classes=num_of_classes,
        iteration=iteration,
    )

    results_dict = {"val": val_results_dict}

    return results_dict, feature_model, linear_classifiers, iteration


def run_eval_linear(
    model,
    output_dir,
    train_dataset_str,
    test_dataset_str,
    batch_size,
    epochs,
    val_epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_epochs,
    learning_rates,
    backbone_learning_rate,
    n_last_blocks_list,
    avgpools,
    autocast_dtype,
    val_dataset_str=None,
    resume=True,
    classifier_fpath=None,
    metric_types=MetricType.MULTILABEL_AUROC,
    fine_tune=False,
    backbone="dinov2",
    peft=None,
    image_size=224,
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str is None:
        raise ValueError("Test dataset cannot be None")

    if "resnet" in backbone or "vgg" in backbone or "dense" in backbone:
        n_last_blocks_list = [1]
        avgpools = [False]

    IMAGENET_DEFAULT_MEAN_MEAN = np.mean(IMAGENET_DEFAULT_MEAN)
    IMAGENET_DEFAULT_STD_MEAN = np.mean(IMAGENET_DEFAULT_STD)
    train_transform = make_glioma_classification_train_transform(
        crop_size=image_size,
        mean=IMAGENET_DEFAULT_MEAN_MEAN,
        std=IMAGENET_DEFAULT_STD_MEAN,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=IMAGENET_DEFAULT_MEAN_MEAN,
        std=IMAGENET_DEFAULT_STD_MEAN,
    )
    train_dataset, val_dataset, test_dataset = make_datasets(
        train_dataset_str=train_dataset_str,
        val_dataset_str=val_dataset_str,
        test_dataset_str=test_dataset_str,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    batch_size = (
        train_dataset.__len__() if batch_size > train_dataset.__len__() else batch_size
    )
    num_of_classes = test_dataset.get_num_classes()
    num_of_classes = 1 if num_of_classes == 2 else num_of_classes
    is_multilabel = test_dataset.is_multilabel()
    is_3d = test_dataset.is_3d()
    collate_fn = None if not is_3d else collate_fn_3d

    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(
        model, n_last_blocks, autocast_ctx, is_3d=is_3d, fine_tune=fine_tune
    )

    sample_input = train_dataset[0][0][0] if is_3d else train_dataset[0][0]
    sample_input = sample_input.unsqueeze(0).cuda()
    sample_output = feature_model.forward_(sample_input)

    if epoch_length is None:
        epoch_length = math.ceil(train_dataset.__len__() / batch_size)
    eval_period_epochs_ = eval_period_epochs * epoch_length
    checkpoint_period = save_checkpoint_frequency * epoch_length

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output=sample_output,
        n_last_blocks_list=n_last_blocks_list,
        learning_rates=learning_rates,
        avgpools=avgpools,
        num_classes=num_of_classes,
        is_3d=is_3d,
    )

    if val_epochs is not None:
        max_iter = epoch_length * val_epochs
    else:
        max_iter = epoch_length * epochs

    if fine_tune:
        logger.info("Finetuning backbone")
        optim_param_groups.append(
            {"params": feature_model.parameters(), "lr": backbone_learning_rate}
        )
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    elif peft == "lora":
        logger.info("Using LoRA for fine tuning")
        config = LoraConfig(
            r=48,
            lora_alpha=16,
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["classifier"],
        )
        feature_model = get_peft_model(feature_model, config)
        tp, ap = trainable_parameters(feature_model)
        logger.info(
            f"LoRA trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}"
        )

        lr_ = backbone_learning_rate
        optim_param_groups.append({"params": feature_model.parameters(), "lr": lr_})
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    elif peft == "bitfit":
        feature_model = bitfit(feature_model)

        tp, ap = trainable_parameters(feature_model)
        logger.info(
            f"BitFit trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}"
        )
        optim_param_groups.append(
            {"params": feature_model.parameters(), "lr": backbone_learning_rate}
        )
        checkpoint_model = nn.Sequential(feature_model, linear_classifiers)
    else:
        checkpoint_model = linear_classifiers
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(optim_param_groups, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_iter, eta_min=0
    )
    checkpointer = Checkpointer(
        checkpoint_model, output_dir, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get(
            "iteration", 0
        )
        + 1
    )

    sampler_type = SamplerType.INFINITE
    train_data_loader, val_data_loader, _ = make_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        sampler_type=sampler_type,
        seed=seed,
        start_iter=start_iter,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=checkpoint_period // 2,
        eval_period=eval_period_epochs_,
        metric_types=metric_types,
        num_of_classes=num_of_classes,
        resume=resume,
        classifier_fpath=classifier_fpath,
        is_multilabel=is_multilabel,
    )

    logger.info("Test Results Dict " + json.dumps(results_dict, indent=2))
    # save to file
    with open(metrics_file_path, "a") as f:
        f.write(
            "ITER: " + str(iteration) + " " + val_data_loader.dataset.split.value + "\n"
        )
        f.write(json.dumps(results_dict) + "\n")

    # write to separate file
    metrics_file_path = os.path.join(output_dir, "results_eval_linear_final.json")
    with open(metrics_file_path, "w") as f:
        f.write(json.dumps(results_dict, indent=2))

    return results_dict


def main(args):
    model, _, autocast_dtype = setup_and_build_model(args)
    train_str, val_str, test_str = (
        args.train_dataset_str,
        args.val_dataset_str,
        args.test_dataset_str,
    )
    run = partial(
        run_eval_linear,
        model=model,
        train_dataset_str=train_str,
        val_dataset_str=val_str,
        test_dataset_str=test_str,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_epochs=args.val_epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_epochs=args.eval_period_epochs,
        learning_rates=args.learning_rates,
        backbone_learning_rate=args.backbone_learning_rate,
        n_last_blocks_list=args.n_last_blocks,
        avgpools=args.avgpools,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        metric_types=args.metric_types,
        fine_tune=args.fine_tune,
        backbone=args.backbone,
        peft=args.peft,
        image_size=args.image_size,
    )

    run(
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    args_parser = get_args_parser(
        description="DINOv2 linear classification evaluation script"
    )
    args = args_parser.parse_args()
    main(args)
