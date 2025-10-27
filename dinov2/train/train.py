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
from itertools import chain, combinations
from pathlib import Path

import torch
import torch.distributed
from fvcore.common.checkpoint import PeriodicCheckpointer

import dinov2.distributed as distributed
from dinov2.data import (
    DataAugmentationDINO,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
)
from dinov2.eval.log_regression import (
    DEFAULT_MAX_ITER as DEFAULT_MAX_ITER_LOG_REGRESSION,
)
from dinov2.eval.log_regression import eval_log_regression_with_model
from dinov2.eval.metrics import MetricType
from dinov2.eval.setup import get_autocast_dtype
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.models.vision_transformer import GliomaDinoViT
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup
from dinov2.utils.dtype import as_torch_dtype
from dinov2.utils.utils import CosineScheduler, load_pretrained_weights
from dinov2.visualization.train.vis_loss import (
    gather_missing_sequence_results,
    vis_loss_and_metrics,
)

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_eval_all_sequences(cfg, model, iteration):
    mri_sequences = ["t1", "t1c", "t2", "flair"]

    mri_sequence_combinations = list(powerset(mri_sequences, min_size=2, max_size=4))

    mri_sequence_combinations += ["random"]  # add random combination

    for mri_sequence_combination in mri_sequence_combinations[::-1]:
        mri_sequence_combination_str = "-".join(mri_sequence_combination)
        mri_sequence_combination_str = f"_mri_sequences-{mri_sequence_combination_str}"

        do_eval(
            cfg,
            model,
            f"{iteration}{mri_sequence_combination_str}",
            mri_sequence_combination,
        )


def powerset(iterable, min_size=0, max_size=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_size, max_size + 1)
    )


def do_eval(cfg, model, iteration, cur_mri_sequences=None):
    logger.info("#" * 40)
    logger.info(f"Starting evaluation at iteration {iteration}")

    val_results_dict, train_features, train_labels, val_features, val_labels = (
        eval_log_regression_with_model(
            # dino head is not used in the evaluation
            model=model.teacher.backbone,
            # needs supervised dataset_str
            train_dataset_str=cfg.evaluation.train_dataset_path,
            val_dataset_str=cfg.evaluation.val_dataset_path,
            finetune_dataset_str=None,
            autocast_dtype=get_autocast_dtype(cfg),
            finetune_on_val=False,
            metric_types=[MetricType(mt) for mt in cfg.evaluation.metric_types],
            train_dtype=as_torch_dtype("float32"),
            max_train_iters=DEFAULT_MAX_ITER_LOG_REGRESSION,
            num_workers=15,
            mri_sequences=cur_mri_sequences,
        )
    )

    results_dict = {"val": val_results_dict}
    logger.info(f"Results at iteration {iteration}: {results_dict}")

    eval_dir = os.path.join(cfg.train.output_dir, "eval", iteration)
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # freeze backbone for the first freeze_backbone_epochs
    freeze_backbone_epochs = cfg.optim.freeze_backbone_epochs
    if freeze_backbone_epochs > 0:
        logger.info(f"Freezing backbone for {freeze_backbone_epochs} epochs.")
        for p in model.student.backbone.parameters():
            p.requires_grad = False

    # checkpointer
    checkpointer = FSDPCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=cfg.train.saveckp_freq * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size

    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=n_tokens // 2,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,  # // 2,
        local_crops_size=cfg.crops.local_crops_size,  # // 2,
        intensity_aug_name=cfg.crops.intensity_aug,
        crop_from_tumor_foreground=cfg.crops.crop_from_tumor_foreground,
        max_blur_radius=cfg.crops.max_blur_radius,
        gamma_range=cfg.crops.gamma_range,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        mask_per_channel=cfg.ibot.mask_per_channel,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # append whether to append segmentation map to train.dataset_path

    cfg.train.dataset_path = (
        cfg.train.dataset_path
        + f":append_label_mask={cfg.crops.crop_from_tumor_foreground}"
    )
    # append percenage labels
    cfg.train.dataset_path = (
        cfg.train.dataset_path + f":percentage_labels={cfg.train.percentage_labels}"
    )

    # setup data loader
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if not isinstance(model.student.backbone.module, GliomaDinoViT):
        assert len(dataset.mri_sequences) in [
            1,
            3,
        ], (
            f"Only 1 or 3 MRI sequences are supported for regular {type(model.student.backbone.module)}."
        )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            logger.info(f"Reached max iteration {max_iter} ({iteration} > {max_iter}).")
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # unfreeze backbone after freeze_backbone_epochs
        if iteration == (freeze_backbone_epochs * OFFICIAL_EPOCH_LENGTH):
            logger.info("Unfreezing backbone.")
            for p in model.student.backbone.parameters():
                p.requires_grad = True

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {
            k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()
        }

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info(
                "NaN detected",
                [k for k, v in loss_dict_reduced.items() if math.isnan(v)],
            )
            raise AssertionError
        losses_reduced = sum(
            loss for k, loss in loss_dict_reduced.items() if "labeled_acc" not in k
        )
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(
            backbone_frozen=float(
                iteration < freeze_backbone_epochs * OFFICIAL_EPOCH_LENGTH
            )
        )
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if (
            cfg.evaluation.eval_period_iterations > 0
            and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
        ):
            do_eval(cfg, model, f"training_{iteration}")
            do_test(cfg, model, f"training_{iteration}")
            vis_loss_and_metrics(cfg)
            # keep best checkpoint only
            keep_only_best_ckpt(cfg)

            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()

    # restore the best checkpoint
    restore_best_ckpt(cfg, model)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def restore_best_ckpt(cfg, model, use_final=False):
    eval_dir = os.path.join(cfg.train.output_dir, "eval")
    if not use_final:
        best_metric_val, best_iter, _ = extract_best_checkpoint_iteration(eval_dir)
        if best_iter == -1:
            logger.info("No best checkpoint found.")
            return
        logger.info(f"Best val mcc: {best_metric_val} at iteration {best_iter}")
    else:
        best_iter = "final"
    backbone = model.teacher.backbone if isinstance(model, SSLMetaArch) else model
    load_pretrained_weights(
        backbone, os.path.join(eval_dir, best_iter, "teacher_checkpoint.pth"), "teacher"
    )
    return best_iter


def keep_only_best_ckpt(cfg):
    eval_dir = os.path.join(cfg.train.output_dir, "eval")
    best_val_acc, best_iter, all_results = extract_best_checkpoint_iteration(eval_dir)
    logger.info(f"Best val acc: {best_val_acc} at iteration {best_iter}")

    for iter, _ in all_results:
        if iter != best_iter:
            teacher_ckp_fp = os.path.join(eval_dir, iter, "teacher_checkpoint.pth")

            if os.path.exists(teacher_ckp_fp):
                os.remove(teacher_ckp_fp)
                logger.info(f"Removing checkpoint {teacher_ckp_fp}")


def extract_best_checkpoint_iteration(eval_dir):
    all_results = []
    for iteration in os.listdir(eval_dir):
        if not iteration.startswith("training_"):
            continue
        if not (os.path.isdir(os.path.join(eval_dir, iteration))):
            continue
        results_fp = os.path.join(eval_dir, iteration, "results.json")
        with open(results_fp, "r") as f:
            results = json.load(f)

        all_results.append((iteration, results))
    logger.info(f"Number of results: {len(all_results)}")

    metric_key = "mcc"
    if len(all_results) == 0:
        return -1, -1, all_results
    best_metric_val = all_results[0][1]["val"][metric_key]
    best_iter = all_results[0][0]
    for iter, results in all_results:
        cur_metric_val = results["val"].get(metric_key, -1)
        if cur_metric_val > best_metric_val:
            best_metric_val = cur_metric_val
            best_iter = iter
    return best_metric_val, best_iter, all_results


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    train_out_dir = Path(cfg.train.output_dir).parent

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        best_iter = restore_best_ckpt(cfg, model, use_final=False)

        if args.eval == "missing_sequences":
            do_eval_all_sequences(cfg, model, "best")
            gather_missing_sequence_results(train_out_dir)
        else:
            do_eval(cfg, model, f"manual_{best_iter}")
        return

    training_out = do_train(cfg, model, resume=not args.no_resume)
    logger.info("Training completed.")
    logger.info(f"Training results: {json.dumps(training_out, indent=2)}")

    do_eval_all_sequences(cfg, model, "final")
    do_test(cfg, model, "final")
    vis_loss_and_metrics(cfg)
    gather_missing_sequence_results(train_out_dir)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
