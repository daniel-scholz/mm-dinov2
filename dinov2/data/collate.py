# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    mask_per_channel=True,
    n_tokens=None,
    mask_generator=None,
):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )

    collated_local_crops = torch.stack(
        [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )

    collated_cls_labels = torch.stack([s[1] for s in samples_list])

    collated_cls_labels = torch.stack([s[1] for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    nc = collated_global_crops.size(1)
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    # create masks list, each element is a tensor of shape (H,W) with True/False values
    # H*W is the number of tokens in the image

    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(N * random.uniform(prob_min, prob_max)))
            )
        )
        upperbound += int(
            N * prob_max
        )  # store the upperbound for the number of masked tokens
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)

    if mask_per_channel:
        upperbound_full = 0
        max_nc_masked = 1
        masks_list_full = []
        # for each crop sample, mask out a random number of channels
        for i in range(0, B):
            # sample a random number of channels to mask out
            n_masked_channels = max_nc_masked  # random.randint(0, max_nc_masked)

            # create a mask tensor with shape (nc, N)
            mask_full = torch.cat(
                [
                    torch.zeros(nc - n_masked_channels, N, dtype=torch.bool),
                    torch.ones(n_masked_channels, N, dtype=torch.bool),
                ],
                dim=0,
            )

            # randomly choose which channels to mask out
            mask_full = mask_full[torch.randperm(nc)]

            masks_list_full.append(mask_full.flatten())
            upperbound_full += n_masked_channels * N

        collated_masks_full = torch.stack(masks_list_full).flatten(1)

        collated_masks = collated_masks.repeat(1, nc)
        upperbound *= nc
        upperbound += upperbound_full
        # final masks are the union of the two masks, so patch masks and full channel masks
        collated_masks = collated_masks_full | collated_masks

    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )
    # if any of the crops contain nan, fill with min
    collated_global_crops[torch.isnan(collated_global_crops)] = (
        collated_global_crops.min()
    )
    collated_local_crops[torch.isnan(collated_local_crops)] = collated_local_crops.min()

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "collate_cls_labels": collated_cls_labels,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
