# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import monai.data.meta_obj
import monai.transforms
from monai.transforms.intensity.array import RandAdjustContrast, RandHistogramShift
from torchvision import transforms

from dinov2.data.monai_transforms.rc_aug import RandConvAugmentation
from dinov2.data.transforms import BSPlines, ClampTransform

from .transforms import (
    GaussianBlur,
    MaybeToTensor,
    RandomResizeForegroundCrop,
    make_normalize_transform,
)

monai.data.meta_obj.set_track_meta(False)
logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        max_blur_radius=1,
        gamma_range=(0.75, 1.5),
        global_crops_size=224,
        local_crops_size=96,
        intensity_aug_name: str = "color_jittering",
        crop_from_tumor_foreground: bool = False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        self.max_blur_radius = max_blur_radius
        self.gamma_range = gamma_range

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        RandResizeCropClass = (
            RandomResizeForegroundCrop
            if crop_from_tumor_foreground
            else transforms.RandomResizedCrop
        )
        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                RandResizeCropClass(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # needs clamping because of the bicubic interpolation
                ClampTransform(0, 1),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                RandResizeCropClass(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                ClampTransform(0, 1),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        if intensity_aug_name == "color_jittering":
            # color distorsions / blurring
            intensity_aug = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    # transforms.RandomGrayscale(p=0.2),
                ]
            )
        elif intensity_aug_name == "rc":
            intensity_aug = RandConvAugmentation(
                spatial_dims=2,
                rc_type="gin",
                upsampling_size=512,
                n_layers=4,
                n_hidden_chans=4,
            )
        elif intensity_aug_name == "bspline":
            intensity_aug = BSPlines(n_knots=5, prob=0.75)
        elif intensity_aug_name == "histogram":
            intensity_aug = RandHistogramShift(prob=1.0, num_control_points=(3, 10))
        else:

            def intensity_aug(x):
                return x

            # raise ValueError(f"Unknown intensity augmentation: {intensity_aug}")

        global_transfo1_extra = GaussianBlur(p=1.0, radius_max=self.max_blur_radius)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1, radius_max=1),
                # transforms.RandomSolarize(threshold=0.5, p=0.2),
                RandAdjustContrast(prob=1.0, gamma=self.gamma_range),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5, radius_max=self.max_blur_radius)

        # normalization
        self.normalize = transforms.Compose(
            [
                MaybeToTensor(),
                # close to the values used in the original DINO code
                # for ImageNet for grayscale images
                make_normalize_transform(mean=0.45, std=0.22),
            ]
        )

        self.global_transfo1 = transforms.Compose(
            [
                intensity_aug,
                global_transfo1_extra,
                self.normalize,
            ]
        )
        self.global_transfo2 = transforms.Compose(
            [
                intensity_aug,
                global_transfo2_extra,
                self.normalize,
            ]
        )
        self.local_transfo = transforms.Compose(
            [
                intensity_aug,
                local_transfo_extra,
                self.normalize,
            ]
        )

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        # for k, v in output.items():
        #     for i, img in enumerate(v):

        #         save_image(img, f"test_{k}_{i}.png")

        return output
