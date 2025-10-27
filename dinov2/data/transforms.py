# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import numpy as np
import torch
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
    Transform,
)
from scipy.interpolate import BSpline, splrep
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(
            kernel_size=9, sigma=(radius_min, radius_max)
        )
        super().__init__(transforms=[transform], p=keep_p)


class RescaleImage:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # Convert to tensor
            image = torch.from_numpy(image)
        elif torch.is_tensor(image):
            pass
        else:
            raise TypeError("Input should be of type numpy.ndarray or torch.Tensor")

        # Rescale the tensor to [0, 1]
        min_val = (
            image.reshape(image.shape[0], -1)
            .min(dim=1)[0]
            .reshape(-1, *(1,) * (image.ndim - 1))
        )
        max_val = (
            image.reshape(image.shape[0], -1)
            .max(dim=1)[0]
            .reshape(-1, *(1,) * (image.ndim - 1))
        )
        return (image - min_val) / (max_val - min_val)


class MaybeToTensor(transforms.PILToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        if isinstance(pic, np.ndarray):
            pic = torch.from_numpy(pic)
            return pic.permute(2, 0, 1)

        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RandResizeSpatialCrop(Transform):
    def __init__(
        self,
        crop_size: int,
        scale: tuple[float] = (0.75, 1),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        # stretch by scale
        # self.affine = RandAffine(
        #     prob=1.0,
        #     translate_range=0,
        #     rotate_range=0,
        #     scale_range=tuple(s - 1 for s in scale),
        #     # cache_grid=True,
        #     # spatial_size=crop_size,
        # )
        min_crop_size = int(crop_size * scale[0])
        max_crop_size = int(crop_size * scale[1])

        self.cropper = RandSpatialCrop(
            roi_size=min_crop_size,
            max_roi_size=max_crop_size,
            random_size=True,
            random_center=True,
            *args,
            **kwargs,
        )
        self.crop_size = crop_size

        self.interpolation_mode = interpolation
        self.antialias = antialias

    def __call__(self, data):
        # assuming shape is (C, H, W(, D))

        cropped_image = self.cropper(data)
        # Resize the cropped image to the original size
        resized_image = torch.nn.functional.interpolate(
            cropped_image.unsqueeze(0),
            size=self.crop_size,
            mode=self.interpolation_mode.value
            if cropped_image.ndim == 3
            else "trilinear",
            antialias=self.antialias and cropped_image.ndim == 3,
        ).squeeze(0)
        return resized_image


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    # return transforms.Normalize(mean=mean, std=std)
    return NormalizeIntensity(subtrahend=mean, divisor=std, nonzero=True)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        transforms.RandomResizedCrop(
            (crop_size, crop_size),
            scale=(0.75, 1),
            interpolation=interpolation,
            antialias=True,
        ),
    ]

    if hflip_prob > 0.0 and False:
        transforms_list.append(
            # transforms.RandomHorizontalFlip(hflip_prob)
            RandFlip(prob=hflip_prob, spatial_axis=1)
        )
    transforms_list.extend(
        [
            MaybeToTensor(),
            # RescaleImage(),
            ScaleIntensity(minv=0, maxv=1, channel_wise=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


def make_glioma_classification_train_transform(
    *, crop_size: int = 224, **train_transform_kwargs
):
    glioma_train_transforms = []
    train_transform = make_classification_train_transform(
        crop_size=crop_size, **train_transform_kwargs
    )
    glioma_train_transforms.append(train_transform)

    intensity_augmentations = Compose(
        [
            RandAdjustContrast(prob=0.5, gamma=(0.5, 1.5)),
            ScaleIntensity(minv=0, maxv=1),
        ]
    )

    affine_kwargs = {
        "prob": 1.0,
        "translate_range": 30,
        "rotate_range": np.deg2rad(180),
        "scale_range": 0.0,
        "cache_grid": True,
        "spatial_size": (crop_size, crop_size),
    }
    affine_augmentations = RandAffine(
        mode="bilinear",
        **affine_kwargs,
    )
    glioma_train_transform = Compose(
        [
            intensity_augmentations,
            affine_augmentations,
        ]
        + glioma_train_transforms
    )
    return glioma_train_transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(
            (resize_size, resize_size), interpolation=interpolation, antialias=True
        ),
        transforms.CenterCrop((crop_size, crop_size)),
        MaybeToTensor(),
        Resize(
            resize_size,
            #    mode=interpolation,
            # anti_aliasing=True,
            size_mode="longest",
        ),
        CenterSpatialCrop(crop_size),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def make_segmentation_train_transforms(
    *,
    resize_size: int = 448,
    vflip_prob: float = 0.25,
    hflip_prob: float = 0.25,
    rot_deg: float = 90,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    train_transforms_list = [
        transforms.Resize(
            (resize_size, resize_size), interpolation=interpolation, antialias=True
        )
    ]
    target_transforms_list = [
        transforms.Resize(
            (resize_size, resize_size),
            antialias=True,
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )
    ]
    if vflip_prob > 0:
        train_transforms_list.append(transforms.RandomVerticalFlip(vflip_prob))
        target_transforms_list.append(transforms.RandomVerticalFlip(vflip_prob))
    if hflip_prob > 0:
        train_transforms_list.append(transforms.RandomVerticalFlip(hflip_prob))
        target_transforms_list.append(transforms.RandomVerticalFlip(hflip_prob))
    if rot_deg > 0:
        train_transforms_list.append(transforms.RandomRotation(rot_deg))
        target_transforms_list.append(transforms.RandomRotation(rot_deg))

    train_transforms_list.extend(
        [
            MaybeToTensor(),
            RescaleImage(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    target_transforms_list.append(MaybeToTensor())

    return (
        transforms.Compose(train_transforms_list),
        transforms.Compose(target_transforms_list),
    )


def make_segmentation_eval_transforms(
    *,
    resize_size: int = 448,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    train_transforms_list = [
        transforms.Resize(
            (resize_size, resize_size), interpolation=interpolation, antialias=True
        ),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    target_transform_list = [
        transforms.Resize(
            (resize_size, resize_size),
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
            antialias=True,
        ),
        MaybeToTensor(),
    ]
    return (
        transforms.Compose(train_transforms_list),
        transforms.Compose(target_transform_list),
    )


class BSPlines(torch.nn.Module):
    def __init__(self, n_knots=5, prob=0.5):
        super().__init__()
        self.prob = prob
        self.n_knots = n_knots
        self.k = 3  # cubic spline
        assert n_knots > self.k, f"n_knots must be greater than {self.k}"

    def forward(self, imgs: torch.Tensor):
        if np.random.rand() > self.prob:
            return imgs

        x = np.linspace(0, 1, self.n_knots)
        y = np.random.rand(self.n_knots)

        # rescale to 0,1
        # y = (y - y.min()) / (y.max() - y.min())
        y[0] = 0
        y[-1] = 1

        # if np.random.rand() > 0.5:
        #     y = y[::-1]
        t, c, k = splrep(x, y)
        bspline = BSpline(t, c, k)
        imgs_aug_np = bspline(imgs.cpu().numpy())

        imgs_aug = torch.from_numpy(imgs_aug_np).to(
            device=imgs.device, dtype=imgs.dtype
        )

        # rescale to 0,1
        # imgs_aug = imgs_aug - imgs_aug.min()
        # imgs_aug = imgs_aug / imgs_aug.max()

        # mask background to 0
        imgs_aug[imgs == 0] = 0

        return imgs_aug


def ClampTransform(min, max):
    def f(x):
        return torch.clamp(x, min, max)

    return f


class RandomResizeForegroundCrop(torch.nn.Module):
    def __init__(
        self,
        spatial_size: tuple[int, int],
        scale: tuple[float, float],
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        if not isinstance(spatial_size, tuple):
            spatial_size = (spatial_size, spatial_size)
        self.resizer = transforms.Resize(
            spatial_size,
            interpolation=interpolation,
            antialias=antialias,
        )
        self.ratio_lower = ratio[0]
        self.ratio_upper = ratio[1]

        self.scale_lower = scale[0]
        self.scale_upper = scale[1]

    def forward(self, img):
        """img in shape (C, H, W)"""

        # get the foreground mask in last channel
        img, fg_mask = img[:-1], img[-1] > 0

        # get the list of coordinates of the foreground pixels
        fg_coords = torch.nonzero(fg_mask)

        # randomly sample a scale
        scale = np.random.uniform(self.scale_lower, self.scale_upper)

        # get the crop size
        crop_size = [s * scale for s in img.shape[1:]]

        # scale by random ratio
        ratio = np.random.uniform(self.ratio_lower, self.ratio_upper)
        # multiply the width by the ratio but keep the height the same
        crop_size = [int(crop_size[0] * ratio), int(crop_size[1])]

        # randomly sample a coordinate
        coord = fg_coords[np.random.randint(0, fg_coords.size(0))]

        # crop the image with the center at the sampled coordinate

        # get the crop coordinates
        y_crop_min = max(coord[0] - crop_size[0] // 2, 0)
        x_crop_min = max(coord[1] - crop_size[1] // 2, 0)

        x_crop_max = x_crop_min + crop_size[0]
        y_crop_max = y_crop_min + crop_size[1]

        x_crop_max = min(x_crop_max, img.shape[1])
        y_crop_max = min(y_crop_max, img.shape[2])

        # slice the image
        img_cropped = img[:, y_crop_min:y_crop_max, x_crop_min:x_crop_max]

        # Ensure that the resized image maintains the correct dimensions
        img_resized = self.resizer(img_cropped)

        return img_resized
