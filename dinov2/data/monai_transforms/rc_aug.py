from functools import partial
from typing import Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlassOfGIN(nn.Module):
    """Single GIN transformation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_hidden_chans,
        spatial_dims,
        n_layers,
        upsampling_fn,
        downsampling_fn,
        rng: torch.Generator,
        rotationally_symmetric: bool = False,
        normalization: Literal["fro", "minmax"] = "minmax",
        alpha_range: tuple[float, float] = (0.0, 1.0),
        kernel_size=3,
        padding=1,
        **conv_kwargs,
    ):
        super().__init__()
        self.rng = rng
        self.spatial_dims = spatial_dims

        self.upsampling_fn = upsampling_fn
        self.downsampling_fn = downsampling_fn
        layers = []
        for i_iter in range(n_layers):
            # intialize conv layer
            rand_conv = nn.Conv2d(
                in_channels=in_channels if i_iter == 0 else n_hidden_chans,
                out_channels=n_hidden_chans if i_iter != n_layers - 1 else out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=0,
                padding_mode="replicate",
                **conv_kwargs,
            )
            # init weights with N(0,1)
            self.init_rand_convs(rand_conv, rotationally_symmetric)

            layers.append(rand_conv)
            if i_iter != n_layers - 1:
                layers.append(self.non_linearity())  # default value: 0.01
                # layers.append(nn.SiLU())
        # print(f"Layers in GIN: {layers}")
        self.transform = nn.Sequential(*layers)
        self.rand_interpolate = RandInterpolate(alpha_range=alpha_range, rng=self.rng)

        if normalization == "fro":
            self.normalize_image = self.normalize_image_fro
        elif normalization == "minmax":
            self.normalize_image = self.normalize_image_minmax
        else:
            raise ValueError(f"Invalid normalization: {normalization}")

    def non_linearity(self):
        return nn.LeakyReLU(negative_slope=1e-2)

    def init_rand_convs(
        self,
        rand_conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        rotationally_symmetric: bool,
    ):
        in_channels, out_channels = (
            rand_conv.weight.size(1),
            rand_conv.weight.size(0),
        )

        if not rotationally_symmetric:
            nn.init.normal_(
                rand_conv.weight,
                mean=0,
                std=1,
                # generator=self.rng,
            )

        if rotationally_symmetric:
            with torch.no_grad():
                if isinstance(rand_conv, nn.Conv1d):
                    raise NotImplementedError("1D not implemented")
                if isinstance(rand_conv, nn.Conv2d):
                    # Define the size of the image
                    rand_conv_symmetric = sample_symmetric_rand_conv_2d(
                        rand_conv, in_channels, out_channels, self.rng
                    )

                    # assign weight to rand_conv
                    rand_conv.weight.data = rand_conv_symmetric

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Augment, interpolate, and normalize the original image."""
        # GIN augmented the original image
        img_size = img.shape[-self.spatial_dims :]

        gin_img = (img - 0.5) * 2  # normalize to [-1, 1]
        gin_img = self.upsampling_fn(gin_img)
        with torch.no_grad():
            gin_img = self.transform(gin_img)
        gin_img = self.downsampling_fn(gin_img, img_size)

        # normalize image to have same frobenius norm as original image
        gin_img = self.normalize_image(img, gin_img)

        # interpolate between original and augmented image
        # alpha is constant for all images in the batch to mimic scanner properties
        gin_img = self.rand_interpolate(img, gin_img)

        return gin_img

    def normalize_image_fro(
        self, og_img: torch.Tensor, gin_img: torch.Tensor
    ) -> torch.Tensor:
        gin_img = gin_img / torch.norm(gin_img, p="fro") * torch.norm(og_img, p="fro")
        return gin_img

    def normalize_image_minmax(
        self, og_img: torch.Tensor, gin_img: torch.Tensor
    ) -> torch.Tensor:
        """minmax normalize the batch to [0, 1] and scale to [-1, 1]"""

        gin_img = (gin_img - gin_img.min()) / (gin_img.max() - gin_img.min())

        return gin_img


class RandInterpolate(nn.Module):
    def __init__(
        self, rng: torch.Generator, alpha_range: tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        self.alpha: torch.Tensor

        self.alpha_range = alpha_range
        self.rng = rng
        self.register_buffer("alpha", self.sample_alpha(1))

    def sample_alpha(self, n_samples):
        """uniform randomly sample batch size many alphas between 0 and 1"""
        if self.alpha_range[0] != self.alpha_range[1]:
            # sample alpha from uniform distribution in range [alpha_range[0], alpha_range[1]]
            alpha = (
                torch.rand(n_samples, device=self.rng.device, generator=self.rng)
                * (self.alpha_range[1] - self.alpha_range[0])
                + self.alpha_range[0]
            )
        else:
            # set alpha to alpha_range[0]
            alpha = torch.ones(n_samples, device=self.rng.device) * self.alpha_range[0]

        return alpha

    def forward(self, og_img: torch.Tensor, gin_img: torch.Tensor) -> torch.Tensor:
        # mix the original image with the augmented image
        broadcast_ones = [1] * (og_img.ndim - 1)
        alpha = self.alpha.view(-1, *broadcast_ones).to(dtype=og_img.dtype)

        # interpolate between original and augmented image
        gin_img = alpha * gin_img + (1 - alpha) * og_img
        return gin_img


@torch.no_grad()
def sample_symmetric_rand_conv_2d(
    rand_conv: nn.Conv2d, in_channels: int, out_channels: int, rng: torch.Generator
) -> torch.Tensor:
    width, height = (
        rand_conv.weight.shape[-2],
        rand_conv.weight.shape[-1],
    )

    center_x, center_y = width / 2, height / 2

    # Create a meshgrid
    x = torch.linspace(0, width, width)
    y = torch.linspace(0, height, height)
    c_in = torch.linspace(0, in_channels, in_channels)
    c_out = torch.linspace(0, out_channels, out_channels)
    C_out, C_in, X, Y = torch.meshgrid(c_out, c_in, x, y, indexing="ij")

    distance = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    dist_bins = torch.linspace(0, torch.max(distance), width)
    digitized_dist_map = torch.bucketize(distance, dist_bins, right=True) - 1
    num_chans = torch.tensor(in_channels * out_channels, device=rand_conv.weight.device)

    rand_conv_symmetric = torch.zeros_like(rand_conv.weight.data)
    for k in range(width):
        # sample from normal distribution zero mean and std 1
        cur_dist_mask = digitized_dist_map == k
        n_cur = cur_dist_mask.sum()
        rand_conv_symmetric[cur_dist_mask] = torch.randn(  # type: ignore
            num_chans,
            device=rand_conv.weight.device,
            dtype=rand_conv.weight.dtype,
            generator=rng,
        ).repeat_interleave(n_cur // num_chans)

    return rand_conv_symmetric


class RandConvAugmentation(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        rc_type: Literal["linear", "gin"],
        rotationally_symmetric: bool = False,
        n_layers: int = 4,
        n_hidden_chans: int = 2,
        alpha_range: tuple[float, float] = (0.0, 1.0),
        normalization: Literal["fro", "minmax"] = "minmax",
        do_updownsampling: bool = True,
        resize_mode: str = "bilinear",
        upsampling_size: int = 2048,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_dims = spatial_dims
        self.n_layers = n_layers
        self.n_hidden_chans = n_hidden_chans
        self.alpha_range = alpha_range
        self.normalization = normalization
        self.rc_type = rc_type
        self.rotationally_symmetric = rotationally_symmetric
        self.rng_map = {}

        self._init_updownsampling(do_updownsampling, resize_mode, upsampling_size)

    def _init_updownsampling(
        self, do_updownsampling: bool, resize_mode: str, upsampling_size: int
    ):
        if do_updownsampling:
            self.upsampling = torch.nn.Upsample(
                size=upsampling_size,
                mode=resize_mode,
                align_corners=resize_mode != "nearest",
            )
            self.downsampling = partial(
                F.interpolate,
                mode=resize_mode,
                align_corners=resize_mode != "nearest",
            )
        else:

            def _identity_with_args(x: Any, *args, **kwargs) -> Any:
                return x

            self.upsampling = _identity_with_args
            self.downsampling = _identity_with_args

    def forward(self, img: torch.Tensor):
        """Implementation of global intensity non-linear (GIN) augmentation."""

        needs_batch_dim = (img.ndim - 1) == (self.spatial_dims)
        # add batch dim if missing, -1 is for the channel dim
        if needs_batch_dim:
            img = img[None]

        # all channels to batch dim
        num_channels = img.size(1)
        img = img.flatten(0, 1)[:, None]
        in_channels = img.size(1)
        out_channels = in_channels

        # sample newly in each forward pass
        gin_transform = self._init_transform(
            in_channels, out_channels, device=img.device, dtype=img.dtype
        )
        gin_img = gin_transform(img)
        bg = img == 0
        gin_img[bg] = 0

        gin_img = gin_img.view(-1, num_channels, *gin_img.shape[2:])
        if needs_batch_dim:
            gin_img = gin_img.squeeze(0)

        return gin_img

    def get_rng(self, device: torch.device) -> torch.Generator:
        if device not in self.rng_map:
            rng = torch.Generator(device=device).manual_seed(device.index or 0)
            self.rng_map[device] = rng

        return self.rng_map[device]

    def sample_n_transforms(
        self, in_channels: int, out_channels: int, n_transforms: int, **conv_kwargs
    ) -> list[GlassOfGIN]:
        return [
            self._init_transform(in_channels, out_channels, **conv_kwargs)
            for _ in range(n_transforms)
        ]

    def _init_transform(
        self, in_channels: int, out_channels: int, **conv_kwargs
    ) -> GlassOfGIN:
        return GlassOfGIN(
            in_channels=in_channels,
            out_channels=out_channels,
            n_hidden_chans=self.n_hidden_chans,
            spatial_dims=self.spatial_dims,
            n_layers=self.n_layers,
            rotationally_symmetric=self.rotationally_symmetric,
            normalization=self.normalization,
            alpha_range=self.alpha_range,
            upsampling_fn=self.upsampling,
            downsampling_fn=self.downsampling,
            rng=self.get_rng(conv_kwargs.get("device", torch.device("cpu"))),
            **conv_kwargs,
        )


class RandConvAugmentationd(RandConvAugmentation):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in self.keys:
            bg = data[key] == 0
            data[key] = super().forward(data[key][None]).squeeze(0)
            data[key][bg] = 0

        return data
