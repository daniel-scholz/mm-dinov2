import torch
import torch.nn as nn
import torch.nn.functional as F


class RandResizeCrop(nn.Module):
    def __init__(self, min_scale: float, max_scale: float):
        super().__init__()
        self.min_scale = torch.tensor(min_scale)
        self.max_scale = torch.tensor(max_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[2:]
        spatial_size_torch = torch.tensor(spatial_size, device=x.device, dtype=x.dtype)
        dims = len(spatial_size_torch)

        self.min_scale = self.min_scale.to(x.device, x.dtype)
        self.max_scale = self.max_scale.to(x.device, x.dtype)

        # sample one scale per dimension
        scale = (
            torch.rand(dims, device=x.device, dtype=x.dtype)
            * (self.max_scale - self.min_scale)
            + self.min_scale
        )
        new_size = (scale * spatial_size_torch).int()

        # compute crop bounds
        start = (
            torch.rand(dims, device=x.device, dtype=x.dtype)
            * (spatial_size_torch - new_size)
        ).int()
        end = start + new_size

        # crop
        slices = [slice(s, e) for s, e in zip(start, end)]

        x_cropped = x[..., *slices]

        x_resized = F.interpolate(x_cropped, size=spatial_size, mode="bilinear")

        return x_resized
