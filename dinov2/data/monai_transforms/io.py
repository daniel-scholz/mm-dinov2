from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform


class SubjectDirToFPsDict(MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, subject_dir: Path) -> dict[str, Path]:
        d = {}

        for key in self.keys:
            fn = f"sub-{subject_dir.name}_ses-preop_space-sri_{key}.nii.gz"
            d[key] = subject_dir / "preop" / fn
        return d


def LoadAxialTumorSlice(fp: Path, slice_idx: int, slice_: list[slice]) -> torch.Tensor:
    return LoadTumorSliceWithAxis(fp, 2, slice_idx, slice_)


def LoadTumorSliceWithAxis(
    fp: Path, slice_idx: int, slice_: list[slice], axis: int
) -> torch.Tensor:
    slice_idx = slice(slice_idx, slice_idx + 1)  # to work with nifti slicer
    nii_img = nib.nifti1.load(str(fp))
    if axis == 0:
        return torch.tensor(
            nii_img.slicer[slice_idx, slice_[0], slice_[1]].get_fdata().squeeze(0)
        )
    elif axis == 1:
        return torch.tensor(
            nii_img.slicer[slice_[0], slice_idx, slice_[1]].get_fdata().squeeze(1)
        )
    elif axis == 2:
        return torch.tensor(
            nii_img.slicer[slice_[0], slice_[1], slice_idx].get_fdata().squeeze(2)
        )


class LoadTumorSliced(MapTransform):
    def __init__(
        self,
        keys: list[str],
        tumor_key: str,
        spatial_size: tuple[int, int],
        axes: list[int] = [0, 1, 2],
        allow_missing_keys: bool = False,
        min_tumor_size: int = 500,
        select_random_slices: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.tumor_key = tumor_key
        self.spatial_size = spatial_size
        self.axes = np.array(axes)

        self.select_random_slices = select_random_slices
        self.min_tumor_size = min_tumor_size

    def __call__(self, data: KeysCollection) -> dict[str, torch.Tensor]:
        # extract and crop segmentation map
        axis = np.random.choice(self.axes)
        tumor_label_map_slice, com_tumor, slice_idx = self.extract_tumor_slice(
            data, axis
        )
        crop_slice = calculate_crop_slices(
            com_tumor, self.spatial_size, tumor_label_map_slice.shape[-2:]
        )
        tumor_label_map_slice = tumor_label_map_slice[crop_slice]

        d = dict(data)
        d[self.tumor_key] = tumor_label_map_slice
        for key in self.key_iterator(d):
            if key != self.tumor_key:
                d[key] = LoadTumorSliceWithAxis(d[key], slice_idx, crop_slice, axis)

        return d

    def extract_tumor_slice(
        self, data: KeysCollection, axis: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        tumor_label_map = nib.nifti1.load(str(data[self.tumor_key])).get_fdata()
        # get slice with most tumor pixels
        if axis == 2:
            # axial
            slice_idx = self.get_tumor_slice_from_label_map(
                tumor_label_map,
                axis=(0, 1),
            )
            tumor_label_map_slice = torch.tensor(tumor_label_map[..., slice_idx])
        elif axis == 1:
            # coronal
            slice_idx = self.get_tumor_slice_from_label_map(
                tumor_label_map,
                axis=(0, 2),
            )
            tumor_label_map_slice = torch.tensor(tumor_label_map[:, slice_idx, :])
        elif axis == 0:
            # sagittal
            slice_idx = self.get_tumor_slice_from_label_map(
                tumor_label_map,
                axis=(1, 2),
            )
            tumor_label_map_slice = torch.tensor(tumor_label_map[slice_idx, :, :])

        # get center of mass of tumor
        com_tumor = calc_center_of_mass(tumor_label_map_slice)

        return tumor_label_map_slice, com_tumor, slice_idx

    def get_tumor_slice_from_label_map(
        self,
        tumor_label_map: torch.Tensor,
        axis: tuple[int, int],
    ):
        num_tumor_voxels_in_slice = np.sum(tumor_label_map, axis=axis)
        if not self.select_random_slices:
            return np.argmax(num_tumor_voxels_in_slice).item()

        # determine slices with enough tumor voxels
        valid_slices = np.where(num_tumor_voxels_in_slice >= self.min_tumor_size)[0]

        if len(valid_slices) == 0:
            # if no slice has enough tumor voxels, select slice with most tumor voxels
            return np.argmax(num_tumor_voxels_in_slice).item()

        # randomly sample slice with at least min_tumor_size tumor voxels
        return np.random.choice(valid_slices).item()


def calculate_crop_slices(
    com: torch.Tensor,
    spatial_crop_size: tuple[int, int],
    spatial_img_size: Sequence[int],
) -> list[slice]:
    """Calculate crop slices for a given center of mass and crop size."""

    com_int = com.int()
    spatial_crop_size_torch = torch.tensor(spatial_crop_size).to(com_int.device).long()
    spatial_img_size_torch = torch.tensor(spatial_img_size).to(com_int.device).long()

    crop_start = torch.clamp(com_int - spatial_crop_size_torch // 2, 0)
    crop_end = crop_start + spatial_crop_size_torch
    if (crop_end > spatial_img_size_torch).any():
        crop_end = torch.min(crop_end, spatial_img_size_torch)
        crop_start = crop_end - spatial_crop_size_torch

    slice_ = [slice(start, end) for start, end in zip(crop_start, crop_end)]

    return slice_


def calc_center_of_mass(img: torch.Tensor, vmin=0, vmax=1) -> torch.Tensor:
    """
    Returns centre of mass of the last dim-many dimensions of x3d.
    In:  (D x) H x W
    Out:  3
    """
    img = img.float()

    spatial_shape = img.shape
    meshgrid = torch.meshgrid(
        [torch.arange(s, device=img.device) for s in spatial_shape],
        indexing="ij",
    )
    coords = torch.stack([m.flatten() for m in meshgrid], dim=-1).float()

    x3d_norm = (img - vmin) / (vmax - vmin)

    x3d_list = torch.flatten(x3d_norm)

    fg_mask = x3d_list > vmin

    coords = coords[fg_mask]
    x3d_list = x3d_list[fg_mask]

    return torch.mean(coords, dim=0)
