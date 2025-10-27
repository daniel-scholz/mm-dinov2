import logging
from pathlib import Path

if __name__ == "__main__":
    import sys

    code_dir = Path("~/coding/DINOv2ForRadiology").expanduser()
    assert code_dir.exists()
    print(f"Adding {code_dir} to sys.path")
    sys.path.append(code_dir.as_posix())

import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRangePercentilesd

from dinov2.data.datasets.medical_dataset import MedicalVisionDataset
from dinov2.data.monai_transforms.glioma import SubjectDirToLabel
from dinov2.data.monai_transforms.io import LoadTumorSliced, SubjectDirToFPsDict

logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1162,
            _Split.VAL: 296,
            _Split.TEST: 214,
        }
        return split_lengths[self]


class GliomaSupervised(MedicalVisionDataset):
    Split = _Split
    spatial_size = (96, 96)

    def __init__(
        self,
        *,
        split: "GliomaSupervised.Split",
        root: str,
        mri_sequences: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # needs to be string because of yaml config
        random_axes: bool = False,
        random_slices: bool = False,
        append_label_mask: bool = False,
        percentage_labels: float = 1.0,
    ) -> None:
        if mri_sequences is not None:
            self.mri_sequences = mri_sequences.split(",")
        else:
            self.mri_sequences = [
                "flair",
                "t1c",
                "t2",
                # "t1"
            ]

        self.random_slices = random_slices
        self.random_axes = random_axes
        self.append_label_mask = append_label_mask

        super().__init__(split, root, transforms, transform, target_transform)

        self.percentage_labels = percentage_labels
        self.class_names = np.array(["astrocytoma", "glioblastoma", "oligendroglioma"])

        self._randomly_drop_labels()

    def _randomly_drop_labels(self):
        if self.percentage_labels < 1.0:
            # stratify by class
            for i in range(len(self.class_names)):
                class_indices = np.where(self.labels["WHO2021_Int"] == i)[0]
                num_labels = len(class_indices)
                num_labels_to_keep = max(
                    int(np.round(self.percentage_labels * num_labels)), 1
                )

                # randomly set labels to -1
                random_indices = np.random.choice(
                    class_indices, size=num_labels - num_labels_to_keep, replace=False
                )
                self.labels.iloc[random_indices, 0] = -1

    def _init_images(self):
        super()._init_images()
        self._init_labels()
        self._init_load_transform()

    def _init_load_transform(self):
        self.img_load_transform = Compose(
            [
                SubjectDirToFPsDict(keys=[*self.mri_sequences, "seg"]),
                LoadTumorSliced(
                    keys=[*self.mri_sequences, "seg"],
                    tumor_key="seg",
                    spatial_size=self.spatial_size,
                    axes=[0, 1, 2]
                    if self.split == self.Split.TRAIN and self.random_axes
                    else [2],
                    select_random_slices=self.random_slices
                    and self.split == self.Split.TRAIN,
                ),
                ScaleIntensityRangePercentilesd(
                    keys=self.mri_sequences,
                    b_min=0.0,
                    b_max=1.0,
                    lower=1,
                    upper=99,
                    clip=True,
                ),
            ]
        )

    def _init_labels(self):
        labels_fp = self._split_dir + os.sep + ".." + os.sep + "phenoData.csv"

        with open(labels_fp, "r", encoding="utf-8") as f:
            df = pd.read_csv(f)

        keep_cols = ["WHO2021_Int", "Patient"]
        df = df[keep_cols]
        df = df.set_index("Patient")
        self.labels = df

        # find intersection between images and labels
        self._filter_labels()

        self.label_load_transform = SubjectDirToLabel(self.labels, "WHO2021_Int")

    def _filter_labels(self):
        # remove rows with missing labels
        self.labels = self.labels.dropna(subset=["WHO2021_Int"])

        images_in_labels = self.labels.index.intersection(self.images)

        self.labels = self.labels.loc[images_in_labels]

        self.images = images_in_labels

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return False

    def is_multilabel(self) -> bool:
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        subject_dir = self._split_dir + os.sep + self.images[index]
        subject_dir = Path(subject_dir)

        subject_dict = self.img_load_transform(subject_dir)

        image = torch.stack([subject_dict[key] for key in self.mri_sequences], dim=0)

        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        # add tumor mask in last channel
        if self.append_label_mask:
            image = torch.cat([image, subject_dict["seg"].unsqueeze(0)], dim=0)

        return image

    def _check_size(self):
        num_of_images = len(self.images)
        logger.info(
            f"{self._split.length - num_of_images} scans are missing from {self._split.value.upper()} set"
        )

    def get_target(self, index: int) -> torch.Tensor:
        subject_dir = self._split_dir + os.sep + self.images[index]
        subject_dir = Path(subject_dir)
        label = self.label_load_transform(subject_dir)
        return label["label"]

    def get_target_name(self, index: int) -> str:
        label = self.get_target(index)
        return self.class_names[label] if label >= 0 else "Unknown"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image)
        # set nans to 0 in image
        if isinstance(image, torch.Tensor):
            image[torch.isnan(image)] = 0
        return image, target
