from pathlib import Path

if __name__ == "__main__":
    import sys

    code_dir = Path("~/coding/DINOv2ForRadiology").expanduser()
    assert code_dir.exists()
    print(f"Adding {code_dir} to sys.path")
    sys.path.append(code_dir.as_posix())

from enum import Enum

import numpy as np
import pandas as pd

from dinov2.data.datasets.glioma_supervised import GliomaSupervised


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 2661,
            _Split.VAL: 666,
            _Split.TEST: 244,
        }
        return split_lengths[self]


class GliomaSSL(GliomaSupervised):
    Split = _Split

    def _filter_labels(self):
        """keep all images. fill missing labels with -1"""

        # instead of finding interesting labels, we just return all labels, fill the missing one with -1
        self.labels.fillna(-1, inplace=True)

        # add rows to labels that are in images but not in labels

        missing_patients_in_labels = np.setdiff1d(self.images, self.labels.index)

        # add missing patient to labels
        if len(missing_patients_in_labels) > 0:
            new_labels_col = pd.Series(
                [-1] * len(missing_patients_in_labels),
                index=missing_patients_in_labels,
                name=self.labels.columns[0],
            )

            self.labels = pd.concat([self.labels, new_labels_col])

        # intersect with images
        self.labels = self.labels.loc[self.images]


if __name__ == "__main__":

    def main():
        dataset_root_dir = Path("~/datasets/glioma_public_splits").expanduser()
        dataset = GliomaSSL(split=GliomaSSL.Split.TRAIN, root=str(dataset_root_dir))
        print(f"Number of classes: {dataset.get_num_classes()}")

        image = dataset.get_image_data(0)

        print(f"Image shape: {image.shape}")

    main()
