from pathlib import Path

import pandas as pd
import torch
from monai.transforms.transform import Transform


class SubjectDirToLabel(Transform):
    def __init__(self, labels: pd.DataFrame, label_col: str):
        super().__init__()
        self.labels = labels
        self.label_col = label_col

    def __call__(self, subject_dir: Path) -> dict[str, torch.Tensor]:
        if subject_dir.name not in self.labels.index:
            return {"label": torch.tensor(-1, dtype=torch.long)}
        label_np = self.labels.loc[subject_dir.name, self.label_col]
        # map nan to -1
        if pd.isna(label_np):
            label_np = -1
        return {"label": torch.tensor(label_np, dtype=torch.long)}
