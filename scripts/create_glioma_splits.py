import multiprocessing
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    fast = True
    file_fp = Path(__file__).resolve()
    cwd = file_fp.parent.parent
    data_base_dir = Path("~/datasets/glioma_public").expanduser()
    splits_base_dir = Path("~/datasets/glioma_public_splits").expanduser()
    splits_base_dir.mkdir(exist_ok=True, parents=True)

    labels_old_fp = data_base_dir.joinpath("phenoData.csv")
    labels_new_fp = splits_base_dir / "phenoData.csv"

    shutil.copy(labels_old_fp, labels_new_fp)

    assert data_base_dir.exists(), f"Data directory {data_base_dir} does not exist"

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Processing {split} split")
        split_dir = splits_base_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

        with open(cwd / f"glioma_splits/{split}.txt", "r") as f:
            subject_ids = f.readlines()

        subject_ids = [s_id.strip() for s_id in subject_ids]
        print(f"Found {len(subject_ids)} subjects in {split} split")

        if fast:
            pool = multiprocessing.Pool()
            pbar = tqdm(
                [(data_base_dir, split_dir, s_id) for s_id in subject_ids],
                desc=f"{split} split",
            )
            pool.starmap(link_subject_directory, pbar)
            pool.close()
        else:
            for s_id in tqdm(subject_ids, desc=f"{split} split"):
                s_id: str
                link_subject_directory(data_base_dir, split_dir, s_id)


def link_subject_directory(data_base_dir: Path, split_dir: Path, s_id: str):
    """Link the subject directory from the data_base_dir to the split_dir"""

    for subset_dir in data_base_dir.iterdir():
        if subset_dir.is_dir():
            subject_dir = subset_dir / s_id
            if subject_dir.exists():
                split_subject_dir = split_dir / s_id

                if not split_subject_dir.exists():
                    split_subject_dir.symlink_to(subject_dir)

                break

    else:
        print(f"{s_id=}")


if __name__ == "__main__":
    main()
