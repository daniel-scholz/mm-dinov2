[![MICCAI 2025 — Paper page](https://img.shields.io/badge/MICCAI%202025-Paper%20Page-blue?style=for-the-badge)](https://papers.miccai.org/miccai-2025/0573-Paper1896.html) [![Download PDF](https://img.shields.io/badge/Download%20PDF-PDF-red?style=for-the-badge&logo=adobereader)](https://papers.miccai.org/miccai-2025/paper/1896_paper.pdf)

# MM-DINOv2: Adapting Foundation Models for Multi-Modal Medical Image Analysis (MICCAI2025)

Daniel Scholz, Ayhan Can Erdur, Viktoria Ehm, Anke Meyer-Baese, Jan C. Peeken, Daniel Rueckert, Benedikt Wiestler

In this work, we experiement with adapting DINOv2 for multi-modal medical image analysis.
Specifically, we apply our approach to glioma subtype classification, an important clinical task.

## Datasets

Please refer to [DATASETS.md](docs/DATASETS.md) for details on datasets and preprocessing.

# Setup

## uv

Create a new virtual environment using [uv](https://docs.astral.sh/uv/):

`uv sync`.
This will create a new virtual environment at `.venv` and install all dependencies in `pyproject.toml`.

## pip

Alternatively, you can create a new virtual environment using `venv` or `conda` and install the dependencies using pip: `pip install -r requirements.txt`

The requirements have been exported using `uv export --no-hashes --format requirements-txt > requirements.txt`.

## Download pre-trained weights

1. Download the pre-trained DINOv2 ViT-base weights from [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth).
2. Create the directory `out/eval/training_2499` if it doesn't exist.
3. Place the downloaded weights file (`dinov2_vitb14_pretrain.pth`) inside `out/eval/training_2499`.

## Running Experiments

The following commands assume you have activated the virtual environment created during setup.

### Run Training

To start a new training run, execute one of the following commands. Make sure to update the `--output-dir` for new experiments to avoid overwriting previous results.

**Using `python`:**

```bash
python dinov2/train/train.py \
    --config-file dinov2/configs/train/glioma_vitb14_mm-dino_v2.yaml \
    --output-dir out/train/glioma_vitb14_mm-dino_v2
```

**Using `uv`:**

```bash
uv run python dinov2/train/train.py \
    --config-file dinov2/configs/train/glioma_vitb14_mm-dino_v2.yaml \
    --output-dir out/train/glioma_vitb14_mm-dino_v2
```

### Run Evaluation

To evaluate a trained model, you can run one of the following commands above commands with the `--eval-only`, pointing to the configuration and checkpoint of your trained model.

## Citing

If you use this repository in your work, please consider citing our paper and the original works.

```
@InProceedings{SchDan_MMDINOv2_MICCAI2025,
        author = { Scholz, Daniel AND Erdur, Ayhan Can AND Ehm, Viktoria AND Meyer-Baese, Anke AND Peeken, Jan C. AND Rueckert, Daniel AND Wiestler, Benedikt},
        title = { { MM-DINOv2: Adapting Foundation Models for Multi-Modal Medical Image Analysis } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15967},
        month = {September},
        page = {320 -- 330}
}

@misc{baharoon2023general,
      title={Towards General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks},
      author={Mohammed Baharoon and Waseem Qureshi and Jiahong Ouyang and Yanwu Xu and Abdulrhman Aljouie and Wei Peng},
      year={2023},
      eprint={2312.02366},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE). See the `LICENSE` file for more details.
