# SUF-HRL

Official code skeleton for **Spatially-Aware Uncertainty Feedback for Hard-Region Learning in Remote Sensing Semantic Segmentation**.

SUF-HRL turns prediction uncertainty from a post-hoc reliability indicator into a spatial feedback signal for hard-region learning. The core pipeline is:

1. SegFormer-B2 semantic segmentation baseline.
2. Maximum-softmax-probability (MSP) uncertainty prior.
3. Residual uncertainty refinement from decoder features.
4. Spatial uncertainty objectives for local alignment and boundary concentration.
5. Top-k hard-region supervision guided by learned spatial uncertainty.

## Repository structure

```text
SUF-HRL/
├── configs/                  # Dataset configs and MMSegmentation baseline configs
├── sufh_rl/                  # Core package
│   ├── models/               # SegFormer baseline and SUF-HRL model
│   ├── losses/               # Dice, focal, top-k, local, and boundary losses
│   ├── datasets/             # Potsdam, Vaihingen, and LoveDA dataloaders
│   ├── metrics/              # mIoU, boundary mIoU, BFUR, DSCG, MSAD
│   └── utils/                # Config and reproducibility helpers
├── tools/                    # Training, evaluation, and visualization entry points
├── scripts/                  # Example shell commands
└── docs/                     # Dataset preparation and figure notes
```

## Installation

```bash
conda create -n sufhrl python=3.10 -y
conda activate sufhrl
pip install -r requirements.txt
```

This code uses HuggingFace SegFormer. The paper experiments use `nvidia/mit-b2`.

## Dataset preparation

The code expects each dataset to be converted into the following layout:

```text
/path/to/dataset/
├── processed_multiclass/
│   ├── images/
│   │   ├── sample_0001.png
│   │   └── ...
│   └── labels/
│       ├── sample_0001.png
│       └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each split file contains one sample id per line, without file extension.

More details are provided in [`docs/dataset_preparation.md`](docs/dataset_preparation.md).

## Training

Edit the dataset root in a config file, for example `configs/potsdam.yaml`, and run:

```bash
python tools/train.py --config configs/potsdam.yaml --method suf_hrl
```

Other supported method flags include:

```text
baseline, focal, ohem, loss_topk, msp_topk, entropy_topk, suf_hrl
```

Example scripts are provided in `scripts/`:

```bash
bash scripts/train_potsdam.sh
bash scripts/train_vaihingen.sh
bash scripts/train_loveda.sh
```

## Evaluation

Global segmentation metrics:

```bash
python tools/evaluate.py \
  --config configs/potsdam.yaml \
  --checkpoint outputs/potsdam_suf_hrl/checkpoints/best.pth
```

Boundary mIoU:

```bash
python tools/eval_boundary_miou.py \
  --config configs/potsdam.yaml \
  --checkpoint outputs/potsdam_suf_hrl/checkpoints/best.pth \
  --widths 3 5 7
```

Uncertainty quality:

```bash
python tools/eval_uncertainty_quality.py \
  --config configs/vaihingen.yaml \
  --checkpoint outputs/vaihingen_suf_hrl/checkpoints/best.pth \
  --source learned
```

Qualitative visualization:

```bash
python tools/make_qualitative_figures.py \
  --config configs/potsdam.yaml \
  --checkpoint outputs/potsdam_suf_hrl/checkpoints/best.pth \
  --out-dir docs/figures/potsdam_examples
```

## Paper figures

The final manuscript figures and example visualizations can be placed in `docs/figures/`. Raw datasets and large checkpoints should not be committed to this repository.

## Notes

This first public version is a cleaned research-code release. It keeps the main SUF-HRL implementation, training logic, evaluation metrics, and dataset interfaces, while removing local AutoDL paths, temporary experiment logs, checkpoints, and raw data.

## Citation

```bibtex
@article{wang2026sufhrl,
  title={Spatially-Aware Uncertainty Feedback for Hard-Region Learning in Remote Sensing Semantic Segmentation},
  author={Wang, Mingrui and Yang, Ronghua and Fan, Hongchao and Wu, Hao and Li, Jiarui},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```
