# SUF-HRL

Official implementation of **Spatially-Aware Uncertainty Feedback for
Hard-Region Learning in Remote Sensing Semantic Segmentation**.

SUF-HRL turns prediction uncertainty from a post-hoc reliability
indicator into a **spatial feedback signal** for hard-region learning.
Instead of selecting isolated pixels only by loss or raw confidence,
SUF-HRL learns a spatially structured uncertainty map and uses it to
guide additional supervision around boundary-adjacent and error-prone
regions.

------------------------------------------------------------------------

## Overview

SUF-HRL is designed for high-resolution remote sensing semantic
segmentation, where errors are often concentrated in structured hard
regions, such as:

-   object boundaries,
-   small objects,
-   shadow regions,
-   ambiguous land-cover transitions.

The framework contains:

1.  **SegFormer-B2 segmentation baseline**
2.  **MSP uncertainty prior**
3.  **Residual uncertainty refinement branch**
4.  **Spatial uncertainty objectives**
5.  **Uncertainty-guided top-k hard-region supervision**

The learned uncertainty is not only used as a reliability indicator, but
also serves as a feedback signal for improving difficult regions.

------------------------------------------------------------------------

## Framework

The main pipeline:

    Input Image
         |
         v
    SegFormer Encoder
         |
         v
    Multi-scale Decoder
         |
         +----------------+
         |                |
     Segmentation     Residual
     Prediction       Uncertainty
         |                |
         +-------+--------+
                 |
         Spatial Uncertainty Map
                 |
          Hard-region Selection
                 |
            SUF-HRL Loss

------------------------------------------------------------------------

## Spatial uncertainty indicators

SUF-HRL evaluates uncertainty quality using both conventional
uncertainty metrics and spatially-aware indicators.

The three spatial uncertainty indicators are:

-   **BFUR (Boundary-Focused Uncertainty Ratio)**\
    Measures uncertainty concentration inside boundary regions. Higher
    values indicate stronger boundary awareness.

-   **DSCG (Distance-Stratified Calibration Gap)**\
    Measures whether uncertainty follows the prediction difficulty
    variation at different distances from object boundaries. Lower
    values indicate better spatial calibration.

-   **MSAD (Multi-Scale Local Alignment Distance)**\
    Measures local alignment between uncertainty maps and error
    distributions across multiple spatial scales. Lower values indicate
    better uncertainty-error alignment.

------------------------------------------------------------------------

# Repository Structure

    SUF-HRL/

    ├── configs/                  # Dataset configurations
    ├── sufh_rl/                  # Core implementation
    │   ├── models/               # SegFormer baseline and SUF-HRL model
    │   ├── losses/               # Segmentation and uncertainty losses
    │   ├── datasets/             # Dataset interfaces
    │   ├── metrics/              # Evaluation metrics
    │   └── utils/                # Utilities
    ├── tools/                    # Training, evaluation and visualization
    ├── scripts/                  # Training scripts
    ├── docs/                     # Dataset preparation and documentation
    ├── tests/                    # Repository verification tests
    └── requirements.txt

------------------------------------------------------------------------

# Installation

Example environment:

    conda create -n sufhrl python=3.10 -y
    conda activate sufhrl
    pip install -r requirements.txt

The implementation uses HuggingFace SegFormer with the `nvidia/mit-b2`
backbone.

------------------------------------------------------------------------

# Dataset Preparation

The code expects datasets in the following format:

    dataset/
    ├── processed_multiclass/
    │   ├── images/
    │   └── labels/
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt

Each split file contains one sample identifier per line.

More details:

    docs/dataset_preparation.md

------------------------------------------------------------------------

# Training

Example:

``` bash
python tools/train.py \
--config configs/potsdam.yaml \
--method suf_hrl
```

Provided scripts:

``` bash
bash scripts/train_potsdam.sh
bash scripts/train_vaihingen.sh
bash scripts/train_loveda.sh
```

------------------------------------------------------------------------

# Evaluation

After downloading checkpoints, run:

## Potsdam

``` bash
PYTHONPATH=. \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python tools/eval_potsdam.py \
--checkpoint checkpoints/potsdam_sufhrl_b2.pth
```

## Vaihingen

``` bash
PYTHONPATH=. \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python tools/eval_vaihingen.py \
--checkpoint checkpoints/vaihingen_sufhrl_b2.pth
```

## LoveDA

``` bash
PYTHONPATH=. \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python tools/eval_loveda.py \
--checkpoint checkpoints/loveda_sufhrl_b2.pth
```

------------------------------------------------------------------------

# Table III Spatial Uncertainty Evaluation

Table III evaluates:

-   Error AUROC
-   Error AUPR
-   UCE
-   BFUR
-   DSCG
-   MSAD

Run:

``` bash
python tools/tableIII/tableIII_spatial_uncertainty_eval.py
```

Required checkpoints:

    checkpoints/tableIII/

    ├── vaihingen_baseline_seed0_tableIII.pth
    └── vaihingen_sufhrl_seed0_tableIII.pth

------------------------------------------------------------------------

# Qualitative Visualization

Generate qualitative comparison figures:

``` bash
PYTHONPATH=. \
python tools/make_qualitative_comparison.py \
--dataset potsdam \
--checkpoint checkpoints/potsdam_sufhrl_b2.pth \
--split test \
--num-cases 4 \
--output-dir results/qualitative
```

------------------------------------------------------------------------

# Pretrained Checkpoints

Large model weights are provided separately and are not included in this
repository.

Available checkpoints:

## Main B2 models

-   Potsdam SUF-HRL
-   Vaihingen SUF-HRL
-   LoveDA SUF-HRL

## Table III models

-   Vaihingen baseline
-   Vaihingen SUF-HRL

## MIT-B5 generalization models

-   Potsdam MIT-B5 SUF-HRL
-   Vaihingen MIT-B5 SUF-HRL
-   LoveDA MIT-B5 SUF-HRL

Checkpoint download links:

(Insert Google Drive / Release links here)

------------------------------------------------------------------------

# Notes

This public version is a cleaned research-code release.

It keeps:

-   SUF-HRL model implementation
-   training pipeline
-   evaluation scripts
-   uncertainty metrics
-   dataset interfaces
-   visualization tools

Large datasets, checkpoints, logs, and temporary experiment outputs are
distributed separately.
