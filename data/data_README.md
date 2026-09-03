# Dataset Directory Structure

This directory is reserved for semantic segmentation datasets used in SUF-HRL.

The original datasets are not included in this repository due to their large storage requirements and original dataset licenses.

Please prepare datasets following:

```
docs/dataset_preparation.md
```

After preprocessing, the expected structure is:

```
data/
├── potsdam/
│   └── processed_multiclass/
│       ├── images/
│       └── labels/
│
├── vaihingen/
│   └── processed_multiclass/
│       ├── images/
│       └── labels/
│
└── loveda/
    └── processed_multiclass/
        ├── images/
        └── labels/
```

---

## Potsdam

6-class semantic segmentation:

```
0: Impervious surfaces
1: Building
2: Low vegetation
3: Tree
4: Car
5: Clutter / Background
```

Training:

```bash
python tools/train.py --config configs/potsdam.yaml --method suf_hrl
```

Evaluation:

```bash
python tools/eval_potsdam.py --checkpoint checkpoints/potsdam_sufhrl_b2.pth
```

---

## Vaihingen

5-class evaluation protocol:

```
0: Impervious surfaces
1: Building
2: Low vegetation
3: Tree
4: Car
```

The clutter/background class is excluded during evaluation.

Training:

```bash
python tools/train.py --config configs/vaihingen.yaml --method suf_hrl
```

Evaluation:

```bash
python tools/eval_vaihingen.py --checkpoint checkpoints/vaihingen_sufhrl_b2.pth
```

---

## LoveDA

7-class semantic segmentation:

```
0: Background
1: Building
2: Road
3: Water
4: Barren
5: Forest
6: Agriculture
```

Training:

```bash
python tools/train.py --config configs/loveda.yaml --method suf_hrl
```

Evaluation:

```bash
python tools/eval_loveda.py --checkpoint checkpoints/loveda_sufhrl_b2.pth
```

---

## General Requirements

Each dataset should contain:

```
processed_multiclass/
├── images/
│   ├── image_001.png
│   └── ...
└── labels/
    ├── image_001.png
    └── ...
```

Requirements:

- Images and labels must have matching filenames.
- Images should be RGB images.
- Labels should contain integer class IDs.
- Ignore label value:

```
255
```

---

## Configuration

Dataset paths are configured in:

```
configs/potsdam.yaml
configs/vaihingen.yaml
configs/loveda.yaml
```

Default roots:

```
data/potsdam
data/vaihingen
data/loveda
```

Modify `dataset.root` if datasets are stored elsewhere.

---

The repository provides dataset loaders, training scripts, evaluation scripts, and preprocessing interfaces. Original datasets must be downloaded separately from their official sources.
