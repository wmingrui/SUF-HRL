# Dataset preparation

This repository does not redistribute Potsdam, Vaihingen, or LoveDA data. Please download the datasets from their official sources and convert them to indexed PNG labels.

## Expected layout

```text
/path/to/dataset/
├── processed_multiclass/
│   ├── images/
│   │   ├── <sample_id>.png
│   │   └── ...
│   └── labels/
│       ├── <sample_id>.png
│       └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Label conventions

### ISPRS Potsdam

```text
0: impervious surface
1: building
2: low vegetation
3: tree
4: car
5: clutter/background
255: ignore
```

### ISPRS Vaihingen

```text
0: impervious surface
1: building
2: low vegetation
3: tree
4: car
5: clutter/background
255: ignore
```

The paper reports class-averaged metrics following the specified evaluation protocol.

### LoveDA

```text
0: background
1: building
2: road
3: water
4: barren
5: forest
6: agriculture
255: ignore
```

## Config update

After preparing a dataset, set the root path in the corresponding YAML file:

```yaml
dataset:
  root: /path/to/potsdam
```
