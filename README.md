# CS898BA Image Classification Project

## Purpose

This branch is for classification of fish species within a given data set using a baseline PyTorch CNN.

## Dataset

The extracted fish dataset is stored in `data/Fish` with six class folders:

- Bete
- Cray
- Discus
- Gold
- Guppy
- Oscar

## Pipeline

```mermaid
flowchart LR
	A[Load images from data/Fish] --> B[Stratified train / val / test split]
	B --> C[Resize, normalize, augment train only]
	C --> D[Baseline CNN in PyTorch]
	D --> E[Final training and test evaluation]
	E --> F[Save baseline weights, curves, and reports]
```

## Run

From the project root, run:

```bash
python -m src.train
```

Implementation status:

- Baseline PyTorch CNN
- Stratified train / validation / test split
- Train-only augmentation
- Baseline artifacts in `outputs/`

Artifacts are written to `outputs/`:

- `outputs/models/baseline_cnn.pt`
- `outputs/history/baseline_history.json`
- `outputs/plots/baseline_accuracy_curve.png`
- `outputs/plots/baseline_loss_curve.png`
- `outputs/reports/baseline_classification_report.txt`
- `outputs/reports/split_summary.json`
