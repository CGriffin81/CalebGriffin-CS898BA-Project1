# CS898BA Image Classification Project

## Purpose

This branch is for classification of fish species within a given data set using a PyTorch CNN.

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
	C --> D[Custom CNN in PyTorch]
	D --> E[Hyperparameter search on validation set]
	E --> F[Final training and test evaluation]
	F --> G[Save weights, curves, report, confusion matrix]
```

## Run

From the project root, run:

```bash
python -m src.train
```

Implementation status:

- PyTorch CNN
- Stratified train / validation / test split
- Train-only augmentation
- Saved model, plots, and reports in `outputs/`

Artifacts are written to `outputs/`:

- `outputs/models/fish_cnn_best.pt`
- `outputs/plots/training_curves.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/reports/classification_report.txt`
- `outputs/reports/hyperparameter_search.json`
- `outputs/reports/split_summary.json`
