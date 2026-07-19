"""Controlled hyperparameter grid search for the fish species CNN.

Grid: 3 learning rates × 2 batch sizes × 2 dropout values = 12 experiments.
Evaluation uses validation split only. Test set is never touched.
Best model selected by lowest validation loss.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .model import FishCNN, ModelConfig
    from .train import (
        build_dataloaders,
        compute_baseline_class_weights,
        create_model,
        evaluate,
        load_dataset,
        plot_metric_curve,
        set_seed,
        stratified_split,
        train_model,
    )
except ImportError:  # supports direct script execution
    from model import FishCNN, ModelConfig
    from train import (
        build_dataloaders,
        compute_baseline_class_weights,
        create_model,
        evaluate,
        load_dataset,
        plot_metric_curve,
        set_seed,
        stratified_split,
        train_model,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fish"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Required search grid — do not modify
LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64]
DROPOUTS = [0.3, 0.5]


def run_grid_search(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    history_dir = output_dir / "history"
    report_dir = output_dir / "reports"
    plot_dir = output_dir / "plots"
    for directory in (model_dir, history_dir, report_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Load dataset once; all experiments share the same split and normalization
    images, labels, class_names = load_dataset(Path(args.data_dir), args.image_size)
    train_indices, val_indices, test_indices = stratified_split(
        labels, 0.7, 0.15, 0.15, args.seed
    )

    class_weights = compute_baseline_class_weights(labels[train_indices], len(class_names))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    grid = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROPOUTS))
    assert len(grid) == 12, "Grid must be exactly 12 experiments"

    results: list[dict] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_config: dict | None = None
    best_history: dict | None = None

    for experiment_index, (lr, batch_size, dropout) in enumerate(grid, start=1):
        print(f"[{experiment_index:02d}/12] lr={lr}  batch={batch_size}  dropout={dropout}")

        set_seed(args.seed)  # reset seed per experiment for reproducibility

        train_loader, val_loader, _test_loader, _mean, _std = build_dataloaders(
            images, labels, train_indices, val_indices, test_indices, batch_size, args.seed
        )

        config = ModelConfig(
            input_channels=3,
            num_classes=len(class_names),
            dropout=dropout,
            learning_rate=lr,
            image_size=args.image_size,
        )
        model = create_model(config, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history, _ = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        )

        # Evaluate on validation set only — test set never used during tuning
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)

        result = {
            "experiment": experiment_index,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        results.append(result)
        print(f"         val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_config = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            best_history = history

    # Save all experiment results
    (report_dir / "hyperparameter_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Save best configuration
    assert best_config is not None
    (report_dir / "best_config.json").write_text(
        json.dumps(best_config, indent=2), encoding="utf-8"
    )

    # Retrain best config from scratch on train split, then save
    set_seed(args.seed)
    best_lr = best_config["learning_rate"]
    best_bs = best_config["batch_size"]
    best_dropout = best_config["dropout"]

    train_loader, val_loader, _test_loader, mean, std = build_dataloaders(
        images, labels, train_indices, val_indices, test_indices, best_bs, args.seed
    )
    optimized_config = ModelConfig(
        input_channels=3,
        num_classes=len(class_names),
        dropout=best_dropout,
        learning_rate=best_lr,
        image_size=args.image_size,
    )
    optimized_model = create_model(optimized_config, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(optimized_model.parameters(), lr=best_lr)
    optimized_history, _ = train_model(
        optimized_model, train_loader, val_loader, criterion, optimizer, device, args.epochs
    )

    # Save optimized model weights
    optimized_model_path = model_dir / "optimized_cnn.pt"
    torch.save(
        {
            "model_state_dict": optimized_model.state_dict(),
            "config": optimized_model.config.__dict__,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "class_names": class_names,
        },
        optimized_model_path,
    )

    # Save optimized training history
    (history_dir / "optimized_history.json").write_text(
        json.dumps(optimized_history, indent=2), encoding="utf-8"
    )

    # Save optimized training curves
    plot_metric_curve(
        optimized_history["train_loss"],
        optimized_history["val_loss"],
        "Optimized Loss",
        "Loss",
        plot_dir / "optimized_loss_curve.png",
    )
    plot_metric_curve(
        optimized_history["train_accuracy"],
        optimized_history["val_accuracy"],
        "Optimized Accuracy",
        "Accuracy",
        plot_dir / "optimized_accuracy_curve.png",
    )

    print("\nGrid search complete.")
    print(f"Best config: lr={best_lr}  batch={best_bs}  dropout={best_dropout}")
    print(f"Best val loss: {best_val_loss:.4f}  val acc: {best_config['val_accuracy']:.4f}")
    print(f"Optimized model saved to: {optimized_model_path}")
    print(f"Results saved to: {report_dir / 'hyperparameter_results.json'}")
    print(f"Best config saved to: {report_dir / 'best_config.json'}")
    print(f"Optimized history saved to: {history_dir / 'optimized_history.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search hyperparameter tuning for fish CNN.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_grid_search(args)


if __name__ == "__main__":
    main()
