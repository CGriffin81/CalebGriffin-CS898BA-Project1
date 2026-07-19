"""Final evaluation script for the fish species CNN project.

Loads saved baseline and optimized model weights.
Evaluates both on the held-out test set only.
No retraining occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

try:
    from .model import FishCNN, ModelConfig
    from .train import (
        build_dataloaders,
        compute_baseline_class_weights,
        compute_metrics,
        evaluate,
        load_dataset,
        plot_confusion_matrix,
        plot_metric_curve,
        set_seed,
        stratified_split,
    )
except ImportError:  # supports direct script execution
    from model import FishCNN, ModelConfig
    from train import (
        build_dataloaders,
        compute_baseline_class_weights,
        compute_metrics,
        evaluate,
        load_dataset,
        plot_confusion_matrix,
        plot_metric_curve,
        set_seed,
        stratified_split,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fish"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[FishCNN, np.ndarray, np.ndarray, list[str]]:
    """Load a saved model checkpoint and return model + normalization stats."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    config = ModelConfig(**config_dict)
    model = FishCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)
    class_names: list[str] = checkpoint["class_names"]
    return model, mean, std, class_names


def compute_aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute per-class, macro, and weighted precision/recall/F1."""
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = []
    precisions, recalls, f1s, supports = [], [], [], []
    for idx, name in enumerate(class_names):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        support = float(confusion[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int(support)})
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    accuracy = float((y_true == y_pred).mean())
    total = sum(supports)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total)

    return {
        "accuracy": accuracy,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "weighted": {"precision": weighted_precision, "recall": weighted_recall, "f1": weighted_f1},
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def write_classification_report(metrics: dict, class_names: list[str], output_path: Path) -> None:
    lines = ["Classification Report", ""]
    lines.append(f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for entry in metrics["per_class"]:
        lines.append(
            f"{entry['class']:<16}{entry['precision']:>10.3f}{entry['recall']:>10.3f}"
            f"{entry['f1']:>10.3f}{entry['support']:>10d}"
        )
    lines.append("")
    lines.append(f"Accuracy: {metrics['accuracy']:.3f}")
    lines.append("")
    m = metrics["macro"]
    lines.append(f"Macro avg     {m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}")
    w = metrics["weighted"]
    lines.append(f"Weighted avg  {w['precision']:>10.3f}{w['recall']:>10.3f}{w['f1']:>10.3f}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_eval_grid(
    baseline_history: dict,
    optimized_history: dict,
    optimized_confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """5-panel README-ready grid: 2 baseline curves, 2 optimized curves, confusion matrix."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    b_epochs = np.arange(1, len(baseline_history["train_loss"]) + 1)
    o_epochs = np.arange(1, len(optimized_history["train_loss"]) + 1)

    # Baseline loss
    axes[0, 0].plot(b_epochs, baseline_history["train_loss"], label="Train")
    axes[0, 0].plot(b_epochs, baseline_history["val_loss"], label="Val")
    axes[0, 0].set_title("Baseline — Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Baseline accuracy
    axes[0, 1].plot(b_epochs, baseline_history["train_accuracy"], label="Train")
    axes[0, 1].plot(b_epochs, baseline_history["val_accuracy"], label="Val")
    axes[0, 1].set_title("Baseline — Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Optimized loss
    axes[1, 0].plot(o_epochs, optimized_history["train_loss"], label="Train")
    axes[1, 0].plot(o_epochs, optimized_history["val_loss"], label="Val")
    axes[1, 0].set_title("Optimized — Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Optimized accuracy
    axes[1, 1].plot(o_epochs, optimized_history["train_accuracy"], label="Train")
    axes[1, 1].plot(o_epochs, optimized_history["val_accuracy"], label="Val")
    axes[1, 1].set_title("Optimized — Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    # Optimized confusion matrix (spans last column, both rows)
    ax_cm = fig.add_subplot(1, 3, 3)
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.set_visible(False)

    img = ax_cm.imshow(optimized_confusion, cmap="Blues")
    ax_cm.set_xticks(np.arange(len(class_names)))
    ax_cm.set_yticks(np.arange(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Optimized Model — Confusion Matrix (Test Set)")
    fig.colorbar(img, ax=ax_cm)
    for r in range(optimized_confusion.shape[0]):
        for c in range(optimized_confusion.shape[1]):
            ax_cm.text(c, r, str(optimized_confusion[r, c]), ha="center", va="center", fontsize=9)

    fig.suptitle("Fish Species CNN — Final Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(args: argparse.Namespace) -> None:
    set_seed(42)
    device = torch.device("cpu")  # evaluation only — CPU is sufficient

    output_dir = Path(args.output_dir)
    report_dir = output_dir / "reports"
    plot_dir = output_dir / "plots"
    history_dir = output_dir / "history"
    for d in (report_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load saved histories (no retraining)
    baseline_history: dict = json.loads((history_dir / "baseline_history.json").read_text(encoding="utf-8"))
    optimized_history: dict = json.loads((history_dir / "optimized_history.json").read_text(encoding="utf-8"))
    best_config: dict = json.loads((report_dir / "best_config.json").read_text(encoding="utf-8"))

    # Load checkpoints
    baseline_model, baseline_mean, baseline_std, class_names = load_checkpoint(
        output_dir / "models" / "baseline_cnn.pt", device
    )
    optimized_model, opt_mean, opt_std, _ = load_checkpoint(
        output_dir / "models" / "optimized_cnn.pt", device
    )

    # Rebuild the same test split used during training
    images, labels, _ = load_dataset(Path(args.data_dir), args.image_size)
    train_indices, val_indices, test_indices = stratified_split(labels, 0.7, 0.15, 0.15, 42)

    class_weights = compute_baseline_class_weights(labels[train_indices], len(class_names))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Build test loaders using each model's saved normalization stats
    from .train import FishDataset  # noqa: PLC0415
    from torch.utils.data import DataLoader

    def make_test_loader(mean: np.ndarray, std: np.ndarray, batch_size: int = 32) -> DataLoader:
        ds = FishDataset(images, labels, test_indices, mean, std, augment=False, seed=42)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    baseline_test_loader = make_test_loader(baseline_mean, baseline_std)
    optimized_test_loader = make_test_loader(opt_mean, opt_std)

    # Evaluate on test set only
    print("Evaluating baseline model on test set...")
    _, _, b_true, b_pred = evaluate(baseline_model, baseline_test_loader, criterion, device)

    print("Evaluating optimized model on test set...")
    _, _, o_true, o_pred = evaluate(optimized_model, optimized_test_loader, criterion, device)

    # Compute full metrics
    baseline_metrics = compute_aggregate_metrics(b_true, b_pred, class_names)
    optimized_metrics = compute_aggregate_metrics(o_true, o_pred, class_names)

    # Write classification reports
    write_classification_report(
        baseline_metrics, class_names, report_dir / "baseline_classification_report.txt"
    )
    write_classification_report(
        optimized_metrics, class_names, report_dir / "optimized_classification_report.txt"
    )

    # Individual training curve plots (from saved histories)
    plot_metric_curve(
        baseline_history["train_loss"], baseline_history["val_loss"],
        "Baseline Loss", "Loss", plot_dir / "baseline_loss_curve.png",
    )
    plot_metric_curve(
        baseline_history["train_accuracy"], baseline_history["val_accuracy"],
        "Baseline Accuracy", "Accuracy", plot_dir / "baseline_accuracy_curve.png",
    )
    plot_metric_curve(
        optimized_history["train_loss"], optimized_history["val_loss"],
        "Optimized Loss", "Loss", plot_dir / "optimized_loss_curve.png",
    )
    plot_metric_curve(
        optimized_history["train_accuracy"], optimized_history["val_accuracy"],
        "Optimized Accuracy", "Accuracy", plot_dir / "optimized_accuracy_curve.png",
    )

    # Optimized confusion matrix (test set)
    opt_confusion = np.array(optimized_metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(opt_confusion, class_names, plot_dir / "optimized_confusion_matrix.png")

    # README-ready 5-panel evaluation grid
    plot_eval_grid(
        baseline_history,
        optimized_history,
        opt_confusion,
        class_names,
        plot_dir / "final_evaluation_grid.png",
    )

    # Comparison JSON
    def delta(opt_val: float, base_val: float) -> float:
        return round(opt_val - base_val, 6)

    comparison = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_precision": baseline_metrics["macro"]["precision"],
            "macro_recall": baseline_metrics["macro"]["recall"],
            "macro_f1": baseline_metrics["macro"]["f1"],
            "weighted_precision": baseline_metrics["weighted"]["precision"],
            "weighted_recall": baseline_metrics["weighted"]["recall"],
            "weighted_f1": baseline_metrics["weighted"]["f1"],
        },
        "optimized": {
            "accuracy": optimized_metrics["accuracy"],
            "macro_precision": optimized_metrics["macro"]["precision"],
            "macro_recall": optimized_metrics["macro"]["recall"],
            "macro_f1": optimized_metrics["macro"]["f1"],
            "weighted_precision": optimized_metrics["weighted"]["precision"],
            "weighted_recall": optimized_metrics["weighted"]["recall"],
            "weighted_f1": optimized_metrics["weighted"]["f1"],
        },
        "improvement": {
            "accuracy": delta(optimized_metrics["accuracy"], baseline_metrics["accuracy"]),
            "macro_f1": delta(optimized_metrics["macro"]["f1"], baseline_metrics["macro"]["f1"]),
            "weighted_f1": delta(optimized_metrics["weighted"]["f1"], baseline_metrics["weighted"]["f1"]),
        },
        "optimized_improved": optimized_metrics["accuracy"] > baseline_metrics["accuracy"],
        "best_config": best_config,
    }
    (report_dir / "final_model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    print("\n--- Evaluation Results ---")
    print(f"Baseline  accuracy: {baseline_metrics['accuracy']:.4f}  macro-F1: {baseline_metrics['macro']['f1']:.4f}")
    print(f"Optimized accuracy: {optimized_metrics['accuracy']:.4f}  macro-F1: {optimized_metrics['macro']['f1']:.4f}")
    print(f"Improvement:        {comparison['improvement']['accuracy']:+.4f} accuracy  {comparison['improvement']['macro_f1']:+.4f} macro-F1")
    print(f"\nArtifacts written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and optimized fish CNN models.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
