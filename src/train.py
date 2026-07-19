"""Training entry point for the fish species CNN project."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from .model import FishCNN, ModelConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from model import FishCNN, ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fish"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def discover_dataset(data_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((image_path, class_dir.name))
    if not samples:
        raise ValueError(f"No images found under {data_dir}")
    return samples


def load_image(image_path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_dataset(data_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples = discover_dataset(data_dir)
    class_names = sorted({label for _, label in samples})
    class_to_index = {label: index for index, label in enumerate(class_names)}

    images = [load_image(path, image_size) for path, _ in samples]
    labels = [class_to_index[label] for _, label in samples]
    return np.stack(images), np.asarray(labels, dtype=np.int64), class_names


def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        count = len(label_indices)
        train_count = int(round(count * train_ratio))
        val_count = int(round(count * val_ratio))
        if train_count + val_count >= count:
            val_count = max(1, count - train_count - 1)
        test_count = count - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

        train_indices.extend(label_indices[:train_count])
        val_indices.extend(label_indices[train_count:train_count + val_count])
        test_indices.extend(label_indices[train_count + val_count:train_count + val_count + test_count])

    return np.asarray(train_indices), np.asarray(val_indices), np.asarray(test_indices)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    for class_index, count in enumerate(counts):
        weights[class_index] = total / (num_classes * count) if count > 0 else 0.0
    weights /= weights.mean() if weights.mean() > 0 else 1.0
    return weights


def standardize(train_images: np.ndarray, *other_images: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_images.mean(axis=(0, 2, 3), keepdims=True)
    std = train_images.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, 1e-6)
    standardized = [(train_images - mean) / std]
    standardized.extend((images - mean) / std for images in other_images)
    return tuple(standardized)


def apply_augmentation(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = batch.copy()
    for index in range(augmented.shape[0]):
        image = augmented[index]
        if rng.random() < 0.5:
            image = np.flip(image, axis=1)
        brightness = rng.uniform(0.9, 1.1)
        contrast = rng.uniform(0.9, 1.1)
        image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)
        image = np.clip(image * brightness, 0.0, 1.0)
        augmented[index] = image
    return augmented


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
):
    indices = np.arange(len(images))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, str]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion[true_label, predicted_label] += 1

    lines = ["Classification report", ""]
    lines.append(f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for class_index, class_name in enumerate(class_names):
        true_positive = float(confusion[class_index, class_index])
        false_positive = float(confusion[:, class_index].sum() - true_positive)
        false_negative = float(confusion[class_index, :].sum() - true_positive)
        support = float(confusion[class_index, :].sum())
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        lines.append(f"{class_name:<16}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{support:>10.0f}")

    accuracy = float((y_true == y_pred).mean())
    lines.append("")
    lines.append(f"Accuracy: {accuracy:.3f}")
    return confusion, "\n".join(lines)


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    figure.colorbar(image, ax=axis)

    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            axis.text(col_index, row_index, str(confusion[row_index, col_index]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def train_model(
    model: FishCNN,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    class_weights: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    augment: bool,
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_state: dict[str, np.ndarray] | None = None
    best_val_accuracy = -np.inf

    for _epoch in range(epochs):
        batch_losses: list[float] = []
        for batch_images, batch_labels in iterate_minibatches(train_images, train_labels, batch_size, rng, shuffle=True):
            if augment:
                batch_images = apply_augmentation(batch_images, rng)
            loss = model.train_batch(batch_images, batch_labels, class_weights=class_weights)
            batch_losses.append(loss)

        train_predictions = model.predict(train_images)
        val_predictions = model.predict(val_images)
        train_logits = model.predict_logits(train_images)
        val_logits = model.predict_logits(val_images)

        train_probs = softmax(train_logits)
        val_probs = softmax(val_logits)
        _train_loss = float(-np.log(train_probs[np.arange(len(train_labels)), train_labels] + 1e-12).mean())
        val_loss = float(-np.log(val_probs[np.arange(len(val_labels)), val_labels] + 1e-12).mean())

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(float((train_predictions == train_labels).mean()))
        history["val_accuracy"].append(float((val_predictions == val_labels).mean()))

        if history["val_accuracy"][-1] > best_val_accuracy:
            best_val_accuracy = history["val_accuracy"][-1]
            best_state = {
                "conv1_weights": model.conv1.weights.copy(),
                "conv1_bias": model.conv1.bias.copy(),
                "conv2_weights": model.conv2.weights.copy(),
                "conv2_bias": model.conv2.bias.copy(),
                "dense1_weights": model.dense1.weights.copy() if model.dense1 is not None else None,
                "dense1_bias": model.dense1.bias.copy() if model.dense1 is not None else None,
                "dense2_weights": model.dense2.weights.copy() if model.dense2 is not None else None,
                "dense2_bias": model.dense2.bias.copy() if model.dense2 is not None else None,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.conv1.weights = best_state["conv1_weights"]
    model.conv1.bias = best_state["conv1_bias"]
    model.conv2.weights = best_state["conv2_weights"]
    model.conv2.bias = best_state["conv2_bias"]
    if model.dense1 is not None and model.dense2 is not None:
        model.dense1.weights = best_state["dense1_weights"]
        model.dense1.bias = best_state["dense1_bias"]
        model.dense2.weights = best_state["dense2_weights"]
        model.dense2.bias = best_state["dense2_bias"]

    return history, best_state


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    plot_dir = output_dir / "plots"
    for directory in (model_dir, report_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images, labels, class_names = load_dataset(Path(args.data_dir), args.image_size)
    train_indices, val_indices, test_indices = stratified_split(
        labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_images = images[train_indices].transpose(0, 3, 1, 2)
    val_images = images[val_indices].transpose(0, 3, 1, 2)
    test_images = images[test_indices].transpose(0, 3, 1, 2)
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    train_images, val_images, test_images = standardize(train_images, val_images, test_images)
    class_weights = compute_class_weights(train_labels, len(class_names))

    candidate_configs = [
        ModelConfig(3, len(class_names), conv1_filters=8, conv2_filters=16, hidden_dim=64, learning_rate=0.001),
        ModelConfig(3, len(class_names), conv1_filters=12, conv2_filters=24, hidden_dim=64, learning_rate=0.0007),
        ModelConfig(3, len(class_names), conv1_filters=8, conv2_filters=16, hidden_dim=96, learning_rate=0.0007),
    ]

    tuning_results: list[dict[str, float]] = []
    best_config = candidate_configs[0]
    best_val_accuracy = -np.inf

    for config in candidate_configs:
        model = FishCNN(config, seed=args.seed)
        history, _ = train_model(
            model,
            train_images,
            train_labels,
            val_images,
            val_labels,
            class_weights,
            epochs=args.tune_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            augment=True,
        )
        validation_accuracy = history["val_accuracy"][-1]
        tuning_results.append({
            "conv1_filters": float(config.conv1_filters),
            "conv2_filters": float(config.conv2_filters),
            "hidden_dim": float(config.hidden_dim),
            "learning_rate": float(config.learning_rate),
            "validation_accuracy": float(validation_accuracy),
        })
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            best_config = config

    final_model = FishCNN(best_config, seed=args.seed)
    history, _ = train_model(
        final_model,
        train_images,
        train_labels,
        val_images,
        val_labels,
        class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=True,
    )

    test_predictions = final_model.predict(test_images)
    confusion, report = compute_metrics(test_labels, test_predictions, class_names)

    final_model.save_weights(str(model_dir / "fish_cnn_best.npz"))
    plot_history(history, plot_dir / "training_curves.png")
    plot_confusion_matrix(confusion, class_names, plot_dir / "confusion_matrix.png")

    (report_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (report_dir / "hyperparameter_search.json").write_text(json.dumps(tuning_results, indent=2), encoding="utf-8")
    (report_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "classes": class_names,
                "train_size": int(len(train_labels)),
                "val_size": int(len(val_labels)),
                "test_size": int(len(test_labels)),
                "class_weights": class_weights.tolist(),
                "best_config": best_config.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training complete")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Test accuracy: {float((test_predictions == test_labels).mean()):.3f}")
    print(f"Model saved to: {model_dir / 'fish_cnn_best.npz'}")
    print(f"Reports saved to: {report_dir}")
    print(f"Plots saved to: {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom CNN on the fish dataset.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--tune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
