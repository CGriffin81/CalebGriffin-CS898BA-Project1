"""Training entry point for the fish species CNN project."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .model import FishCNN, ModelConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from model import FishCNN, ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fish"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        val_indices.extend(label_indices[train_count : train_count + val_count])
        test_indices.extend(label_indices[train_count + val_count : train_count + val_count + test_count])

    return np.asarray(train_indices), np.asarray(val_indices), np.asarray(test_indices)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    for class_index, count in enumerate(counts):
        weights[class_index] = total / (num_classes * count) if count > 0 else 0.0
    weights /= weights.mean() if weights.mean() > 0 else 1.0
    return weights


def compute_baseline_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return compute_class_weights(labels, num_classes)


def compute_normalization_stats(train_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_images.mean(axis=(0, 1, 2), keepdims=False).reshape(1, 1, -1)
    std = train_images.std(axis=(0, 1, 2), keepdims=False).reshape(1, 1, -1)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = image.copy()

    # Random horizontal flip
    if rng.random() < 0.5:
        augmented = np.flip(augmented, axis=1)

    # Minor rotation: ±15 degrees
    if rng.random() < 0.5:
        angle = rng.uniform(-15.0, 15.0)
        h, w = augmented.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Brightness and contrast adjustment
    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)
    augmented = np.clip((augmented - 0.5) * contrast + 0.5, 0.0, 1.0)
    augmented = np.clip(augmented * brightness, 0.0, 1.0)
    return np.ascontiguousarray(augmented)


class FishDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
        seed: int,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        image = self.images[index]
        if self.augment:
            image = apply_augmentation(image, self.rng)
        normalized = (image - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return tensor, label


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


def plot_metric_curve(
    train_values: list[float],
    val_values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(train_values) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train_values, label=f"Train {ylabel.lower()}")
    axis.plot(epochs, val_values, label=f"Val {ylabel.lower()}")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.legend()
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


def build_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_images = images[train_indices]
    mean, std = compute_normalization_stats(train_images)

    train_dataset = FishDataset(images, labels, train_indices, mean, std, augment=True, seed=seed)
    val_dataset = FishDataset(images, labels, val_indices, mean, std, augment=False, seed=seed)
    test_dataset = FishDataset(images, labels, test_indices, mean, std, augment=False, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, mean, std


def create_model(config: ModelConfig, device: torch.device) -> FishCNN:
    return FishCNN(config).to(device)


def evaluate(
    model: FishCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == targets).sum().item())
            total_count += batch_size
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    return total_loss / max(1, total_count), total_correct / max(1, total_count), y_true, y_pred


def train_model(
    model: FishCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -np.inf

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((predictions == targets).sum().item())
            running_count += batch_size

        train_loss = running_loss / max(1, running_count)
        train_accuracy = running_correct / max(1, running_count)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)
    return history, best_state


def save_baseline_artifacts(
    model: FishCNN,
    history: dict[str, list[float]],
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    class_names: list[str],
    test_loss: float,
    test_accuracy: float,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
) -> None:
    model_dir = output_dir / "models"
    history_dir = output_dir / "history"
    plot_dir = output_dir / "plots"
    report_dir = output_dir / "reports"

    for directory in (model_dir, history_dir, plot_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "class_names": class_names,
        },
        model_path,
    )

    history_path = history_dir / "baseline_history.json"
    history_payload = {
        **history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    plot_metric_curve(history["train_loss"], history["val_loss"], "Baseline Loss", "Loss", plot_dir / "baseline_loss_curve.png")
    plot_metric_curve(history["train_accuracy"], history["val_accuracy"], "Baseline Accuracy", "Accuracy", plot_dir / "baseline_accuracy_curve.png")

    confusion, report = compute_metrics(test_labels, test_predictions, class_names)
    (report_dir / "baseline_classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(confusion, class_names, plot_dir / "baseline_confusion_matrix.png")

    print(f"Baseline model saved to: {model_path}")
    print(f"Baseline history saved to: {history_path}")
    print(f"Baseline plots saved to: {plot_dir}")
    print(f"Baseline report saved to: {report_dir / 'baseline_classification_report.txt'}")


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

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

    train_loader, val_loader, test_loader, mean, std = build_dataloaders(
        images,
        labels,
        train_indices,
        val_indices,
        test_indices,
        args.batch_size,
        args.seed,
    )

    class_weights = compute_baseline_class_weights(labels[train_indices], len(class_names))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    baseline_config = ModelConfig(3, len(class_names), image_size=args.image_size)
    final_model = create_model(baseline_config, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=baseline_config.learning_rate, weight_decay=baseline_config.weight_decay)
    history, _ = train_model(
        final_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
    )

    test_loss, test_accuracy, test_labels, test_predictions = evaluate(final_model, test_loader, criterion, device)
    save_baseline_artifacts(
        final_model,
        history,
        output_dir,
        mean,
        std,
        class_names,
        test_loss,
        test_accuracy,
        test_labels,
        test_predictions,
    )

    (report_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "classes": class_names,
                "train_size": int(len(train_indices)),
                "val_size": int(len(val_indices)),
                "test_size": int(len(test_indices)),
                "class_weights": class_weights.tolist(),
                "device": str(device),
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training complete")
    print(f"Device: {device}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Model saved to: {model_dir / 'baseline_cnn.pt'}")
    print(f"Reports saved to: {report_dir}")
    print(f"Plots saved to: {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN on the fish dataset.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
