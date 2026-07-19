"""Custom CNN model implemented with NumPy.

The project runs in a dependency-light environment, so the model is built
from scratch with manual forward and backward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import json

import numpy as np
from scipy.signal import convolve2d


def _he_scale(fan_in: int) -> float:
    return float(np.sqrt(2.0 / max(1, fan_in)))


class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rng: np.random.Generator) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        scale = _he_scale(in_channels * kernel_size * kernel_size)
        self.weights = rng.normal(
            0.0,
            scale,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self._input_padded: np.ndarray | None = None
        self._windows: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pad = self.kernel_size // 2
        self._input_padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        self._windows = np.lib.stride_tricks.sliding_window_view(
            self._input_padded,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3),
        )
        outputs = np.einsum("bchwkl,fckl->bfhw", self._windows, self.weights, optimize=True)
        return outputs + self.bias[None, :, None, None]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._windows is None or self._input_padded is None:
            raise RuntimeError("Conv2D backward called before forward")

        self.grad_bias = grad_output.sum(axis=(0, 2, 3)).astype(np.float32)
        self.grad_weights = np.einsum("bchwkl,bfhw->fckl", self._windows, grad_output, optimize=True).astype(np.float32)

        grad_input_padded = np.zeros_like(self._input_padded, dtype=np.float32)
        for batch_index in range(grad_output.shape[0]):
            for output_channel in range(self.out_channels):
                upstream = grad_output[batch_index, output_channel]
                for input_channel in range(self.in_channels):
                    grad_input_padded[batch_index, input_channel] += convolve2d(
                        upstream,
                        self.weights[output_channel, input_channel],
                        mode="full",
                    ).astype(np.float32)

        pad = self.kernel_size // 2
        if pad == 0:
            return grad_input_padded
        return grad_input_padded[:, :, pad:-pad, pad:-pad]


class ReLU:
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._mask = inputs > 0
        return np.maximum(inputs, 0.0).astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU backward called before forward")
        return grad_output * self._mask


class MaxPool2D:
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self._input_shape: tuple[int, int, int, int] | None = None
        self._mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = inputs.shape
        pooled_height = height // self.stride
        pooled_width = width // self.stride
        trimmed = inputs[:, :, : pooled_height * self.stride, : pooled_width * self.stride]
        blocks = trimmed.reshape(
            batch_size,
            channels,
            pooled_height,
            self.stride,
            pooled_width,
            self.stride,
        )
        outputs = blocks.max(axis=(3, 5))
        self._mask = blocks == outputs[:, :, :, None, :, None]
        self._input_shape = inputs.shape
        return outputs.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None or self._input_shape is None:
            raise RuntimeError("MaxPool2D backward called before forward")

        grad_blocks = grad_output[:, :, :, None, :, None]
        max_counts = self._mask.sum(axis=(3, 5), keepdims=True)
        max_counts = np.maximum(max_counts, 1)
        grad_trimmed = (self._mask * grad_blocks) / max_counts

        batch_size, channels, height, width = self._input_shape
        output = np.zeros((batch_size, channels, height, width), dtype=np.float32)
        pooled_height = height // self.stride
        pooled_width = width // self.stride
        output[:, :, : pooled_height * self.stride, : pooled_width * self.stride] = grad_trimmed.reshape(
            batch_size,
            channels,
            pooled_height * self.stride,
            pooled_width * self.stride,
        )
        return output


class Flatten:
    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Flatten backward called before forward")
        return grad_output.reshape(self._input_shape)


class Dense:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        scale = _he_scale(in_features)
        self.weights = rng.normal(0.0, scale, size=(out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self._inputs: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._inputs = inputs
        return inputs @ self.weights.T + self.bias[None, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("Dense backward called before forward")
        self.grad_weights = (grad_output.T @ self._inputs).astype(np.float32)
        self.grad_bias = grad_output.sum(axis=0).astype(np.float32)
        return grad_output @ self.weights


@dataclass
class ModelConfig:
    input_channels: int
    num_classes: int
    conv1_filters: int = 8
    conv2_filters: int = 16
    hidden_dim: int = 64
    kernel_size: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0


class FishCNN:
    def __init__(self, config: ModelConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(config.input_channels, config.conv1_filters, config.kernel_size, self.rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(config.conv1_filters, config.conv2_filters, config.kernel_size, self.rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        self.flatten = Flatten()
        self.dense1: Dense | None = None
        self.relu3 = ReLU()
        self.dense2: Dense | None = None

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self._optimizer_step = 0
        self._adam_state: dict[str, dict[str, np.ndarray]] = {}

    def _ensure_dense_layers(self, inputs: np.ndarray) -> None:
        if self.dense1 is not None and self.dense2 is not None:
            return
        flat_size = inputs.reshape(inputs.shape[0], -1).shape[1]
        self.dense1 = Dense(flat_size, self.config.hidden_dim, self.rng)
        self.dense2 = Dense(self.config.hidden_dim, self.config.num_classes, self.rng)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(inputs)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        self._ensure_dense_layers(x)
        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.relu3.forward(x)
        return self.dense2.forward(x)

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(inputs)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(inputs), axis=1)

    def _apply_adam(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        state = self._adam_state.setdefault(
            name,
            {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
            },
        )
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
        m_hat = state["m"] / (1.0 - beta1 ** self._optimizer_step)
        v_hat = state["v"] / (1.0 - beta2 ** self._optimizer_step)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def _parameter_pairs(self) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        if self.dense1 is None or self.dense2 is None:
            raise RuntimeError("Dense layers are not initialized")

        yield "conv1.weights", self.conv1.weights, self.conv1.grad_weights
        yield "conv1.bias", self.conv1.bias, self.conv1.grad_bias
        yield "conv2.weights", self.conv2.weights, self.conv2.grad_weights
        yield "conv2.bias", self.conv2.bias, self.conv2.grad_bias
        yield "dense1.weights", self.dense1.weights, self.dense1.grad_weights
        yield "dense1.bias", self.dense1.bias, self.dense1.grad_bias
        yield "dense2.weights", self.dense2.weights, self.dense2.grad_weights
        yield "dense2.bias", self.dense2.bias, self.dense2.grad_bias

    def backward(self, grad_logits: np.ndarray) -> None:
        if self.dense1 is None or self.dense2 is None:
            raise RuntimeError("Dense layers are not initialized")

        grad = self.dense2.backward(grad_logits)
        grad = self.relu3.backward(grad)
        grad = self.dense1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        _ = self.conv1.backward(grad)

    def step(self) -> None:
        self._optimizer_step += 1
        for name, param, grad in self._parameter_pairs():
            if self.weight_decay and name.endswith("weights"):
                grad = grad + self.weight_decay * param
            self._apply_adam(name, param, grad)

    def loss_and_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        logits = self.forward(inputs)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        batch_size = inputs.shape[0]
        sample_weights = np.ones(batch_size, dtype=np.float32)
        if class_weights is not None:
            sample_weights = class_weights[targets]
        normalizer = float(sample_weights.sum()) if float(sample_weights.sum()) > 0 else float(batch_size)
        target_probabilities = probabilities[np.arange(batch_size), targets]
        loss = -np.sum(sample_weights * np.log(target_probabilities + 1e-12)) / normalizer

        grad_logits = probabilities.copy()
        grad_logits[np.arange(batch_size), targets] -= 1.0
        grad_logits *= (sample_weights / normalizer)[:, None]
        return float(loss), grad_logits.astype(np.float32)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> float:
        loss, grad_logits = self.loss_and_gradients(inputs, targets, class_weights=class_weights)
        self.backward(grad_logits)
        self.step()
        return loss

    def save_weights(self, path: str) -> None:
        if self.dense1 is None or self.dense2 is None:
            raise RuntimeError("Dense layers are not initialized")

        payload = {
            "config": np.array(json.dumps(self.config.__dict__)),
            "conv1_weights": self.conv1.weights,
            "conv1_bias": self.conv1.bias,
            "conv2_weights": self.conv2.weights,
            "conv2_bias": self.conv2.bias,
            "dense1_weights": self.dense1.weights,
            "dense1_bias": self.dense1.bias,
            "dense2_weights": self.dense2.weights,
            "dense2_bias": self.dense2.bias,
        }
        np.savez_compressed(path, **payload)

    def load_weights(self, path: str) -> None:
        archive = np.load(path, allow_pickle=False)
        self.conv1.weights = archive["conv1_weights"]
        self.conv1.bias = archive["conv1_bias"]
        self.conv2.weights = archive["conv2_weights"]
        self.conv2.bias = archive["conv2_bias"]
        if self.dense1 is None or self.dense2 is None:
            raise RuntimeError("Dense layers are not initialized")
        self.dense1.weights = archive["dense1_weights"]
        self.dense1.bias = archive["dense1_bias"]
        self.dense2.weights = archive["dense2_weights"]
        self.dense2.bias = archive["dense2_bias"]
