"""PyTorch CNN for fish classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_channels: int
    num_classes: int
    conv1_filters: int = 16
    conv2_filters: int = 32
    hidden_dim: int = 128
    kernel_size: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


class FishCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        padding = config.kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, config.conv1_filters, kernel_size=config.kernel_size, padding=padding),
            nn.BatchNorm2d(config.conv1_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(config.conv1_filters, config.conv2_filters, kernel_size=config.kernel_size, padding=padding),
            nn.BatchNorm2d(config.conv2_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.conv2_filters * 4 * 4, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
