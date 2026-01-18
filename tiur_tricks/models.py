from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """A small CNN for quick experiments."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infer_in_channels(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset in {"mnist", "fashionmnist", "fashion-mnist"}:
        return 1
    return 3


def make_resnet18(in_channels: int, num_classes: int) -> nn.Module:
    # Lazy import so that users can run the quick suite (SmallCNN) even if
    # torchvision is not installed or is temporarily broken in the runtime.
    try:
        from torchvision import models as tv_models  # type: ignore
    except Exception as e:
        raise ImportError(
            "torchvision failed to import, but resnet18 was requested. "
            "In Colab, avoid reinstalling torchvision unless you also install a matching torch build. "
            f"Original error: {e!r}"
        )

    m = tv_models.resnet18(num_classes=num_classes)
    if in_channels != 3:
        # Replace first conv to accept 1-channel input
        old = m.conv1
        m.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
    return m


def make_model(model_name: str, dataset: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    in_ch = infer_in_channels(dataset)

    if model_name == "resnet18":
        return make_resnet18(in_ch, num_classes)

    if model_name in {"small_cnn", "cnn", "smallcnn"}:
        return SmallCNN(in_ch, num_classes)

    raise ValueError(f"Unknown model: {model_name}")
