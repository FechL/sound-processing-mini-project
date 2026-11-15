"""Frozen Tiny CNN architecture shared by training, evaluation and deployment."""

import numpy as np
import torch
from torch import nn


class TinyKWSNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class QuantTinyKWS(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.gap(self.relu2(self.bn2(self.conv2(x))))
        return self.dequant(self.fc(x.flatten(1)))


def load_logmel(root, sample_id, target_frames=96):
    with np.load(
        root / "features/logmel/mp09" / f"{sample_id}.npz", allow_pickle=True
    ) as data:
        x = data["log_mel"].astype(np.float32)
        label = str(data["label"])
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise RuntimeError(f"Log-Mel tidak valid: {sample_id}")
    if len(x) >= target_frames:
        x = x[:target_frames]
    else:
        x = np.pad(
            x, ((0, target_frames - len(x)), (0, 0)), constant_values=float(x.min())
        )
    return x, label
