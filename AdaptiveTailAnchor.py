import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTailAnchor(nn.Module):
    def __init__(self, feature_dim=192, num_classes=100, smoothing_coeff=0.5):
        super().__init__()
        self.anchor_points = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.smoothing_coeff = smoothing_coeff

    def update_anchor(self, new_prototype, class_idx):
        with torch.no_grad():
            self.anchor_points[class_idx] = (
                self.smoothing_coeff * new_prototype +
                (1 - self.smoothing_coeff) * self.anchor_points[class_idx]
            )

    def forward(self, features):
        return torch.matmul(features, self.anchor_points.T)