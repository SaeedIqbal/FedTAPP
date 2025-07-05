# fedta.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTailAnchor(nn.Module):
    """
    Tail anchor module to stabilize feature embeddings under concept drift.
    """

    def __init__(self, feature_dim=768, num_classes=100, smoothing_coeff=0.5):
        super(AdaptiveTailAnchor, self).__init__()
        self.anchor_points = nn.Parameter(F.normalize(torch.randn(num_classes, feature_dim), p=2, dim=1))
        self.smoothing_coeff = smoothing_coeff

    def update_anchor(self, new_prototype, class_idx):
        """
        Update anchor points with moving average based on drift estimation.
        """
        with torch.no_grad():
            new_prototype = F.normalize(new_prototype.unsqueeze(0), p=2, dim=1)
            old_anchor = self.anchor_points[class_idx]
            self.anchor_points.data[class_idx] = F.normalize(
                self.smoothing_coeff * new_prototype + (1 - self.smoothing_coeff) * old_anchor,
                p=2, dim=1
            )

    def forward(self, features):
        """
        Compute similarity between features and tail anchors.
        """
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, self.anchor_points.T)
        return logits


class FedTAPlusPlusClient(nn.Module):
    """
    Client-side model using frozen ViT + tail anchor.
    """

    def __init__(self, feature_dim=768, num_classes=100):
        super(FedTAPlusPlusClient, self).__init__()
        self.feature_extractor = ViTBackbone(model_name='vit_tiny', pretrained=True)
        self.tail_anchor = AdaptiveTailAnchor(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.tail_anchor(features)
        return logits

    def extract_features(self, x):
        return self.feature_extractor(x)