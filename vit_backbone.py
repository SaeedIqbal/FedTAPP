import torch
from torchvision import models

class ViTBackbone(torch.nn.Module):
    """
    Frozen Vision Transformer backbone for feature extraction.
    """

    def __init__(self, model_name='vit_b_16', pretrained=True):
        super(ViTBackbone, self).__init__()
        # Load pre-trained ViT model
        if model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_tiny':
            self.model = models.vit_tiny(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Use encoder features only
        self.feature_extractor = self.model.encoder

    def forward(self, x):
        """
        Extracts features using ViT encoder.
        Returns CLS token embedding.
        """
        batch_size, _, _, _ = x.shape
        patches = self.model.patch_embed(x)
        cls_token = self.model.class_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls_token, patches), dim=1)
        features = self.feature_extractor(patches)
        return features[:, 0, :]  # Return CLS token