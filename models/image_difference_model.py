"""
Image-only CNN difference model for building damage assessment.

This architecture intentionally ignores climate and event-label inputs while
keeping the same forward signature as the multimodal baseline, so it can be
selected from the existing training pipeline with only the architecture name.
"""

from typing import Optional

import torch
import torch.nn as nn

from .shared_components import CNNBackbone


class ImageDifferenceCNNModel(nn.Module):
    """
    CNN-only model that classifies damage from the pre/post image difference.

    Architecture:
    - Shared CNN backbone encodes pre- and post-disaster satellite images
    - Difference embedding is computed from the visual features
    - Classifier predicts the post-disaster damage class from that difference

    Parameters
    ----------
    num_classes : int
        Number of damage classes.
    dropout_rate : float
        Dropout rate for regularization.
    backbone : str
        Visual backbone type ('cnn' only).
    vis_dim : int
        Visual feature and difference embedding dimension.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        backbone: str = 'cnn',
        vis_dim: int = 256,
    ):
        super().__init__()

        assert backbone == 'cnn', "Only CNN backbone supported in ImageDifferenceCNNModel"

        self.visual_backbone = CNNBackbone(out_dim=vis_dim)
        self.difference_dim = vis_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.difference_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_climate: Optional[torch.Tensor] = None,
        event_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using only satellite imagery.

        Parameters
        ----------
        x_pre : torch.Tensor
            Pre-disaster satellite image, shape (B, 3, 64, 64).
        x_post : torch.Tensor
            Post-disaster satellite image, shape (B, 3, 64, 64).
        x_climate : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.
        event_labels : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.

        Returns
        -------
        logits : torch.Tensor
            Class logits, shape (B, num_classes).
        """
        f_pre = self.visual_backbone(x_pre)
        f_post = self.visual_backbone(x_post)
        f_diff = torch.abs(f_post - f_pre)

        return self.classifier(f_diff)


def create_image_difference_model(
    num_classes: int = 4,
    dropout_rate: float = 0.2,
    backbone: str = 'cnn',
    vis_dim: int = 256,
    **kwargs
) -> ImageDifferenceCNNModel:
    """
    Factory function to create the image-only CNN difference model.

    Additional keyword arguments are ignored so the existing config can be
    reused even when it contains multimodal-only options such as climate_dim.
    """
    return ImageDifferenceCNNModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        backbone=backbone,
        vis_dim=vis_dim,
    )
