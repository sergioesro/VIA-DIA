"""
Combined image-difference and weather-GRU model for damage assessment.

This architecture fuses the two lightweight variants:
- CNN feature difference from pre/post satellite images
- GRU + attention context from ERA-5 weather/climate data
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .baseline_model import SimpleAttention, SimpleClimateEncoder
from .shared_components import CNNBackbone
from .weather_gru_model import SimpleGRUTemporalEncoder


class ImageDifferenceWeatherGRUModel(nn.Module):
    """
    Combined image-difference and weather-GRU classifier.

    Architecture:
    - Visual branch: shared CNN encodes pre/post images
    - Difference embedding: absolute difference between visual features
    - Weather branch: spatial encoder + GRU temporal encoder + attention
    - Fusion: concatenate image difference and weather context
    - Classifier: MLP over the fused embedding

    Parameters
    ----------
    num_classes : int
        Number of damage classes.
    dropout_rate : float
        Dropout rate for regularization.
    backbone : str
        Visual backbone type ('cnn' only).
    vis_dim : int
        Visual feature difference dimension.
    climate_dim : int
        Weather temporal feature dimension.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        backbone: str = 'cnn',
        vis_dim: int = 256,
        climate_dim: int = 128,
    ):
        super().__init__()

        assert backbone == 'cnn', "Only CNN backbone supported in ImageDifferenceWeatherGRUModel"

        self.visual_backbone = CNNBackbone(out_dim=vis_dim)

        spatial_dim = 64
        self.climate_spatial = SimpleClimateEncoder(in_channels=10, out_dim=spatial_dim)
        self.climate_temporal = SimpleGRUTemporalEncoder(
            in_dim=spatial_dim,
            out_dim=climate_dim,
            dropout=dropout_rate * 0.5,
        )
        self.climate_attention = SimpleAttention(
            dim=climate_dim,
            num_heads=max(1, climate_dim // 32),
            dropout=dropout_rate * 0.5,
        )

        fusion_dim = vis_dim + climate_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform where appropriate."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_climate: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using satellite images and weather/climate data.

        Parameters
        ----------
        x_pre : torch.Tensor
            Pre-disaster satellite image, shape (B, 3, 64, 64).
        x_post : torch.Tensor
            Post-disaster satellite image, shape (B, 3, 64, 64).
        x_climate : torch.Tensor
            Weather/climate time series, shape (B, 10, T, 20, 20).
        event_labels : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.

        Returns
        -------
        logits : torch.Tensor
            Class logits, shape (B, num_classes).
        att_weights : torch.Tensor
            Weather attention weights, shape (B, T).
        """
        f_pre = self.visual_backbone(x_pre)
        f_post = self.visual_backbone(x_post)
        f_diff = torch.abs(f_post - f_pre)

        z = self.climate_spatial(x_climate)
        z = self.climate_temporal(z)
        f_climate, att_weights = self.climate_attention(z)

        fused = torch.cat([f_diff, f_climate], dim=1)
        logits = self.classifier(fused)
        return logits, att_weights


def create_image_difference_weather_gru_model(
    num_classes: int = 4,
    dropout_rate: float = 0.2,
    backbone: str = 'cnn',
    vis_dim: int = 256,
    climate_dim: int = 128,
    **kwargs
) -> ImageDifferenceWeatherGRUModel:
    """
    Factory function to create the combined image-difference/weather-GRU model.

    Additional keyword arguments are ignored for compatibility with shared
    training configs.
    """
    return ImageDifferenceWeatherGRUModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        backbone=backbone,
        vis_dim=vis_dim,
        climate_dim=climate_dim,
    )
