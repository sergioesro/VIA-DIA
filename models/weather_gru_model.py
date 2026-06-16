"""
Weather-only GRU model for building damage assessment.

This architecture uses only ERA-5 weather/climate tensors and ignores the
pre/post satellite images while preserving the training pipeline interface.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .baseline_model import SimpleAttention, SimpleClimateEncoder


class SimpleGRUTemporalEncoder(nn.Module):
    """
    Simple temporal encoder using a GRU.

    Processes weather time series with recurrence.
    Input:  (B, T, in_dim)
    Output: (B, T, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gru = nn.GRU(
            input_size=out_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        Returns: (B, T, out_dim)
        """
        x = self.proj(x)
        res = x
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = self.norm(x + res)
        return x


class WeatherGRUModel(nn.Module):
    """
    Weather-only model with spatial encoding, GRU temporal encoding, attention.

    Architecture:
    - Weather branch: per-timestep spatial CNN encoder for ERA-5 variables
    - Temporal branch: GRU sequence encoder instead of Conv1d
    - Attention: same simple attention pooling used by the baseline
    - Classifier: MLP over the weather context vector

    Parameters
    ----------
    num_classes : int
        Number of damage classes.
    dropout_rate : float
        Dropout rate for regularization.
    climate_dim : int
        Weather temporal feature dimension.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        climate_dim: int = 128,
    ):
        super().__init__()

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

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(climate_dim, 256),
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
        x_pre: Optional[torch.Tensor],
        x_post: Optional[torch.Tensor],
        x_climate: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using only weather/climate data.

        Parameters
        ----------
        x_pre : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.
        x_post : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.
        x_climate : torch.Tensor
            Weather/climate time series, shape (B, 10, T, 20, 20).
        event_labels : torch.Tensor, optional
            Ignored. Kept for training-pipeline compatibility.

        Returns
        -------
        logits : torch.Tensor
            Class logits, shape (B, num_classes).
        att_weights : torch.Tensor
            Attention weights, shape (B, T).
        """
        z = self.climate_spatial(x_climate)
        z = self.climate_temporal(z)
        f_climate, att_weights = self.climate_attention(z)
        logits = self.classifier(f_climate)
        return logits, att_weights


def create_weather_gru_model(
    num_classes: int = 4,
    dropout_rate: float = 0.2,
    climate_dim: int = 128,
    **kwargs
) -> WeatherGRUModel:
    """
    Factory function to create the weather-only GRU model.

    Additional keyword arguments are ignored so shared configs can include
    image-only or multimodal options such as backbone and vis_dim.
    """
    return WeatherGRUModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        climate_dim=climate_dim,
    )
