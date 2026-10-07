"""
Shallow 2D CNN for EEG connectivity matrices.

Input:  (batch, 5, 19, 19)   -- 5 frequency bands as channels
Output: (batch, 2)            -- raw logits for softmax / CrossEntropyLoss
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class EEGConnectivityCNN(nn.Module):
    def __init__(
        self,
        conv_config: List[Tuple[int, int]],
        dropout: float,
        in_channels: int = 5,
        fc_units: int = 128,
        n_classes: int = 2,
        use_pooling: bool = False,
        input_size: int = 19,
    ):
        super().__init__()
        self.conv_config = conv_config
        self.dropout_p = dropout
        self.use_pooling = use_pooling

        conv_layers = []
        c_in = in_channels
        for c_out, kernel_size in conv_config:
            padding = kernel_size // 2  # 'same' padding, preserves 19x19 spatial size
            conv_layers.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.BatchNorm2d(c_out))
            conv_layers.append(nn.ReLU(inplace=True))
            c_in = c_out


        spatial_size = input_size
        if use_pooling:
            assert spatial_size >= 2, (
                f"Cannot pool: spatial size already {spatial_size} "
                f"(too many conv blocks for input_size={input_size})"
            )
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            spatial_size = spatial_size // 2

        self.conv_blocks = nn.Sequential(*conv_layers)
        self.final_spatial_size = spatial_size  # informational, GAP doesn't need this

        self.last_conv_channels = c_in

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c_in, fc_units)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(fc_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(x)              # (B, C, 19, 19)
        pooled = self.gap(features).flatten(1)       # (B, C)
        pooled = self.dropout1(pooled)
        hidden = self.relu_fc(self.fc1(pooled))
        hidden = self.dropout2(hidden)
        logits = self.output_layer(hidden)            # (B, n_classes)
        return logits

    def forward_with_features(self, x: torch.Tensor):
        """Returns (logits, last_conv_feature_map)"""
        features = self.conv_blocks(x)
        pooled = self.gap(features).flatten(1)
        pooled_d = self.dropout1(pooled)
        hidden = self.relu_fc(self.fc1(pooled_d))
        hidden_d = self.dropout2(hidden)
        logits = self.output_layer(hidden_d)
        return logits, features

    def get_config_dict(self) -> dict:
        
        return {
            "conv_config": self.conv_config,
            "dropout": self.dropout_p,
            "in_channels": self.conv_blocks[0].in_channels,
            "fc_units": self.fc1.out_features,
            "n_classes": self.output_layer.out_features,
            "use_pooling": self.use_pooling,
        }


def build_model_from_config_dict(config_dict: dict) -> EEGConnectivityCNN:
    return EEGConnectivityCNN(
        conv_config=[tuple(c) for c in config_dict["conv_config"]],
        dropout=config_dict["dropout"],
        in_channels=config_dict["in_channels"],
        fc_units=config_dict["fc_units"],
        n_classes=config_dict["n_classes"],
        use_pooling=config_dict.get("use_pooling", False), 
    )
