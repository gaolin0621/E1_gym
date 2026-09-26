from __future__ import annotations

import torch
import torch.nn as nn

class DepthEncoderMLP(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            depth_dim = [24, 24],
            output_dim: int = 128,
    ):
        super().__init__()

        flatten_dim = in_channels * depth_dim[0] * depth_dim[1]
        self.image_compression = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, output_dim),
            nn.ELU(),
        )
        print(f"DepthEncoderMLP: {self.image_compression}")


    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        depth_flat = depth.flatten(start_dim=1)
        return self.image_compression(depth_flat)
