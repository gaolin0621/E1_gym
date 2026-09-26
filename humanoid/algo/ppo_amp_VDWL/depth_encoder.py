from __future__ import annotations

import torch
import torch.nn as nn

# output_dim = 128
# in_channel = 2
# channels = [4]
# kernel_sizes = [3]
# strides = [1]
# paddings = [1]
# nonlinearity = "ReLU"
# use_maxpool = True


class DepthEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            output_dim: int = 128,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # input: [B, 2, 24, 24]
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 自动计算 CNN 输出展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 24, 24)
            cnn_out = self.cnn(dummy)
            flatten_dim = cnn_out.view(1, -1).shape[1]

        self.image_compression = nn.Sequential(
            self.cnn,
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ELU(),
            nn.Linear(256, output_dim),
            nn.ELU(),
        )
        print(f"DepthEncoder: {self.image_compression}")


    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.image_compression(depth)
