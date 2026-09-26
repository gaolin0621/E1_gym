from __future__ import annotations

import torch
import torch.nn as nn

from humanoid.algo.utils import resolve_nn_activation

class CNN(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_size=[14, 24],  # 输入图像高宽，用于推断
                 conv_channels=[16, 32, 64],
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 paddings=[1, 1, 1],
                 activation='elu',
                 output_dim=128,
                 name='cnn'):
        """
        Args:
            input_channels (int): 输入通道数
            input_size= (list[int]),  # 输入图像高宽，用于推断
            conv_channels (list[int]): 每层卷积的输出通道数
            kernel_sizes (list[int]): 卷积核大小
            strides (list[int]): 步长
            paddings (list[int]): 填充
            activation (str): 激活函数名称
            output_dim (int): 线性层输出
            name (str): 模块名（打印结构时用）
        """
        super().__init__()
        self.activation = resolve_nn_activation(activation)

        # 卷积层堆叠
        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(conv_channels):
            layers.append(nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i]
            ))
            layers.append(self.activation)
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # ===== 自动推断 flatten 后的维度 =====
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size[0], input_size[1])
            conv_out = self.conv(dummy)
            flatten_dim = conv_out.view(1, -1).shape[1]
        print(f"{name} conv output shape: {tuple(conv_out.shape[1:])}, flatten_dim={flatten_dim}")
        # 线性输出层
        self.output_layer = nn.Linear(flatten_dim, output_dim)

        print(f"{name}: {self.conv}")

    def forward(self, depth):
        x = self.conv(depth)
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x

class CNN_Time(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_size=[14, 24],  # 输入图像高宽，用于推断
                 conv_channels=[16, 32, 64],
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 paddings=[1, 1, 1],
                 activation='elu',
                 output_dim=128,
                 name='cnn'):
        """
        Args:
            input_channels (int): 输入通道数
            input_size= (list[int]),  # 输入图像高宽，用于推断
            conv_channels (list[int]): 每层卷积的输出通道数
            kernel_sizes (list[int]): 卷积核大小
            strides (list[int]): 步长
            paddings (list[int]): 填充
            activation (str): 激活函数名称
            output_dim (int): 线性层输出
            name (str): 模块名（打印结构时用）
        """
        super().__init__()
        self.activation = resolve_nn_activation(activation)

        # 卷积层堆叠
        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(conv_channels):
            layers.append(nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i]
            ))
            layers.append(self.activation)
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # ===== 自动推断 flatten 后的维度 =====
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size[0], input_size[1])
            conv_out = self.conv(dummy)
            flatten_dim = conv_out.view(1, -1).shape[1]
        print(f"[{name}] Conv output shape: {tuple(conv_out.shape[1:])}, flatten_dim={flatten_dim}")
        # 线性输出层
        self.output_layer = nn.Linear(flatten_dim, output_dim)

        print(f"{name}: {self.conv}")

    def forward(self, depth):
        # 自动展开时间维度
        # [B, T, H, W] → [B*T, H, W]
        is_seq = depth.dim() == 4
        if is_seq:
            B, T, H, W = depth.shape
            x = depth.reshape(B * T, 1, H, W)
        else:
            B, H, W = depth.shape
            x = depth.unsqueeze(1)

        # CNN 前向
        x = self.conv(x)  # [B*T, C_out, H', W']
        x = torch.flatten(x, start_dim=1)  # [B*T, F]
        x = self.output_layer(x)  # [B*T, D]
        # 恢复时间维度
        if is_seq:
            x = x.view(B, T, -1)  # [B, T, D]
        return x