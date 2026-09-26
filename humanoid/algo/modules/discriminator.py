# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self,
                 input_dim,
                 amp_reward_coef,
                 hidden_layer_sizes,
                 device,
                 task_reward_lerp=0.0,
                 loss_type: str = "LSGAN",
                 eta_wgan: float = 0.3
                 ):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

        self.loss_type = loss_type
        self.eta_wgan = eta_wgan

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(
            self,
            expert_state,
            expert_next_state,
            policy_state,
            policy_next_state,
            lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            policy_state (torch.Tensor): Batch of policy states.
            policy_next_state (torch.Tensor): Batch of policy next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        if self.loss_type == "LSGAN":
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            expert_data.requires_grad = True

            disc = self.amp_linear(self.trunk(expert_data))
            ones = torch.ones(disc.size(), device=disc.device)
            grad = autograd.grad(
                outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            # Enforce that the grad norm approaches 0.
            grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
            return grad_pen

        elif self.loss_type == "WGAN":
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            policy_data = torch.cat([policy_state, policy_next_state], dim=-1)

            alpha = torch.rand((expert_data.shape[0], 1), device=expert_data.device)
            data = alpha * expert_data + (1 - alpha) * policy_data
            data = data.detach().requires_grad_(True)

            disc = self.amp_linear(self.trunk(data))
            ones = torch.ones(disc.size(), device=disc.device)

            grad = autograd.grad(
                outputs=disc,
                inputs=data,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_pen = lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()
            return grad_pen

        elif self.loss_type == "BCEWithLogits":
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            expert_data.requires_grad = True

            disc = self.amp_linear(self.trunk(expert_data))
            grad = autograd.grad(
                outputs=disc.sum(), inputs=expert_data, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_pen = 0.5 * lambda_ * (grad.pow(2).sum(dim=1)).mean()
            return grad_pen

        return None

    def predict_amp_reward(self, state, next_state, task_reward, normalizer = None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                # state = normalizer.normalize_torch(state, self.device)
                # next_state = normalizer.normalize_torch(next_state, self.device)
                state = normalizer(state)
                next_state = normalizer(next_state)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            if self.loss_type == "LSGAN":
                amp_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.loss_type == "WGAN":
                amp_reward = self.amp_reward_coef * torch.exp(torch.tanh(self.eta_wgan * d))
            elif self.loss_type == "BCEWithLogits":
                # softplus(logit) == -log(1 - sigmoid(logit))
                amp_reward = nn.functional.softplus(d)
                # prob = 1 / (1 + torch.exp(-d))
                # amp_reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
                amp_reward *= self.amp_reward_coef
            # elif self.loss_type == "ADD":

            reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d, amp_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
