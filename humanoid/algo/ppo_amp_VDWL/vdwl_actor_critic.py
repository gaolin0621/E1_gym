# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Normal
from humanoid.algo.utils import resolve_nn_activation
from humanoid.algo.modules import MLP
from .depth_encoder import DepthEncoder
from .depth_encoder_mlp import DepthEncoderMLP


class VDWLActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        noise_std_type: str = "scalar",
                        init_noise_std=1.0,
                        # policy cfg
                        policy_cfg: Optional[dict] = None,
                        **kwargs):
        if kwargs:
            print("VDWLActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()
        activation = resolve_nn_activation(activation)
        # get policy_cfg
        self.cfg = policy_cfg
        print("\n================== policy cfg ======================")
        print(self.cfg)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_short_obs = self.cfg["num_short_obs"]
        self.num_long_obs = self.cfg["num_long_obs"]
        self.depth_dim = self.cfg["depth_image_dim"]
        self.depth_frame = self.cfg["depth_image_frame"]
        print('num_short_obs = ',self.num_short_obs)
        print('num_long_obs = ', self.num_long_obs)
        print('depth_dim = ', self.depth_dim)
        print('depth_frame = ', self.depth_frame)
        # actor critic input dim
        mlp_input_dim_a = self.num_short_obs + self.cfg["fusion_encoder"]["output_dim"]
        mlp_input_dim_c = self.num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # define depth_encoder cnn
        # cfg = self.cfg["depth_encoder"]
        # self.depth_encoder = DepthEncoder(
        #     in_channels = cfg["in_channels"],
        #     output_dim     = cfg["output_dim"]
        # )

        # define depth_encoder mlp
        cfg = self.cfg["depth_encoder"]
        self.depth_encoder = DepthEncoderMLP(
            in_channels = cfg["in_channels"],
            output_dim     = cfg["output_dim"]
        )


        # define prop_encoder mlp
        cfg = self.cfg["prop_encoder"]
        self.prop_encoder = MLP(
            input_dim  = cfg["input_dim"],
            output_dim = cfg["output_dim"],
            hidden_dims= cfg["hidden_dims"],
            activation = cfg["activation"],
            name       = cfg["name"]
        )
        # define fusion_encoder mlp
        cfg = self.cfg["fusion_encoder"]
        self.fusion_encoder = MLP(
            input_dim  = cfg["input_dim"],
            output_dim = cfg["output_dim"],
            hidden_dims= cfg["hidden_dims"],
            activation = cfg["activation"],
            name       = cfg["name"]
        )

        # define height_decoder mlp
        cfg = self.cfg["height_decoder"]
        self.height_decoder = MLP(
            input_dim  = cfg["input_dim"],
            output_dim = cfg["output_dim"],
            hidden_dims= cfg["hidden_dims"],
            activation = cfg["activation"],
            name       = cfg["name"]
        )
        # define obs_decoder mlp
        cfg = self.cfg["obs_decoder"]
        self.obs_decoder = MLP(
            input_dim  = cfg["input_dim"],
            output_dim = cfg["output_dim"],
            hidden_dims= cfg["hidden_dims"],
            activation = cfg["activation"],
            name       = cfg["name"]
        )

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, depth, **kwargs):
        # -- depth
        depth_latent = self.depth_encoder(depth) # [B, 2, 24, 24]
        # -- prop
        prop_latent = self.prop_encoder(observations[:, -self.num_long_obs:])  # long obs
        # -- fusion
        fusion = torch.cat([prop_latent, depth_latent], dim=-1)
        encode = self.fusion_encoder(fusion)
        # -- actor
        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, depth):
        # -- depth
        depth_latent = self.depth_encoder(depth) # [B, 2, 24, 24]
        # -- prop
        prop_latent = self.prop_encoder(observations[:, -self.num_long_obs:])  # long obs
        # -- fusion
        fusion = torch.cat([prop_latent, depth_latent], dim=-1)
        encode = self.fusion_encoder(fusion)
        # -- actor
        actor_input = torch.cat((observations[:, -self.num_short_obs:], encode), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    # def estimator_inference(self, observations, depth):
    #     # -- depth
    #     depth_latent = self.depth_encoder(depth) # [B, 2, 24, 24]
    #     # -- prop
    #     prop_latent = self.prop_encoder(observations[:, -self.num_long_obs:])  # long obs
    #     # -- fusion
    #     fusion = torch.cat([prop_latent, depth_latent], dim=-1)
    #     encode = self.fusion_encoder(fusion)
    #     vel = encode[:, 0:3]
    #     # -- for est loss
    #     est_vel = vel
    #     est_feet_height = None
    #     est_height = None
    #     est_obs_cur = None
    #     return est_vel, est_feet_height, est_height, est_obs_cur

    def estimator_inference(self, observations, depth):
        # -- depth
        depth_latent = self.depth_encoder(depth) # [B, 2, 24, 24]
        # -- prop
        prop_latent = self.prop_encoder(observations[:, -self.num_long_obs:])  # long obs
        # -- fusion
        fusion = torch.cat([prop_latent, depth_latent], dim=-1)
        encode = self.fusion_encoder(fusion)
        vel = encode[:, 0:3]
        obs_latent = encode[:, 3:3+16]
        heightmap_latent = encode[:,3+16:]
        # -- decoder
        decode_obs = self.obs_decoder(obs_latent)
        decode_heightmap = self.height_decoder(heightmap_latent)

        # -- for est loss
        est_vel = vel
        est_feet_height = None
        est_height = decode_heightmap
        est_obs_cur = decode_obs
        return est_vel, est_feet_height, est_height, est_obs_cur

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
