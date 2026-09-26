from __future__ import annotations

from torch import nn
from humanoid.algo.utils import resolve_nn_activation
# input : long history obs
# output : base lin vel(vt), latent (zt)
class Estimator(nn.Module):
    def __init__(self,
                 input_dim = 512,
                 output_dim = 128,
                 hidden_dims=[256, 128],
                 hidden_dims_vel = [64],
                 hidden_dims_latent = [64],
                 output_dim_vel = 3,
                 output_dim_latent = 9,
                 ):
        super().__init__()
        # define encoder MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(nn.ELU())
        self.encode = nn.Sequential(*layers)
        print(f"encoder: {self.encode}")

        # define encode vel
        layers = []
        layers.append(nn.Linear(output_dim, hidden_dims_vel[0]))
        layers.append(nn.ELU())
        for l in range(len(hidden_dims_vel)):
            if l == len(hidden_dims_vel) - 1:
                layers.append(nn.Linear(hidden_dims_vel[l], output_dim_vel))
            else:
                layers.append(nn.Linear(hidden_dims_vel[l], hidden_dims_vel[l + 1]))
                layers.append(nn.ELU())
        self.encode_vel = nn.Sequential(*layers)
        print(f"encode_vel: {self.encode_vel}")

        # define encode latent
        layers = []
        layers.append(nn.Linear(output_dim, hidden_dims_latent[0]))
        layers.append(nn.Tanh())
        for l in range(len(hidden_dims_latent)):
            if l == len(hidden_dims_latent) - 1:
                layers.append(nn.Linear(hidden_dims_latent[l], output_dim_latent))
            else:
                layers.append(nn.Linear(hidden_dims_latent[l], hidden_dims_latent[l + 1]))
                layers.append(nn.Tanh())
        self.encode_latent = nn.Sequential(*layers)
        print(f"encoder_latent: {self.encode_latent}")

    def forward(self, x):
        distribution = self.encode(x)
        mean_vel = self.encode_vel(distribution)
        mean_latent = self.encode_latent(distribution)
        return mean_vel, mean_latent
