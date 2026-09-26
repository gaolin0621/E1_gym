from __future__ import annotations

from torch import nn
from humanoid.algo.utils import resolve_nn_activation

class MLP(nn.Module):
    def __init__(self,
                 input_dim = 512,
                 output_dim = 3,
                 hidden_dims = [256, 128],
                 activation='elu',
                 name='mlp',
                 ):
        super().__init__()
        self.activation = resolve_nn_activation(activation)
        # define encoder MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                if output_dim != None:
                    layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(self.activation)
        self.mlp = nn.Sequential(*layers)
        print(f"{name}: {self.mlp}")

    def forward(self, x):
        output = self.mlp(x)
        return output
