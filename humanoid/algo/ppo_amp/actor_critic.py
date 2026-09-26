"""Proprioceptive actor-critic for the independent, non-visual AMP task."""

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid.algo.utils import resolve_nn_activation


class AmpActorCritic(nn.Module):
    """Feed-forward actor-critic whose policy input is proprioception only."""

    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        noise_std_type="scalar",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "AmpActorCritic.__init__ ignored arguments: "
                + str(sorted(kwargs.keys()))
            )

        self.num_actor_inputs = int(num_actor_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.noise_std_type = noise_std_type
        nonlinearity = resolve_nn_activation(activation)

        self.actor = self._make_mlp(
            self.num_actor_obs, list(actor_hidden_dims), int(num_actions), nonlinearity
        )
        self.critic = self._make_mlp(
            self.num_critic_obs, list(critic_hidden_dims), 1, nonlinearity
        )
        print(f"Proprioceptive Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown noise_std_type {noise_std_type!r}; expected 'scalar' or 'log'"
            )
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation):
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (nn.Linear(previous_dim, hidden_dim), copy.deepcopy(activation))
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        return nn.Sequential(*layers)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        del dones

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        del kwargs
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        del kwargs
        return self.critic(critic_observations)
