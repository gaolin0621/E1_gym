# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Union, Optional
from humanoid.algo.utils.utils import print_dict_aligned
from .actor_critic import AmpActorCritic
from .rollout_storage import RolloutStorage
from .replay_buffer import ReplayBuffer
from humanoid.algo.utils.motion_loader_x2_v1 import AMPLoader
from humanoid.algo.modules.discriminator import Discriminator
from humanoid.algo.modules.normalizer import EmpiricalNormalization

class AmpPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: Union[AmpActorCritic]
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 learning_rate_print = True,
                 amp_enable=True,
                 # Symmetry parameters
                 symmetry_cfg: Optional[dict] = None,
                 # AMP cfg
                 amp_cfg: Optional[dict] = None,
                 _motion_loader_cls=AMPLoader,
                 **kwargs):
        if kwargs:
            print("ESTPPO.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        # device-related parameters
        self.device = device

        # Discriminator components
        self.amp_cfg = amp_cfg
        if self.amp_cfg["loss_type"] is None:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'LSGAN', 'WGAN', 'BCEWithLogits', or 'ADD'")
        else:
            self.loss_type = self.amp_cfg["loss_type"]
        print_dict_aligned("amp_cfg", self.amp_cfg)
        # init amp loader
        self.amp_data = _motion_loader_cls(
            self.device,
            time_between_frames= self.amp_cfg["step_dt"],
            preload_transitions=True,
            num_preload_transitions=self.amp_cfg["amp_num_preload_transitions"],
            motion_files=self.amp_cfg["amp_motion_files"],
        )
        if self.amp_cfg["normalizer"]:
            self.amp_normalizer = EmpiricalNormalization(shape=self.amp_data.observation_dim).to(self.device)
        else:
            self.amp_normalizer = None

        # init discriminator
        self.discriminator = Discriminator(
            self.amp_data.observation_dim * 2,
            self.amp_cfg["amp_reward_coef"],
            self.amp_cfg["amp_discr_hidden_dims"],
            self.device,
            self.amp_cfg["amp_task_reward_lerp"],
            self.loss_type,
            self.amp_cfg["eta_wgan"],
        ).to(self.device)
        # -- Discriminator components
        self.amp_loss_coef = self.amp_cfg["amp_loss_coef"]
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(self.discriminator.input_dim // 2, self.amp_cfg["amp_replay_buffer_size"], self.device)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        params = [
            {"params": self.actor_critic.parameters(), "name": "policy"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": self.amp_cfg["trunk_weight_decay"], "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": self.amp_cfg["linear_weight_decay"], "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # Create rollout storage
        self.storage = None # initialized later
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.learning_rate_print = learning_rate_print

        # Isaac Gym uses PyTorch 1.13, whose mixed-precision API lives under
        # torch.cuda.amp (torch.amp.GradScaler(device=...) is a PyTorch 2.x API).
        self.amp_enable = bool(amp_enable and str(self.device).startswith("cuda"))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enable)
        # Parse optional symmetry configuration. No visual/privileged-state
        # estimator exists in this independent AMP algorithm.
        self.symmetry = symmetry_cfg or {}
        self.use_sym_loss = self.symmetry.get("sym_loss", False)

        # 预计算系数
        self.sym_coef = self.symmetry.get("sym_coef", 0.0) if self.use_sym_loss else 0.0
        if self.symmetry:
            print_dict_aligned("symmetry_cfg", self.symmetry)

        if self.use_sym_loss:
            act_permutation = self.symmetry["act_permutation"]
            obs_permutation = self.symmetry["obs_permutation"]
            frame_stack = self.symmetry["frame_stack"]

            # --- 动作 permutation 矩阵 ---
            self.act_perm_mat = torch.zeros((len(act_permutation),len(act_permutation)), device=self.device)
            for i, perm in enumerate(act_permutation):
                self.act_perm_mat[int(abs(perm))][i] = np.sign(perm)
            # --- 观测 permutation 堆叠 ---
            obs_permutation_stack = []
            for i in range(frame_stack):
                for p in obs_permutation:
                    obs_permutation_stack.append(np.sign(p) * (abs(p) + i * len(obs_permutation)))
            self.obs_perm_mat = torch.zeros((len(obs_permutation_stack), len(obs_permutation_stack)), device=self.device)
            for i, perm in enumerate(obs_permutation_stack):
                self.obs_perm_mat[int(abs(perm))][i] = np.sign(perm)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device
        )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, amp_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.amp_transition.clear()

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- AMP loss
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
       # -- symmetry loss
        mean_symmetry_loss = 0
        learning_rate_min = 10.
        learning_rate_max = 0.

        # Use AMP for mixed precision
        scaler = self.scaler

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch
            ) = sample

            # -------------------------
            # Forward + loss under autocast
            # -------------------------
            with torch.cuda.amp.autocast(enabled=self.amp_enable):

                # Recompute actions log prob and entropy for current batch of transitions
                # Note: we need to do this because we updated the policy with the new parameters
                # -- actor
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # -- critic
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                # -- entropy
                # we only keep the entropy of the first augmentation (the original one)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                with torch.cuda.amp.autocast(enabled=False):
                    if self.desired_kl is not None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                / (2.0 * torch.square(sigma_batch))
                                - 0.5,
                                axis=-1,
                            )
                            kl_mean = torch.mean(kl)

                            # Update the learning rate
                            # Perform this adaptation only on the main process
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif self.desired_kl / 2.0 > kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Discriminator loss.
                policy_state_raw, policy_next_state_raw = sample_amp_policy
                expert_state_raw, expert_next_state_raw = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        self.amp_normalizer.eval()
                        policy_state = self.amp_normalizer(policy_state_raw)
                        policy_next_state = self.amp_normalizer(policy_next_state_raw)
                        expert_state = self.amp_normalizer(expert_state_raw)
                        expert_next_state = self.amp_normalizer(expert_next_state_raw)
                else:
                    policy_state = policy_state_raw
                    policy_next_state = policy_next_state_raw
                    expert_state = expert_state_raw
                    expert_next_state = expert_next_state_raw

                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                if self.loss_type == "LSGAN":
                    expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = 0.5 * (expert_loss + policy_loss)
                elif self.loss_type == "WGAN":
                    _policy_d = torch.tanh(self.discriminator.eta_wgan * policy_d)
                    _expert_d = torch.tanh(self.discriminator.eta_wgan * expert_d)
                    amp_loss = _policy_d.mean() - _expert_d.mean()
                elif self.loss_type == "BCEWithLogits":
                    loss_fun = torch.nn.BCEWithLogitsLoss()
                    expert_loss = loss_fun(expert_d, torch.ones_like(expert_d))
                    policy_loss = loss_fun(policy_d, torch.zeros_like(policy_d))
                    amp_loss = 0.5 * (expert_loss + policy_loss)

                grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, policy_state, policy_next_state, lambda_=10)
                loss += self.amp_loss_coef * amp_loss + self.amp_loss_coef * grad_pen_loss

                # Symmetry loss
                if self.use_sym_loss:
                    #构造镜像观察
                    mirror_obs = torch.matmul(obs_batch, self.obs_perm_mat)
                    #得到镜像动作输出
                    mirror_act = self.actor_critic.act_inference(mirror_obs.detach())
                    #对动作进行对称重排（左右腿交换等）
                    m_mirror_act = torch.matmul(mirror_act, self.act_perm_mat)
                    #若只更新主策略分支 → detach()；若希望两边更新 → 去掉 detach()
                    sym_loss = (mu_batch - m_mirror_act.detach()).pow(2).mean()
                    # sym_loss = (mu_batch - m_mirror_act).pow(2).mean()
                    # add the loss to the total loss
                    loss += self.sym_coef * sym_loss

                # -------------------------
                # Backward & step with GradScaler
                # -------------------------
                # Compute the gradients
                if self.amp_enable:
                    self.optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    # gradient clipping with scaler
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state_raw)
                    self.amp_normalizer.update(expert_state_raw)

                # Store the losses
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                # -- amp loss
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                if "sym_loss" in locals():
                    mean_symmetry_loss += sym_loss.item()

                learning_rate_min = min(learning_rate_min, self.learning_rate)
                learning_rate_max = max(learning_rate_max, self.learning_rate)
        # ===== 统计平均值 =====
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For AMP
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # -- For Symmetry
        mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function loss": mean_value_loss,
            "surrogate loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp loss": mean_amp_loss,
            "amp_grad_pen loss": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        # 可选添加 symmetry loss
        if self.symmetry.get("sym_loss", False):
            loss_dict["symmetry"] = mean_symmetry_loss

        # 可选添加学习率信息
        if self.learning_rate_print:
            loss_dict.update({
                "lr_min": learning_rate_min,
                "lr_max": learning_rate_max,
            })

        return loss_dict
