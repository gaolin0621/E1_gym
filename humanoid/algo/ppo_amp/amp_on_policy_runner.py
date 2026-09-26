# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
import wandb
from typing import Union
from datetime import datetime
from .amp_ppo import AmpPPO
from .amp_ppo_e1_12dof import E1_12DOFAmpPPO
from .actor_critic import AmpActorCritic
from humanoid.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

class AmpOnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        # wandb
        self.wandb_run_name = (
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )

        # resolve dimensions of observations
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        # evaluate the policy class
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: Union[AmpActorCritic] = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize algorithm
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: Union[AmpPPO, E1_12DOFAmpPPO] = alg_class(
            actor_critic, device=self.device, **self.alg_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # _, _ = self.env.reset() # Todo: 是否需要reset

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        from humanoid.utils import export_policy_as_jit
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            if self.cfg["use_wandb"]:
                wandb.init(
                    entity="2741355724-droid",
                    project=self.all_cfg["runner"]["experiment_name"],
                    sync_tensorboard=True,
                    name=self.wandb_run_name,
                    config=self.all_cfg,
                    mode="online",
                    settings=wandb.Settings(init_timeout=20)
                )
            # Launch Tensorboard summary writer(s)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        amp_obs = self.env.get_amp_obs_for_expert_trans()  # TODO:1
        obs, critic_obs, amp_obs = (
            obs.to(self.device),
            critic_obs.to(self.device),
            amp_obs.to(self.device),
        )
        # -- PPO
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        style_rewbuffer = deque(maxlen=100)
        task_rewbuffer = deque(maxlen=100)
        cur_style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    # Step the environment
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    # Move to device
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )

                    # Account for terminal state transitions
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = amp_obs[reset_env_ids]

                    task_rewards = rewards.clone()
                    rewards, _, style_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )
                    amp_obs = torch.clone(next_amp_obs)

                    next_critic_obs = torch.clone(critic_obs)
                    # process the step
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        # Update rewards
                        cur_reward_sum += rewards
                        cur_style_reward_sum += style_rewards
                        cur_task_reward_sum += task_rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        style_rewbuffer.extend(cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        task_rewbuffer.extend(cur_task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_style_reward_sum[new_ids] = 0
                        cur_task_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(critic_obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    # export policy
                    policy_name = self.log_dir.split('/')[-1] + "_" + f"{it}"
                    export_policy_as_jit(self.alg.actor_critic, os.path.join(self.log_dir, 'policies'), policy_name)
            # Clear episode infos
            ep_infos.clear()

        # Save the final model after training
        if self.log_dir is not None:
            self.current_learning_iteration += 1 # save last idx
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        # export policy
        policy_name = self.log_dir.split('/')[-1] + "_" + f"{self.current_learning_iteration}"
        export_policy_as_jit(self.alg.actor_critic, os.path.join(self.log_dir, 'policies'), policy_name)

    def log(self, locs: dict, width: int = 80, pad: int = 45):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':<{pad}} {value:8.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_style_reward", statistics.mean(locs["style_rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", statistics.mean(locs["task_rewbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        # -- Losses
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f'{key}:':>{pad}} {value:.5f}\n"""

        # -- Rewards
        log_string += ep_string

        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Learning iteration:':>{pad}} {locs['it']}\n"""
                f"""{'Mean style_reward:':>{pad}} {statistics.mean(locs['style_rewbuffer']):.2f}\n"""
                f"""{'Mean task_reward:':>{pad}} {statistics.mean(locs['task_rewbuffer']):.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                f"""{'Time elapsed:':>{pad}} {self.format_time(self.tot_time)}\n"""
                f"""{'ETA:':>{pad}} {self.format_time((
                        self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                        * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                    )
                )}\n"""
            )
        else:
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Learning iteration:':>{pad}} {locs['it']}\n"""
                f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                f"""{'Time elapsed:':>{pad}} {self.format_time(self.tot_time)}\n"""
                f"""{'ETA:':>{pad}} {self.format_time((
                        self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                        * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                    )
                )}\n"""
            )
        print(log_string)

    def format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def save(self, path, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # save model
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        return policy

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
