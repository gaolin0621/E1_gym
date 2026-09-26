import os
import time
from typing import Union

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi,quat_yaw
from humanoid.utils.helpers import class_to_dict
from .e1_12dof_amp_vdwl_cfg import E1_12DOFAMPVDWLCfg
import torch, torchvision
from humanoid.envs.base.base_task import BaseTask
from humanoid.utils.terrain_vision import Terrain
from humanoid.utils.depth_image_viewer import DepthImageViewer
import warp as wp
import xml.etree.ElementTree as ET
import trimesh
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from humanoid.algo.utils.motion_loader_e1_12dof_v1 import AMPLoader
from humanoid.utils.mjlab_math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

@wp.kernel
def depth_draw_batch(
    mesh: wp.uint64,
    boder_size: float,
    cam_poss: wp.array(dtype=wp.vec3),
    cam_rots: wp.array(dtype=wp.quat),
    body_mesh_ids: wp.array(dtype=wp.uint64),
    body_poss: wp.array(dtype=wp.vec3),
    body_rots: wp.array(dtype=wp.quat),
    num_bodies: int,
    fovs: wp.array(dtype=float),
    width: int,
    height: int,
    pixels: wp.array(dtype=float, ndim=2)
):
    tid = wp.tid()
    env_id = tid // (width * height)
    pixel_id = tid % (width * height)

    x = pixel_id % width
    y = pixel_id // width
    y = height - y #- 1

    cam_pos = cam_poss[env_id]
    cam_rot = cam_rots[env_id]

    fovy_rad = wp.radians(fovs[env_id])
    scale_y = wp.tan(fovy_rad * 0.5)
    aspect_ratio = float(width) / float(height)
    sx = 2.0 * (float(x)+0.5) / float(width) - 1.0
    sy = 2.0 * (float(y)-0.5) / float(height) - 1.0
    view_dir = wp.vec3(
        sx * aspect_ratio * scale_y,
        sy * scale_y,
        -1.0
    )

    init_quat = wp.quat_rpy(wp.pi*0.5, 0.0, -wp.pi*0.5)
    rd = wp.quat_rotate(init_quat, wp.normalize(view_dir))
    rd = wp.quat_rotate(cam_rot, rd)
    offset = wp.vec3(boder_size, boder_size, 0.0)
    ro = cam_pos
    ro_terrain = cam_pos + offset

    query = wp.mesh_query_ray(mesh, ro_terrain, rd, 5.0)

    principal_view = wp.vec3(0.0, 0.0, -1.0)
    rd_principal = wp.quat_rotate(init_quat, principal_view)
    rd_principal = wp.quat_rotate(cam_rot, rd_principal)
    rd_principal = wp.normalize(rd_principal)

    multiplier = wp.dot(rd, rd_principal)
    hit_t = wp.float32(5.0)

    if query.result:
        hit_t = query.t

    near_t = wp.float32(0.03)
    body_base = env_id * num_bodies
    for body_id in range(num_bodies):
        body_mesh = body_mesh_ids[body_id]
        if body_mesh != wp.uint64(0):
            body_pos = body_poss[body_base + body_id]
            body_rot_inv = wp.quat_inverse(body_rots[body_base + body_id])
            ro_local = wp.quat_rotate(body_rot_inv, ro - body_pos)
            rd_local = wp.quat_rotate(body_rot_inv, rd)
            body_query = wp.mesh_query_ray(body_mesh, ro_local, rd_local, hit_t)
            if body_query.result:
                if body_query.t > near_t and body_query.t < hit_t:
                    hit_t = body_query.t

    dist = multiplier * hit_t
    pixels[env_id, pixel_id] = dist

@wp.kernel
def feet_ray_cast_kernel(
    mesh: wp.uint64,
    border_size: wp.float32,
    foot_pos: wp.array(dtype=wp.vec3),
    foot_quat: wp.array(dtype=wp.quat),
    max_dist: wp.float32,
    out_dist_front: wp.array(dtype=wp.float32),
    out_dist_back: wp.array(dtype=wp.float32)
):
    tid = wp.tid()

    offset = wp.vec3(border_size, border_size, 0.0)

    view_dir_front = wp.vec3(1.0, 0.0, 0.0)
    rd_front = wp.quat_rotate(foot_quat[tid], view_dir_front)
    query_front = wp.mesh_query_ray(mesh, foot_pos[tid] + offset, rd_front, max_dist)
    out_dist_front[tid] = query_front.t

    view_dir_back = wp.vec3(-1.0, 0.0, 0.0)
    rd_back = wp.quat_rotate(foot_quat[tid], view_dir_back)
    query_back = wp.mesh_query_ray(mesh, foot_pos[tid] + offset, rd_back, max_dist)
    out_dist_back[tid] = query_back.t

class E1_12DOFAMPVDWLEnv(BaseTask):
    def __init__(self, cfg: Union[E1_12DOFAMPVDWLCfg], sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg:Union[E1_12DOFAMPVDWLCfg] = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", True)
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_random_motor_paras()
        self.init_done = True
        # -- amp motion show
        if getattr(self.cfg.env, "amp_motion_files_display", False):
            self._init_amp_motion()
        self.reset()

    # =================================================== MDP（step observation reward reset） ========================================================
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions[:] = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        # every step rand motor strength
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[:] = torch_rand_float(rng[0], rng[1], shape=(self.num_envs, self.num_actions), device=self.device)
        for _ in range(self.cfg.control.decimation):
            action_delayed = self.update_cmd_action_latency_buffer()
            self.torques[:] = self._compute_torques(action_delayed).view(self.torques.shape)
            # -- 计算手臂力矩
            if self.arm_dof_enable:
                self.arm_torques[:] = self._compute_arm_torques()
                all_torques = torch.cat([self.torques, self.arm_torques], dim=-1)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_torques))
            else:
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_warp or self.cfg.depth.use_camera:
            self.extras["depth"] = self.depth_buffer[:, -self.cfg.depth.obs_len:]
        else:
            self.extras["depth"] = None

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factor
        d_gains = self.d_gains * self.kd_factor
        torques = p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos - self.motor_offset) - d_gains * self.dof_vel
        torques = torques * self.motor_strength
        return torch.clip(torques, -self.q_torque_limits, self.q_torque_limits)

    def _compute_arm_torques(self):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.domain_rand.randomize_arm_pos and  ((self.common_step_counter-1) % self.cfg.domain_rand.arm_pos_interval == 0):
            dr = self.cfg.domain_rand
            for i in range(len(dr.min_arm_pos)):
                self.arm_actions[:, i] = torch_rand_float(dr.min_arm_pos[i], dr.max_arm_pos[i], shape=(self.num_envs, 1), device=self.device).squeeze(1)
                self.arm_actions[:, i + len(dr.min_arm_pos)] = torch_rand_float(dr.min_arm_pos[i], dr.max_arm_pos[i], shape=(self.num_envs, 1), device=self.device).squeeze(1)
        self.arm_actions_fil = 0.98 * self.arm_actions_fil + 0.02 * self.arm_actions
        torques = self.arm_kp * (self.arm_actions_fil - self.arm_pos) - self.arm_kd * self.arm_vel
        return torch.clip(torques, -self.arm_torque_limit, self.arm_torque_limit)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # ============================ prepare quantities ==================================
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_euler_xyz[:] = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.left_feet_euler[:] = get_euler_xyz_tensor(self.left_feet_quat)
        self.right_feet_euler[:] = get_euler_xyz_tensor(self.right_feet_quat)
        # -- 计算feet contact
        self.feet_forces[:] = self.contact_forces[:, self.feet_indices]
        self.contact[:] = self.feet_forces[:, :, 2] > 1.
        self.contact_xyz[:] = torch.norm(self.feet_forces[:, :, :3],dim=-1) > 5.0
        self.contact_filt[:] = torch.logical_or(self.contact, self.last_contact)
        self.feet_forces_history[:] = torch.roll(self.feet_forces_history, shifts=-1, dims=1)
        self.feet_forces_history[:, -1] = self.feet_forces
        self.feet_vel_history[:] = torch.roll(self.feet_vel_history, shifts=-1, dims=1)
        self.feet_vel_history[:, -1] = self.rigid_body_vel[:,self.feet_indices]
        # -- 计算feet contact air time
        self.first_contact[:] = (self.current_air_time > 0.) * self.contact
        self.first_air[:] = (self.current_contact_time > 0.) * (~self.contact)
        self.current_air_time[:] += self.dt
        self.current_air_time *= ~self.contact
        self.current_contact_time[:] += self.dt
        self.current_contact_time[:] *= self.contact
        self._update_feet_pos()
        self._update_feet_toe_pos()
        # -- terrain_mask
        self._get_terrain_indices_from_pos()
        # ============================ Event ==================================
        self._post_physics_step_callback()

        # ============================ compute observations, rewards, resets, ... ==========
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # -- ray caster for depth terrain
        self.update_depth_buffer()
        # -- ray caster for feet distance to terrain
        self.update_feet_ray_caster_dist()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # ============ hist data save ==========
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_body_vel[:] = self.rigid_body_vel
        self.last_contact[:] = self.contact[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz and not self.headless:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # =================================== 更新速度指令 =======================================
        if self.cfg.commands.gait_enable:
            self._resample_gait_commands()
        else:
            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)

        self._update_command()
        # =================================== 更新步态指令 =======================================
        if self.cfg.rewards.gait_radio:
            self._gait_style_update()
        self._phase_step_update()
        # =================================== 更新地形高度图 =======================================
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.left_feet_height_maps = self._get_feet_heights(feet_idx=0)
            self.right_feet_height_maps = self._get_feet_heights(feet_idx=1)
            self.left_feet_hold_maps = self._get_feet_hold(feet_idx=0)
            self.right_feet_hold_maps = self._get_feet_hold(feet_idx=1)
            self.base_height_maps = self._get_base_heights()
            self.left_feet_forward_maps, left_feet_forward_points_w = self._get_feet_forward_heights(feet_idx=0)
            self.right_feet_forward_maps, right_feet_forward_points_w = self._get_feet_forward_heights(feet_idx=1)
            self.left_feet_forward_points_w = torch.where(self.first_air[:, 0:1].unsqueeze(1), left_feet_forward_points_w, self.left_feet_forward_points_w)
            self.right_feet_forward_points_w = torch.where(self.first_air[:, 1:2].unsqueeze(1), right_feet_forward_points_w, self.right_feet_forward_points_w)
            self._update_foothold_candidates_on_first_air()

        # =================================== 更新外部推力 =======================================
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots() #TODO：检查是否执行
        else:
            self.rand_push_force.zero_()
            self.rand_push_torque.zero_()

        if self.cfg.domain_rand.apply_force_torque:
            self._apply_force_torque()

    def compute_observations(self):

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1) * self.cfg.rewards.clock_enable
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1) * self.cfg.rewards.clock_enable
        sin_pos[self.stand_command] = 0.
        cos_pos[self.stand_command] = 0.

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        feet_height = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2],  # left_height
            self.rigid_state[:, self.feet_indices[1], 2]  # right_height
        )
        self.base_height[:,0] = self.root_states[:, 2] - feet_height + self.cfg.rewards.feet_height

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            q,  # 12
            dq,  # 12
            self.actions,  # 12
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2

            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            # dim = 6 可选配
            self.env_frictions * 1,  # 1
            # self.base_height, # 1
            self.body_mass / 3.6, # E1 pelvis mass normalization
            # self.feet_front_dist_ray, # 2
            # self.feet_back_dist_ray, # 2
            stance_mask * 1, # 2
            contact_mask * 1, # 2
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            q,  # 12
            dq,  # 12
            self.actions,  # 12
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            base_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            left_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.left_feet_height_maps - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            right_height_maps = (torch.clip(self.root_states[:, 2].unsqueeze(1) - self.right_feet_height_maps - self.cfg.normalization.height_offset, -1, 1.)
                                * self.obs_scales.height_measurements)

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, left_height_maps, right_height_maps, base_height_maps), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + (2. * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        # obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        # self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K

        self.obs_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.frame_stack)], dim=1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        roll_cutoff = torch.abs(self.base_euler_xyz[:,0]) > 1.3
        pitch_cutoff = torch.abs(self.base_euler_xyz[:,1]) > 1.3
        height_cutoff = self.root_states[:, 2] < -5.
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids: torch) -> None:
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            # self._update_terrain_curriculum(env_ids)
            self._update_terrain_curriculum_vel(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)

        self._reset_root_states(env_ids)

        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_contact[env_ids] = False
        self.last_root_vel[env_ids] = 0.
        self.last_rigid_body_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.current_air_time[env_ids] = 0.
        self.current_contact_time[env_ids] = 0.
        self.feet_forces_history[env_ids] = 0.
        self.feet_both_contact_time[env_ids] = 0.
        self.feet_vel_history[env_ids] = 0.
        self.left_foothold_candidates[env_ids] = 0.
        self.right_foothold_candidates[env_ids] = 0.
        self.left_foothold_candidate_valid[env_ids] = False
        self.right_foothold_candidate_valid[env_ids] = False
        self.left_foothold_landing_error[env_ids] = 0.
        self.right_foothold_landing_error[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.commands[env_ids] = 0.
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device) * 0.5
        self.phase[env_ids] = self.gait_start[env_ids]
        # resample command
        if self.cfg.commands.gait_enable:
            self.generate_gait_time(env_ids)
            self._resample_gait_commands()
        else:
            self._resample_commands(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        # reset latency buffers and randomization
        self._reset_latency_buffer(env_ids)
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz[:] = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dr = self.cfg.domain_rand
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(dr.joint_pos_range[0], dr.joint_pos_range[1], (len(env_ids), self.num_actions), device=self.device)
        self.dof_vel[env_ids] = torch_rand_float(dr.joint_vel_range[0], dr.joint_vel_range[1], (len(env_ids), self.num_actions), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        dr = self.cfg.domain_rand
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(dr.pose_xy[0], dr.pose_xy[1], (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base quat
        roll  = torch_rand_float(-0., 0., (len(env_ids), 1), device=self.device).squeeze(1)
        pitch = torch_rand_float(-0., 0., (len(env_ids), 1), device=self.device).squeeze(1)
        yaw   = torch_rand_float(dr.pose_yaw[0], dr.pose_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)
        base_quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[env_ids, 3:7] = base_quat  # [3:7]: base quat
        # base velocities
        self.root_states[env_ids, 7:10] = torch_rand_float(dr.lin_vel[0], dr.lin_vel[1], (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids,10:13] = torch_rand_float(dr.ang_vel[0], dr.ang_vel[1], (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        dr = self.cfg.domain_rand
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch.empty_like(self.root_states[env_ids, :2]).uniform_(dr.pose_xy[0], dr.pose_xy[1])
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:10] = AMPLoader.get_linear_vel_batch(frames)
        self.root_states[env_ids, 10:13] = AMPLoader.get_angular_vel_batch(frames)

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    # ==================================================== Env Init, Simulation Create、Terrain Create ===============================================
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.small_push_interval = np.ceil(self.cfg.domain_rand.small_push_interval_s / self.dt)
        self.cfg.domain_rand.apply_interval = np.ceil(self.cfg.domain_rand.apply_interval_s / self.dt)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _arm_enable_check(self):
        # -- 加载模型时判断是否有手臂自由度
        self.arm_dof_enable = self.num_actions < self.num_dof
        self.num_arms = self.num_dof - self.num_actions
        # self.cfg.env.num_arms
        print("\n============  arm_dof_enable ============")
        print("arm_dof_enable: ", self.arm_dof_enable)
        print("num_arms      : ", self.num_arms)
        print("============  arm_dof_enable ============\n")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # -- check arm dof
        self._arm_enable_check()

        # body and joint paras
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.joint_damping = torch.zeros(self.num_actions, device=self.device)
        self.joint_armature = torch.zeros(self.num_actions, device=self.device)
        self.joint_friction = torch.zeros(self.num_actions, device=self.device)
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.default_dof_pos = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)

        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.body_names = body_names
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        penalized_contact_names = []
        termination_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # find body idx
        self.feet_indices = np.zeros(len(feet_names), dtype=np.longlong)
        self.knee_indices = np.zeros(len(knee_names), dtype=np.longlong)
        self.amp_body_indices = np.zeros(len(self.cfg.env.amp_body_names), dtype=np.longlong)
        self.penalised_contact_indices = np.zeros(len(penalized_contact_names), dtype=np.longlong)
        self.termination_contact_indices = np.zeros(len(termination_contact_names), dtype=np.longlong)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, knee_names[i])
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])
        for i in range(len(self.cfg.env.amp_body_names)):
            self.amp_body_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.env.amp_body_names[i])
            print("self.amp_body_indices:", self.amp_body_indices)
        # joint positions offsets and PD gains
        for i in range(self.num_actions):
            name = self.dof_names[i]
            print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    self.joint_damping[i] = self.cfg.control.joint_damping[dof_name]
                    self.joint_armature[i] = self.cfg.control.joint_armature[dof_name]
                    self.joint_friction[i] = self.cfg.control.joint_friction[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                self.joint_damping[i] = 0.
                self.joint_armature[i] = 0.
                self.joint_friction[i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._show_default_joint_paras()

        # init states
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        # env actor
        self.actor_handles = []
        self.envs = []
        # 相机参数
        if self.cfg.depth.use_camera or self.cfg.depth.use_warp:
            self._init_camera_paras()
            self._create_robot_body_meshes(asset_path, body_names)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(self.cfg.domain_rand.pose_xy[0], self.cfg.domain_rand.pose_xy[1], (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.cfg.depth.use_camera:
                self.attach_camera(i, env_handle, actor_handle)

    def _show_default_joint_paras(self):
        print("\n===================== default joint paras =====================")
        print(f"{'Joint':<30} {'kp':>6} {'kd':>6} {'damping':>10} {'armature':>10} {'friction':>10} {'default_pos':>10}")
        print("-" * 75)
        # 打印每个关节
        for i, name in enumerate(self.dof_names):
            if i >= self.num_actions:
                break
            # kp/kd 从第一个环境取即可（假设每个env相同）
            kp = self.p_gains[0, i].item() if self.p_gains.numel() > 0 else 0.0
            kd = self.d_gains[0, i].item() if self.d_gains.numel() > 0 else 0.0

            # damping, armature, friction
            damping = self.joint_damping[i].item()
            armature = self.joint_armature[i].item()
            friction = self.joint_friction[i].item()

            # default dof pos
            default_pos = self.default_dof_pos[0][i].item()
            print(f"{name:<30} {kp:>6.1f} {kd:>6.1f} {damping:>10.3f} {armature:>10.4f} {friction:>10.4f} {default_pos:>10.3f}")
        print("================================================================\n")

    def _show_robot_paras(self, body_props):
        print("\n===================== body & joint info =====================")
        print(f"{'Index':<6} {'Body Name':<25} {'Mass(kg)':>10}")
        print("-" * 50)
        for i in range(self.num_bodies):
            print(f"{i:<6} {self.body_names[i]:<25} {body_props[i].mass:>10.4f}")

        print("\n" + f"{'Index':<6} {'Joint Name':<25}")
        print("-" * 40)
        for i in range(self.num_dof):
            print(f"{i:<6} {self.dof_names[i]:<25}")
        print("===================== end =====================\n")

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        # -- robot terrain level, type, mask
        self.level_idx = torch.ones(self.num_envs,dtype=torch.long,device=self.device)
        self.type_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.plane_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.slope_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.stair_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        # -- robot terrain level, type, mask
        self.level_idx = torch.ones(self.num_envs,dtype=torch.long,device=self.device)
        self.type_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.plane_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.slope_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.stair_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # warp mesh
        self._create_warp_mesh()

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = np.clip(self.cfg.terrain.max_init_terrain_level, 0, self.cfg.terrain.num_rows)
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows
            self.terrain_levels = torch.randint(0, max_init_level, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _get_terrain_indices_from_pos(self):
        """
        根据机器人世界坐标 (base_pos)，计算对应的地形索引。
        同时区分是平面 (plane) 还是地形 (terrain)。
        Returns:
            level_idx (Tensor): 每个环境的 terrain level 索引
            type_idx  (Tensor): 每个环境的 terrain type 索引
            terrain_mask (Tensor): 0 表示 plane, 1 表示 terrain
        """
        if not self.cfg.terrain.curriculum:
            return
        # 获取配置
        num_rows = self.terrain.cfg.num_rows  # 行 (level)
        num_cols = self.terrain.cfg.num_cols  # 列 (type)
        env_length = self.terrain.env_length  # 每块地形长度
        env_width = self.terrain.env_width  # 每块地形宽度

        # --- 计算行列索引 ---
        self.level_idx = torch.floor(self.base_pos[:, 0] / env_length)
        self.type_idx = torch.floor(self.base_pos[:, 1] / env_width)
        # -- plane设为-1
        self.type_idx[self.type_idx >= num_cols] = -1
        self.type_idx[self.type_idx <= -1]       = -1
        type_mask_plane = self.type_idx.clone()
        type_mask_stair = self.type_idx.clone()
        if self.terrain.plane_type_idx:
            # 将 plane 对应环境 mask 置 -1
            plane_mask = torch.isin(type_mask_plane, torch.tensor(self.terrain.plane_type_idx, device=self.device))
            type_mask_plane[plane_mask] = -1
        if self.terrain.slope_type_idx:
            # 将 slope 对应环境 mask 置 -2
            slope_mask = torch.isin(type_mask_plane, torch.tensor(self.terrain.slope_type_idx, device=self.device))
            type_mask_plane[slope_mask] = -2
        if self.terrain.stair_type_idx:
            # 将 stair 对应环境 mask 置 -3
            stair_mask = torch.isin(type_mask_stair, torch.tensor(self.terrain.stair_type_idx, device=self.device))
            type_mask_stair[stair_mask] = -3
        # --- 判断是否为 terrain ---
        # terrain_mask: 1 表示在step地形区间, 0 表示和在 plane\slope
        # type_idx >= 0, type_idx < (num_cols)
        # level_idx >= 0, level_idx < (num_rows)
        self.terrain_mask[:] = (
                (type_mask_plane >= 0) & (type_mask_plane < num_cols) &
                (self.level_idx >= 0) & (self.level_idx < num_rows)
        )
        self.step_mask[:] = self.terrain_mask[:]
        self.plane_mask[:] = (type_mask_plane == -1)
        self.slope_mask[:] = (type_mask_plane == -2)
        self.stair_mask[:] = (type_mask_stair == -3)
        # if not self.headless:
        #     print("\n===================================")
        #     print("all name:", self.terrain.terrain_type_name)
        #     print("type:",self.type_idx[self.lookat_id].item())
        #     print("level:", self.level_idx[self.lookat_id].item())
        #     print("terrain:", self.terrain_mask[self.lookat_id].item())
        #     print("plane:", self.plane_mask[self.lookat_id].item())
        #     print("slope:", self.slope_mask[self.lookat_id].item())
        #     print("stair:", self.stair_mask[self.lookat_id].item())

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, :self.num_actions, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, :self.num_actions, 1]
        if self.arm_dof_enable:
            self.arm_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, -self.num_arms:, 0]
            self.arm_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, -self.num_arms:, 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rigid_body_pos = self.rigid_state[..., 0:3]
        self.rigid_body_quat = self.rigid_state[..., 3:7]
        self.rigid_body_vel = self.rigid_state[..., 7:10]
        self.rigid_body_ang = self.rigid_state[..., 10:13]
        self.left_feet_quat = self.rigid_state[:, self.feet_indices[0], 3:7]
        self.right_feet_quat = self.rigid_state[:, self.feet_indices[1], 3:7]
        self.left_feet_pos = torch.zeros_like(self.base_pos)
        self.right_feet_pos = torch.zeros_like(self.base_pos)
        self.left_feet_toe_pos = torch.zeros_like(self.base_pos)
        self.right_feet_toe_pos = torch.zeros_like(self.base_pos)
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        if self.arm_dof_enable:
            self.arm_torques = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.arm_actions = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.arm_actions_fil = torch.zeros(self.num_envs, self.num_arms, dtype=torch.float, device=self.device, requires_grad=False)
            self.arm_kp = torch.full((self.num_envs, self.num_arms), 200, dtype=torch.float, device=self.device)
            self.arm_kd = torch.full((self.num_envs, self.num_arms),   2, dtype=torch.float, device=self.device)
            self.cfg.domain_rand.arm_pos_interval = np.ceil(self.cfg.domain_rand.arm_pos_interval_s / self.dt)

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_rigid_body_vel = torch.zeros_like(self.rigid_body_vel)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_walking_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_walking_y_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_walking_z_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.current_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.current_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_both_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_xyz = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_air = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_forces = self.contact_forces[:, self.feet_indices, :]
        history_len = 3
        self.feet_forces_history = torch.zeros(self.num_envs, history_len, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_vel_history = torch.zeros(self.num_envs, history_len, len(self.feet_indices), 3, dtype=torch.float, device=self.device)
        self.single_contact_history = torch.zeros(self.num_envs, 20, dtype=torch.bool, device=self.device) # 0.2s 内有单支撑奖励， 等价于：惩罚超过连续0.2s的双支撑
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.left_feet_euler = get_euler_xyz_tensor(self.left_feet_quat)
        self.right_feet_euler = get_euler_xyz_tensor(self.right_feet_quat)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        # === height maps ===
        self.num_height_points = 0
        self.num_feet_height_points = 0
        self.num_feet_hold_points = 0
        self.num_base_height_points = 0
        self.num_feet_forward_points = 0
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.feet_height_points = self._init_feet_height_points()
            self.feet_hold_points = self._init_feet_hold_points()
            self.base_height_points = self._init_base_height_points()
            self.feet_forward_points = self._init_feet_forward_points()
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_height_maps = torch.zeros(self.num_envs, self.num_feet_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_height_maps = torch.zeros(self.num_envs, self.num_feet_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_hold_maps = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_hold_maps = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_hold_maps_first_contact = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_hold_maps_first_contact = torch.zeros(self.num_envs, self.num_feet_hold_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_height_maps = torch.zeros(self.num_envs, self.num_base_height_points, dtype=torch.float, device=self.device, requires_grad=False)

        self.left_feet_forward_maps = torch.zeros(self.num_envs, self.num_feet_forward_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_forward_maps = torch.zeros(self.num_envs, self.num_feet_forward_points, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_forward_points_w = torch.zeros(self.num_envs, self.num_feet_forward_points, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_feet_forward_points_w = torch.zeros(self.num_envs, self.num_feet_forward_points, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_feet_filter_n_points = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.right_feet_filter_n_points = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        # === foothold candidate buffers ===
        # 每只脚在“第一次离地(first_air)”时，根据前方 1.5m 地形点云生成一组平坦候选点。
        # 缓存使用固定长度 [num_envs, candidate_num, 3]，这样 reward、可视化和 PPO batch 都不需要处理变长列表。
        self.num_foothold_candidates = self.cfg.terrain.foothold_candidate_num
        self.left_foothold_candidates = torch.zeros(
            self.num_envs, self.num_foothold_candidates, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.right_foothold_candidates = torch.zeros_like(self.left_foothold_candidates)
        # valid 标志记录哪些候选是真正由平坦窗口产生的；不足 candidate_num 的位置会用最后一个候选补齐，
        # 但 valid=False 的位置在 reward 中会被屏蔽，避免补齐点重复影响最近距离。
        self.left_foothold_candidate_valid = torch.zeros(
            self.num_envs, self.num_foothold_candidates,
            dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.right_foothold_candidate_valid = torch.zeros_like(self.left_foothold_candidate_valid)
        # 触地(first_contact)瞬间，记录真实落点到最近候选点的水平距离；reward 只在触地帧读取这个误差。
        self.left_foothold_landing_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_foothold_landing_error = torch.zeros_like(self.left_foothold_landing_error)
        # debug 绘制前方点云时使用：从该 index 之后的点属于速度过滤后的搜索区域。
        self.filter_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        # feet ray caster
        self.feet_front_dist_ray = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_back_dist_ray = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        # === 推力 ===
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_small_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_small_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.apply_force = torch.zeros((self.num_envs,self.num_bodies, 3), dtype=torch.float32, device=self.device)
        self.apply_torque = torch.zeros((self.num_envs,self.num_bodies, 3), dtype=torch.float32, device=self.device)

        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_privileged_obs, dtype=torch.float, device=self.device))
        # gait paras
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait), dtype=torch.int, device=self.device, requires_grad=False)
        self.stand_command = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.low_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device) * 0.5
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.cycle_time = torch.full((self.num_envs,), self.cfg.rewards.cycle_time, device=self.device, dtype=torch.float)
        self.add_cycle_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_offset = self.cfg.rewards.phase_offset
        self.stand_radio = torch.full((self.num_envs,), self.cfg.rewards.stand_radio, device=self.device, dtype=torch.float)
        # 记录上一次双足着地瞬间计算出的两脚间距
        self.last_x_dist = torch.zeros(self.num_envs, device=self.device)
        # 记录上一帧的状态：True表示上一帧是单支撑（只有一只脚着地），False表示不是
        self.was_single_support = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.cmd_action_latency_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.range_cmd_action_latency[1]+1,device=self.device)
        self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_latency_buffer(torch.arange(self.num_envs, device=self.device))

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    # =============== 机器人参数随机化：质量、惯量、质心、关节阻尼、电机转子惯量、Kp、Kd、扭矩、外部推力、action delay、 sensor niose ===============================
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = self.cfg.domain_rand.num_buckets
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = self.cfg.domain_rand.num_buckets
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_actions, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.q_torque_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(self.num_actions):
                props["effort"][i] = self.cfg.control.dof_torque_max[i]
                props["velocity"][i] = self.cfg.control.dof_vel_max[i]
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = self.cfg.control.dof_vel_limits[i]
                self.q_torque_limits[i] = self.cfg.control.dof_torque_limits[i]
                self.torque_limits[i] = self.cfg.control.dof_torque_limits[i]
                # set default joint paras
                props["damping"][i] = self.joint_damping[i]
                props["armature"][i] = self.joint_armature[i]
                props["friction"][i] = self.joint_friction[i]
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2.
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            if self.arm_dof_enable:
                self.arm_torque_limit = 30
                self.arm_vel_limit = 10
                self.arm_damping = 0.1
                self.arm_armature = 0.01
                self.arm_friction = 0.00
                self.arm_kp = torch.full((self.num_envs, self.num_arms), 100, dtype=torch.float, device=self.device)
                self.arm_kd = torch.full((self.num_envs, self.num_arms), 2, dtype=torch.float, device=self.device)
                arm_kp = [100, 100, 100, 100, 100, 100, 100, 100]
                arm_kd = [  2,   2,   2,   2,   2,   2,   2,   2]
                for i in range(self.num_arms):
                    props["effort"][i + self.num_actions] = self.arm_torque_limit
                    props["velocity"][i + self.num_actions] = self.arm_vel_limit
                    props["damping"][i + self.num_actions] = self.arm_damping
                    props["armature"][i + self.num_actions] = self.arm_armature
                    props["friction"][i + self.num_actions] = self.arm_friction
                    self.arm_kp[:, i] = arm_kp[i]
                    self.arm_kd[:, i] = arm_kd[i]


        # rand joint damping armature friction
        for i in range(self.num_actions):
            if self.cfg.domain_rand.randomize_joint_damping:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_damping_range[0], self.cfg.domain_rand.joint_damping_range[1])
                if self.cfg.domain_rand.damping_operation == "abs":
                    props["damping"][i] = rd_num
                elif self.cfg.domain_rand.damping_operation == "scale":
                    props["damping"][i] = self.joint_damping[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.damping_operation}' for property randomization. ")

            if self.cfg.domain_rand.randomize_joint_armature:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_armature_range[0], self.cfg.domain_rand.joint_armature_range[1])
                if self.cfg.domain_rand.armature_operation == "abs":
                    props["armature"][i] = rd_num
                elif self.cfg.domain_rand.armature_operation == "scale":
                    props["armature"][i] = self.joint_armature[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.armature_operation}' for property randomization. ")

            if self.cfg.domain_rand.randomize_joint_friction:
                rd_num = np.random.uniform(self.cfg.domain_rand.joint_friction_range[0], self.cfg.domain_rand.joint_friction_range[1])
                if self.cfg.domain_rand.friction_operation == "abs":
                    props["friction"][i] = rd_num
                elif self.cfg.domain_rand.friction_operation == "scale":
                    props["friction"][i] = self.joint_friction[i] * rd_num
                else:
                    raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.friction_operation}' for property randomization. ")
        # show random para
        if env_id == 0:
            print("\n===================== props random paras =========================")
            print("effort: ",props["effort"])
            print("velocity: ", props["velocity"])
            print("damping: ", props["damping"])
            print("armature: ", props["armature"])
            print("friction: ", props["friction"])
            print("===================== props random paras =========================\n")

        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if env_id == 0:
            self._show_robot_paras(props)
            self.rd_mass_body_idx = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.domain_rand.randomize_mass_body_name)
            self.rd_com_body_idx = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.domain_rand.randomize_com_body_name)
            print("rand idx:", self.rd_mass_body_idx, self.rd_com_body_idx)

        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_base_mass_range
            props[self.rd_mass_body_idx].mass += np.random.uniform(rng[0], rng[1])
        self.body_mass[env_id] = props[self.rd_mass_body_idx].mass

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            rng_comx, rng_comy, rng_comz = self.cfg.domain_rand.added_base_com_range
            rand_com_x = np.random.uniform(rng_comx[0], rng_comx[1])
            rand_com_y = np.random.uniform(rng_comy[0], rng_comy[1])
            rand_com_z = np.random.uniform(rng_comz[0], rng_comz[1])
            props[self.rd_com_body_idx].com.x += 1. * rand_com_x
            props[self.rd_com_body_idx].com.y += 1. * rand_com_y
            props[self.rd_com_body_idx].com.z += 1. * rand_com_z

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.multiplied_link_mass_range
            for i in range(len(props)):
                if i == self.rd_mass_body_idx:
                    continue
                props[i].mass *= np.random.uniform(rng[0], rng[1])

        # randomize com of all link
        if self.cfg.domain_rand.randomize_link_com:
            for s in range(len(props)):
                if s == self.rd_com_body_idx:
                    continue
                rng_comx, rng_comy, rng_comz = self.cfg.domain_rand.added_link_com_range
                rand_com_x = np.random.uniform(rng_comx[0], rng_comx[1])
                rand_com_y = np.random.uniform(rng_comy[0], rng_comy[1])
                rand_com_z = np.random.uniform(rng_comz[0], rng_comz[1])
                props[s].com.x += 1.0 * rand_com_x
                props[s].com.y += 1.0 * rand_com_y
                props[s].com.z += 1.0 * rand_com_z

        # randomize inertia of all body
        if self.cfg.domain_rand.randomize_inertia:
            rng = self.cfg.domain_rand.multiplied_inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                props[s].inertia.z.z *= rd_num
        return props

    def _init_random_motor_paras(self):
        # random Kp Kd strength offset of joint
        self.kp_factor = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.kd_factor = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.motor_strength = torch.ones((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        self.motor_offset = torch.zeros((self.num_envs, self.num_actions), device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_pd_factor:
            rng = self.cfg.domain_rand.Kp_factor_range
            self.kp_factor[:] = torch_rand_float(rng[0], rng[1], shape=(self.num_envs, self.num_actions), device=self.device)
            rng = self.cfg.domain_rand.Kd_factor_range
            self.kd_factor[:] = torch_rand_float(rng[0], rng[1], shape=(self.num_envs, self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[:] = torch_rand_float(rng[0], rng[1], shape=(self.num_envs, self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_motor_offset:
            rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offset[:] = torch_rand_float(rng[0], rng[1], shape=(self.num_envs, self.num_actions), device=self.device)

    def randomize_dof_props(self, env_ids):
        # Randomise the motor strength:
        # rand ouput torque
        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[env_ids, :] = torch_rand_float(rng[0], rng[1], shape=(len(env_ids), self.num_actions),
                                                   device=self.device)
        # rand motor position offset
        if self.cfg.domain_rand.randomize_motor_offset:
            rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offset[env_ids, :] = torch_rand_float(rng[0], rng[1], (len(env_ids), self.num_actions), device=self.device)


        # rand kp kd gain
        if self.cfg.domain_rand.randomize_pd_factor:
            rng = self.cfg.domain_rand.Kp_factor_range
            self.kp_factor[env_ids, :] = torch_rand_float(rng[0], rng[1], shape=(len(env_ids), self.num_actions), device=self.device)
            rng = self.cfg.domain_rand.Kd_factor_range
            self.kd_factor[env_ids, :] = torch_rand_float(rng[0], rng[1], shape=(len(env_ids), self.num_actions), device=self.device)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        push_x, push_y = self.cfg.domain_rand.max_push_vel_xy
        self.rand_push_force[:, 0] = torch.empty_like(self.rand_push_force[:, 0]).uniform_(push_x[0], push_x[1])
        self.rand_push_force[:, 1] = torch.empty_like(self.rand_push_force[:, 1]).uniform_(push_y[0], push_y[1])
        self.rand_push_torque.uniform_(-self.cfg.domain_rand.max_push_ang_vel, self.cfg.domain_rand.max_push_ang_vel)
        rand_push_force_w = quat_apply(quat_yaw(self.base_quat), self.rand_push_force[:, :3])

        if self.cfg.domain_rand.push_operation == "add":
            self.root_states[:, 7:9] += rand_push_force_w[:, :2]
            self.root_states[:, 10:13] += self.rand_push_torque
        elif self.cfg.domain_rand.push_operation == "abs":
            self.root_states[:, 7:9] = rand_push_force_w[:, :2]
            self.root_states[:, 10:13] = self.rand_push_torque
        else:
            raise NotImplementedError(f"Unknown operation: '{self.cfg.domain_rand.push_operation}' for property randomization. ")

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _small_push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        self.rand_small_push_force.uniform_(-self.cfg.domain_rand.max_small_push_vel_xy, self.cfg.domain_rand.max_small_push_vel_xy)
        self.rand_small_push_torque.uniform_(-self.cfg.domain_rand.max_small_push_ang_vel, self.cfg.domain_rand.max_small_push_ang_vel)

        self.root_states[:, 7:9] += self.rand_small_push_force[:, :2]
        self.root_states[:, 10:13] += self.rand_small_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _apply_force_torque(self):
        if self.common_step_counter % self.cfg.domain_rand.apply_interval == 0:
            self.apply_force[:, 0, :2].uniform_(-self.cfg.domain_rand.max_apply_force * self.cfg.control.decimation, self.cfg.domain_rand.max_apply_force * self.cfg.control.decimation)
            self.apply_torque[:, 0, :3].uniform_(-self.cfg.domain_rand.max_apply_torque * self.cfg.control.decimation, self.cfg.domain_rand.max_apply_torque * self.cfg.control.decimation)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.apply_force),
            torqueTensor=gymtorch.unwrap_tensor(self.apply_torque),
            space=gymapi.CoordinateSpace.LOCAL_SPACE
        )

    def update_cmd_action_latency_buffer(self):
        if self.cfg.domain_rand.add_cmd_action_latency:
            range_end = self.cfg.domain_rand.range_cmd_action_latency[1]
            self.cmd_action_latency_buffer[:, :, 1:] = self.cmd_action_latency_buffer[:, :, :range_end]
            self.cmd_action_latency_buffer[:, :, 0] = self.actions
            action_delayed = self.cmd_action_latency_buffer[torch.arange(self.num_envs), :,
                             self.cmd_action_latency_simstep.long()]
        else:
            action_delayed = self.actions
        return action_delayed

    def _reset_latency_buffer(self, env_ids):
        if self.cfg.domain_rand.add_cmd_action_latency:
            self.cmd_action_latency_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_cmd_action_latency:
                self.cmd_action_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_cmd_action_latency[0],
                    self.cfg.domain_rand.range_cmd_action_latency[1] + 1, (len(env_ids),), device=self.device)
            else:
                self.cmd_action_latency_simstep[env_ids] = self.cfg.domain_rand.range_cmd_action_latency[1]

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        n_cmd = self.cfg.env.num_commands
        n_dof = self.cfg.env.num_actions
        noise_vec[0                   : n_cmd              ] = 0.  # commands
        noise_vec[n_cmd               : n_cmd + n_dof      ] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[n_cmd + 1*n_dof     : n_cmd + 2*n_dof    ] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[n_cmd + 2*n_dof     : n_cmd + 3*n_dof    ] = 0.  # previous actions
        noise_vec[n_cmd + 3*n_dof     : n_cmd + 3*n_dof + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[n_cmd + 3*n_dof + 3 : n_cmd + 3*n_dof + 5] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        print("noise = :", noise_vec)
        return noise_vec


    #==================================================== 地形高程图 可视化绘图 ==========================================================================
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.cfg.terrain.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self._draw_height_maps(self.lookat_id)
        # self._draw_base_height_maps(self.lookat_id)
        # self._draw_feet_height_maps(self.lookat_id, 0)
        # self._draw_feet_height_maps(self.lookat_id, 1)
        self._draw_feet_hold_maps(self.lookat_id, 0)
        self._draw_feet_hold_maps(self.lookat_id, 1)
        # self._draw_feet_dist_ray(self.lookat_id)
        self._draw_depth_image(self.lookat_id)
        # self._draw_feet_forward_maps(self.lookat_id, 0)
        # self._draw_feet_forward_maps(self.lookat_id, 1)
        self._draw_foothold_candidates(self.lookat_id, 0)
        self._draw_foothold_candidates(self.lookat_id, 1)

    def _draw_height_maps(self, env_idx):
        # ===================== measured_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_base_height_maps(self, env_idx):
        # ===================== measured_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.base_height_maps[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.base_height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_feet_height_maps(self, env_idx, feet_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        if feet_idx == 0:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[0]]).cpu().numpy()
            heights = self.left_feet_height_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.left_feet_quat[i].repeat(heights.shape[0]), self.feet_height_points[i]).cpu().numpy()
        else:
            base_pos = (self.rigid_body_pos[i, self.feet_indices[1]]).cpu().numpy()
            heights = self.right_feet_height_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.right_feet_quat[i].repeat(heights.shape[0]), self.feet_height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_feet_hold_maps(self, env_idx, feet_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 20, 20, None, color=(1, 1, 0))
        if feet_idx == 0:
            base_pos = self.left_feet_pos[i].cpu().numpy()
            heights = self.left_feet_hold_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.left_feet_quat[i].repeat(heights.shape[0]), self.feet_hold_points[i]).cpu().numpy()
        else:
            base_pos = self.right_feet_pos[i].cpu().numpy()
            heights = self.right_feet_hold_maps[i].cpu().numpy()
            height_points = quat_apply_yaw(self.right_feet_quat[i].repeat(heights.shape[0]), self.feet_hold_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_feet_dist_ray(self, env_idx):
        i = env_idx
        gymutil.draw_line(gymapi.Vec3(self.right_feet_toe_pos[i, 0], self.right_feet_toe_pos[i, 1], self.right_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_feet_ray_pos[i, 0], self.right_feet_ray_pos[i, 1], self.right_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        gymutil.draw_line(gymapi.Vec3(self.right_feet_toe_pos[i, 0], self.right_feet_toe_pos[i, 1], self.right_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_feet_ray_pos[i, 0], self.right_feet_ray_pos[i, 1], self.right_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        gymutil.draw_line(gymapi.Vec3(self.right_feet_toe_pos[i, 0], self.right_feet_toe_pos[i, 1], self.right_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_feet_ray_pos[i, 0], self.right_feet_ray_pos[i, 1], self.right_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])

        gymutil.draw_line(gymapi.Vec3(self.right_back_feet_toe_pos[i, 0], self.right_back_feet_toe_pos[i, 1], self.right_back_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_back_feet_ray_pos[i, 0], self.right_back_feet_ray_pos[i, 1], self.right_back_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        gymutil.draw_line(gymapi.Vec3(self.right_back_feet_toe_pos[i, 0], self.right_back_feet_toe_pos[i, 1], self.right_back_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_back_feet_ray_pos[i, 0], self.right_back_feet_ray_pos[i, 1], self.right_back_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        gymutil.draw_line(gymapi.Vec3(self.right_back_feet_toe_pos[i, 0], self.right_back_feet_toe_pos[i, 1], self.right_back_feet_toe_pos[i, 2]),
                          gymapi.Vec3(self.right_back_feet_ray_pos[i, 0], self.right_back_feet_ray_pos[i, 1], self.right_back_feet_ray_pos[i, 2]),
                          gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])

    def _draw_feet_forward_maps(self, env_idx, feet_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        if feet_idx == 0:
            points = self.left_feet_forward_points_w[i,self.filter_indices[i]:,:].cpu().numpy()
        else:
            points = self.right_feet_forward_points_w[i,self.filter_indices[i]:,:].cpu().numpy()
        for j in range(points.shape[0]):
            x = points[j, 0]
            y = points[j, 1]
            z = points[j, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # sphere_geom_1 = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 0, 0))
        # if feet_idx == 0:
        #     points = self.left_feet_forward_points_w[i].cpu().numpy()
        # else:
        #     points = self.right_feet_forward_points_w[i].cpu().numpy()
        # for j in range(points.shape[0]):
        #     x = points[j, 0]
        #     y = points[j, 1]
        #     z = points[j, 2]
        #     sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #     gymutil.draw_lines(sphere_geom_1, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_foothold_candidates(self, env_idx, feet_idx):
        """
        绘制某只脚当前缓存的候选落脚点。

        颜色约定：
        - 绿色大球：真实有效候选，由“平坦窗口”中心点生成。
        - 灰色小球：固定长度缓存的补齐位置，或者当前还没有有效候选的位置。

        这里只读缓存，不重新计算候选；候选的更新时机由 first_air 事件控制。
        """
        i = env_idx
        valid_geom = gymutil.WireframeSphereGeometry(0.02, 20, 20, None, color=(0, 1, 0))
        invalid_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0.35, 0.35, 0.35))

        if feet_idx == 0:
            points = self.left_foothold_candidates[i].cpu().numpy()
            valid = self.left_foothold_candidate_valid[i].cpu().numpy()
        else:
            points = self.right_foothold_candidates[i].cpu().numpy()
            valid = self.right_foothold_candidate_valid[i].cpu().numpy()

        for j in range(points.shape[0]):
            geom = valid_geom if valid[j] else invalid_geom
            x = points[j, 0]
            y = points[j, 1]
            z = points[j, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _update_feet_pos(self):
        # rigid_body_pos 中的 feet link 位置近似在踝关节处；真实脚底接触点相对踝关节约有 0.1m 向下偏移。
        # cfg.rewards.feet_height 当前为 0.1，这里把踝关节位置转换成脚底点位置，后续落脚误差和脚底采样都使用该点。
        feet_offset_local = torch.tensor(
            [0.0, 0.0, -self.cfg.rewards.feet_height],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        left_feet_offset_world = quat_apply(self.left_feet_quat, feet_offset_local)
        right_feet_offset_world = quat_apply(self.right_feet_quat, feet_offset_local)
        self.left_feet_pos = (
                self.rigid_body_pos[:, self.feet_indices[0]]
                + left_feet_offset_world
        )
        self.right_feet_pos = (
                self.rigid_body_pos[:, self.feet_indices[1]]
                + right_feet_offset_world
        )

    def _update_feet_toe_pos(self):
        # rigid_body_pos 中的 feet link 位置近似在踝关节处；真实脚底接触点相对踝关节约有 0.1m 向下偏移。
        # cfg.rewards.feet_height 当前为 0.1，这里把踝关节位置转换成脚底点位置，后续落脚误差和脚底采样都使用该点。
        feet_offset_local = torch.tensor(
            [0.15, 0.0, -0.08],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        left_feet_offset_world = quat_apply(self.left_feet_quat, feet_offset_local)
        right_feet_offset_world = quat_apply(self.right_feet_quat, feet_offset_local)
        self.left_feet_toe_pos = (
                self.rigid_body_pos[:, self.feet_indices[0]]
                + left_feet_offset_world
        )
        self.right_feet_toe_pos = (
                self.rigid_body_pos[:, self.feet_indices[1]]
                + right_feet_offset_world
        )


        back_feet_offset_local = torch.tensor(
            [-0.12, 0.0, -0.08],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        right_back_feet_offset_world = quat_apply(self.right_feet_quat, back_feet_offset_local)
        self.right_back_feet_toe_pos = (
                self.rigid_body_pos[:, self.feet_indices[1]]
                + right_back_feet_offset_world
        )
        # 计算射线
        ray_offset_local = torch.tensor(
            [0.5, 0.0, 0.0],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        right_ray_offset_world = quat_apply_yaw(self.right_feet_quat, ray_offset_local)
        self.right_feet_ray_pos = self.right_feet_toe_pos + right_ray_offset_world

        back_ray_offset_local = torch.tensor(
            [-0.5, 0.0, 0.0],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        right_back_ray_offset_world = quat_apply_yaw(self.right_feet_quat, back_ray_offset_local)
        self.right_back_feet_ray_pos = self.right_back_feet_toe_pos + right_back_ray_offset_world

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_base_height_points(self):
        y = torch.tensor(self.cfg.terrain.base_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.base_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.feet_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.feet_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_hold_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.feet_hold_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.feet_hold_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_hold_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_hold_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_forward_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.feet_forward_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.feet_forward_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_forward_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_forward_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_base_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_base_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_feet_heights(self, feet_idx=0 ,env_ids=None):
        # feet_idx = 0: 左腿
        # feet_idx = 1: 右腿
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_feet_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            feet_quat = self.rigid_body_quat[env_ids, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_height_points), self.feet_height_points[env_ids]) + (self.rigid_body_pos[env_ids, self.feet_indices[feet_idx]]).unsqueeze(1)
        else:
            feet_quat = self.rigid_body_quat[:, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_height_points), self.feet_height_points) + (self.rigid_body_pos[:, self.feet_indices[feet_idx]]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_feet_hold(self, feet_idx=0 ,env_ids=None):
        # feet_idx = 0: 左腿
        # feet_idx = 1: 右腿
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_feet_hold_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        if feet_idx == 0:
            feet_pos = self.left_feet_pos
        else:
            feet_pos = self.right_feet_pos

        if env_ids:
            feet_quat = self.rigid_body_quat[env_ids, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_hold_points), self.feet_hold_points[env_ids]) + feet_pos.unsqueeze(1)
        else:
            feet_quat = self.rigid_body_quat[:, self.feet_indices[feet_idx]]
            points = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_hold_points), self.feet_hold_points)  + feet_pos.unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_feet_forward_heights(self, feet_idx=0 ,env_ids=None):
        # feet_idx = 0: 左腿
        # feet_idx = 1: 右腿
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_feet_forward_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        if feet_idx == 0:
            feet_pos = self.left_feet_pos
        else:
            feet_pos = self.right_feet_pos

        if env_ids:
            feet_quat = self.rigid_body_quat[env_ids, self.feet_indices[feet_idx]]
            points_w = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_forward_points), self.feet_forward_points[env_ids]) + feet_pos.unsqueeze(1)
        else:
            feet_quat = self.rigid_body_quat[:, self.feet_indices[feet_idx]]
            points_w = quat_apply_yaw(feet_quat.repeat(1, self.num_feet_forward_points), self.feet_forward_points) + feet_pos.unsqueeze(1)

        points = points_w + self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights_w = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        if feet_idx == 0:
            new_points_w = torch.zeros_like(self.left_feet_forward_points_w)
            new_points_w[:] = points_w
            new_points_w[:,:,2] = heights_w
            # self.left_feet_forward_points_w = torch.where(self.first_contact[:,1:2].unsqueeze(1), new_points_w, self.left_feet_forward_points_w)
        else:
            new_points_w = torch.zeros_like(self.left_feet_forward_points_w)
            new_points_w[:] = points_w
            new_points_w[:,:,2] = heights_w
            # self.right_feet_forward_points_w = torch.where(self.first_contact[:, 0:1].unsqueeze(1), new_points_w, self.right_feet_forward_points_w)

        return heights_w, new_points_w

    def terrain_heights(self, base_pos):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(len(base_pos), dtype=torch.float, device=self.device)
        else:
            x = self.terrain.border + base_pos[:, 0].cpu().numpy() / self.terrain.cfg.horizontal_scale
            y = self.terrain.border + base_pos[:, 1].cpu().numpy() / self.terrain.cfg.horizontal_scale
            x1 = np.floor(x).astype(int)
            y1 = np.floor(y).astype(int)
            x1 = np.clip(x1, 0, self.terrain.height_field_raw.shape[0]-2)
            y1 = np.clip(y1, 0, self.terrain.height_field_raw.shape[1]-2)
            x2 = x1 + 1
            y2 = y1 + 1
            return torch.tensor(
                (
                    (x2 - x) * (y2 - y) * self.terrain.height_field_raw[x1, y1]
                    + (x - x1) * (y2 - y) * self.terrain.height_field_raw[x2, y1]
                    + (x2 - x) * (y - y1) * self.terrain.height_field_raw[x1, y2]
                    + (x - x1) * (y - y1) * self.terrain.height_field_raw[x2, y2]
                )
                * self.terrain.cfg.vertical_scale,
                dtype=torch.float,
                device=self.device,
            )

    def _get_ankle_heights(self):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.rigid_state[:, self.feet_indices, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights


    #==================================================== gym api: camera \ nvidia warp: ray caster camera =============================================================
    def _init_camera_paras(self):
        self.cam_handles = []
        self.camera_props = []

        self.cam_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.cam_quat = torch.zeros(self.num_envs, 4, device=self.device)

        self.cam_pos_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.cam_rot_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.cam_fov_y = torch.empty(self.num_envs, device=self.device).uniform_(self.cfg.depth.vertical_fov[0], self.cfg.depth.vertical_fov[1])
        self.depth_width = self.cfg.depth.original[0]
        self.depth_height = self.cfg.depth.original[1]
        self.warp_cam_pos = wp.from_torch(self.cam_pos, dtype=wp.vec3)
        self.warp_cam_quat = wp.from_torch(self.cam_quat, dtype=wp.quat)
        self.warp_fov_y = wp.from_torch(self.cam_fov_y, dtype=wp.float32)

        self.wp_pixels = wp.zeros((self.num_envs, self.depth_width * self.depth_height), dtype=float, device=self.device)
        self.depth_graph = None

        for i in range(len(self.cfg.depth.pos_min)):
            self.cam_pos_offset[:, i] = torch.empty(self.num_envs, device=self.device).uniform_(self.cfg.depth.pos_min[i], self.cfg.depth.pos_max[i])
            self.cam_rot_offset[:, i] = torch.empty(self.num_envs, device=self.device).uniform_(self.cfg.depth.rot_min[i], self.cfg.depth.rot_max[i])
        self.cam_quat_offset = quat_from_euler_xyz(self.cam_rot_offset[:, 0], self.cam_rot_offset[:, 1], self.cam_rot_offset[:, 2])

        self.camera_depth_tensor = torch.zeros(self.num_envs,
                                        self.cfg.depth.original[1],
                                        self.cfg.depth.original[0]).to(self.device)

        self.depth_original = torch.zeros(self.num_envs,
                                        self.cfg.depth.original[1],
                                        self.cfg.depth.original[0]).to(self.device)

        self.depth_resized = torch.zeros(self.num_envs,
                                        self.cfg.depth.resized[1],
                                        self.cfg.depth.resized[0]).to(self.device)

        self.depth_clip = torch.zeros(self.num_envs,
                                            self.cfg.depth.original[1],
                                            self.cfg.depth.original[0]).to(self.device)

        self.depth_crop = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_gaussian = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_dropout = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_artifact = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_blur = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_noise = torch.zeros(self.num_envs,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

        self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.crop_size[1],
                                            self.cfg.depth.crop_size[0]).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.horizontal_fov = self.cam_fov_h[i]
            camera_props.enable_tensors = True
            camera_props.use_collision_geometry = False

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(*self.cam_pos_offset[i].cpu().numpy())
            local_transform.r = gymapi.Quat.from_euler_zyx(self.cam_rot_offset[i,2], self.cam_rot_offset[i,1], self.cam_rot_offset[i,0])
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_warp_mesh(self):
        wp.init()
        warp_vertices = wp.array(self.terrain.vertices, dtype=wp.vec3, device=self.device)
        warp_triangles = wp.array(self.terrain.triangles.flatten(order='C'), dtype=int, device=self.device)
        self.mesh = wp.Mesh(points=warp_vertices, indices=warp_triangles)

    def _build_robot_mesh_topology(self):
        all_vertices = []
        all_faces = []
        vertex_link_ids = []

        vertex_offset = 0

        for link_id, link in enumerate(self.robot_links):
            v = link.vertices_local.astype(np.float32)
            f = link.faces.astype(np.int32)

            all_vertices.append(v)
            all_faces.append(f + vertex_offset)
            vertex_link_ids.append(
                np.full(len(v), link_id, dtype=np.int32)
            )

            vertex_offset += len(v)

        self.robot_vertices_local = np.concatenate(all_vertices, axis=0)
        self.robot_faces = np.concatenate(all_faces, axis=0)
        self.vertex_link_ids = np.concatenate(vertex_link_ids, axis=0)

    def _rpy_to_matrix_np(self, rpy):
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return rz @ ry @ rx

    def _parse_xyz_np(self, value, default):
        if value is None:
            return np.array(default, dtype=np.float32)
        return np.array([float(x) for x in value.split()], dtype=np.float32)

    def _create_robot_body_meshes(self, asset_path, body_names):
        self.robot_body_mesh_ids = torch.zeros(len(body_names), dtype=torch.int64, device=self.device)
        self.robot_body_meshes = []

        if not self.cfg.depth.use_warp or not asset_path.lower().endswith(".urdf"):
            return

        asset_root = os.path.dirname(asset_path)
        root = ET.parse(asset_path).getroot()
        link_visual_meshes = {}
        for link in root.findall("link"):
            link_name = link.attrib.get("name")
            visual = link.find("visual")
            if link_name is None or visual is None:
                continue
            geometry = visual.find("geometry")
            mesh = None if geometry is None else geometry.find("mesh")
            if mesh is None or "filename" not in mesh.attrib:
                continue

            origin = visual.find("origin")
            xyz = self._parse_xyz_np(None if origin is None else origin.attrib.get("xyz"), [0.0, 0.0, 0.0])
            rpy = self._parse_xyz_np(None if origin is None else origin.attrib.get("rpy"), [0.0, 0.0, 0.0])
            scale = self._parse_xyz_np(mesh.attrib.get("scale"), [1.0, 1.0, 1.0])
            link_visual_meshes[link_name] = (mesh.attrib["filename"], xyz, rpy, scale)

        for body_id, body_name in enumerate(body_names):
            if body_name not in link_visual_meshes:
                continue

            mesh_filename, xyz, rpy, scale = link_visual_meshes[body_name]
            mesh_path = mesh_filename
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.normpath(os.path.join(asset_root, mesh_path))
            if not os.path.exists(mesh_path):
                continue

            loaded = trimesh.load_mesh(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            vertices = np.asarray(loaded.vertices, dtype=np.float32)
            faces = np.asarray(loaded.faces, dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                continue

            rot = self._rpy_to_matrix_np(rpy)
            vertices = (vertices * scale) @ rot.T + xyz
            warp_vertices = wp.array(vertices, dtype=wp.vec3, device=self.device)
            warp_faces = wp.array(faces.reshape(-1), dtype=int, device=self.device)
            body_mesh = wp.Mesh(points=warp_vertices, indices=warp_faces)
            self.robot_body_meshes.append(body_mesh)
            self.robot_body_mesh_ids[body_id] = int(body_mesh.id)

    def update_depth_buffer(self):
        if self.common_step_counter % self.cfg.depth.update_interval != 0:
            return
        # ===================== warp camera ===========================
        if self.cfg.depth.use_warp:
            width = self.cfg.depth.original[0]
            height = self.cfg.depth.original[1]
            buffer_len = self.cfg.depth.buffer_len
            # 更新相机在世界坐标的位置和姿态
            self.cam_pos[:] = self.base_pos + quat_rotate(self.base_quat, self.cam_pos_offset)
            self.cam_quat[:] = quat_mul(self.base_quat, self.cam_quat_offset)
            # warp 更新内容
            self.warp_cam_pos.assign(wp.from_torch(self.cam_pos, dtype=wp.vec3))
            self.warp_cam_quat.assign(wp.from_torch(self.cam_quat, dtype=wp.quat))
            # Warp 数组（一次创建，后续复用）
            self.wp_body_mesh_ids = wp.from_torch(self.robot_body_mesh_ids, dtype=wp.uint64)
            self.wp_body_poss = wp.from_torch(self.rigid_state[:, :, 0:3].contiguous().view(-1, 3), dtype=wp.vec3)
            self.wp_body_rots = wp.from_torch(self.rigid_state[:, :, 3:7].contiguous().view(-1, 4), dtype=wp.quat)

            # 批量渲染所有环境
            wp.launch(
                kernel=depth_draw_batch,
                dim=self.num_envs * width * height,
                inputs=[
                    self.mesh.id,  # mesh
                    self.cfg.terrain.border_size,
                    self.warp_cam_pos,  # cam_poss
                    self.warp_cam_quat,  # cam_rots
                    self.wp_body_mesh_ids,  # body_mesh_ids
                    self.wp_body_poss,  # body_poss
                    self.wp_body_rots,  # body_rots
                    self.num_bodies,  # num_bodies
                    self.warp_fov_y,  # fovs
                    width,  # width
                    height,  # height
                    self.wp_pixels  # pixels
                ],
                device=self.device,
            )
            # 转回 PyTorch tensor 并 reshape
            self.depth_original[:] = wp.to_torch(self.wp_pixels).view(self.num_envs, height, width)

        # ===================== gym api camera ========================
        if self.cfg.depth.use_camera:
            self.gym.step_graphics(self.sim)  # required to render in headless mode
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for i in range(self.num_envs):
                camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH)
                self.depth_original[i] = gymtorch.wrap_tensor(camera_depth_tensor) * -1
            self.gym.end_access_image_tensors(self.sim)
        # ===================== 处理 depth images ===========================
        # nan inf 检查
        self._set_nan_inf_zero()
        # 裁切，缩放
        self._depth_resized_crop_clip(self.depth_original)
        # 噪声
        self._apply_depth_noise()
        # 归一化
        self._depth_norm(self.depth_noise)
        # depth buffer
        self._refresh_depth_buffer()

    def _apply_depth_noise(self):
        if not self.cfg.depth.noise.enable:
            self.depth_noise = self.depth_clip
            return
        self.depth_gaussian = self._apply_gaussian_noise(self.depth_clip)
        self.depth_dropout = self._apply_dropout_noise(self.depth_gaussian)
        # self.depth_artifact = self._add_depth_artifacts_fixed_windows(
        #     self.depth_dropout,
        #     num_windows=1,
        #     window_h=3,
        #     window_w=2,
        #     noise_value=3.0,
        #     device=self.device,
        # )
        # self.depth_blur = self._gaussian_blur_noise(self.depth_artifact)
        self.depth_noise = self.depth_dropout

    def _apply_gaussian_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise to the depth images."""
        std_dev = self.cfg.depth.noise.depth_std
        std_dev += depth_images * self.cfg.depth.noise.depth_std_multiplier
        noise = torch.randn_like(depth_images) * std_dev
        noisy_depth = depth_images + noise
        return noisy_depth

    def _apply_dropout_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Applies dropout noise to the depth data."""
        dropout_mask = torch.rand_like(depth_images) < self.cfg.depth.noise.dropout_prob
        noisy_depth = depth_images.clone()
        noisy_depth[dropout_mask] = self.cfg.depth.noise.dropout_value

        # dropout_mask = torch.rand_like(depth_images) < self.cfg.depth.noise.dropout_prob
        # noisy_depth[dropout_mask] = self.cfg.depth.noise.dropout_value_max

        return noisy_depth

    def _gaussian_blur_noise(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to the image data."""
        # depth_images (N, H, W)
        # Create GaussianBlur transform
        blur_transform = GaussianBlur(kernel_size=5, sigma=0.5)
        # Apply Gaussian blur
        blurred = blur_transform(depth_images)
        return blurred

    def _add_depth_artifacts_fixed_windows(
            self,
            data,
            num_windows=3,
            window_h=5,
            window_w=5,
            noise_value=3.0,
            device=None,
    ):
        """
        Add fixed-size depth artifacts.

        For each depth image, randomly select `num_windows` windows with size
        `window_size x window_size`, and set the depth values inside these windows
        to `noise_value`.

        Args:
            data: Tensor with shape (N, H, W)
            num_windows: number of artifact windows per depth image
            window_h: artifact window height
            window_w: artifact window width
            noise_value: depth value assigned to artifact regions
            device: torch device

        Returns:
            data with artificial depth artifacts
        """
        if device is None:
            device = data.device

        N, H, W = data.shape
        data_ = data.clone()

        half_h = window_h // 2
        half_w = window_w // 2

        # Random center coordinates for each artifact window
        center_y = torch.randint(
            low=half_h,
            high=H - half_h,
            size=(N, num_windows),
            device=device,
        )

        center_x = torch.randint(
            low=half_w,
            high=W - half_w,
            size=(N, num_windows),
            device=device,
        )

        for i in range(num_windows):
            for n in range(N):
                cy = center_y[n, i]
                cx = center_x[n, i]

                y0 = cy - half_h
                y1 = y0 + window_h
                x0 = cx - half_w
                x1 = x0 + window_w

                data_[n, y0:y1, x0:x1] = noise_value

        return data_

    def _add_depth_artifacts(self,data, artifacts_prob, artifacts_height_mean_std, artifacts_width_mean_std, device, noise_value=0.0):
        """Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
        should be an artifacts, the mask is 1.
        """

        N, H, W = data.shape

        def _clip(data, dim):
            return torch.clip(data, 0.0, (H, W)[dim])

        # random patched artifacts
        artifacts_mask = torch_rand_float(0.0, 1.0, (N, H * W), device=device).view(N, H, W) < artifacts_prob
        artifacts_mask = artifacts_mask & (data[:, :, :] > 0.0)
        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32)  # (n_, 3) n_ <= N * H * W

        if len(artifacts_coord) == 0:
            return data

        artifacts_size = (
            torch.clip(
                artifacts_height_mean_std[0]
                + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_height_mean_std[1],
                0.0,
                H,
            ),
            torch.clip(
                artifacts_width_mean_std[0]
                + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_width_mean_std[1],
                0.0,
                W,
            ),
        )  # (n_,), (n_,)

        artifacts_top = _clip(artifacts_coord[:, 1] - artifacts_size[0] / 2, 0)
        artifacts_left = _clip(artifacts_coord[:, 2] - artifacts_size[1] / 2, 1)
        artifacts_bottom = _clip(artifacts_coord[:, 1] + artifacts_size[0] / 2, 0)
        artifacts_right = _clip(artifacts_coord[:, 2] + artifacts_size[1] / 2, 1)

        # create one-hot encoding for environment IDs
        env_ids = artifacts_coord[:, 0].long()
        env_onehot = torch.zeros((len(artifacts_coord), N), device=device)
        env_onehot[torch.arange(len(artifacts_coord)), env_ids] = 1.0

        # batch generate all artifacts
        num_artifacts = len(artifacts_coord)
        tops_expanded = artifacts_top[:, None, None]
        lefts_expanded = artifacts_left[:, None, None]
        bottoms_expanded = artifacts_bottom[:, None, None]
        rights_expanded = artifacts_right[:, None, None]

        # build the source patch
        source_patch = torch.zeros((num_artifacts, 1, 25, 25), device=device)
        source_patch[:, :, 1:24, 1:24] = 1.0

        # build the grid
        grid = torch.zeros((num_artifacts, H, W, 2), device=device)
        grid[..., 0] = torch.linspace(-1, 1, W, device=device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device=device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights_expanded - lefts_expanded) / (rights_expanded - lefts_expanded)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms_expanded - tops_expanded) / (bottoms_expanded - tops_expanded)

        # sample using the grid and form the artifacts for the entire depth image
        all_artifacts = F.grid_sample(
            source_patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(
            1
        )  # (num_artifacts, H, W)

        # combine the artifacts with the environment one-hot encoding
        # env_onehot: (num_artifacts, N)
        # all_artifacts: (num_artifacts, H, W)
        # final_masks: (N, H, W)
        final_masks = torch.einsum("an,ahw->nhw", env_onehot, all_artifacts)
        final_masks = torch.clamp(final_masks, 0, 1)

        data = data * (1 - final_masks) + final_masks * noise_value

        return data


    def _set_nan_inf_zero(self):
        self.depth_original[torch.isnan(self.depth_original)] = 0.
        self.depth_original[torch.isinf(self.depth_original)] = 0.

    def _process_depth_image(self):
        self.depth_resized = F.interpolate(
            self.depth_original.unsqueeze(1),
            size=(
                self.cfg.depth.resized[1],
                self.cfg.depth.resized[0]
            ),
            mode='nearest'
        ).squeeze(1)
        # 图像大小裁切
        x = self.cfg.depth.crop_rang_x
        y = self.cfg.depth.crop_rang_y
        self.depth_crop = self.depth_resized[:, y[0]:y[1], x[0]:x[1]]
        # 深度裁切
        # self.depth_clip = torch.where(self.depth_crop > self.cfg.depth.far_clip, 0, self.depth_crop)  # zero clip model
        self.depth_clip = torch.where(self.depth_crop > self.cfg.depth.far_clip, self.cfg.depth.far_clip, self.depth_crop)  # zero clip model
        self.depth_clip = torch.clamp(self.depth_clip, self.cfg.depth.near_clip, self.cfg.depth.far_clip)
        # 归一化
        self.depth_norm = self.depth_clip / self.cfg.depth.far_clip - 0.5

    def _depth_resized_crop_clip(self, data):
        self.depth_resized = F.interpolate(
            data.unsqueeze(1),
            size=(
                self.cfg.depth.resized[1],
                self.cfg.depth.resized[0]
            ),
            mode='nearest'
        ).squeeze(1)
        # 图像大小裁切
        x = self.cfg.depth.crop_rang_x
        y = self.cfg.depth.crop_rang_y
        self.depth_crop = self.depth_resized[:, y[0]:y[1], x[0]:x[1]]
        # 深度裁切
        # self.depth_clip = torch.where(self.depth_crop > self.cfg.depth.far_clip, 0, self.depth_crop)  # zero clip model
        self.depth_clip = torch.where(self.depth_crop > self.cfg.depth.far_clip, self.cfg.depth.far_clip, self.depth_crop)  # zero clip model
        self.depth_clip = torch.clamp(self.depth_clip, self.cfg.depth.near_clip, self.cfg.depth.far_clip)
        return self.depth_clip

    def _depth_norm(self, data):
        # self.depth_norm = data / self.cfg.depth.far_clip - 0.5
        self.depth_norm = data / self.cfg.depth.far_clip
        return self.depth_norm

    def _refresh_depth_buffer(self):
        init_flg = self.episode_length_buf <= 1
        self.depth_buffer[init_flg] = torch.stack([self.depth_norm[init_flg]] * self.cfg.depth.buffer_len, dim=1)
        self.depth_buffer[:] = torch.cat([self.depth_buffer[:, 1:], self.depth_norm.unsqueeze(1)], dim=1)

    def get_depth(self):
        return self.depth_buffer[:, -self.cfg.depth.obs_len:]

    def _draw_depth_image(self, i):
        if (self.cfg.depth.use_camera or  self.cfg.depth.use_warp) and not self.headless:
            viewer = DepthImageViewer(window_name="camera depth image",window_width=640,window_height=480)
            viewer.display(self.depth_original[i])

            # viewer_1 = DepthImageViewer(window_name="camera depth clip",window_width=480,window_height=480)
            # viewer_1.display(self.depth_clip[i])

            # viewer_2 = DepthImageViewer(window_name="camera depth gaussian",window_width=480,window_height=480)
            # viewer_2.display(self.depth_gaussian[i])
            #
            # viewer_3 = DepthImageViewer(window_name="camera depth dropout",window_width=480,window_height=480)
            # viewer_3.display(self.depth_dropout[i])

            # viewer_4 = DepthImageViewer(window_name="camera depth artifact",window_width=480,window_height=480)
            # viewer_4.display(self.depth_artifact[i])
            # #
            # viewer_5 = DepthImageViewer(window_name="camera depth blur",window_width=480,window_height=480)
            # viewer_5.display(self.depth_blur[i])
            viewer_5 = DepthImageViewer(window_name="camera depth noise",window_width=480,window_height=480)
            viewer_5.display(self.depth_noise[i])

    def update_feet_ray_caster_dist(self):
        if not self.cfg.terrain.feet_ray_enable:
            return
        # 提取 yaw 四元数
        left_feet_quat = quat_yaw(self.left_feet_quat)
        right_feet_quat = quat_yaw(self.right_feet_quat)
        foot_quat = torch.cat([left_feet_quat, right_feet_quat], dim=0)  # [N*2,4]
        # 脚位置
        # left_pos = self.rigid_body_pos[:, self.feet_indices[0]].clone()
        # right_pos = self.rigid_body_pos[:, self.feet_indices[1]].clone()
        # left_pos[:, 2] -= 0.08
        # right_pos[:, 2] -= 0.08
        # foot_pos = torch.cat([left_pos, right_pos], dim=0)  # [N*2,3]
        foot_pos = torch.cat([self.left_feet_toe_pos, self.right_feet_toe_pos], dim=0)  # [N*2,3]

        out_dist_front = torch.zeros(self.num_envs * 2, device=self.device)
        out_dist_back = torch.zeros(self.num_envs * 2, device=self.device)
        max_dist = self.cfg.terrain.feet_far_clip
        # -- warp 计算射线与地形接触距离
        wp.launch(
            feet_ray_cast_kernel,
            dim=self.num_envs * 2,
            inputs=[self.mesh.id, self.cfg.terrain.border_size, foot_pos, foot_quat, max_dist, out_dist_front, out_dist_back]
        )
        # -- 提取dist
        self.feet_front_dist_ray[:] = out_dist_front.view(2, self.num_envs).T  # shape [N_envs, 2], col0=left, col1=right
        self.feet_back_dist_ray[:] = out_dist_back.view(2, self.num_envs).T
        # -- dist clip： dist=0，设置为最大值
        self.feet_front_dist_ray[:] = torch.where(self.feet_front_dist_ray==0, max_dist, self.feet_front_dist_ray)
        self.feet_back_dist_ray[:] = torch.where(self.feet_back_dist_ray == 0, max_dist, self.feet_back_dist_ray)
        # print("=============================================")
        # torch.set_printoptions(precision=3)
        # print(self.feet_front_dist_ray[self.lookat_id, 0], self.feet_back_dist_ray[self.lookat_id, 0])
        # print(self.feet_front_dist_ray[self.lookat_id, 1], self.feet_back_dist_ray[self.lookat_id, 1])

    # =========================================================== Walk Gait ==========================================================================
    def _get_phase(self):
        return self.phase

    def _phase_step_update(self):
        self.low_speed = (torch.norm(self.base_lin_vel[:, :2], dim=1) <= 0.5)
        self.stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold) * self.low_speed
        self.phase += self.dt / (self.cycle_time + self.add_cycle_time)
        self.phase[self.stand_command] = 0

        self.phase_left = self.phase % 1
        self.phase_right = (self.phase + self.phase_offset) % 1
        self.phase_left[self.stand_command] = 0.01
        self.phase_right[self.stand_command] = 0.01

    def _gait_style_update(self):
        vel = torch.abs(self.commands[:, 0])
        self.cycle_time[vel < 0.1] = 0.7
        self.cycle_time[(0.1 <= vel)] = self.cfg.rewards.cycle_time

        self.stand_radio[vel < 0.5] = 0.60
        # self.stand_radio[(0.5 <= vel) & (vel <= 1.2)] = self.cfg.rewards.stand_radio
        self.stand_radio[(0.5 <= vel)] = self.cfg.rewards.stand_radio

    def _get_gait_phase(self):
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = self.phase_left <= self.stand_radio
        # right foot stance
        stance_mask[:, 1] = self.phase_right <= self.stand_radio
        return stance_mask


    # ============================================= Event Command  Sensor Noise ======================================================================
    def _update_command(self):
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.ranges.ang_vel_yaw[1])

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :] = 0.0

        walking_x_env_ids = self.is_walking_x_env.nonzero(as_tuple=False).flatten()
        self.commands[walking_x_env_ids, 1] = 0.0
        self.commands[walking_x_env_ids, 2] = 0.0

        walking_y_env_ids = self.is_walking_y_env.nonzero(as_tuple=False).flatten()
        self.commands[walking_y_env_ids, 0] = 0.0
        self.commands[walking_y_env_ids, 2] = 0.0

        walking_z_env_ids = self.is_walking_z_env.nonzero(as_tuple=False).flatten()
        self.commands[walking_z_env_ids, 0] = 0.0
        self.commands[walking_z_env_ids, 1] = 0.0

        self._zero_small_commands()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        cmd_rand = torch.rand(len(env_ids), device=self.device)

        p_s = self.cfg.commands.standing_env_radio
        p_x = p_s + self.cfg.commands.walking_x_env_radio
        p_y = p_x + self.cfg.commands.walking_y_env_radio
        p_z = p_y + self.cfg.commands.walking_z_env_radio

        self.is_standing_env[env_ids] = cmd_rand <= p_s
        self.is_walking_x_env[env_ids] = (cmd_rand > p_s) & (cmd_rand <= p_x)
        self.is_walking_y_env[env_ids] = (cmd_rand > p_x) & (cmd_rand <= p_y)
        self.is_walking_z_env[env_ids] = (cmd_rand > p_y) & (cmd_rand <= p_z)

    def _resample_gait_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:, i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return

        # rand sample
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                                    gait_time_range[1],
                                                    (len(envs), 1), device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone()
        scaled_tensor[:, 0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_stand_command(self, env_ids):
        self.is_standing_env[env_ids] = True
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_sagittal_command(self, env_ids):
        self.is_standing_env[env_ids] = False
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.is_standing_env[env_ids] = False
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_rotate_command(self, env_ids):
        self.is_standing_env[env_ids] = False
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self, env_ids):
        self.is_standing_env[env_ids] = False
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _zero_small_commands(self):
        # set small commands to zero
        # self.commands[:, :] *= (torch.abs(self.commands[:, :]) >= self.cfg.commands.min_vel)
        self.commands[:, 0:2] *= (torch.abs(self.commands[:, 0:2]) >= self.cfg.commands.min_vel)


    # =========================================================== 课程学习 =============================================================================
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_terrain_curriculum_vel(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        if "tracking_lin_vel" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        elif "tracking_lin_vel_x" in self.episode_sums.keys():
            move_down = (
                self.episode_sums["tracking_lin_vel_x"][env_ids] / self.max_episode_length_s
                < (self.reward_scales["tracking_lin_vel_x"] / self.dt) * 0.5
            ) * ~move_up
            self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        else:
            print("no tracking_lin_vel in reward for terrain_curriculum")
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    # ============================================================ AMP ===============================================================================
    def _init_amp_motion(self):
        self.amp_loader = AMPLoader(
            motion_files=self.cfg.env.amp_motion_files_display,
            device=self.device,
            time_between_frames=self.dt
        )

    def visualize_amp_motion(self, motion_idx=0):
        device = self.device
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=device)
        motion_count = len(self.amp_loader.trajectories_full)
        if motion_idx < 0 or motion_idx >= motion_count:
            raise ValueError(
                f"motion_index={motion_idx} is out of range; expected 0..{motion_count - 1}"
            )

        data_len = int(self.amp_loader.trajectory_num_frames[motion_idx])
        frame_duration = float(self.amp_loader.trajectory_frame_durations[motion_idx])
        frame_stride = int(self.amp_loader.trajectory_transition_steps[motion_idx])
        motion_name = self.amp_loader.trajectory_names[motion_idx]
        print(
            f"Replaying AMP motion [{motion_idx}] {motion_name}: "
            f"{data_len} frames at {1.0 / frame_duration:.1f} Hz, "
            f"viewer stride={frame_stride} at {1.0 / self.dt:.1f} Hz."
        )
        current_idx = 0
        while True:
            frame = self.amp_loader.get_full_frame_at_time(motion_idx, current_idx)
            # 0       3          7         21          27            30            33        47
            # root_pos, root_quat, joint_pos,  foot_pos, root_lin_vel, root_ang_vel, joint_vel
            root_pos = AMPLoader.get_root_pos(frame)
            # Motion files store root quaternion as wxyz, whereas Isaac Gym
            # actor root states use xyzw.
            quat_wxyz = AMPLoader.get_root_rot(frame)
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            self.dof_pos[:, 0:self.num_actions] = AMPLoader.get_joint_pose(frame)
            self.dof_vel[:, 0:self.num_actions] = AMPLoader.get_joint_vel(frame)

            # Isaac Gym root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            self.root_states[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
            self.root_states[:, 3:7] = torch.tile(quat_xyzw.unsqueeze(0), (self.num_envs, 1))
            # Root velocities in the AMP file are body-frame values, while
            # Isaac Gym root states require world-frame values. They are not
            # needed for kinematic motion replay, so keep them at zero.
            self.root_states[:, 7:13] = 0.0

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids), len(env_ids))

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids), len(env_ids))
            self.render()
            self.gym.simulate(self.sim)

            # self.gym.refresh_actor_root_state_tensor(self.sim)
            # self.gym.refresh_net_contact_force_tensor(self.sim)
            # self.gym.refresh_rigid_body_state_tensor(self.sim)

            time.sleep(self.dt)
            current_idx += frame_stride
            if current_idx >= data_len:
                current_idx = 0

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        joint_pos = self.dof_pos.clone()
        joint_vel = self.dof_vel.clone()
        base_lin_vel_b = self.base_lin_vel.clone()
        base_ang_vel_b = self.base_ang_vel.clone()
        body_pos_b, body_ori_b = self.amp_robot_body_pos_ori_b()

        amp_obs = torch.cat((
            joint_pos,
            joint_vel,
            base_lin_vel_b,
            base_ang_vel_b,
            body_pos_b,
            body_ori_b,
            ), dim=-1)
        return amp_obs

    def amp_robot_body_pos_ori_b(self):
        root_pos_w = self.base_pos.clone()  # (num_envs, 3)
        root_quat_w = self.base_quat.clone()  # (num_envs, 4)
        body_pos_w = self.rigid_body_pos[:, self.amp_body_indices].clone()  # (num_envs, num_bodies, 3)
        body_quat_w = self.rigid_body_quat[:, self.amp_body_indices].clone()  # (num_envs, num_bodies, 4)

        root_quat_w = root_quat_w[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        body_quat_w = body_quat_w[:, :, [3, 0, 1, 2]]  # xyzw -> wxyz

        num_bodies = body_pos_w.shape[1]

        pos_b, ori_b = subtract_frame_transforms(
            root_pos_w[:, None, :].expand(-1, num_bodies, -1),
            root_quat_w[:, None, :].expand(-1, num_bodies, -1),
            body_pos_w,
            body_quat_w,
        )
        mat = matrix_from_quat(ori_b)
        return pos_b.reshape(self.num_envs, -1), mat[..., :2].reshape(mat.shape[0], -1)

    # ============================================================ Rewards ===========================================================================
    # ============================================================ Rewards reference motion tracking==================================================
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_height - self._get_ankle_heights()
        pos_error = torch.square(feet_z - self.cfg.rewards.target_feet_height) * ~contact
        pos_error[self.stand_command] = 0.
        return torch.sum(pos_error, dim=1)

    def _reward_feet_swing_under_target(self):
        feet_height = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_height - self._get_ankle_heights()
        height_err = feet_height - self.cfg.rewards.target_feet_height
        mask = height_err <= 0  # 只保留低于目标高度的脚
        reward = torch.sum(torch.square(height_err) * ~self.contact * mask, dim=1)
        reward[self.stand_command] = 0.
        return reward

    def _reward_feet_single_contact_time(self):
        """足部接触奖励"""
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 单足接触
        single_feet_contact = torch.logical_xor(self.contact_filt[:, 0], self.contact_filt[:, 1])
        # 双足接触
        both_feet_contact = torch.logical_and(self.contact_filt[:, 0], self.contact_filt[:, 1])

        # 更新双足接触时间
        self.feet_both_contact_time[both_feet_contact] += self.dt
        self.feet_both_contact_time *= both_feet_contact

        # 奖励条件：单足接触或双足接触时间小于0.2秒，或站立命令
        rew_filter = torch.logical_or(single_feet_contact, self.feet_both_contact_time < 0.2)
        rew_filter = torch.logical_or(rew_filter, self.stand_command)
        rew[rew_filter] = 1.0
        return rew

    def _reward_feet_gait_contact(self):
        stance_mask = self._get_gait_phase().bool()
        contact = self.contact_forces[:, self.feet_indices, 2] > 20.0
        # reward: contact == stance
        reward = (contact == stance_mask).sum(dim=1).to(dtype=torch.float)
        # stand command override
        reward[self.stand_command] = 2.0
        return reward

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - self.cfg.rewards.feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~self.contact_filt
        rew_airTime[self.stand_command] = 0.
        return rew_airTime

    def _reward_feet_air_time_positive(self):
        air_time = self.current_air_time
        contact_time = self.current_contact_time
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        reward = torch.clamp(reward, max=self.cfg.rewards.feet_air_time)
        # no reward for zero command
        reward[self.stand_command] = 0.
        return reward

    def _reward_feet_gait_stance(self):
        """
        Calculates reward for each foot during the stance phase of the gait.
        Fully vectorized for speed.
        """
        # 支撑相掩码 (num_envs, 2)
        stance_mask = self._get_gait_phase()
        # 获取脚在 Z 方向的速度分量 (num_envs, 2, 2)
        # 取 feet_indices 对应的脚, 第7:9维作为 velocity_x, velocity_y
        foot_velocities = self.rigid_state[:, self.feet_indices, 7:9]
        # foot_velocities = self.rigid_state[:, self.feet_indices, 7:10]
        # 计算速度大小并归一化
        foot_vel_mag = torch.norm(foot_velocities, dim=-1) / 5  # [num_envs, 2]
        # 奖励计算：1 - exp(-|v|)
        reward_vel = (1 - torch.exp(-foot_vel_mag)) * stance_mask  # [num_envs, 2]
        # 总奖励
        total_reward = reward_vel.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        return total_reward

    def _reward_feet_gait_swing(self):
        """
        Calculates reward for the swing phase of the gait.
        Fully vectorized for speed.
        """
        # 获取摆动相掩码 (num_envs, 2)
        swing_mask = 1 - self._get_gait_phase()
        # 获取左右脚接触力 (num_envs, 2, 3)
        foot_contact_forces = self.contact_forces[:, self.feet_indices, 0:3]
        # 计算每只脚接触力的范数并归一化
        contact_force_mag = torch.norm(foot_contact_forces, dim=-1) / 50  # [num_envs, 2]
        # 奖励计算：1 - exp(-|force|) 并乘以摆动相掩码
        reward_force = (1 - torch.exp(-contact_force_mag)) * swing_mask  # [num_envs, 2]
        # 总奖励：左右脚相加
        total_reward = reward_force.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        return total_reward

    def _reward_feet_gait_stance_mask(self):
        """
        Calculates reward for each foot during the stance phase of the gait.
        Fully vectorized for speed.
        """
        # 支撑相掩码 (num_envs, 2)
        stance_mask = self._get_gait_phase()
        # 获取脚在 Z 方向的速度分量 (num_envs, 2, 2)
        # 取 feet_indices 对应的脚, 第7:9维作为 velocity_x, velocity_y
        foot_velocities = self.rigid_state[:, self.feet_indices, 7:9]
        # foot_velocities = self.rigid_state[:, self.feet_indices, 7:10]
        # 计算速度大小并归一化
        foot_vel_mag = torch.norm(foot_velocities, dim=-1) / 5  # [num_envs, 2]
        # 奖励计算：1 - exp(-|v|)
        reward_vel = (1 - torch.exp(-foot_vel_mag)) * stance_mask  # [num_envs, 2]
        # 总奖励
        total_reward = reward_vel.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        vel_mask = torch.abs(self.commands[:, 0]) > 0.3
        total_reward[vel_mask] *= self.cfg.rewards.feet_gait_stance_mask
        return total_reward

    def _reward_feet_gait_swing_mask(self):
        """
        Calculates reward for the swing phase of the gait.
        Fully vectorized for speed.
        """
        # 获取摆动相掩码 (num_envs, 2)
        swing_mask = 1 - self._get_gait_phase()
        # 获取左右脚接触力 (num_envs, 2, 3)
        foot_contact_forces = self.contact_forces[:, self.feet_indices, 0:3]
        # 计算每只脚接触力的范数并归一化
        contact_force_mag = torch.norm(foot_contact_forces, dim=-1) / 50  # [num_envs, 2]
        # 奖励计算：1 - exp(-|force|) 并乘以摆动相掩码
        reward_force = (1 - torch.exp(-contact_force_mag)) * swing_mask  # [num_envs, 2]
        # 总奖励：左右脚相加
        total_reward = reward_force.sum(dim=1)
        total_reward[self.stand_command] = 0.0
        vel_mask = torch.abs(self.commands[:, 0]) > 0.3
        total_reward[vel_mask] *= self.cfg.rewards.feet_gait_swing_mask
        return total_reward

    def _reward_feet_slide(self):
        # scale: -0.25
        contact = self.feet_forces_history.norm(dim=-1).max(dim=1)[0] > 1.
        feet_vel = self.rigid_body_vel[:, self.feet_indices, :2]
        reward = torch.sum(feet_vel.norm(dim=-1) * contact, dim=1)
        return reward

    def _reward_feet_slide_xyz(self):
        # scale: -0.25
        contact = self.feet_forces_history.norm(dim=-1).max(dim=1)[0] > 1.
        feet_vel = self.rigid_body_vel[:, self.feet_indices, :3]
        reward = torch.sum(feet_vel.norm(dim=-1) * contact, dim=1)
        return reward

    def _reward_feet_y_distance(self):
        """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
        left_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[0]] - self.base_pos)
        right_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[1]] - self.base_pos)
        y_distance_b = torch.abs(left_foot_b[:, 1] - right_foot_b[:, 1])
        rew = torch.clip(self.cfg.rewards.close_feet_threshold - y_distance_b, min=0, max=1)
        return rew

    def _reward_feet_y_distance_l1(self):
        """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
        left_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[0]] - self.base_pos)
        right_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[1]] - self.base_pos)
        y_distance_b = torch.abs(left_foot_b[:, 1] - right_foot_b[:, 1])
        rew = torch.abs(self.cfg.rewards.close_feet_threshold - y_distance_b)

        y_vel_flag = (torch.abs(self.commands[:, 1]) < 0.3)
        rew *= y_vel_flag
        return rew

    def _reward_feet_y_distance_exp(self):
        """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
        left_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[0]] - self.base_pos)
        right_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[1]] - self.base_pos)
        y_distance = torch.abs(left_foot_b[:, 1] - right_foot_b[:, 1])
        y_distance_error = torch.abs(y_distance - self.cfg.rewards.close_feet_threshold)
        rew = torch.exp(-10 * y_distance_error)

        y_vel_flag = (torch.abs(self.commands[:, 1]) < 0.3)
        rew *= y_vel_flag
        return rew

    def _reward_feet_x_distance(self):
        """
        Penalize foot x-distance when commanded x-velocity is low to maintain reasonable spacing.
        Quat rotation is done separately for left and right foot.
        """
        left_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[0]] - self.base_pos)
        right_foot_b = quat_rotate_inverse(self.base_quat, self.rigid_body_pos[:, self.feet_indices[1]] - self.base_pos)
        x_distance_b = torch.abs(left_foot_b[:, 0]) + torch.abs(right_foot_b[:, 0])
        vel_flag = torch.abs(self.commands[:, 0]) < 0.2
        rew = torch.clip(x_distance_b, 0, 0.2) * vel_flag
        return rew

    def _reward_feet_step_limit(self):
        """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
        x_step_max = self.cfg.rewards.feet_step_limit[0]
        y_step_max = self.cfg.rewards.feet_step_limit[1]
        z_step_max = self.cfg.rewards.feet_step_limit[2]
        leftfoot = self.rigid_state[:, self.feet_indices[0], 0:3] - self.root_states[:, 0:3]
        rightfoot = self.rigid_state[:, self.feet_indices[1], 0:3] - self.root_states[:, 0:3]
        leftfoot_b = quat_rotate_inverse(self.base_quat, leftfoot)
        rightfoot_b = quat_rotate_inverse(self.base_quat, rightfoot)
        x_distance_b = torch.abs(leftfoot_b[:, 0] - rightfoot_b[:, 0]) - x_step_max
        y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1]) - y_step_max
        z_distance_b = torch.abs(leftfoot_b[:, 2] - rightfoot_b[:, 2]) - z_step_max
        rew = torch.clip(x_distance_b, min=0, max=1)
        rew += torch.clip(y_distance_b, min=0, max=1)
        rew += torch.clip(z_distance_b, min=0, max=1)
        # print(torch.clip(z_distance_b[self.lookat_id], min=0, max=1))
        return rew

    def _reward_feet_too_near(self):
        # 惩罚小于close_feet_threshold
        feet_pos = self.rigid_body_pos[:, self.feet_indices, :]
        distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
        return (self.cfg.rewards.close_feet_threshold - distance).clamp(min=0)

    def _reward_no_fly(self):
        none_contact = torch.sum(1. * self.contact, dim=1) == 0
        single_contact = torch.sum(1. * self.contact, dim=1) == 1
        double_contact = torch.sum(1. * self.contact, dim=1) == 2
        single_contact[self.stand_command] = 1
        return 1.0 * single_contact.float() + 0.3 * double_contact.float() - 1. * none_contact.float()

    def _reward_stand_still(self):
        """静止站立惩罚"""
        # 对于站立命令，惩罚关节位置偏差和关节速度
        rew = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        rew += torch.sum(torch.square(self.dof_vel), dim=1)
        rew[~self.stand_command] = 0.
        return rew

    # ==================================== vel tracking  =========================================== #
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the x and y axes
        with separate sigmas.
        """
        err_x = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        err_y = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])

        return torch.exp(
            - err_x * self.cfg.rewards.tracking_sigma_x
            - err_y * self.cfg.rewards.tracking_sigma_y
        )

    def _reward_tracking_lin_vel_l2(self):
        lin_vel_error = torch.sum(torch.square(
            self.commands[:,0:2] - self.base_lin_vel[:, 0:2]), dim=1)
        return lin_vel_error

    def _reward_tracking_stuck(self):

        return ((self.commands[:, 0] > 0.3).float() *
                ((self.base_lin_vel[:, 0] < 0.15).float() +
                (self.base_lin_vel[:, 0] < 0.0).float() +
                (self.base_lin_vel[:, 0] < -0.15).float()))

    def _reward_tracking_dont_wait(self):
        rew = ((self.commands[:, 0] > 0.30).float() *
               ((self.base_lin_vel[:, 0] < 0.30).float() +
                (self.base_lin_vel[:, 0] < 0.15).float() +
                (self.base_lin_vel[:, 0] < 0.00).float()))
        # rew = (self.commands[:, 0] > 0.30).float() * (self.base_lin_vel[:, 0] < 0.30).float()
        # rew += (self.commands[:, 0] < -0.30).float() * (self.base_lin_vel[:, 0] > -0.30).float()
        return rew

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_z)

    def _reward_base_ang_vel_xy_exp(self):
        ang_vel_xy_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

        return torch.exp(-ang_vel_xy_error * self.cfg.rewards.base_ang_vel_xy_sigma)

    def _reward_tracking_ang_vel_z_l2(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return ang_vel_error

    # ==================================== base pos  =========================================== #
    def _reward_base_height_exp(self):
        # 1.0
        # Penalize base height away from target
        terrain_height = torch.mean(self.base_height_maps, dim=-1)
        base_height = self.root_states[:, 2] - terrain_height
        base_height_error = (base_height - self.cfg.rewards.base_height_target)
        reward = torch.exp(-10 * torch.abs(base_height_error))
        return reward

    def _reward_base_height_feet_exp(self):
        # 1.0
        feet_height = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2],  # left_height
            self.rigid_state[:, self.feet_indices[1], 2]  # right_height
        )
        base_height = self.root_states[:, 2] - feet_height + self.cfg.rewards.feet_height
        base_height_error = (base_height - self.cfg.rewards.base_height_target)
        reward = torch.exp(-10 * torch.abs(base_height_error))
        return reward

    def _reward_base_height_l1(self):
        # Penalize base height away from target
        terrain_height = torch.mean(self.base_height_maps, dim=-1)
        base_height = self.root_states[:, 2] - terrain_height
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)
        return reward

    def _reward_base_feet_height_l1(self):
        feet_height = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2],  # left_height
            self.rigid_state[:, self.feet_indices[1], 2]  # right_height
        )
        base_height = self.root_states[:, 2] - feet_height + self.cfg.rewards.feet_height
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)
        return reward

    def _reward_base_feet_height_l2(self):
        feet_height = torch.minimum(
            self.rigid_state[:, self.feet_indices[0], 2],  # left_height
            self.rigid_state[:, self.feet_indices[1], 2]  # right_height
        )
        base_height = self.root_states[:, 2] - feet_height + self.cfg.rewards.feet_height
        reward = torch.square(base_height - self.cfg.rewards.base_height_target)
        return reward

    def _reward_base_gravity_exp(self):
        # 1.0
        ori = torch.norm(self.projected_gravity[:, :2], dim=1)
        reward = torch.exp(-10 * ori)
        reward[self.stand_command] *= 0
        return reward

    def _reward_base_gravity_exp_mask(self):
        # 1.0
        mask = torch.abs(self.commands[:,0]) < 0.2
        ori = torch.norm(self.projected_gravity[:, :2], dim=1)
        reward = torch.exp(-10 * ori)
        reward[self.stand_command] *= 0
        reward[mask] *= 0
        return reward

    def _reward_base_gravity(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew

    def _reward_base_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return rew

    def _reward_base_ang_vel_xy_mask(self):
        vel_mask = self.commands[:, 0].abs() > 0.3
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew[vel_mask] *= self.cfg.rewards.base_ang_vel_xy_mask
        return rew

    def _reward_base_ang_vel_xy_stand_mask(self):
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew[self.stand_command] *= self.cfg.rewards.base_ang_vel_xy_stand_mask
        return rew

    def _reward_base_ang_vel_z(self):
        mask = torch.abs(self.commands[:, 2]) < 0.1
        rew =  torch.square(self.base_ang_vel[:, 2]) * mask
        return rew

    def _reward_base_acc_l2(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        reward = torch.sum(torch.square(root_acc), dim=1)
        return reward


    def _reward_base_acc_exp(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_hip_roll_action(self):
        """Penalize hip roll joint actions."""
        y_vel_flag = torch.abs(self.commands[:, 1]) < 0.1
        return torch.sum(torch.abs(self.actions[:, [1, 7]]), dim=1) * y_vel_flag

    def _reward_hip_yaw_action(self):
        """Penalize hip yaw joint actions."""
        yaw_vel_flag = torch.abs(self.commands[:, 2]) < 0.1
        return torch.sum(torch.abs(self.actions[:, [2, 8]]), dim=1) * yaw_vel_flag

    def _reward_hip_action_l2(self):
        """Penalize hip roll joint actions."""
        y_vel_flag = (torch.abs(self.commands[:, 1]) < 0.1) & (torch.abs(self.base_lin_vel[:, 1]) < 0.2)
        yaw_vel_flag = torch.abs(self.commands[:, 2]) < 0.1
        rew = torch.sum(torch.square(self.actions[:, [1, 7]]) * ~self.contact, dim=1) * y_vel_flag
        rew += torch.sum(torch.square(self.actions[:, [2, 8]]) * ~self.contact, dim=1) * yaw_vel_flag
        return rew

    def _reward_ankle_pitch_action(self):
        return torch.sum(torch.abs((self.actions[:, [4, 10]])), dim=1)

    def _reward_yaw_pos(self):
        yaw_pos = self.dof_pos[:, [2, 8]].clamp(max=0)
        reward = yaw_pos.norm(dim=1)
        return reward

    def _reward_hip_pos_exp(self):
        # 判断是否有显著侧向或转向速度命令
        move_y = (self.commands[:, 1].abs() > 0.1)
        move_yaw = (self.commands[:, 2].abs() > 0.1)
        # 取出髋关节 yaw 和 roll 的偏差
        joint_diff = self.dof_pos - self.default_dof_pos
        hip_yaw_err = joint_diff[:, [2, 8]].norm(dim=1)
        hip_roll_err = joint_diff[:, [1, 7]].norm(dim=1)
        # 对应动作时放宽惩罚
        hip_yaw_err *= torch.where(move_yaw, 0.1, 1.0)
        hip_roll_err *= torch.where(move_y, 0.1, 1.0)
        # 合并并生成奖励
        error = (hip_yaw_err + hip_roll_err).clamp(max=10)
        return torch.exp(-10 * error)

    def _reward_hip_pos(self):
        return torch.norm(self.dof_pos[:, [1, 2, 7, 8]], dim=1)

    def _reward_hip_pos_l1(self):
        return torch.sum(torch.abs(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_hip_pos_l2(self):
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_hip_roll_exp(self):
        move_y = (self.commands[:, 1].abs() > 0.2)
        joint_diff = self.dof_pos - self.default_dof_pos
        hip_roll_err = joint_diff[:, [1, 7]].norm(dim=1)
        hip_roll_err *= torch.where(move_y, 0.01, 1.0)
        # 合并并生成奖励
        error = hip_roll_err.clamp(max=10)
        return torch.exp(-5 * error)

    def _reward_hip_yaw_exp(self):
        move_yaw = (self.commands[:, 2].abs() > 0.2)
        joint_diff = self.dof_pos - self.default_dof_pos
        hip_yaw_err = joint_diff[:, [2, 8]].norm(dim=1)
        hip_yaw_err *= torch.where(move_yaw, 0.01, 1.0)
        error = hip_yaw_err.clamp(max=10)
        return torch.exp(-5 * error)

    def _reward_hip_pos_l2_mask(self):
        """Penalize hip roll joint actions."""
        y_vel_flag = (torch.abs(self.commands[:, 1]) < 0.2)
        yaw_vel_flag = (torch.abs(self.commands[:, 2]) < 0.2)
        rew_y = torch.sum(torch.square(self.dof_pos[:, [1, 7]]), dim=1) * y_vel_flag
        rew_yaw = torch.sum(torch.square(self.dof_pos[:, [2, 8]]), dim=1) * yaw_vel_flag
        return rew_y + rew_yaw

    def _reward_hip_yaw_pos_mask(self):
        """Penalize hip yaw joint actions."""
        yaw_vel_flag = (torch.abs(self.commands[:, 2]) < 0.3)
        rew_yaw = torch.sum(torch.abs(self.dof_pos[:, [2, 8]]), dim=1) * yaw_vel_flag
        return rew_yaw

    def _reward_hip_roll_pos_mask(self):
        """Penalize hip roll joint actions."""
        y_vel_flag = (torch.abs(self.commands[:, 1]) < 0.3)
        rew_y = torch.sum(torch.abs(self.dof_pos[:, [1, 7]]), dim=1) * y_vel_flag
        return rew_y

    def _reward_knee_pos_swing(self):
        # 摆动腿，前半周期，惩罚过小的膝关节弯曲，确保抬腿，拟人行走
        vel_mask = self.commands[:, 0] > 0.1
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (self.stand_radio <= self.phase_left) & (self.phase_left <= 0.5 + 0.5 * self.stand_radio)
        swing_mask[:, 1] = (self.stand_radio <= self.phase_right) & (self.phase_right <= 0.5 + 0.5 * self.stand_radio)
        target_pos = self.cfg.rewards.target_knee_swing_pos
        pos_error = (self.dof_pos[:, [3, 9]] - target_pos)
        pos_error = torch.clamp(pos_error, min=-0.2)  # 将误差限制在[-0.1, inf)
        reward = pos_error * ~self.contact * swing_mask
        reward[self.stand_command] = 0.
        return torch.sum(reward, dim=1) * vel_mask

    def _reward_knee_pos_swing_v1(self):
        # 摆动腿，前半周期，惩罚过小的膝关节弯曲，确保抬腿，拟人行走
        vel_mask = self.commands[:, 0] > 0.1
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (0.3 + 0.7*self.stand_radio <= self.phase_left) & (self.phase_left <= 0.7 + 0.3 * self.stand_radio)
        swing_mask[:, 1] = (0.3 + 0.7*self.stand_radio <= self.phase_right) & (self.phase_right <= 0.7 + 0.3 * self.stand_radio)
        target_pos = self.cfg.rewards.target_knee_swing_pos
        pos_error = (self.dof_pos[:, [3, 9]] - target_pos)
        pos_error = torch.clamp(pos_error, min=-0.2)  # 将误差限制在[-0.1, inf)
        reward = pos_error * ~self.contact * swing_mask
        reward[self.stand_command] = 0.
        return torch.sum(reward, dim=1) * vel_mask

    def _reward_knee_pos_swing_v2(self):
        # 摆动腿，前半周期，惩罚过小的膝关节弯曲，确保抬腿，拟人行走
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (0.3 + 0.7*self.stand_radio <= self.phase_left) & (self.phase_left <= 0.7 + 0.3 * self.stand_radio)
        swing_mask[:, 1] = (0.3 + 0.7*self.stand_radio <= self.phase_right) & (self.phase_right <= 0.7 + 0.3 * self.stand_radio)
        target_pos = self.cfg.rewards.target_knee_swing_pos
        pos_error = (self.dof_pos[:, [3, 9]] - target_pos)
        pos_error = torch.clamp(pos_error, min=-0.2)  # 将误差限制在[-0.1, inf)
        reward = pos_error * ~self.contact * swing_mask
        reward[self.stand_command] = 0.
        return torch.sum(reward, dim=1)

    def _reward_knee_pos_stand(self):
        # 支撑腿，前半周期，确保膝盖打直，拟人行走
        # vel_mask = torch.abs(self.commands[:, 0]) > 0.5
        stand_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stand_mask[:, 0] = (0 <= self.phase_left) & (self.phase_left <= 0.2 * self.stand_radio)
        stand_mask[:, 1] = (0 <= self.phase_right) & (self.phase_right <=  0.2 * self.stand_radio)
        target_pos = -0.3
        pos_error = (target_pos - self.dof_pos[:, [3, 9]])
        pos_error = torch.clamp(pos_error, min=0)  # 将误差限制在[0, inf)
        reward = pos_error * self.contact * stand_mask
        return torch.sum(reward, dim=1)  # * (~self.stand_command)

    def _reward_ankle_pitch_pos(self):
        # 踝关节保持默认位置，脚尖和脚后跟触地，拟人行走
        vel_mask = torch.abs(self.commands[:,0]) > 0.3
        terrain_mask = self.plane_mask
        return torch.norm(self.dof_pos[:, [4, 10]] - self.default_dof_pos[:, [4, 10]], dim=1) * vel_mask * terrain_mask

    def _reward_ankle_pitch_pos_v1(self):
        # 踝关节保持默认位置，脚尖和脚后跟触地，拟人行走
        vel_mask = torch.abs(self.commands[:,0]) > 0.3
        terrain_mask = self.plane_mask | self.step_mask
        return torch.norm(self.dof_pos[:, [4, 10]] - self.default_dof_pos[:, [4, 10]], dim=1) * vel_mask  *  terrain_mask

    def _reward_ankle_pitch_pos_mask(self):
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (self.stand_radio <= self.phase_left) & (self.phase_left <= 0.5 + 0.5 * self.stand_radio)
        swing_mask[:, 1] = (self.stand_radio <= self.phase_right) & (self.phase_right <= 0.5 + 0.5 * self.stand_radio)
        # 踝关节保持默认位置，脚尖和脚后跟触地，拟人行走
        vel_mask = torch.abs(self.commands[:,0]) > 0.3
        return torch.norm((self.dof_pos[:, [4, 10]] - self.default_dof_pos[:, [4, 10]]) * swing_mask, dim=1) * vel_mask

    def _reward_feet_ori(self):
        terrain_mask = self.step_mask
        left_gravity = quat_rotate_inverse(self.left_feet_quat, self.gravity_vec)
        right_gravity = quat_rotate_inverse(self.right_feet_quat, self.gravity_vec)
        reward = torch.norm(left_gravity[:, 0:1], dim=1)
        reward += torch.norm(right_gravity[:, 0:1], dim=1)
        return reward * terrain_mask

    def _reward_feet_ori_first_contact(self):
        terrain_mask = self.plane_mask | self.step_mask
        left_gravity = quat_rotate_inverse(self.left_feet_quat, self.gravity_vec)
        right_gravity = quat_rotate_inverse(self.right_feet_quat, self.gravity_vec)
        reward = torch.norm(left_gravity[:, 0:1], dim=1) * self.first_contact[:, 0]
        reward += torch.norm(right_gravity[:, 0:1], dim=1) * self.first_contact[:, 1]
        return reward  * terrain_mask

    def _reward_feet_ori_mask(self):
        # 低速时，脚面水平
        vel_mask = torch.abs(self.commands[:,0]) < 0.3
        left_quat = self.rigid_state[:, self.feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_state[:, self.feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        reward = torch.sum(torch.square(left_gravity[:, 0:1]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, 0:1]), dim=1)**0.5
        return reward * vel_mask

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        return torch.norm(joint_diff, dim=1)

    # ==================================== energy  =========================================== #
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(0, ), dim=1)

    def _reward_feet_contact_forces_xy(self):
        threshold = self.cfg.rewards.max_contact_force_xy
        feet_force_xy = self.feet_forces[:, :, :2]
        # print(feet_force_xy[self.lookat_id])
        reward = torch.sum((torch.norm(feet_force_xy, dim=-1) - threshold).clip(0,), dim=1)
        return reward

    def _reward_feet_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.rigid_state[:, self.feet_indices, 7:10] * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_feet_contact_no_vel_max_hist(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_vel_max = torch.max(torch.abs(self.feet_vel_history), dim=1)[0]
        contact_feet_vel = feet_vel_max * contact.unsqueeze(-1)
        reward = torch.sum(torch.square(contact_feet_vel), dim=(1, 2))
        return reward

    def _reward_feet_acc(self):
        feet_acc = (self.last_rigid_body_vel[:, self.feet_indices] - self.rigid_body_vel[:, self.feet_indices]) / self.dt
        reward = torch.sum(torch.norm(feet_acc, dim=2), dim=1)
        return reward

    def _reward_feet_acc_z(self):
        feet_acc_z = (self.last_rigid_body_vel[:, self.feet_indices, 2] - self.rigid_body_vel[:, self.feet_indices, 2]) / self.dt
        feet_acc_z = (torch.abs(feet_acc_z) - self.cfg.rewards.max_feet_acc_z).clip(0, )
        reward = torch.sum(torch.abs(feet_acc_z), dim=-1)
        return reward

    def _reward_feet_vel_z(self):
        feet_vel_z = (torch.abs(self.rigid_body_vel[:, self.feet_indices, 2]) - self.cfg.rewards.max_feet_vel_z).clip(0,)
        reward = torch.sum(torch.abs(feet_vel_z), dim=1)
        return reward

    def _reward_feet_swing_vel(self):
        # 摆动腿，后半周期，脚向下的速度大小不高于0.2，触地轻柔
        # 创建摆动阶段的掩码（仅在后半周期奖励）
        swing_mask = torch.zeros((self.num_envs, 2), device=self.device)
        swing_mask[:, 0] = (0.5 + 0.5*self.stand_radio <= self.phase_left)
        swing_mask[:, 1] = (0.5 + 0.5*self.stand_radio <= self.phase_right)
        swing_feet_vel = self.rigid_state[:, self.feet_indices, 9] * swing_mask * (self.rigid_state[:, self.feet_indices, 9]<-0.2)
        return torch.sum(torch.square(swing_feet_vel), dim=1)

    def _reward_action_rate(self):
        return  torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        return torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    def _reward_action_hip_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        action_diff_1 = self.last_actions - self.actions
        action_diff_2 = self.actions + self.last_last_actions - 2. * self.last_actions
        term_1 = torch.sum(torch.square(action_diff_1[:, [1,2,7,8]]), dim=1)
        term_2 = torch.sum(torch.square(action_diff_2[:, [1,2,7,8]]), dim=1)
        return term_1 + term_2

    def _reward_dof_torque(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_torque_ankle(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques[:,[4, 10]]), dim=1)

    def _reward_dof_vel_knee(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.dof_vel[:,[3, 9]]), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_power(self):
        # scale -1e-5
        return torch.sum((torch.abs(self.dof_vel)*torch.abs(self.torques)), dim=1)

    def _reward_dof_power_distribution(self):
        power = torch.abs(self.torques * self.dof_vel)
        return torch.var(power, dim=1)

    def _reward_dof_energy(self):
        # scale -1e-3
        return torch.norm((torch.abs(self.dof_vel) * torch.abs(self.torques)), dim=-1)

    # ==================================== safety  =========================================== #
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.q_torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.),dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        # print(self.contact_forces[self.lookat_id, self.feet_indices, 0])
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_hold(self):
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 左脚和右脚的地形高度
        left_height_maps = torch.clip(self.left_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.left_feet_hold_maps, -1.0, 1.0)
        right_height_maps = torch.clip(self.right_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.right_feet_hold_maps, -1.0, 1.0)
        # 超过 threshold 认为地形不平整
        threshold = 0.05
        left_exceed_mask = (left_height_maps > threshold).float()
        right_exceed_mask = (right_height_maps > threshold).float()
        # 计算每只脚的超阈值比例
        left_exceed_ratio = left_exceed_mask.mean(dim=1) * self.contact[:, 0]
        right_exceed_ratio = right_exceed_mask.mean(dim=1) * self.contact[:, 1]
        reward = (left_exceed_ratio>0.4).float() + (right_exceed_ratio>0.4).float()
        # print((left_exceed_ratio > 0.3).float(),(right_exceed_ratio > 0.3).float())
        # if not self.headless:
        #     print("_reward_feet_hold:",
        #           f"{left_exceed_ratio[self.lookat_id].cpu().numpy():.4f},",
        #           f"{right_exceed_ratio[self.lookat_id].cpu().numpy():.4f}"
        #           )
        #         # -- vel mask
        forward_mask = self.commands[:, 0] > 0.3
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_hold_v1(self):
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 只在第一次接触时更新左脚和右脚的地形高度
        left_height_maps = torch.clip(self.left_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.left_feet_hold_maps, -1.0, 1.0)
        right_height_maps = torch.clip(self.right_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.right_feet_hold_maps, -1.0, 1.0)
        self.left_feet_hold_maps_first_contact = torch.where(self.first_contact[:, 0].unsqueeze(1), left_height_maps, self.left_feet_hold_maps_first_contact)
        self.right_feet_hold_maps_first_contact = torch.where(self.first_contact[:, 1].unsqueeze(1), right_height_maps, self.right_feet_hold_maps_first_contact)
        # 超过 threshold 认为地形不平整
        threshold = 0.05
        left_exceed_mask = (self.left_feet_hold_maps_first_contact > threshold).float()
        right_exceed_mask = (self.right_feet_hold_maps_first_contact > threshold).float()
        # 计算每只脚的超阈值比例
        left_exceed_ratio = left_exceed_mask.mean(dim=1)
        right_exceed_ratio = right_exceed_mask.mean(dim=1)
        reward = (left_exceed_ratio>0.4).float() + (right_exceed_ratio>0.4).float()
        # if not self.headless:
        #     print("_reward_feet_hold:",
        #           f"{left_exceed_ratio[self.lookat_id].cpu().numpy():.4f},",
        #           f"{right_exceed_ratio[self.lookat_id].cpu().numpy():.4f}"
        #           )
        #         # -- vel mask
        forward_mask = self.commands[:, 0] > 0.2
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_hold_v2(self):
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 只在第一次接触时更新左脚和右脚的地形高度
        left_height_maps = torch.clip(self.left_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.left_feet_hold_maps, -1.0, 1.0)
        right_height_maps = torch.clip(self.right_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.right_feet_hold_maps, -1.0, 1.0)
        self.left_feet_hold_maps_first_contact = torch.where(self.first_contact[:, 0].unsqueeze(1), left_height_maps, self.left_feet_hold_maps_first_contact)
        self.right_feet_hold_maps_first_contact = torch.where(self.first_contact[:, 1].unsqueeze(1), right_height_maps, self.right_feet_hold_maps_first_contact)
        # 超过 threshold 认为地形不平整
        threshold = 0.08
        left_exceed_mask = (self.left_feet_hold_maps_first_contact > threshold).float()
        right_exceed_mask = (self.right_feet_hold_maps_first_contact > threshold).float()
        # 计算每只脚的超阈值比例
        left_exceed_ratio = left_exceed_mask.mean(dim=1)
        right_exceed_ratio = right_exceed_mask.mean(dim=1)
        reward = torch.where(left_exceed_ratio < 0.35, 0, left_exceed_ratio)
        reward += torch.where(right_exceed_ratio < 0.35, 0, right_exceed_ratio)
        # if not self.headless:
        #     print("_reward_feet_hold:",
        #           f"{left_exceed_ratio[self.lookat_id].cpu().numpy():.4f},",
        #           f"{right_exceed_ratio[self.lookat_id].cpu().numpy():.4f}"
        #           )
        forward_mask = self.commands[:, 0] > 0.2
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_hold_first_contact(self):
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 左脚和右脚的地形高度
        left_height_maps = torch.clip(self.left_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.left_feet_hold_maps, -1.0, 1.0)
        right_height_maps = torch.clip(self.right_feet_hold_maps.max(dim=1).values.unsqueeze(1) - self.right_feet_hold_maps, -1.0, 1.0)
        # 超过 threshold 认为地形不平整
        threshold = 0.08
        left_exceed_mask = (left_height_maps > threshold).float()
        right_exceed_mask = (right_height_maps > threshold).float()
        # 计算每只脚的超阈值比例
        left_exceed_ratio = left_exceed_mask.mean(dim=1) * self.first_contact[:, 0]
        right_exceed_ratio = right_exceed_mask.mean(dim=1) * self.first_contact[:, 1]

        reward = (left_exceed_ratio > 0.4).float() + (right_exceed_ratio > 0.4).float()
        forward_mask = self.commands[:, 0] > 0.3
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_dist_safety(self):
        """
        奖励函数：脚部与前后地形距离的安全性

        逻辑说明：
        - 若脚前方或后方距离小于安全阈值（可能撞到障碍），则给予惩罚。
        - 仅在脚部与地面接触时计算该奖励（避免悬空脚干扰）。
        - 最终乘以 terrain_mask，确保仅在有效地形区域生效。
        """
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 前后方向的安全距离阈值
        safe_front_dist = self.cfg.terrain.feet_front_threshold
        safe_back_dist = self.cfg.terrain.feet_back_threshold
        # 计算前后方距离小于阈值的接触脚数量（触地状态下）
        reward_front = torch.sum((self.feet_front_dist_ray < safe_front_dist) * self.contact, dim=1)
        reward_back = torch.sum((self.feet_back_dist_ray < safe_back_dist) * self.contact, dim=1)
        # print(reward_front[self.lookat_id])
        # 汇总前后方向的惩罚
        reward = reward_front + reward_back*0
        # 仅在地形有效区域生效
        forward_mask = self.commands[:, 0] > 0.3
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_dist_safety_v1(self):
        """
        奖励函数：脚部与前后地形距离的安全性

        逻辑说明：
        - 若脚前方或后方距离小于安全阈值（可能撞到障碍），则给予惩罚。
        - 仅在脚部与地面接触时计算该奖励（避免悬空脚干扰）。
        - 最终乘以 terrain_mask，确保仅在有效地形区域生效。
        """
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 前后方向的安全距离阈值
        safe_front_dist = self.cfg.terrain.feet_front_threshold
        safe_back_dist = self.cfg.terrain.feet_back_threshold
        # 计算前后方距离小于阈值的接触脚数量（触地状态下）
        reward_front = torch.sum((self.feet_front_dist_ray < safe_front_dist), dim=1)
        reward_back = torch.sum((self.feet_back_dist_ray < safe_back_dist), dim=1)
        # print(self.feet_front_dist_ray[self.lookat_id])
        # 汇总前后方向的惩罚
        reward = reward_front + reward_back*0
        # 仅在地形有效区域生效
        forward_mask = self.commands[:, 0] > 0.3
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_dist_safety_first_contact(self):
        """
        奖励函数：脚部与前后地形距离的安全性

        逻辑说明：
        - 若脚前方或后方距离小于安全阈值（可能撞到障碍），则给予惩罚。
        - 仅在脚部与地面接触时计算该奖励（避免悬空脚干扰）。
        - 最终乘以 terrain_mask，确保仅在有效地形区域生效。
        """
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 前后方向的安全距离阈值
        safe_front_dist = self.cfg.terrain.feet_front_threshold
        safe_back_dist = self.cfg.terrain.feet_back_threshold
        # 计算前后方距离小于阈值的接触脚数量（触地状态下）
        reward_front = torch.sum((self.feet_front_dist_ray < safe_front_dist) * self.first_contact, dim=1)
        reward_back = torch.sum((self.feet_back_dist_ray < safe_back_dist) * self.first_contact, dim=1)
        # print(reward_front[self.lookat_id])
        # 汇总前后方向的惩罚
        reward = reward_front + reward_back*0
        # 仅在地形有效区域生效
        forward_mask = self.commands[:, 0] > 0.1
        return reward * self.step_mask * forward_mask * enable

    def _reward_feet_dist_safety_continuous(self):
        enable = self.common_step_counter > self.cfg.rewards.feet_safety_enable_step
        # 前后方向的安全距离阈值
        safe_front_dist = self.cfg.terrain.feet_front_threshold
        # 计算前后方距离小于阈值的接触脚数量（触地状态下）
        reward_front = torch.sum((safe_front_dist - self.feet_front_dist_ray).clip(min=0) * self.contact, dim=1)
        # 汇总前后方向的惩罚
        reward = reward_front
        # 仅在地形有效区域生效
        forward_mask = self.commands[:, 0] > 0.3
        return reward * self.step_mask * forward_mask * enable

    def _build_foothold_candidates(self, feet_forward_points_w):
        """
        从某只脚前方的地形采样点云中，提取固定数量的平坦落脚候选点。

        输入:
            feet_forward_points_w: [num_envs, num_forward_points, 3]
                第 0/1/2 维分别是世界坐标 x/y/z，z 已经替换为地形高度。

        输出:
            candidates: [num_envs, num_foothold_candidates, 3]
                固定长度候选缓存。有效候选不足时，用最后一个有效候选补齐；完全没有有效候选时，
                使用搜索起点附近的 fallback 点补齐，保持后续计算不会出现未定义值。
            valid: [num_envs, num_foothold_candidates]
                True 表示该位置是真正由平坦窗口生成的候选；False 表示补齐或 fallback。

        算法时序:
            1. 起脚时已经冻结 feet_forward_points_w，候选搜索基于这份冻结点云。
            2. 按 |vx| * filter_time 过滤近处点，避免把脚刚离开的附近位置当成下一步落点。
            3. 从过滤后的搜索起点开始，用 0.25m 窗口、0.04m stride 滑动。
            4. 窗口内 max(z)-min(z) 小于阈值，则窗口中心点进入候选缓存。
            5. 每个 env 最多保存固定数量候选，便于 GPU 并行 reward。
        """
        cfg = self.cfg.terrain
        num_points = feet_forward_points_w.shape[1]
        num_candidates = self.num_foothold_candidates

        # 根据当前前向速度估计“太近”的距离。这里使用 abs(vx)，允许后续扩展倒走逻辑；
        # 当前 reward 仍会用 forward_mask 限制主要在前进台阶场景生效。
        filter_dist = torch.abs(self.commands[:, 0]) * cfg.foothold_filter_time
        filter_dist = torch.clamp(filter_dist, cfg.foothold_filter_min, cfg.foothold_filter_max)
        filter_indices = torch.floor(filter_dist / cfg.foothold_point_spacing).to(dtype=torch.long)
        filter_indices = torch.clamp(filter_indices, 0, num_points - 1)

        # 保存给 debug draw 使用：绿色前方点云从该 index 之后开始画。
        self.filter_indices = filter_indices.clone()

        window_pts = max(int(cfg.foothold_window_size / cfg.foothold_point_spacing), 1)
        stride_pts = max(int(cfg.foothold_window_stride / cfg.foothold_point_spacing), 1)
        max_steps = max((num_points - window_pts) // stride_pts + 1, 1)

        # sample_indices: [num_envs, max_steps, window_pts]
        # 每个环境的窗口起点不同，因为 filter_indices 由该环境的速度指令决定。
        steps = torch.arange(max_steps, device=self.device).view(1, -1) * stride_pts
        window_starts = filter_indices.view(-1, 1) + steps
        offsets = torch.arange(window_pts, device=self.device).view(1, 1, -1)
        sample_indices = window_starts.unsqueeze(-1) + offsets
        valid_window = sample_indices < num_points
        sample_indices = torch.clamp(sample_indices, 0, num_points - 1)

        # 取出每个滑窗内的高度，计算窗口平坦度。
        heights = feet_forward_points_w[:, :, 2]
        expanded_heights = heights.unsqueeze(1).expand(-1, max_steps, -1)
        window_heights = torch.gather(expanded_heights, dim=2, index=sample_indices)
        roughness = window_heights.max(dim=-1).values - window_heights.min(dim=-1).values
        is_flat = (roughness < cfg.foothold_flatness_threshold) & valid_window.all(dim=-1)

        # 候选点取窗口中心。center_indices: [num_envs, max_steps]
        center_indices = torch.clamp(window_starts + window_pts // 2, 0, num_points - 1)
        center_indices_3d = center_indices.unsqueeze(-1).expand(-1, -1, 3)
        window_centers = torch.gather(feet_forward_points_w, dim=1, index=center_indices_3d)

        # 对每个 env 的平坦窗口按从近到远排序，保留前 num_candidates 个。
        # flat_rank 为第几个平坦窗口；非平坦窗口的 rank 值会被 selected_flat 屏蔽。
        flat_rank = torch.cumsum(is_flat.to(torch.long), dim=1) - 1
        selected_flat = is_flat & (flat_rank < num_candidates)

        candidates = torch.zeros(
            self.num_envs, num_candidates, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        valid = torch.zeros(
            self.num_envs, num_candidates,
            dtype=torch.bool, device=self.device, requires_grad=False
        )

        # 只把真正选中的平坦窗口写入缓存。这里不用 scatter 写所有窗口，
        # 是为了避免非平坦窗口把同一个 slot 的有效候选覆盖掉。
        selected_env_ids, selected_step_ids = selected_flat.nonzero(as_tuple=True)
        if selected_env_ids.numel() > 0:
            selected_candidate_ids = flat_rank[selected_env_ids, selected_step_ids]
            candidates[selected_env_ids, selected_candidate_ids] = window_centers[selected_env_ids, selected_step_ids]
            valid[selected_env_ids, selected_candidate_ids] = True

        # 用最后一个有效候选补齐固定长度缓存；如果没有任何有效候选，则 fallback 到搜索起点附近。
        valid_count = valid.sum(dim=1)
        has_valid = valid_count > 0
        last_valid_idx = torch.clamp(valid_count - 1, min=0).to(dtype=torch.long)
        last_valid_points = candidates[
            torch.arange(self.num_envs, device=self.device),
            last_valid_idx
        ]

        fallback_idx = torch.clamp(filter_indices + window_pts // 2, 0, num_points - 1)
        fallback_points = feet_forward_points_w[
            torch.arange(self.num_envs, device=self.device),
            fallback_idx
        ]
        fill_points = torch.where(has_valid.view(-1, 1), last_valid_points, fallback_points)

        candidates = torch.where(
            valid.unsqueeze(-1),
            candidates,
            fill_points.unsqueeze(1).expand(-1, num_candidates, -1)
        )

        return candidates, valid

    def _update_foothold_candidates_on_first_air(self):
        """
        在摆动腿第一次离地(first_air)时刷新该腿候选缓存。

        这样候选点代表“起脚瞬间看到的前方可踩区域”，随后整个摆动相保持不变；
        触地时再用真实落点和这组候选点计算最近距离。
        """
        left_candidates, left_valid = self._build_foothold_candidates(self.left_feet_forward_points_w)
        right_candidates, right_valid = self._build_foothold_candidates(self.right_feet_forward_points_w)

        left_air = self.first_air[:, 0].view(-1, 1, 1)
        right_air = self.first_air[:, 1].view(-1, 1, 1)
        self.left_foothold_candidates[:] = torch.where(left_air, left_candidates, self.left_foothold_candidates)
        self.right_foothold_candidates[:] = torch.where(right_air, right_candidates, self.right_foothold_candidates)

        left_air_valid = self.first_air[:, 0].view(-1, 1)
        right_air_valid = self.first_air[:, 1].view(-1, 1)
        self.left_foothold_candidate_valid[:] = torch.where(left_air_valid, left_valid, self.left_foothold_candidate_valid)
        self.right_foothold_candidate_valid[:] = torch.where(right_air_valid, right_valid, self.right_foothold_candidate_valid)

    def _update_foothold_landing_error_on_first_contact(self):
        """
        在脚第一次触地(first_contact)时，计算真实落点到最近有效候选点的 x-z 平面距离。

        当前速度指令主要沿 x 方向直线通过地形：
        - x 表示前进方向落点是否踩到候选区域附近；
        - z 表示落点高度是否接近候选平台高度；
        - y 暂时不纳入距离，避免侧向微小摆动干扰前向台阶通过任务。

        注意：left_feet_pos/right_feet_pos 已经在 _update_feet_pos() 中从踝关节向脚底偏移了 0.1m，
        因此这里直接使用它们和候选地形点比较，不再额外减脚底偏移。
        """
        large_dist = 1e6
        left_foot_xz = self.left_feet_pos[:, [0, 2]].unsqueeze(1)
        right_foot_xz = self.right_feet_pos[:, [0, 2]].unsqueeze(1)
        # print("xy:", left_foot_xz[self.lookat_id])
        left_dist = torch.norm(left_foot_xz - self.left_foothold_candidates[:, :, [0, 2]], dim=-1)
        right_dist = torch.norm(right_foot_xz - self.right_foothold_candidates[:, :, [0, 2]], dim=-1)

        left_dist = torch.where(self.left_foothold_candidate_valid, left_dist, torch.full_like(left_dist, large_dist))
        right_dist = torch.where(self.right_foothold_candidate_valid, right_dist, torch.full_like(right_dist, large_dist))

        left_min = left_dist.min(dim=1).values
        right_min = right_dist.min(dim=1).values

        # 2cm 死区：落点已经非常接近候选区域时视为命中，避免微小姿态/高度噪声产生持续惩罚。
        deadband = 0.01
        left_min = torch.where(left_min < deadband, torch.zeros_like(left_min), left_min)
        right_min = torch.where(right_min < deadband, torch.zeros_like(right_min), right_min)

        # 没有有效候选时不产生落脚惩罚，避免“无可踩点”的地形给策略错误梯度。
        left_has_valid = self.left_foothold_candidate_valid.any(dim=1)
        right_has_valid = self.right_foothold_candidate_valid.any(dim=1)
        left_min = torch.where(left_has_valid, left_min, torch.zeros_like(left_min))
        right_min = torch.where(right_has_valid, right_min, torch.zeros_like(right_min))

        self.left_foothold_landing_error[:] = torch.where(
            self.first_contact[:, 0],
            left_min,
            self.left_foothold_landing_error
        )
        self.right_foothold_landing_error[:] = torch.where(
            self.first_contact[:, 1],
            right_min,
            self.right_foothold_landing_error
        )

    def _draw_target_footholds_left(self, env_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 0))
        points = self.target_footholds_left[i].cpu().numpy()
        x = points[0]
        y = points[1]
        z = points[2]
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_target_footholds_right(self, env_idx):
        # ===================== feet_heights ==========================
        i = env_idx
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 0))
        points = self.target_footholds_right[i].cpu().numpy()
        x = points[0]
        y = points[1]
        z = points[2]
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _reward_feet_hold_placement(self):
        """
        落脚位置惩罚：脚第一次触地时，计算真实落点到最近“平坦候选点”的水平距离。

        使用方式：
        - 候选点在 first_air 时刷新，表示摆动开始时规划出的可踩区域集合。
        - 距离在 first_contact 时刷新，表示这一步真实踩到了候选集合附近多远。
        - 这里只返回距离作为 cost，建议 reward scale 使用负数。

        这样不强迫策略踩某一个固定目标点，而是允许它落在任意一个合适的平坦候选附近。
        """
        self._update_foothold_landing_error_on_first_contact()

        left_rew = self.left_foothold_landing_error * self.first_contact[:, 0]
        right_rew = self.right_foothold_landing_error * self.first_contact[:, 1]
        # print(self.first_contact[self.lookat_id])
        reward = left_rew + right_rew

        forward_mask = (self.commands[:, 0] > 0.2) & (torch.abs(self.commands[:, 3]) < 0.3)
        return reward * self.step_mask * forward_mask

    def _reward_feet_hold_placement_exp(self):
        """
        论文形式的落脚候选集合奖励：

            r_fh = sum_f I_td^f * exp(-d_xz^f * s_xz)

        其中：
        - I_td^f 是 touchdown 指示量，这里使用 first_contact；
        - d_xz^f 是真实脚底触地点到最近候选点的 x-z 平面距离；
        - s_xz 是距离尺度，配置为 cfg.rewards.feet_hold_placement_s_xz；
        - 候选集合只在 liftoff(first_air) 更新，摆动期间保持固定，避免候选点抖动。

        注意：
        - 如果某只脚当前没有有效候选，则该脚不产生 exp 奖励，避免 fallback 点带来虚假高奖励。
        - 该函数返回正奖励，配置中的 reward scale 使用正数，例如 2.0。
        """
        self._update_foothold_landing_error_on_first_contact()
        s_xz = max(self.cfg.rewards.feet_hold_placement_sigma_xz, 1e-6)
        left_reward = torch.exp(-self.left_foothold_landing_error * s_xz)   #* self.contact[:, 0]
        right_reward = torch.exp(-self.right_foothold_landing_error * s_xz) #* self.contact[:, 1]
        reward = left_reward + right_reward
        # print(left_reward[self.lookat_id])
        # forward_mask = (self.commands[:, 0] > 0.2) & (torch.abs(self.commands[:, 3]) < 0.3)
        forward_mask = (self.commands[:, 0] > 0.2) & (torch.norm(self.commands[:, 1:3], dim=-1) < 0.2)
        return reward * self.step_mask * forward_mask

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_x_symmetry(self):
        """
        约束左右腿行走的步长一致。
        策略：仅在“从单支撑切换到双支撑”的瞬间（即一只脚刚落地的时刻）
              记录当前两脚间距，并与上一次切换时的间距进行对比。
        """
        # 1. 获取接触状态
        # feet_indices[0]: 左脚, feet_indices[1]: 右脚
        left_contact = self.contact_filt[:, 0]
        right_contact = self.contact_filt[:, 1]
        # 2. 判断当前状态
        is_double_support = left_contact & right_contact
        is_single_support = (left_contact != right_contact)  # 异或：只有一个为True
        # 3. 检测“单 -> 双”的转换事件
        # 条件：当前是双支撑 AND 上一帧是单支撑
        transition_event = is_double_support & self.was_single_support
        # 4. 计算当前两脚在基座坐标系下的 X 轴距离
        left_foot_pos = self.rigid_body_pos[:, self.feet_indices[0]]
        right_foot_pos = self.rigid_body_pos[:, self.feet_indices[1]]
        # 转换到基座坐标系
        left_foot_rel = quat_rotate_inverse(self.base_quat, left_foot_pos - self.base_pos)
        right_foot_rel = quat_rotate_inverse(self.base_quat, right_foot_pos - self.base_pos)
        # 计算当前间距 (绝对值)
        current_dist = torch.abs(left_foot_rel[:, 0] - right_foot_rel[:, 0])

        # 5. 计算惩罚
        # 差值 = |当前距离 - 上次距离|
        diff = torch.abs(current_dist - self.last_x_dist)
        # 惩罚项：超过阈值的差值
        stride_diff_threshold = 0.01
        reward = torch.clamp(diff - stride_diff_threshold, min=0.0)
        reward[~transition_event] *= 0.0 # 只保留切换时的奖励。

        # if transition_event[self.lookat_id].cpu().numpy():
        #     print("==================================================================")
        #     print("transition_event:", transition_event[self.lookat_id].cpu().numpy())
        #     print("current_dist:", current_dist[self.lookat_id].cpu().numpy())
        #     print("last_x_dist:", self.last_x_dist[self.lookat_id].cpu().numpy())
        #     print("diff:", diff[self.lookat_id].cpu().numpy())

        # 6. 更新记忆状态
        self.last_x_dist[transition_event] = current_dist[transition_event]
        # 7. 更新上一帧状态 (用于下一帧判断)
        # 将当前的 is_single_support 赋值给 was_single_support
        self.was_single_support = is_single_support
        # 8. (可选) 速度过滤
        is_moving = (torch.abs(self.commands[:, 0]) > 0.3) * (torch.norm(self.commands[:, [0,2]],dim=-1) < 0.5)
        return reward * is_moving
