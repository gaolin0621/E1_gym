from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from tqdm import tqdm
from humanoid.algo.utils import mjlab_math as math_utils

class AMPLoader:
    def __init__(self,
                 motion_files: str,
                 body_names: Sequence[str],
                 anchor_name: str,
                 all_body_names: Sequence[str],
                 device: str = "cuda:0"):
        """Load AMP motion data.

        Args:
            motion_files: Path to a single .npz file or a directory of .npz files.
            body_names: Names of the target bodies to track.
            anchor_name: Name of the anchor (root) body.
            all_body_names: Ordered list of *all* body names in the model.
                The index of each name in this list must match the body
                dimension in the .npz arrays.
            device: Torch device.
        """

        # resolve name -> index
        all_names_list = list(all_body_names)
        self._body_indexes = [all_names_list.index(n) for n in body_names]
        self._anchor_indexes = all_names_list.index(anchor_name)
        self._num_bodies = len(self._body_indexes)
        self._num_dof = 0
        
        # 存储所有motion的数据列表
        self._joint_pos_list = []
        self._joint_vel_list = []
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        self._motion_weights_list = []

        # 处理每个motion文件
        for motion_idx, motion_file in enumerate(motion_files):
            motion_name = motion_file.split(".")[0]
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_file)
            
            if motion_idx == 0:
                self.fps = data["fps"]
            
            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            
            time_step_total = _dof_pos.shape[0]
            
            # 为当前motion初始化存储
            _body_pos_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_quat_b = torch.zeros((time_step_total, self._num_bodies, 4), dtype=torch.float32, device=device)
            _body_ori_b = torch.zeros((time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device)
            _body_lin_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_ang_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            
            # 处理所有帧
            for frame_idx in tqdm(range(time_step_total), desc=f"Preloading AMP data for {motion_name}"):
                # 获取当前帧的anchor和body数据
                tgt_anchor_pos_w = _body_pos_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_anchor_quat_w = _body_quat_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_body_pos_w = _body_pos_w[frame_idx, self._body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, self._body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, self._body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, self._body_indexes, :]

                # 计算body相对于anchor的位置和姿态 (局部坐标系)
                tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                    math_utils.subtract_frame_transforms(
                        tgt_anchor_pos_w,
                        tgt_anchor_quat_w,
                        tgt_body_pos_w,
                        tgt_body_quat_w,
                    )
                )

                # 将姿态四元数转换为旋转矩阵的前两列
                mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                # 将速度转换到每个body自己的局部坐标系
                tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_lin_vel_w,
                )

                tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_ang_vel_w,
                )

                # 存储当前帧的局部坐标系数据
                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_quat_b[frame_idx] = tgt_robot_body_quat_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b
            
            # 将当前motion的数据添加到列表
            self._joint_pos_list.append(_dof_pos)
            self._joint_vel_list.append(_dof_vel)
            self._body_pos_b_list.append(_body_pos_b)
            self._body_quat_b_list.append(_body_quat_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)

        # 为了向后兼容，使用第一个motion的数据作为默认值
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]
        self._num_dof = self._joint_pos_list[0].shape[1]


    @property
    def observation_dim(self) -> int:
        num_bodies = len(self._body_indexes)
        obs_dim = 2 * self._num_dof + (3 + 6 + 3 + 3) * num_bodies   # dof_pos dof_vel pos, mat[:,:2], lin_vel, ang_vel
        return obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        num_motions = len(self._body_pos_b_list)
        
        for batch_idx in range(num_mini_batch):
            # 按顺序循环选择motion文件
            motion_idx = batch_idx % num_motions
            
            # 获取当前motion的数据
            current_joint_pos = self._joint_pos_list[motion_idx]
            current_joint_vel = self._joint_vel_list[motion_idx]
            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]
            
            # 从当前motion中随机采样
            idxs = torch.randint(0, current_time_step_total, (mini_batch_size,), device=current_body_pos_b.device)
            idxs = torch.clamp(idxs, max=current_time_step_total - 1)

            batch_joint_pos = current_joint_pos[idxs]  # (mini_batch_size, num_bodies, n_dof)
            batch_joint_vel = current_joint_vel[idxs]  # (mini_batch_size, num_bodies, n_dof)
            batch_body_pos_b = current_body_pos_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ori_b = current_body_ori_b[idxs]  # (mini_batch_size, num_bodies, 6)
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            s = torch.cat(
                [
                    batch_joint_pos.reshape(mini_batch_size, -1),
                    batch_joint_vel.reshape(mini_batch_size, -1),
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)

            next_idxs = (idxs + 1)
            next_idxs = torch.clamp(next_idxs, max=current_time_step_total - 1)
            batch_next_joint_pos = current_joint_pos[next_idxs]  # (mini_batch_size, num_bodies, n_dof)
            batch_next_joint_vel = current_joint_vel[next_idxs]  # (mini_batch_size, num_bodies, n_dof)
            batch_next_body_pos_b = current_body_pos_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ori_b = current_body_ori_b[next_idxs]  # (mini_batch_size, num_bodies, 6)
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            s_next = torch.cat(
                [
                    batch_next_joint_pos.reshape(mini_batch_size, -1),
                    batch_next_joint_vel.reshape(mini_batch_size, -1),
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)
            yield s, s_next

if __name__ == "__main__":
    all_body_names = [
        'pelvis',
        'left_hip_pitch_link',
        'left_hip_roll_link',
        'left_hip_yaw_link',
        'left_knee_link',
        'left_ankle_pitch_link',
        'left_ankle_roll_link',
        'right_hip_pitch_link',
        'right_hip_roll_link',
        'right_hip_yaw_link',
        'right_knee_link',
        'right_ankle_pitch_link',
        'right_ankle_roll_link',
        'waist_roll_link',
        'torso_link',
        'left_shoulder_pitch_link',
        'left_shoulder_roll_link',
        'left_shoulder_yaw_link',
        'left_elbow_link',
        'left_wrist_roll_link',
        'left_wrist_pitch_link',
        'left_wrist_yaw_link',
        'right_shoulder_pitch_link',
        'right_shoulder_roll_link',
        'right_shoulder_yaw_link',
        'right_elbow_link',
        'right_wrist_roll_link',
        'right_wrist_pitch_link',
        'right_wrist_yaw_link'
    ]
    body_names = [
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link"
    ]
    anchor_name = "torso_link"

    motion_files = [
    '/home/liangzhiyuan/RL/droidup/AMP_for_DroidUp/humanoid/envs/datasets/x3_zq_npz/017_walk_xyz.npz'
    ]
    loader = AMPLoader(
    motion_files=motion_files,
    body_names=body_names,
    anchor_name=anchor_name,
    all_body_names=all_body_names,
    device = "cuda:0",
    )
