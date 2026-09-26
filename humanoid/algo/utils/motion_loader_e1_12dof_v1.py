"""AMP motion loader for E1 12-DOF 73-column JSON/TXT motions.

The motion files retain their native frame duration.  Expert transitions are
sampled at the policy time step, so 100 Hz files use a two-frame stride for a
50 Hz policy.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import torch


class AMPLoader:
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    JOINT_VEL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    KNEE_POS_SIZE = 6
    FOOT_POS_SIZE = 6
    KNEE_MAT_SIZE = 12
    FOOT_MAT_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    LINEAR_VEL_START_IDX = JOINT_VEL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE
    KNEE_POSE_START_IDX = ANGULAR_VEL_END_IDX
    KNEE_POSE_END_IDX = KNEE_POSE_START_IDX + KNEE_POS_SIZE
    FOOT_POSE_START_IDX = KNEE_POSE_END_IDX
    FOOT_POSE_END_IDX = FOOT_POSE_START_IDX + FOOT_POS_SIZE
    KNEE_MAT_START_IDX = FOOT_POSE_END_IDX
    KNEE_MAT_END_IDX = KNEE_MAT_START_IDX + KNEE_MAT_SIZE
    FOOT_MAT_START_IDX = KNEE_MAT_END_IDX
    FOOT_MAT_END_IDX = FOOT_MAT_START_IDX + FOOT_MAT_SIZE

    FULL_FRAME_SIZE = FOOT_MAT_END_IDX  # 73
    AMP_OBS_SIZE = JOINT_POS_SIZE + JOINT_VEL_SIZE + 3 + 3 + 12 + 24  # 66

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1_000_000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        del data_dir
        self.device = device
        self.time_between_frames = float(time_between_frames)
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_transition_steps = []
        self.trajectory_num_frames = []

        if not motion_files:
            raise ValueError("No E1 12-DOF AMP motion files were configured")

        for index, motion_file in enumerate(motion_files):
            with open(motion_file, encoding="utf-8") as stream:
                motion_json = json.load(stream)
            motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)
            if motion_data.ndim != 2 or motion_data.shape[1] != self.FULL_FRAME_SIZE:
                raise ValueError(
                    f"{motion_file} has frame shape {motion_data.shape}; expected [N, {self.FULL_FRAME_SIZE}]"
                )
            if motion_data.shape[0] < 3:
                raise ValueError(f"{motion_file} must contain at least three frames")
            if not np.isfinite(motion_data).all():
                raise ValueError(f"{motion_file} contains NaN or Inf values")

            frame_duration = float(motion_json["FrameDuration"])
            stride_float = self.time_between_frames / frame_duration
            transition_step = int(round(stride_float))
            if transition_step < 1 or not np.isclose(stride_float, transition_step, rtol=0.0, atol=1.0e-4):
                raise ValueError(
                    f"Policy dt {self.time_between_frames:g}s is not an integer multiple of "
                    f"{motion_file} frame duration {frame_duration:g}s"
                )
            if motion_data.shape[0] <= transition_step:
                raise ValueError(f"{motion_file} is shorter than one policy transition")

            motion_data = self.data_process(motion_data)
            full_tensor = torch.as_tensor(motion_data, dtype=torch.float32, device=device)
            self.trajectories_full.append(full_tensor)
            self.trajectories.append(full_tensor[:, self.JOINT_POSE_START_IDX : self.FOOT_MAT_END_IDX])
            self.trajectory_names.append(motion_file.rsplit(".", 1)[0])
            self.trajectory_idxs.append(index)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_transition_steps.append(transition_step)
            self.trajectory_num_frames.append(motion_data.shape[0])
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            print(
                f"Loaded {traj_len:.3f}s motion from {motion_file} "
                f"({1.0 / frame_duration:.1f} Hz, AMP transition stride={transition_step})."
            )

        weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        if np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("AMP motion weights must be non-negative and have a positive sum")
        self.trajectory_weights = weights / weights.sum()
        self.trajectory_idxs = np.asarray(self.trajectory_idxs, dtype=np.int64)
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float64)
        self.trajectory_transition_steps = np.asarray(self.trajectory_transition_steps, dtype=np.int64)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float64)
        self.trajectory_num_frames = np.asarray(self.trajectory_num_frames, dtype=np.int64)

        self.preload_transitions = preload_transitions
        if preload_transitions:
            print(f"Preloading {num_preload_transitions} E1 AMP transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            frame_idxs = self.traj_time_sample_batch(traj_idxs)
            next_frame_idxs = frame_idxs + self.trajectory_transition_steps[traj_idxs]
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, frame_idxs)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, next_frame_idxs)
            print("Finished preloading E1 AMP transitions")

    def data_process(self, motion_data):
        return np.hstack(
            [
                self.get_root_pos_batch(motion_data),
                self.get_root_rot_batch(motion_data),
                self.get_joint_pose_batch(motion_data),
                self.get_joint_vel_batch(motion_data),
                self.get_linear_vel_batch(motion_data),
                self.get_angular_vel_batch(motion_data),
                self.get_body_pose_batch(motion_data),
                self.get_body_mat_batch(motion_data),
            ]
        )

    def weighted_traj_idx_sample(self):
        return int(np.random.choice(self.trajectory_idxs, p=self.trajectory_weights))

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample_batch(self, traj_idxs):
        traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
        valid_counts = self.trajectory_num_frames[traj_idxs] - self.trajectory_transition_steps[traj_idxs]
        return (np.random.uniform(size=len(traj_idxs)) * valid_counts).astype(np.int64)

    def get_trajectory(self, traj_idx):
        return self.trajectories_full[traj_idx]

    def get_full_frame_at_time_batch(self, traj_idxs, times_step):
        return torch.stack([self.trajectories_full[int(t)][int(f)] for t, f in zip(traj_idxs, times_step)])

    def get_full_frame_at_time(self, traj_idx, time_step):
        return self.trajectories_full[traj_idx][time_step]

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        frame_idxs = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, frame_idxs)

    def compute_obs_dim(self):
        return self.AMP_OBS_SIZE

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        if not self.preload_transitions:
            raise RuntimeError("feed_forward_generator requires preload_transitions=True")
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
            yield self._amp_observation(self.preloaded_s[idxs]), self._amp_observation(self.preloaded_s_next[idxs])

    @classmethod
    def _amp_observation(cls, frames):
        return torch.cat(
            (
                frames[:, cls.JOINT_POSE_START_IDX : cls.JOINT_POSE_END_IDX],
                frames[:, cls.JOINT_VEL_START_IDX : cls.JOINT_VEL_END_IDX],
                frames[:, cls.LINEAR_VEL_START_IDX : cls.LINEAR_VEL_END_IDX],
                frames[:, cls.ANGULAR_VEL_START_IDX : cls.ANGULAR_VEL_END_IDX],
                frames[:, cls.KNEE_POSE_START_IDX : cls.FOOT_POSE_END_IDX],
                frames[:, cls.KNEE_MAT_START_IDX : cls.FOOT_MAT_END_IDX],
            ),
            dim=-1,
        )

    @property
    def observation_dim(self):
        return self.compute_obs_dim()

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @classmethod
    def get_root_pos(cls, pose):
        return pose[cls.ROOT_POS_START_IDX : cls.ROOT_POS_END_IDX]

    @classmethod
    def get_root_pos_batch(cls, poses):
        return poses[:, cls.ROOT_POS_START_IDX : cls.ROOT_POS_END_IDX]

    @classmethod
    def get_root_rot(cls, pose):
        return pose[cls.ROOT_ROT_START_IDX : cls.ROOT_ROT_END_IDX]

    @classmethod
    def get_root_rot_batch(cls, poses):
        return poses[:, cls.ROOT_ROT_START_IDX : cls.ROOT_ROT_END_IDX]

    @classmethod
    def get_joint_pose(cls, pose):
        return pose[cls.JOINT_POSE_START_IDX : cls.JOINT_POSE_END_IDX]

    @classmethod
    def get_joint_pose_batch(cls, poses):
        return poses[:, cls.JOINT_POSE_START_IDX : cls.JOINT_POSE_END_IDX]

    @classmethod
    def get_joint_vel(cls, pose):
        return pose[cls.JOINT_VEL_START_IDX : cls.JOINT_VEL_END_IDX]

    @classmethod
    def get_joint_vel_batch(cls, poses):
        return poses[:, cls.JOINT_VEL_START_IDX : cls.JOINT_VEL_END_IDX]

    @classmethod
    def get_linear_vel(cls, pose):
        return pose[cls.LINEAR_VEL_START_IDX : cls.LINEAR_VEL_END_IDX]

    @classmethod
    def get_linear_vel_batch(cls, poses):
        return poses[:, cls.LINEAR_VEL_START_IDX : cls.LINEAR_VEL_END_IDX]

    @classmethod
    def get_angular_vel(cls, pose):
        return pose[cls.ANGULAR_VEL_START_IDX : cls.ANGULAR_VEL_END_IDX]

    @classmethod
    def get_angular_vel_batch(cls, poses):
        return poses[:, cls.ANGULAR_VEL_START_IDX : cls.ANGULAR_VEL_END_IDX]

    @classmethod
    def get_body_pose(cls, pose):
        return pose[cls.KNEE_POSE_START_IDX : cls.FOOT_POSE_END_IDX]

    @classmethod
    def get_body_pose_batch(cls, poses):
        return poses[:, cls.KNEE_POSE_START_IDX : cls.FOOT_POSE_END_IDX]

    @classmethod
    def get_body_mat(cls, pose):
        return pose[cls.KNEE_MAT_START_IDX : cls.FOOT_MAT_END_IDX]

    @classmethod
    def get_body_mat_batch(cls, poses):
        return poses[:, cls.KNEE_MAT_START_IDX : cls.FOOT_MAT_END_IDX]
