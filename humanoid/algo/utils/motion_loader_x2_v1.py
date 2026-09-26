import glob
import json
import torch
import numpy as np


class AMPLoader:
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 10
    JOINT_VEL_SIZE = 10
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

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir="",
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.amp_obs_dim = 0
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                motion_data = self.data_process(motion_data)

                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(
                    torch.tensor(motion_data[:,AMPLoader.JOINT_POSE_START_IDX:AMPLoader.FOOT_MAT_END_IDX],
                                 dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, :],
                    dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times_step = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times_step)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times_step+1)
            print(f'Finished preloading')

    def data_process(self, motion_data):
        root_pos = AMPLoader.get_root_pos_batch(motion_data)
        root_rot = AMPLoader.get_root_rot_batch(motion_data)
        joint_pos = AMPLoader.get_joint_pose_batch(motion_data)
        joint_vel = AMPLoader.get_joint_vel_batch(motion_data)
        lin_vel = AMPLoader.get_linear_vel_batch(motion_data)
        ang_vel = AMPLoader.get_angular_vel_batch(motion_data)
        body_pos_b = AMPLoader.get_body_pose_batch(motion_data)
        body_mat_b = AMPLoader.get_body_mat_batch(motion_data)
        return np.hstack([root_pos, root_rot, joint_pos, joint_vel, lin_vel, ang_vel, body_pos_b, body_mat_b])

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random frame index for each trajectory independently."""
        n_frames = self.trajectory_num_frames[traj_idxs]
        frame_samples = (np.random.uniform(size=len(traj_idxs)) * n_frames).astype(np.int64)
        frame_samples = np.clip(frame_samples, 0, n_frames - 2)
        return frame_samples.astype(np.int64)

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_full_frame_at_time_batch(self, traj_idxs, times_step):
        states = torch.stack([self.trajectories_full[t][f] for t, f in zip(traj_idxs, times_step)])
        return states

    def get_full_frame_at_time(self, traj_idx, time_step):
        """Returns full frame for the given trajectory at the specified time."""
        return self.trajectories_full[traj_idx][time_step]

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times_step = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times_step)

    def compute_obs_dim(self):
        return 10 + 10 + 3 + 3 + 6*2*1 + 12*2*1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)

            joint_pos = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX: AMPLoader.JOINT_POSE_END_IDX]
            joint_vel = self.preloaded_s[idxs, AMPLoader.JOINT_VEL_START_IDX: AMPLoader.JOINT_VEL_END_IDX]
            root_lin_vel_b = self.preloaded_s[idxs, AMPLoader.LINEAR_VEL_START_IDX: AMPLoader.LINEAR_VEL_END_IDX]
            root_ang_vel_b = self.preloaded_s[idxs, AMPLoader.ANGULAR_VEL_START_IDX: AMPLoader.ANGULAR_VEL_END_IDX]
            body_pos_b = self.preloaded_s[idxs, AMPLoader.KNEE_POSE_START_IDX: AMPLoader.FOOT_POSE_END_IDX]
            body_mat_b = self.preloaded_s[idxs, AMPLoader.KNEE_MAT_START_IDX: AMPLoader.FOOT_MAT_END_IDX]
            s = torch.cat([
                joint_pos,
                joint_vel,
                root_lin_vel_b,
                root_ang_vel_b,
                body_pos_b,
                body_mat_b,
            ],dim=-1)

            joint_pos_n = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX: AMPLoader.JOINT_POSE_END_IDX]
            joint_vel_n = self.preloaded_s_next[idxs, AMPLoader.JOINT_VEL_START_IDX: AMPLoader.JOINT_VEL_END_IDX]
            root_lin_vel_b_n = self.preloaded_s_next[idxs, AMPLoader.LINEAR_VEL_START_IDX: AMPLoader.LINEAR_VEL_END_IDX]
            root_ang_vel_b_n = self.preloaded_s_next[idxs, AMPLoader.ANGULAR_VEL_START_IDX: AMPLoader.ANGULAR_VEL_END_IDX]
            body_pos_b_n = self.preloaded_s_next[idxs, AMPLoader.KNEE_POSE_START_IDX: AMPLoader.FOOT_POSE_END_IDX]
            body_mat_b_n = self.preloaded_s_next[idxs, AMPLoader.KNEE_MAT_START_IDX: AMPLoader.FOOT_MAT_END_IDX]
            s_next = torch.cat([
                joint_pos_n,
                joint_vel_n,
                root_lin_vel_b_n,
                root_ang_vel_b_n,
                body_pos_b_n,
                body_mat_b_n,
            ],dim=-1)

            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.compute_obs_dim()

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_body_pose(pose):
        return pose[AMPLoader.KNEE_POSE_START_IDX : AMPLoader.FOOT_POSE_END_IDX]

    def get_body_pose_batch(poses):
        return poses[:, AMPLoader.KNEE_POSE_START_IDX : AMPLoader.FOOT_POSE_END_IDX]

    def get_body_mat(pose):
        return pose[AMPLoader.KNEE_MAT_START_IDX : AMPLoader.FOOT_MAT_END_IDX]

    def get_body_mat_batch(poses):
        return poses[:, AMPLoader.KNEE_MAT_START_IDX : AMPLoader.FOOT_MAT_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX : AMPLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX : AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX : AMPLoader.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX : AMPLoader.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

if __name__ == "__main__":
    amp_loader=AMPLoader(
        device='cuda:0',
        time_between_frames=0.02,
        data_dir="",
        preload_transitions=True,
        num_preload_transitions=100,
        motion_files=["/home/liangzhiyuan/RL/droidup/AMP_for_DroidUp/humanoid/envs/datasets/x3_amp_v1/017_walk_xyz.txt",
                      "/home/liangzhiyuan/RL/droidup/AMP_for_DroidUp/humanoid/envs/datasets/x3_amp_v1/017_walk_xyz_1.txt"],
    )
    print(amp_loader.get_full_frame_at_time(0,0))
