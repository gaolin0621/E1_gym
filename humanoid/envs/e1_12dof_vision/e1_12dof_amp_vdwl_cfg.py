from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from humanoid.utils.helpers import create_point_list

CLOCK_INPUT = 2              # sin cos
CMD_DIM = 3                  # vx vy v_yaw
PROPRIOCEPTION_DIM = 36 + 5  # 12 joint positions/velocities/actions + gyro + roll/pitch
Feet_height_obs_dim = 42     # left_feet_map + right_feet_map
Height_obs_dim = 121         # base_height_map
PRIVILEGED_DIM = 3 + 3 + 3   # vel ...

Body_pos_ori_b_Enable = 0
Body_pos_ori_b_dim = 4 * (3 + 6) * Body_pos_ori_b_Enable

# MJCF depth_cam is mounted at [0.09, 0.0, 0.18] in torso_link.
# waist_yaw_joint is fixed at [0.0, 0.0, 0.12] with identity rotation, so
# Isaac Gym collapses torso_link into pelvis and the equivalent root-relative
# camera pose is [0.09, 0.0, 0.30]. The MJCF xyaxes optical axis points 70 deg
# below horizontal. These are the nominal values; the original VDWL camera
# domain randomization remains centered on them.
DEPTH_CAM_POS_PELVIS = (0.09, 0.0, 0.30)
DEPTH_CAM_RPY_PELVIS = (0.0, 1.2217304763960306, 0.0)
# D435 640x480 (VGA) depth mode uses an approximately 62-degree vertical FOV.
# Only the camera extrinsics above are aligned to the MJCF depth_cam.
DEPTH_CAM_FOV_Y_DEG = 62.0

Single_obs_dim = CLOCK_INPUT + CMD_DIM + PROPRIOCEPTION_DIM
Single_priv_obs_dim = Single_obs_dim + PRIVILEGED_DIM + Feet_height_obs_dim + Height_obs_dim

Feet_hold_x, Feet_hold_y = create_point_list(resolution=0.01,
                                             range_x=(-0.10, 0.10),
                                             range_y=(-0.04, 0.04),
                                             debug=False)
Base_point_x, Base_point_y = create_point_list(resolution=0.05,
                                               range_x=(-0.4, 0.4),
                                               range_y=(-0.4, 0.4),
                                               debug=False)
Terrain_point_x, Terrain_point_y = create_point_list(resolution=0.1,
                                                     range_x=(0, 1.0),
                                                     range_y=(-0.5, 0.5),
                                                     debug=False)
Feet_forward_x, Feet_forward_y = create_point_list(resolution=0.01,
                                              range_x=(0.0, 1.5),
                                             range_y=(0, 0),
                                               debug=False)
class E1_12DOFAMPVDWLCfg(LeggedRobotCfg):
    """
    Independent VDWL/AMP vision configuration for the E1 12-DOF robot.
    """
    class env(LeggedRobotCfg.env):
        reference_state_initialization = True
        amp_motion_files_display = [
            f"{LEGGED_GYM_ROOT_DIR}/humanoid/envs/datasets/e1/txt_v1/walk1_100hz.txt",
            f"{LEGGED_GYM_ROOT_DIR}/humanoid/envs/datasets/e1/txt_v1/stair1_cut_100hz.txt",
        ]
        amp_body_names = [
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link"
        ]
        body_pos_ori_b_enable = Body_pos_ori_b_Enable
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = Single_obs_dim  # 46
        num_single_privileged_obs = Single_priv_obs_dim
        num_observations = int(frame_stack * num_single_obs)  # 690
        num_privileged_obs = int(c_frame_stack * num_single_privileged_obs)  # 654
        num_commands = CMD_DIM + CLOCK_INPUT
        num_actions = 12
        num_arms = 0
        num_envs = 4096
        episode_length_s = 20  # episode length in seconds

        lin_vel_idx = num_single_obs  # 46:49
        feet_height_idx = lin_vel_idx + PRIVILEGED_DIM  # 55:97
        height_idx = feet_height_idx + Feet_height_obs_dim  # 97:218

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        horizontal_scale = 0.10 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25  # [m]
        edge_width_thresh = 0.05
        simplify_grid = True

        measure_heights = True
        # -- loss 重构
        measured_points_x = Terrain_point_x
        measured_points_y = Terrain_point_y
        feet_points_x = [-0.09, -0.06, -0.03, 0., 0.03, 0.06, 0.09]
        feet_points_y = [-0.04, 0., 0.04]
        # for reward feet hold
        feet_hold_x = Feet_hold_x
        feet_hold_y = Feet_hold_y
        # for base_height in terrain
        base_points_x = Base_point_x
        base_points_y = Base_point_y
        # for feet forward to stair
        feet_forward_x = Feet_forward_x
        feet_forward_y = Feet_forward_y
        # foothold candidate search
        foothold_candidate_num = 30          # 每只脚缓存的候选落脚点数量，固定 shape 便于并行环境和 reward 计算
        foothold_point_spacing = 0.01        # 前方地形采样间距，需要和 Feet_forward_x 的 resolution 保持一致
        foothold_filter_time = 0.3           # 起脚时按照 |vx| * time 过滤过近候选，避免把脚下附近点当成下一步目标
        foothold_filter_min = 0.02           # 低速时仍保留最小前向过滤距离，避免候选贴脚过近
        foothold_filter_max = 0.60           # 高速时限制过滤距离，避免跨过当前可踩台阶直接找太远候选
        foothold_window_size = 0.26          # 平坦性检测窗口长度，约等于脚掌前后可用支撑长度
        foothold_window_stride = 0.03        # 滑窗步长，决定候选点密度
        foothold_flatness_threshold = 0.08   # 窗口内最大高度差小于该值时认为足够平坦
        # feet ray caster
        feet_ray_enable = True
        feet_far_clip = 0.5 # m

        feet_front_threshold = 0.03 # m
        feet_back_threshold = 0.03  # m

        static_friction = 1.0
        dynamic_friction = 0.9
        restitution = 0.1

        slop_range = [0.0, 0.5]
        step_width = 0.31
        step_height = [0.00, 0.20]
        step_height_1 = [0.00, 0.30]
        step_height_w60cm =[0.0, 0.30]
        discrete_obstacles_height = [0.05, 0.15]
        rough_height = [0.00,  0.02]
        downsampled_scale = 0.2
        step_stone = [0.05, 0.20]
        wave_amplitude = [0.1, 0.4]
        gap_size = [0.1, 0.6]
        pit_depth_size = [0.1, 0.35]
        curriculum = True
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 40  # number of terrain cols (types)
        max_init_terrain_level = 1  # starting curriculum state
        terrain_dict = {"smooth slope": 0.1,
                        "rough slope": 0.0,
                        "stairs up": 0.2,
                        "stairs down": 0.2,
                        "discrete": 0.0,
                        "stairs up w1m": 0.1,
                        "stairs down w1m": 0.1,
                        "stepping stones": 0.0,
                        "plane": 0.1,
                        "gap":   0.1, # 0.30m
                        "gap_1": 0.1, # 0.45m
                        "gap_2": 0.00,  # 0.60m
                        "stairs up w60cm": 0.00,
                        "stairs down w60cm": 0.00,
                        "pit": 0.0, }
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = 0.7 # slopes above this threshold will be corrected to vertical surfaces

    class depth:
        use_camera = False
        camera_num_envs = 100 # camera_num_envs = min(camera_num_envs, env_nums)
        use_warp = True

        # Root-relative pose after the fixed torso joint is collapsed.
        # Position randomization: +/- 1 cm on x/y/z.
        pos_min = [DEPTH_CAM_POS_PELVIS[0] - 0.01,
                   DEPTH_CAM_POS_PELVIS[1] - 0.01,
                   DEPTH_CAM_POS_PELVIS[2] - 0.01]
        pos_max = [DEPTH_CAM_POS_PELVIS[0] + 0.01,
                   DEPTH_CAM_POS_PELVIS[1] + 0.01,
                   DEPTH_CAM_POS_PELVIS[2] + 0.01]
        # Orientation randomization: +/- 0.1 rad around the exact XML pose.
        rot_min = [DEPTH_CAM_RPY_PELVIS[0] - 0.1,
                   DEPTH_CAM_RPY_PELVIS[1] - 0.1,
                   DEPTH_CAM_RPY_PELVIS[2] - 0.1]
        rot_max = [DEPTH_CAM_RPY_PELVIS[0] + 0.1,
                   DEPTH_CAM_RPY_PELVIS[1] + 0.1,
                   DEPTH_CAM_RPY_PELVIS[2] + 0.1]
        # 针孔相机模型参数
        vertical_fov = [DEPTH_CAM_FOV_Y_DEG - 2.0, DEPTH_CAM_FOV_Y_DEG + 2.0]
        original = [32, 24] #[w, h]
        resized = [32, 24] #[w, h]
        crop_size = [24, 24]  # 裁切 [4, 4， 0, 0] 左右上下裁切去除异常值; [w, h]
        crop_rang_x = [4, 28] # 4:32
        crop_rang_y = [0, 24] # 0:24
        # D435配置
        # 640*480:         fov_x=75, fov_y=62
        # 640*480, 对齐rgb: fov_x=69, fov_y=42
        # 1280×720         fov_x=87, fov_y=58

        # 深度裁切范围
        near_clip = 0.1 # m
        far_clip = 3.0 # m

        # update_interval = 10  # 10*0.01s = 0.1
        update_interval = 5  # 5*0.02s = 0.1
        buffer_len = 4
        obs_len = 2
        # depth image noise
        class noise:
            enable = True
            depth_std = 0.01  # 噪声标准差（米）
            depth_std_multiplier = 0.015  # 随距离增加的噪声放大
            dropout_prob = 0.02  # 2% 像素丢失
            dropout_value = -1.0  # 丢失像素值

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        standing_env_radio = 0.1
        walking_x_env_radio = 0.3
        walking_y_env_radio = 0.1
        walking_z_env_radio = 0.1
        heading_command = True  # if true: compute ang vel command from heading error

        gait_enable = True
        gait = ["walk_sagittal", "stand", "walk_sagittal", "walk_omnidirectional"]  # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,6],
                           "stand": [2,6],
                           "walk_omnidirectional": [2,6]}
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        min_vel = 0.1
        class ranges:
            lin_vel_x   = [-1.0, 1.2]  # min max [m/s]
            lin_vel_y   = [-0.6, 0.6]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.91]  # x,y,z [m]
        pos = [0.0, 0.0, 0.80]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': -0.20,
            'left_hip_roll_joint': 0.,
            'left_hip_yaw_joint': 0.,
            'left_knee_joint': 0.46,
            'left_ankle_pitch_joint': -0.26,
            'left_ankle_roll_joint': 0.,
            'right_hip_pitch_joint': -0.20,
            'right_hip_roll_joint': 0.,
            'right_hip_yaw_joint': 0.,
            'right_knee_joint': 0.46,
            'right_ankle_pitch_joint': -0.26,
            'right_ankle_roll_joint': 0.,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 80,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 200,
                     'ankle_pitch': 80,
                     'ankle_roll': 60,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 3,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 5,
                     'ankle_pitch': 3,
                     'ankle_roll': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # ==============joint damping armature ==============
        joint_damping = {
            'hip_yaw': 0.1,
            'hip_roll': 0.1,
            'hip_pitch': 0.1,
            'knee': 0.1,
            'ankle_pitch': 0.1,
            'ankle_roll': 0.1,
        }
        joint_armature = {
            'hip_yaw': 0.01,
            'hip_roll': 0.01,
            'hip_pitch': 0.01,
            'knee': 0.01,
            'ankle_pitch': 0.01,
            'ankle_roll': 0.01,
        }
        joint_friction = {
            'hip_yaw': 0.0,
            'hip_roll': 0.0,
            'hip_pitch': 0.0,
            'knee': 0.0,
            'ankle_pitch': 0.0,
            'ankle_roll': 0.0,
        }
        # ============== motor limit and dof limit ==============
        # -- max dof vel and torque
        dof_torque_max = [120., 120., 36., 120., 36., 30.,
                          120., 120., 36., 120., 36., 30.]

        dof_vel_max = [12.04, 12.04, 13.61, 12.04, 13.61, 15.71,
                       12.04, 12.04, 13.61, 12.04, 13.61, 15.71]

        # -- soft limit
        dof_torque_limits = [120., 120., 36., 120., 36., 30.,
                             120., 120., 36., 120., 36., 30.]

        dof_vel_limits = [12.04, 12.04, 13.61, 12.04, 13.61, 15.71,
                          12.04, 12.04, 13.61, 12.04, 13.61, 15.71]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 2  # 100hz 2*0.05 = 0.01s
        decimation = 4  # 50hz  4*0.05 = 0.02s

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/e1/urdf/E1_12dof.urdf'
        name = "e1_12dof"
        foot_name = "ankle_roll_link"
        knee_name = "knee_link"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis", "knee"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        fix_base_link = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">


    class domain_rand:
        # 瞬时扰动
        push_robots = True
        push_operation = "abs" # "abs" "add"
        push_interval_s = 7.
        max_push_vel_xy =  [[-1.0, 1.0],
                            [-1.0, 1.0]]
        max_push_ang_vel = 0.0

        small_push_robots = False
        small_push_interval_s = 4.
        max_small_push_vel_xy = 0.5
        max_small_push_ang_vel = 0.2

        # 持续推力
        apply_force_torque = False
        apply_interval_s = 7.
        max_apply_force = 2.0
        max_apply_torque = 2.0

        # 摩擦系数
        randomize_friction = True
        num_buckets = 256
        friction_range = [-0.5, 1.5]
        # 恢复系数
        randomize_restitution = False
        restitution_range = [0., 0.2]
        # base质量
        randomize_base_mass = True
        randomize_mass_body_name = "pelvis" # "pelvis" "torso_link"
        added_base_mass_range = [-1., 3.]
        # base质心
        randomize_base_com = True
        randomize_com_body_name = "pelvis"
        added_base_com_range = [[-0.02, 0.02],
                                [-0.02, 0.02],
                                [-0.02, 0.02]]
        # 其他link质量
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        # 其他link质心
        randomize_link_com = True
        added_link_com_range = [[-0.01, 0.01],
                                [-0.01, 0.01],
                                [-0.01, 0.01]]
        # 惯量
        randomize_inertia = True
        multiplied_inertia_range = [0.9, 1.1]
        # PD刚度阻尼
        randomize_pd_factor = True
        Kp_factor_range = [0.8, 1.2]
        Kd_factor_range = [0.8, 1.2]
        # torque
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        # dof offset
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        # joint 阻尼
        randomize_joint_damping = True
        damping_operation = "abs"        # ["scale", "abs"]
        joint_damping_range = [0.1, 0.2]
        # joint 静摩擦
        randomize_joint_friction = False
        friction_operation = "abs"  # ["scale", "abs"]
        joint_friction_range = [0.001, 0.005]
        # 电枢惯量
        randomize_joint_armature = True
        armature_operation = "abs"  # ["scale", "abs"]
        joint_armature_range = [0.01, 0.02]
        # action延迟
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 4] # [0 - 2*5] ms
        # action噪声
        add_action_noise= False
        action_noise = 0.001
        # base state 重置
        reset_base_pose = True
        pose_xy = [-0.5, 0.5]
        pose_yaw =[-3.14*0, 3.14*0]
        lin_vel = [-0.0, 0.0]
        ang_vel = [-0.0, 0.0]
        # joint state 重置
        reset_joint = True
        joint_pos_range = [0.5, 1.5] # scale
        joint_vel_range = [-1, 1] # range rad/s
        # arm noise
        randomize_arm_pos = False
        arm_pos_interval_s = 10.
        min_arm_pos = [-1.5,  0.0, -0.5, -4.00] # rad
        max_arm_pos = [ 1.5,  0.5,  0.5,  0.90] # rad

    class rewards:
        # base_height_target = 0.90  # 0.15 rad
        base_height_target = 0.65
        feet_height = 0.058
        target_feet_height = 0.10  # desired swing height above the local terrain
        target_knee_swing_pos = 0.9 # rad

        clock_enable = 1 # 1: 使用[sin、cos]时钟信号,  0: 不使用时钟信号
        cycle_time = 1.20                    # sec; averaged from walk/stair AMP motions
        stand_radio = 0.65
        gait_radio = False
        # feet_air_time = cycle_time * (1 - stand_radio)
        feet_air_time = 0.42                 # cycle_time * (1 - stand_radio)
        phase_offset = 0.5
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma_x = 4
        tracking_sigma_y = 4
        tracking_sigma_z = 4
        base_ang_vel_xy_sigma = 0.5
        max_contact_force = 500  # forces above this value are penalized
        max_contact_force_xy = 100  # forces above this value are penalized
        max_feet_vel_z = 3.0
        max_feet_acc_z = 50.0
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        soft_dof_vel_limit = 0.90
        close_feet_threshold = 0.24 - 0.01

        feet_safety_enable_step = 0 * 24 # after this count
        # feet_safety_enable_step = 5000 * 24  # after this count

        feet_step_limit = [0.45, 0.40, 0.30]

        feet_gait_stance_mask = 0.2
        feet_gait_swing_mask = 0.2
        base_lin_vel_z_mask = 0.2
        base_ang_vel_xy_mask = 0.5
        base_ang_vel_xy_stand_mask = 2

        class scales:
            # survival = 0.5
            #==================== gait style ========================
            feet_swing_under_target = -5
            feet_gait_contact = 0.5
            feet_gait_stance_mask = -2.0
            feet_gait_swing_mask = -2.0
            feet_slide = -0.4
            # feet_slide_xyz = -0.2
            # feet_x_distance = -4.
            feet_y_distance = -2.
            # feet_x_symmetry = -20.
            # feet_step_limit = -2.
            #==================== vel tracking =======================
            tracking_lin_vel = 4.0
            tracking_ang_vel = 3.0
            tracking_lin_vel_l2 = -0.5
            tracking_ang_vel_z_l2 = -0.5
            tracking_dont_wait = -2.0
            # ==================== base pos ==========================
            # base_feet_height_l1 = -6
            # base_gravity_exp = 0.5
            # base_gravity_exp_mask = 1.0
            base_gravity = -10.
            base_lin_vel_z = -1.0
            base_ang_vel_xy = -0.5

            hip_pos_exp = 0.5
            hip_pos = -1.0
            hip_yaw_pos_mask = -2.0
            hip_roll_pos_mask = -2.0

            # knee_pos_swing_v1 = -0.5
            # ankle_pitch_pos_v1 = -0.1
            # ankle_pitch_pos_mask = -0.1
            feet_ori = -1.0
            # feet_ori_first_contact = -20.
            #======================== energy ============================
            feet_contact_forces = -0.01
            # feet_contact_forces_xy = -0.01
            # feet_contact_no_vel = -0.2
            # feet_contact_no_vel_max_hist = -0.5
            # feet_acc = -0.002
            # feet_vel_z = -1.0

            action_rate = -0.02
            action_smoothness = -0.005
            action_hip_smoothness = -0.01

            # dof_torque = -1e-5
            dof_vel = -5e-4
            # dof_vel_knee = -1e-3
            dof_acc = -2.5e-7
            dof_energy = -1e-3
            #======================= reward for safety ===================
            dof_pos_limits = -10
            # dof_torque_limits = -0.01
            # dof_vel_limits = -0.5

            feet_stumble = -5
            # feet_dist_safety = -1.
            # feet_dist_safety_v1 = -2.
            # feet_hold = -3.0
            feet_hold_v1 = -2.0
            # feet_hold_v2 = -2.0
            # feet_hold_first_contact = -20
            # feet_hold_placement = -100.0
            feet_hold_placement_exp = 1.0
            termination = -200

        feet_hold_placement_sigma_xz = 5.  # 论文 exp 奖励中的距离尺度 s_xz，d_xz 越小 exp(-d_xz * s_xz) 越接近 1


    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        height_offset = 0.65
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.50
            ang_vel = 0.50
            lin_vel = 0.10
            quat    = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005    # 200 Hz
        substeps = 1  # 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # 0.1 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

Short_obs = 5 * E1_12DOFAMPVDWLCfg.env.num_single_obs # input actor
Long_obs = 15 * E1_12DOFAMPVDWLCfg.env.num_single_obs # input state_encoder
Depth_image_dim = [E1_12DOFAMPVDWLCfg.depth.crop_size[1],
                   E1_12DOFAMPVDWLCfg.depth.crop_size[0]] # input depth_cnn [h, w]
Depth_image_frame = E1_12DOFAMPVDWLCfg.depth.obs_len  # n

class E1_12DOFAMPVDWLCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'VDWL_AmpOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        noise_std_type = "scalar"  # 'scalar' or 'log'
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # for ActorCritic_XXX
        class policy_cfg:
            # 长短历史
            num_short_obs = Short_obs
            num_long_obs = Long_obs
            # 图像大小
            depth_image_dim = Depth_image_dim # h, w
            depth_image_frame = Depth_image_frame # n
            # CNN
            class depth_encoder:
                name = 'depth_encoder'
                # cnn [B, n, h, w]
                in_channels = Depth_image_frame # n
                # mip head
                output_dim = 128 # depth_latent

            class prop_encoder:
                name = 'prop_encoder'
                # long obs
                input_dim = Long_obs
                hidden_dims = [256, 128]
                # prop_latent
                output_dim = 64
                activation = 'elu'

            class fusion_encoder:
                name = 'fusion_encoder'
                # prop_latent + depth_latent
                input_dim = 128 + 64
                hidden_dims = [256, 128]
                # vel , z_obs, z_heightmap
                output_dim = 3 + 16 + 32
                activation = 'elu'

            # -- 显示重构 height_map
            class height_decoder:
                name = 'height_decoder'
                input_dim = 32 # height_map_latent
                hidden_dims = [128, 256]
                output_dim = Height_obs_dim  # height_map
                activation = 'elu'

            # -- 显示重构 obs
            class obs_decoder:
                name = 'obs_decoder'
                input_dim = 16 # osb_latent
                hidden_dims = [64, 128]
                output_dim = E1_12DOFAMPVDWLCfg.env.num_single_obs  # obs
                activation = 'elu'


    class algorithm(LeggedRobotCfgPPO.algorithm):
        # -- value function
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        # -- surrogate loss
        desired_kl = 0.01
        entropy_coef = 0.01
        gamma = 0.99
        lam = 0.95
        max_grad_norm = 1.0
        # -- training
        learning_rate = 0.001
        num_learning_epochs =  5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_mini_batches
        schedule = 'adaptive'  # adaptive, fixed
        learning_rate_print = False
        amp_enable = True # 混合精度训练
        # -- Amp cfg
        class amp_cfg:
            # The source demonstrations are 100 Hz. The E1 loader samples every
            # second frame so each AMP transition still matches this 50 Hz policy.
            step_dt = 0.02
            amp_motion_files = [
                f"{LEGGED_GYM_ROOT_DIR}/humanoid/envs/datasets/e1/txt_v1/walk1_100hz.txt",
                f"{LEGGED_GYM_ROOT_DIR}/humanoid/envs/datasets/e1/txt_v1/stair1_cut_100hz.txt",
            ]
            amp_num_preload_transitions = 200000
            amp_replay_buffer_size = 100000
            # amp parameter
            normalizer = True
            amp_reward_coef = 0.4
            amp_task_reward_lerp = 0.7
            amp_discr_hidden_dims = [512, 256, 128]
            amp_loss_coef = 1.0
            loss_type = "LSGAN" # "LSGAN", "WGAN", "BCEWithLogits"
            eta_wgan = 0.5 # 0.1 ~ 0.5 is the proper range of selection

            trunk_weight_decay = 10e-4 # 10e-4  1.0e-4
            linear_weight_decay = 10e-2 # 10e-2 1.0e-2
            learning_rate = 1.0e-4

        # -- Symmetry Augmentation
        class symmetry_cfg:
            sym_loss = False
            obs_permutation = [
                -0.0001, -1, 2, -3, -4,
                11,-12,-13,14,15,-16, 5,-6,-7,8,9,-10,
                23,-24,-25,26,27,-28, 17,-18,-19,20,21,-22,
                35,-36,-37,38,39,-40, 29,-30,-31,32,33,-34,
                -41,42,-43,-44,45
            ]
            act_permutation = [6,-7,-8,9,10,-11, 0.0001,-1,-2,3,4,-5]
            frame_stack = E1_12DOFAMPVDWLCfg.env.frame_stack
            sym_coef = 1.0
        # -- est cfg
        class priv_est_cfg:
            obs_recon_loss = True
            obs_cur_idx = E1_12DOFAMPVDWLCfg.env.num_single_privileged_obs * (E1_12DOFAMPVDWLCfg.env.c_frame_stack- 1)
            obs_cur_dim = Single_obs_dim

            lin_vel_loss = True
            lin_vel_idx = E1_12DOFAMPVDWLCfg.env.num_single_privileged_obs * (E1_12DOFAMPVDWLCfg.env.c_frame_stack- 1) + E1_12DOFAMPVDWLCfg.env.lin_vel_idx
            lin_vel_dim = 3

            feet_height_loss = False
            feet_height_idx = E1_12DOFAMPVDWLCfg.env.num_single_privileged_obs * (E1_12DOFAMPVDWLCfg.env.c_frame_stack- 1) + E1_12DOFAMPVDWLCfg.env.feet_height_idx
            feet_height_dim = Feet_height_obs_dim

            height_recon_loss = True
            height_idx = E1_12DOFAMPVDWLCfg.env.num_single_privileged_obs * (E1_12DOFAMPVDWLCfg.env.c_frame_stack- 1) + E1_12DOFAMPVDWLCfg.env.height_idx
            height_dim = Height_obs_dim

    class runner:
        policy_class_name = 'VDWLActorCritic'
        algorithm_class_name = 'VDWL_AmpPPO_E1_12DOF'
        num_steps_per_env = 24  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        save_envs = 'e1_12dof_vision'
        experiment_name = 'e1_12dof_amp_vdwl'
        run_name = 'e1_12dof_amp_vdwl_100hz_expert_50hz_policy'
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        use_wandb = False
