from humanoid.envs import *
from humanoid.utils import get_args, task_registry

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 15
    env_cfg.commands.gait_enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = 2 # level
    env_cfg.terrain.num_cols = 1 # type
    env_cfg.terrain.max_init_terrain_level = 1  # starting curriculum state
    env_cfg.terrain.terrain_dict = {"smooth slope": 0.,
                                    "rough slope": 0.,
                                    "stairs up": 0.0,
                                    "stairs down": 0.0,
                                    "discrete": 0.0,
                                    "large stairs up": 0.0,
                                    "large stairs down": 0.0,
                                    "stepping stones": 0.0,
                                    "plane": 1., }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = True

    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 4.
    env_cfg.domain_rand.max_push_vel_xy = 1.
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.apply_force_torque = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_link_com = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_pd_factor = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_damping = True
    env_cfg.domain_rand.randomize_joint_armature = True
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.add_action_noise = False
    env_cfg.domain_rand.randomize_arm_pos = False
    env_cfg.domain_rand.arm_pos_interval_s = 3.

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    env.visualize_amp_motion(args.motion_index)

if __name__ == '__main__':
    args = get_args()
    play(args)
