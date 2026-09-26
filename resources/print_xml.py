import mujoco
import numpy as np

def print_robot_physics(xml_path):
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    print_link_data(model)
    print_joint_data(model)
    print_dof_data(model)


def print_link_data(model):
    print(f"\n--- 机器人模型信息: {model.nbody} link---")
    print("=" * 100)
    print("📦 连杆物理参数 (Link/Body Data)")
    print("=" * 100)
    print(
        f"{'ID':<4} | {'name':<25} | {'mass (kg)':<10} | {'com X':<8} | {'com Y':<8} | {'com Z':<8} | {'Ixx':<8} | {'Iyy':<8} | {'Izz':<8}")
    print("-" * 100)
    total_mass = 0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        com = model.body_ipos[i]
        inertia = model.body_inertia[i]
        total_mass += mass
        print(
            f"{i:<4} | {name:<25} | {mass:<10.4f} | {com[0]:<8.4f} | {com[1]:<8.4f} | {com[2]:<8.4f} | {inertia[0]:<8.4f} | {inertia[1]:<8.4f} | {inertia[2]:<8.4f}")
    print("total_mass:", f"{total_mass:<8.4f}\n")

def print_joint_data(model):
    """打印所有关节 (Joint) 的物理参数"""
    print(f"--- 机器人模型信息: {model.njnt} joint, {model.nv} dof---")
    if model.njnt == 0:
        print("\n⚠️ 模型中没有检测到关节 (Joints)。")
        return

    print("=" * 130)
    print("🔗 关节物理参数 (Joint Data)")
    print("=" * 130)
    print(
        f"{'ID':<4} | {'name':<28} | {'link':<28} | {'type':<8} | {'pos_min':<8} | {'pos_max':<8} | {'actfrc_min':<10} | {'actfrc_max':<10}")
    print("-" * 130)

    type_map = {0: "Free", 1: "Slide", 2: "Ball", 3: "Hinge"}

    for i in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

        body_id = model.jnt_bodyid[i]
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        j_type_id = model.jnt_type[i]
        j_type_str = type_map.get(j_type_id, f"Unknown({j_type_id})")

        jnt_range = model.jnt_range[i]
        jnt_actfrcrange = model.jnt_actfrcrange[i]

        print(
            f"{i:<4} | {j_name:<28} | {b_name:<28} | {j_type_str:<8} | {jnt_range[0]:<8} | {jnt_range[1]:<8} | {jnt_actfrcrange[0]:<10} | {jnt_actfrcrange[1]:<10}")

def print_dof_data(model):
    """打印所有dof参数"""
    print(f"\n--- 机器人模型信息: {model.nv} dof---")

    print("=" * 100)
    print("🔗 dof 物理参数 (dof Data)")
    print("=" * 100)
    print(
        f"{'ID':<4} | {'name':<28} | {'damping':<8} | {'armature':<8} | {'frictionloss':<10}")
    print("-" * 100)
    for i in range(5, model.nv):
        jnt_id = model.dof_jntid[i]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        damping = model.dof_damping[i]
        armature = model.dof_armature[i]
        frictionloss = model.dof_frictionloss[i]

        print(
            f"{i:<4} | {jnt_name:<28} | {damping:<8} | {armature:<8} | {frictionloss:<10}")


if __name__ == "__main__":
    # 尝试加载一个本地文件，如果没有请修改路径
    # xml_file = "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/resources/robots/x3h/x3h_14dof.xml"
    xml_file = "/home/liangzhiyuan/RL/droidup/x2-gym-terrain/resources/robots/Moya-01/Moya01-V1.urdf"
    print_robot_physics(xml_file)