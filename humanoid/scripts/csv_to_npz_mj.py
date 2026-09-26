import mujoco
import mujoco.viewer
import time
import csv
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')  # delimiter 可改为 '\t' 等
        header = next(reader)  # 如果有表头
        rows = [row for row in reader]

    try:
        data = np.array(rows, dtype=float)
    except ValueError:
        # 如果存在非数值项，退回到 object dtype
        data = np.array(rows, dtype=object)
    return header, data

def process_csv(model,data,csv_data):
    viewer = mujoco.viewer.launch_passive(model, data)
    body_pos = np.zeros((csv_data.shape[0],model.nbody - 1,3))
    body_quat = np.zeros((csv_data.shape[0],model.nbody - 1,4)) # xyzw
    for i in range(csv_data.shape[0]):
        frame = csv_data[i]
        data.qpos[:3] = frame[:3]
        data.qpos[2] -= 0.065
        quat = frame[3:7]
        quat = quat[[3,0,1,2]] # wxyz
        data.qpos[3:7] = quat
        data.qpos[7:] = frame[7:]
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[1:]
        body_quat[i] = data.xquat[1:]
        viewer.sync()
        # time.sleep(0.005)

    joint_pos = csv_data[:,7:]
    body_quat = body_quat[:,:,[1,2,3,0]] # xyzw
    return joint_pos,body_pos,body_quat

def process_joint_vel(joint_pos,sample_rate):
    num_frames = joint_pos.shape[0]

    if num_frames < 2:
        return np.zeros_like(joint_pos)

    dt = 1.0 / sample_rate
    joint_vel = []

    vel = (joint_pos[1] - joint_pos[0]) / dt
    joint_vel.append(vel)

    for i in range(1,num_frames - 1):
        vel = (joint_pos[i + 1] - joint_pos[i - 1]) / (2 * dt)
        joint_vel.append(vel)

    vel = (joint_pos[-1] - joint_pos[-2] / dt)
    joint_vel.append(vel)

    return np.array(joint_vel)

def process_body_pos_vel(body_pos_w,sample_rate):
    num_frames = body_pos_w.shape[0]
    if  num_frames < 2:
        return np.zeros_like(body_pos_w)

    dt = 1.0 / sample_rate
    body_vel = []

    vel = (body_pos_w[1] -  body_pos_w[0]) / dt
    body_vel.append(vel)

    for i in range(1, num_frames - 1):
        vel = (body_pos_w[i + 1] -  body_pos_w[i - 1]) / (2 * dt)
        body_vel.append(vel)

    vel = (body_pos_w[-1] -  body_pos_w[-2] / dt)
    body_vel.append(vel)

    return np.array(body_vel)

def _quaternion_inverse(q):
    """四元数求逆"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def _quaternion_multiply(q1, q2):
    """四元数乘法"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z,w])

def _quaternion_to_angular_velocity(q0, q1, dt):
    """
    从两个四元数计算角速度
    ω = 2 * log(q1 * q0⁻¹) / dt
    对于小旋转，近似为：ω ≈ 2 * (q1 - q0) / dt
    """
    # 简化实现：使用轴角表示
    # 实际实现可能需要更精确的四元数对数运算
    num_bodies = q0.shape[0]
    ang_vel = np.zeros((num_bodies, 3))
    
    for i in range(num_bodies):
        # 计算相对旋转：q_rel = q1 * q0⁻¹
        q0_inv = _quaternion_inverse(q0[i])
        q_rel =_quaternion_multiply(q1[i], q0_inv)
        
        # 四元数转轴角
        angle = 2 * np.arccos(np.clip(q_rel[3], -1.0, 1.0))  # w在索引3

        if angle < 1e-6:
            ang_vel[i] = np.zeros(3)
        else:
            axis = q_rel[:3] / np.sin(angle / 2)  # xyz在前三个索引
            ang_vel[i] = axis * angle / dt
    
    return ang_vel

def process_body_angular_velocity(body_quat_w,sample_rate):
    """计算body角速度（四元数差分）"""

    # body_quat_list =  body_quat_w
    num_frames = body_quat_w.shape[0]
    if num_frames < 2:
        return np.zeros((body_quat_w.shape[0],body_quat_w.shape[1],3))
    
    dt = 1.0 /  sample_rate
    ang_vel = []
    
    # 第一帧
    q0 =  body_quat_w[0]
    q1 =  body_quat_w[1]
    vel = _quaternion_to_angular_velocity(q0, q1, dt)
    ang_vel.append(vel)
    
    # 中间帧
    for i in range(1, num_frames - 1):
        q_prev =  body_quat_w[i - 1]
        q_next =  body_quat_w[i + 1]
        vel = _quaternion_to_angular_velocity(q_prev, q_next, 2 * dt)
        ang_vel.append(vel)
    
    # 最后一帧
    qn2 =  body_quat_w[-2]
    qn1 =  body_quat_w[-1]
    vel = _quaternion_to_angular_velocity(qn2, qn1, dt)
    ang_vel.append(vel)
    
    return np.array(ang_vel)

def _resample_linear_array(data_array: np.ndarray, 
                          original_times: np.ndarray, 
                          target_times: np.ndarray) -> np.ndarray:
    # 确保数据是2D或3D
    if data_array.ndim == 2:
        # [frames, joints] - 直接插值
        interpolator = interp1d(original_times, data_array, axis=0, 
                               kind='linear', fill_value='extrapolate')
        return interpolator(target_times)
    
    elif data_array.ndim == 3:
        # [frames, bodies, dim] - 对每个body单独插值
        num_bodies = data_array.shape[1]
        dim = data_array.shape[2]
        
        # 预分配结果数组
        result = np.zeros((len(target_times), num_bodies, dim))
        
        for i in range(num_bodies):
            body_data = data_array[:, i, :]
            interpolator = interp1d(original_times, body_data, axis=0,
                                   kind='linear', fill_value='extrapolate')
            result[:, i, :] = interpolator(target_times)
        
        return result
    
    else:
        raise ValueError(f"不支持的数组维度: {data_array.ndim}")

def _resample_quaternions_array(quat_array: np.ndarray,
                               original_times: np.ndarray,
                               target_times: np.ndarray) -> np.ndarray:
    num_bodies = quat_array.shape[1]
    
    # 预分配结果数组
    result = np.zeros((len(target_times), num_bodies, 4))
    
    for i in range(num_bodies):
        body_quats = quat_array[:, i, :]
        
        # 检查并修复无效四元数
        norms = np.linalg.norm(body_quats, axis=1)
        invalid_mask = norms < 1e-6
        
        
        # 确保所有四元数都单位化
        body_quats = body_quats / np.linalg.norm(body_quats, axis=1, keepdims=True)
        
        # 使用scipy的Slerp
        try:
            rotations = R.from_quat(body_quats)
            slerp = Slerp(original_times, rotations)
            resampled_rotations = slerp(target_times)
            result[:, i, :] = resampled_rotations.as_quat()
        except ValueError as e:
            print(f"Error in body {i}: {e}")
            print(f"Problematic quaternions shape: {body_quats.shape}")
            print(f"Norms: {np.linalg.norm(body_quats, axis=1)}")
            raise
    
    return result

def resample_data_array(data_array: np.ndarray, 
                       original_fps: float, 
                       target_fps: float,
                       is_quaternion: bool = False) -> np.ndarray:
    if original_fps == target_fps:
        return data_array
    
    # 计算时间轴
    num_frames = data_array.shape[0]
    original_duration = num_frames / original_fps
    target_num_frames = int(original_duration * target_fps)
    
    original_times = np.arange(num_frames) / original_fps
    target_times = np.arange(target_num_frames) / target_fps
    target_times = np.minimum(target_times, original_times[-1])
    
    if is_quaternion:
        # 四元数数据使用球面线性插值
        return _resample_quaternions_array(data_array, original_times, target_times)
    else:
        # 其他数据使用线性插值
        return _resample_linear_array(data_array, original_times, target_times)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",type=str,required=True)
    parser.add_argument("--input_fps",type=int,default=100)
    parser.add_argument("--output_fps",type=int,default=50)
    parser.add_argument("--output_path",type=str,required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)

    input_fps = args.input_fps
    output_fps = args.output_fps

    output_dir.mkdir(exist_ok=True,parents=True)
    csv_files = sorted(list(input_dir.glob("*.csv")))
    print(f"find {len(csv_files)} csv files")

    xml_path = "/home/liangzhiyuan/RL/IL/AMP_mjlab/src/assets/robots/droid_x3/xmls/scene_x3_14dof.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"link_{i}_name: {name}")
    for i in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"joint_{i}_name: {j_name}")

    for file in csv_files:
        output_file = f"{str(output_dir / file.stem)}.npz"
        # --- read csv data ---
        _,data_input = read_csv(file)
        # --- input_fps data ---
        joint_pos, body_pos, body_quat = process_csv(model, data, data_input)
        joint_vel = process_joint_vel(joint_pos,input_fps)
        body_pos_vel = process_body_pos_vel(body_pos,input_fps)
        body_quat_vel = process_body_angular_velocity(body_quat,input_fps)
        # --- output_fps data ---
        joint_pos = resample_data_array(joint_pos,input_fps,output_fps,False)
        body_pos = resample_data_array(body_pos,input_fps,output_fps,False)
        body_quat = resample_data_array(body_quat,input_fps,output_fps,True)

        joint_vel = resample_data_array(joint_vel,input_fps,output_fps,False)
        body_pos_vel = resample_data_array(body_pos_vel,input_fps,output_fps,False)
        body_quat_vel = resample_data_array(body_quat_vel,input_fps,output_fps,False)

        body_quat = body_quat[:,:,[3,0,1,2]] # xyzw -> wxyz

        np.savez(
            file=output_file,
            fps=output_fps,
            order="mujoco",
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_pos_vel,
            body_ang_vel_w=body_quat_vel
        )

if __name__ == "__main__":
    main()