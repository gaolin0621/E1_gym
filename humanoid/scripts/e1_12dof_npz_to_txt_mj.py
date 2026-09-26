"""Convert retargeted E1 12-DoF MuJoCo qpos NPZ files to AMP TXT files.

The output frame layout intentionally matches the X2 vision AMP dataset:

  root_pos_w(3), root_quat_w(4), dof_pos(12), dof_vel(12),
  root_lin_vel_b(3), root_ang_vel_b(3), body_pos_b(4*3),
  body_mat6_b(4*6)

The four tracked bodies are the left/right knees followed by the left/right
feet. Therefore an E1 frame has 73 values (the X2 frame has 69 because X2
has ten actuated joints).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "e1_12dof"
DEFAULT_OUTPUT = PROJECT_ROOT / "humanoid" / "envs" / "datasets" / "e1" / "txt_v1"
DEFAULT_XML = PROJECT_ROOT / "resources" / "robots" / "e1" / "mjcf" / "E1_12dof.xml"

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Keep the X2 order: two knees, then two feet. E1's foot rigid body is the
# ankle-roll link because the sole collision geometry is attached to it.
AMP_BODY_NAMES = (
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

FRAME_DIM = 3 + 4 + 12 + 12 + 3 + 3 + 4 * 3 + 4 * 6


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., (1, 2, 3, 0)]


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., (3, 0, 1, 2)]


def _normalise_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).copy()
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("Input contains a zero-length root quaternion")
    quat /= norm
    # q and -q are equivalent. A continuous sign sequence makes interpolation
    # deterministic and avoids a discontinuity in finite differences.
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat


def _resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] != 19:
        raise ValueError(f"Expected qpos shape (frames, 19), got {qpos.shape}")
    if len(qpos) < 2:
        raise ValueError("At least two qpos frames are required")
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("Input and output FPS must be positive")

    if np.isclose(input_fps, output_fps):
        result = np.asarray(qpos, dtype=np.float64).copy()
        result[:, 3:7] = _normalise_quaternions(result[:, 3:7])
        return result

    source_time = np.arange(len(qpos), dtype=np.float64) / input_fps
    last_time = source_time[-1]
    target_time = np.arange(
        int(np.floor(last_time * output_fps + 1.0e-9)) + 1,
        dtype=np.float64,
    ) / output_fps

    root_pos_and_joints = np.concatenate((qpos[:, :3], qpos[:, 7:]), axis=1)
    linear = interp1d(source_time, root_pos_and_joints, axis=0, kind="linear")(
        target_time
    )

    root_quat_wxyz = _normalise_quaternions(qpos[:, 3:7])
    rotations = Rotation.from_quat(_wxyz_to_xyzw(root_quat_wxyz))
    root_quat = _xyzw_to_wxyz(Slerp(source_time, rotations)(target_time).as_quat())

    return np.concatenate((linear[:, :3], root_quat, linear[:, 3:]), axis=1)


def _differentiate(values: np.ndarray, dt: float) -> np.ndarray:
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, dt, axis=0, edge_order=edge_order)


def _body_angular_velocity(rotations: Rotation, dt: float) -> np.ndarray:
    """Return angular velocity in the instantaneous root/body frame."""
    count = len(rotations)
    velocity = np.zeros((count, 3), dtype=np.float64)
    if count == 1:
        return velocity

    velocity[0] = (rotations[0].inv() * rotations[1]).as_rotvec() / dt
    velocity[-1] = -(
        rotations[-1].inv() * rotations[-2]
    ).as_rotvec() / dt
    for index in range(1, count - 1):
        previous = (rotations[index].inv() * rotations[index - 1]).as_rotvec()
        following = (rotations[index].inv() * rotations[index + 1]).as_rotvec()
        velocity[index] = (following - previous) / (2.0 * dt)
    return velocity


def _npz_joint_order(npz: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if "robot_joint_names" not in npz.files:
        raise ValueError("NPZ is missing robot_joint_names")
    return tuple(str(name) for name in npz["robot_joint_names"].tolist())


def _load_qpos(path: Path, input_fps_override: float | None) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as npz:
        if "qpos" not in npz.files:
            raise ValueError(f"{path} does not contain qpos")
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        source_names = _npz_joint_order(npz)
        if set(source_names) != set(JOINT_NAMES):
            missing = sorted(set(JOINT_NAMES) - set(source_names))
            extra = sorted(set(source_names) - set(JOINT_NAMES))
            raise ValueError(f"Joint set mismatch; missing={missing}, extra={extra}")

        # qpos columns follow robot_joint_names. Reorder explicitly so a file
        # with the same joints in another order cannot silently corrupt AMP.
        reorder = [source_names.index(name) for name in JOINT_NAMES]
        qpos = np.concatenate((qpos[:, :7], qpos[:, 7:][:, reorder]), axis=1)

        if input_fps_override is not None:
            input_fps = float(input_fps_override)
        elif "fps" in npz.files:
            input_fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
        else:
            raise ValueError(f"{path} does not contain fps; pass --input-fps")
    return qpos, input_fps


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        names.append(str(name))
    return tuple(names)


def _build_frames(model: mujoco.MjModel, qpos: np.ndarray, output_fps: float) -> np.ndarray:
    if model.nq != 19 or model.nv != 18 or model.nu != 12:
        raise ValueError(
            "E1_12dof.xml must expose nq=19, nv=18 and nu=12; "
            f"got nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    model_joint_names = _model_joint_names(model)
    if model_joint_names != JOINT_NAMES:
        raise ValueError(f"MJCF joint order mismatch: {model_joint_names} != {JOINT_NAMES}")

    body_ids = []
    for name in AMP_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MJCF is missing AMP body {name!r}")
        body_ids.append(body_id)

    root_quat_wxyz = _normalise_quaternions(qpos[:, 3:7])
    qpos = qpos.copy()
    qpos[:, 3:7] = root_quat_wxyz
    root_rotations = Rotation.from_quat(_wxyz_to_xyzw(root_quat_wxyz))
    dt = 1.0 / output_fps

    root_lin_vel_w = _differentiate(qpos[:, :3], dt)
    root_lin_vel_b = root_rotations.inv().apply(root_lin_vel_w)
    root_ang_vel_b = _body_angular_velocity(root_rotations, dt)
    joint_vel = _differentiate(qpos[:, 7:], dt)

    body_pos_b = np.empty((len(qpos), len(body_ids), 3), dtype=np.float64)
    body_mat6_b = np.empty((len(qpos), len(body_ids), 6), dtype=np.float64)
    data = mujoco.MjData(model)

    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        root_rotation = root_rotations[frame_index]
        positions_w = np.asarray(data.xpos[body_ids], dtype=np.float64).copy()
        quaternions_wxyz = np.asarray(data.xquat[body_ids], dtype=np.float64).copy()
        body_pos_b[frame_index] = root_rotation.inv().apply(
            positions_w - pose[None, :3]
        )

        body_rotations_w = Rotation.from_quat(_wxyz_to_xyzw(quaternions_wxyz))
        body_rotations_b = root_rotation.inv() * body_rotations_w
        # Match the online AMP observation exactly.  matrix_from_quat(...)[..., :2]
        # selects the first two COLUMNS of each 3x3 rotation matrix, resulting
        # in a [body, 3, 2] tensor before flattening.  Selecting [:, :2, :]
        # here would instead store the first two rows and gives the discriminator
        # an artificial expert/policy format difference.
        body_mat6_b[frame_index] = body_rotations_b.as_matrix()[:, :, :2].reshape(-1, 6)

    frames = np.concatenate(
        (
            qpos[:, :3],
            qpos[:, 3:7],
            qpos[:, 7:],
            joint_vel,
            root_lin_vel_b,
            root_ang_vel_b,
            body_pos_b.reshape(len(qpos), -1),
            body_mat6_b.reshape(len(qpos), -1),
        ),
        axis=1,
    )
    if frames.shape[1] != FRAME_DIM:
        raise AssertionError(f"Expected {FRAME_DIM} values per frame, got {frames.shape[1]}")
    if not np.isfinite(frames).all():
        raise ValueError("Converted motion contains NaN or Inf")
    return frames


def _write_txt(
    output_path: Path,
    frames: np.ndarray,
    output_fps: float,
    motion_weight: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_names = ", ".join(AMP_BODY_NAMES)
    motion_data = (
        "root_pos_w,root_quat_w,dof_pos,dof_vel,root_lin_vel_b,"
        "root_ang_vel_b, body_pos_b, body_mat6_b"
    )

    # Keep the same human-readable layout as the X2 files: metadata occupies
    # one line per field and every complete AMP frame occupies one line.
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        stream.write('  "LoopMode": "Wrap",\n')
        stream.write(f'  "FrameDuration": {1.0 / output_fps:.6f},\n')
        stream.write('  "EnableCycleOffsetPosition": true,\n')
        stream.write('  "EnableCycleOffsetRotation": true,\n')
        stream.write(f'  "body_names": {json.dumps(body_names)},\n')
        stream.write(f'  "motion data": {json.dumps(motion_data)},\n')
        stream.write(f'  "TotalTime": {len(frames) / output_fps:.5f},\n')
        stream.write(f'  "MotionWeight": {motion_weight:.1f},\n')
        stream.write('  "Frames": [\n')
        for index, frame in enumerate(frames):
            values = ", ".join(f"{value:.6f}" for value in frame)
            comma = "," if index + 1 < len(frames) else ""
            stream.write(f"    [{values}]{comma}\n")
        stream.write("  ]\n}\n")


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError(f"Input file must be NPZ: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    files = sorted(input_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {input_path}")
    return files


def _output_file(output_path: Path, source: Path, output_fps: float, single: bool) -> Path:
    if single and output_path.suffix.lower() == ".txt":
        return output_path
    fps_label = f"{output_fps:g}".replace(".", "p")
    return output_path / f"{source.stem}_{fps_label}hz.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert E1 12-DoF qpos NPZ motion files to X2-style AMP TXT"
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"NPZ file or directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output TXT file or directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=DEFAULT_XML,
        help=f"12-DoF MJCF used for forward kinematics (default: {DEFAULT_XML})",
    )
    parser.add_argument(
        "--input-fps",
        "--input_fps",
        type=float,
        default=None,
        help="Override NPZ fps metadata",
    )
    parser.add_argument(
        "--output-fps",
        "--output_fps",
        type=float,
        default=None,
        help="Optional output frame rate; default preserves each NPZ fps",
    )
    parser.add_argument("--motion-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml_path.resolve()))
    sources = _input_files(args.input_path.resolve())
    output_path = args.output_path.resolve()

    print(
        f"[INFO] MJCF={args.xml_path.resolve()} "
        f"nq={model.nq} nv={model.nv} nu={model.nu} frame_dim={FRAME_DIM}"
    )
    for source in sources:
        qpos, input_fps = _load_qpos(source, args.input_fps)
        output_fps = input_fps if args.output_fps is None else float(args.output_fps)
        resampled_qpos = _resample_qpos(qpos, input_fps, output_fps)
        frames = _build_frames(model, resampled_qpos, output_fps)
        destination = _output_file(output_path, source, output_fps, len(sources) == 1)
        _write_txt(destination, frames, output_fps, args.motion_weight)
        print(
            f"[OK] {source.name}: {len(qpos)} @ {input_fps:g} Hz -> "
            f"{len(frames)} @ {output_fps:g} Hz, {frames.shape[1]} dims -> {destination}"
        )


if __name__ == "__main__":
    main()
