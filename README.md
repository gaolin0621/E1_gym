# E1 12-DOF Visual AMP Training

This directory contains the standalone E1 12-DOF visual parkour training task.

## Environment

The verified local environment uses Python 3.8, PyTorch 1.13.1,
torchvision 0.14.1, CUDA 11.7 and Isaac Gym Preview 4. Install Isaac Gym
manually first, then install this project from its source directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Use an NVIDIA driver compatible with the CUDA runtime used by PyTorch. The
project uses NVIDIA Warp for batched depth ray casting.

## Training

Run from the repository root:

```bash
python humanoid/scripts/train.py --task e1_12dof_amp_vdwl --headless
```

The task is also the default, so `--task` may be omitted. Logs are written to
`logs/e1_12dof_amp_vdwl/`.

The AMP discriminator currently trains from:

- `humanoid/envs/datasets/e1/txt_v1/walk1_100hz.txt`
- `humanoid/envs/datasets/e1/txt_v1/stair1_cut_100hz.txt`

Both motion files are 100 Hz. The loader samples them with a two-frame stride
to match the 50 Hz policy step.

## Play and export

```bash
python humanoid/scripts/play_amp_VDWL.py --task e1_12dof_amp_vdwl --num_envs 1
```

Checkpoint selection follows the runner settings in
`humanoid/envs/e1_12dof_vision/e1_12dof_amp_vdwl_cfg.py` or the corresponding
command-line resume arguments.
