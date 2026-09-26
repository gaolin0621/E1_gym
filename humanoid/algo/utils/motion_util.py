# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for processing motion clips."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np

from humanoid.algo.utils import pose3d
from pybullet_utils import transformations


def standardize_quaternion(q):
  """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

  Args:
    q: A quaternion to be standardized.

  Returns:
    A quaternion with q.w >= 0.

  """
  if q[-1] < 0:
    q = -q
  return q


def normalize_rotation_angle(theta):
  """Returns a rotation angle normalized between [-pi, pi].

  Args:
    theta: angle of rotation (radians).

  Returns:
    An angle of rotation normalized between [-pi, pi].

  """
  norm_theta = theta
  if np.abs(norm_theta) > np.pi:
    norm_theta = np.fmod(norm_theta, 2 * np.pi)
    if norm_theta >= 0:
      norm_theta += -2 * np.pi
    else:
      norm_theta += 2 * np.pi

  return norm_theta


def calc_heading(q):
  """Returns the heading of a rotation q, specified as a quaternion.

  The heading represents the rotational component of q along the vertical
  axis (z axis).

  Args:
    q: A quaternion that the heading is to be computed from.

  Returns:
    An angle representing the rotation about the z axis.

  """
  ref_dir = np.array([1, 0, 0])
  rot_dir = pose3d.QuaternionRotatePoint(ref_dir, q)
  heading = np.arctan2(rot_dir[1], rot_dir[0])
  return heading


def calc_heading_rot(q):
  """Return a quaternion representing the heading rotation of q along the vertical axis (z axis).

  Args:
    q: A quaternion that the heading is to be computed from.

  Returns:
    A quaternion representing the rotation about the z axis.

  """
  heading = calc_heading(q)
  q_heading = transformations.quaternion_about_axis(heading, [0, 0, 1])
  return q_heading

def quat_rotate_inverse_array(q, v):
    """
    q: 四元数数组 [N, 4]，每行是 [x, y, z, w]
    v: 向量数组 [N, 3]，每行是 [vx, vy, vz]
    返回: 旋转后的向量数组 [N, 3]
    """
    q_vec = q[:, :3]  # [N, 3] - [x, y, z]
    q_w = q[:, 3]  # [N,] - w

    # 修正广播问题
    a = v * (2.0 * q_w ** 2 - 1.0)[:, np.newaxis]
    b = 2.0 * q_w[:, np.newaxis] * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=1)[:, np.newaxis]

    return a - b + c
