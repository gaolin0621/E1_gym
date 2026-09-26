"""Locally maintained E1 21-DOF robot asset."""

from .e1_21dof import E1_21DOF_ACTION_SCALE as E1_21DOF_ACTION_SCALE
from .e1_21dof import get_e1_21dof_robot_cfg as get_e1_21dof_robot_cfg

__all__ = ["E1_21DOF_ACTION_SCALE", "get_e1_21dof_robot_cfg"]
