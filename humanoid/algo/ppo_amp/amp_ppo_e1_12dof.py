"""E1 12-DOF specialization of the independent AMP PPO algorithm."""

from humanoid.algo.utils.motion_loader_e1_12dof_v1 import AMPLoader as E1_12DOFAMPLoader

from .amp_ppo import AmpPPO


class E1_12DOFAmpPPO(AmpPPO):
    """Use the E1 12-DOF/66-dimensional AMP data layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _motion_loader_cls=E1_12DOFAMPLoader, **kwargs)
