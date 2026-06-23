"""Goalkeeper task for T1 robot – registers goalkeeper_t1 in the task registry."""
from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass

from .task import GoalkeeperT1ControllerCfg
from .controller import GoalkeeperMujocoController


@configclass
class GoalkeeperT1Cfg(GoalkeeperT1ControllerCfg):
    """Goalkeeper policy: model_2000 checkpoint, MuJoCo sim2sim."""
    pass


# Override the controller constructor used by deploy.py when --mujoco flag is given.
# We monkey-patch the deploy dispatch by registering our custom controller in a
# post-init hook so it's selected instead of the standard MujocoController.
_cfg = GoalkeeperT1Cfg()
_cfg._mujoco_controller_cls = GoalkeeperMujocoController

register_task("goalkeeper_t1", _cfg)
