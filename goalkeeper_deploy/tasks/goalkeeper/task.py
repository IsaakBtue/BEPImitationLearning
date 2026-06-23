"""Goalkeeper policy and task configuration for Booster T1.

Observation layout (87 dims per step, history=10 → flattened 870 → model input):
  [0:3]   base_lin_vel_b   – robot base linear velocity in body frame
  [3:6]   base_ang_vel_b   – robot base angular velocity in body frame
  [6:29]  joint_pos_rel    – joint_pos - training_default_pos  (23 joints)
  [29:52] joint_vel        – joint velocities                   (23 joints)
  [52:75] last_action      – raw actor output from previous step (23)
  [75:78] ball_pos_b       – ball position in robot body frame (zeroed when hidden)
  [78:81] ball_vel_b       – ball velocity in robot body frame (zeroed when hidden)
  [81:84] left_hand_pos_b  – left hand position in robot body frame
  [84:87] right_hand_pos_b – right hand position in robot body frame

Joint order (same as training MuJoCo XML and T1_23DOF_CFG.joint_names):
  AAHead_yaw, Head_pitch,
  Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
  Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
  Waist,
  Left_Hip_Pitch,  Left_Hip_Roll,  Left_Hip_Yaw,
  Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
  Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw,
  Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
"""
from __future__ import annotations

import math
import os
from dataclasses import MISSING
from pathlib import Path

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, PrepareStateCfg, RobotCfg,
)
from booster_deploy.utils.isaaclab.configclass import configclass

# --------------------------------------------------------------------
# Joint metadata
# --------------------------------------------------------------------

JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",  "Left_Hip_Roll",  "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# Training default pose (from T1_STANDING_KEYFRAME in t1_constants.py).
# Observations use  joint_pos - TRAINING_DEFAULT  so this must match training.
# Action targets =  TRAINING_DEFAULT + action * ACTION_SCALE.
TRAINING_DEFAULT_POS = [
    0.0,   # AAHead_yaw
    0.0,   # Head_pitch
    -0.21, # Left_Shoulder_Pitch
    -0.41, # Left_Shoulder_Roll
    0.0,   # Left_Elbow_Pitch
    0.0,   # Left_Elbow_Yaw
    -0.11, # Right_Shoulder_Pitch
    1.07,  # Right_Shoulder_Roll
    -0.13, # Right_Elbow_Pitch
    0.21,  # Right_Elbow_Yaw
    0.0,   # Waist
    -0.3,  # Left_Hip_Pitch
    0.0,   # Left_Hip_Roll
    0.0,   # Left_Hip_Yaw
    0.6,   # Left_Knee_Pitch
    -0.3,  # Left_Ankle_Pitch
    0.0,   # Left_Ankle_Roll
    -0.3,  # Right_Hip_Pitch
    0.0,   # Right_Hip_Roll
    0.0,   # Right_Hip_Yaw
    0.6,   # Right_Knee_Pitch
    -0.3,  # Right_Ankle_Pitch
    0.0,   # Right_Ankle_Roll
]

# Per-joint action scale from T1_ACTION_SCALE in t1_constants.py:
#   scale = 0.25 * effort / stiffness  (arms/legs)
#   scale = effort / stiffness          (head: no 0.25 factor)
ACTION_SCALE = [
    7.0 / 20.0,              # AAHead_yaw    = 0.350
    7.0 / 20.0,              # Head_pitch    = 0.350
    0.25 * 36.0 / 15.0,     # Left_Shoulder_Pitch  = 0.600
    0.25 * 36.0 / 15.0,     # Left_Shoulder_Roll   = 0.600
    0.25 * 36.0 / 15.0,     # Left_Elbow_Pitch     = 0.600
    0.25 * 36.0 / 15.0,     # Left_Elbow_Yaw       = 0.600
    0.25 * 36.0 / 15.0,     # Right_Shoulder_Pitch = 0.600
    0.25 * 36.0 / 15.0,     # Right_Shoulder_Roll  = 0.600
    0.25 * 36.0 / 15.0,     # Right_Elbow_Pitch    = 0.600
    0.25 * 36.0 / 15.0,     # Right_Elbow_Yaw      = 0.600
    0.25 * 40.0 / 80.0,     # Waist               = 0.125
    0.25 * 55.0 / 120.0,    # Left_Hip_Pitch       ≈ 0.1146
    0.25 * 40.0 / 80.0,     # Left_Hip_Roll        = 0.125
    0.25 * 40.0 / 80.0,     # Left_Hip_Yaw         = 0.125
    0.25 * 65.0 / 200.0,    # Left_Knee_Pitch      = 0.08125
    0.25 * 50.0 / 50.0,     # Left_Ankle_Pitch     = 0.250
    0.25 * 50.0 / 40.0,     # Left_Ankle_Roll      = 0.3125
    0.25 * 55.0 / 120.0,    # Right_Hip_Pitch      ≈ 0.1146
    0.25 * 40.0 / 80.0,     # Right_Hip_Roll       = 0.125
    0.25 * 40.0 / 80.0,     # Right_Hip_Yaw        = 0.125
    0.25 * 65.0 / 200.0,    # Right_Knee_Pitch     = 0.08125
    0.25 * 50.0 / 50.0,     # Right_Ankle_Pitch    = 0.250
    0.25 * 50.0 / 40.0,     # Right_Ankle_Roll     = 0.3125
]

# PD gains matching the training actuator stiffness/damping:
#   kp = stiffness
#   kd = 2 * 2.0 * sqrt(stiffness * armature)
#      = 2 * 2.0 * stiffness / (2π*10)   ≈ 0.06366 * stiffness
_K = 4.0 / (2.0 * math.pi * 10.0)   # = 0.06366

KP = [
    20.0,  20.0,               # head
    15.0,  15.0,  15.0,  15.0, # left arm
    15.0,  15.0,  15.0,  15.0, # right arm
    80.0,                      # waist
    120.0, 80.0,  80.0, 200.0, 50.0, 40.0,  # left leg
    120.0, 80.0,  80.0, 200.0, 50.0, 40.0,  # right leg
]
KD = [_K * k for k in KP]

EFFORT = [7, 7, 36, 36, 36, 36, 36, 36, 36, 36, 40, 55, 40, 40, 65, 50, 50, 55, 40, 40, 65, 50, 50]

# Prepare-state: stiffer gains + upright standing pose for entering custom mode
_PREPARE_KP = [40.0, 40.0, 40.0, 50.0, 20.0, 20.0, 40.0, 50.0, 20.0, 20.0, 350.0,
               350.0, 350.0, 180.0, 350.0, 350.0, 350.0,
               350.0, 350.0, 180.0, 350.0, 350.0, 350.0]
_PREPARE_KD = [0.65, 0.65, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2, 5.0,
               7.5, 7.5, 3.0, 5.5, 5.0, 5.0,
               7.5, 7.5, 3.0, 5.5, 5.0, 5.0]
_PREPARE_POS = [0, 0, 0, -1.4, 0, 0, 0, 1.4, 0, 0, 0,
                -0.1, 0, 0, 0.2, -0.1, 0,
                -0.1, 0, 0, 0.2, -0.1, 0]


# --------------------------------------------------------------------
# Quaternion helper (MuJoCo convention: [w, x, y, z])
# --------------------------------------------------------------------

def _quat_rot_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v from world frame to body frame using quaternion q=[w,x,y,z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    v0, v1, v2 = v[0], v[1], v[2]
    # R^T @ v  where R is the body-to-world rotation matrix
    r0 = (1 - 2*(y*y + z*z))*v0 + 2*(x*y + w*z)*v1 + 2*(x*z - w*y)*v2
    r1 = 2*(x*y - w*z)*v0 + (1 - 2*(x*x + z*z))*v1 + 2*(y*z + w*x)*v2
    r2 = 2*(x*z + w*y)*v0 + 2*(y*z - w*x)*v1 + (1 - 2*(x*x + y*y))*v2
    return torch.stack([r0, r1, r2])


# --------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------

class GoalkeeperPolicy(Policy):
    """Goalkeeper inference policy for Booster T1.

    Loads a TorchScript actor model, maintains a 10-step observation
    history, computes all required observations (including body-frame ball
    and hand positions), and returns joint position targets.
    """

    OBS_DIM = 87
    HISTORY = 10

    def __init__(self, cfg: "GoalkeeperPolicyCfg", controller: BaseController) -> None:
        super().__init__(cfg, controller)
        self.cfg = cfg

        # Resolve checkpoint path (absolute or relative to task directory)
        ckpt = cfg.checkpoint_path
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(self.task_path, ckpt)

        self._model: torch.jit.ScriptModule = torch.jit.load(ckpt, map_location="cpu")
        self._model.eval()

        self.default_pos = torch.tensor(TRAINING_DEFAULT_POS, dtype=torch.float32)
        self.action_scale = torch.tensor(ACTION_SCALE, dtype=torch.float32)

        self.obs_history: torch.Tensor = torch.zeros(
            self.HISTORY, self.OBS_DIM, dtype=torch.float32)
        self.last_action: torch.Tensor = torch.zeros(len(JOINT_NAMES), dtype=torch.float32)

    def reset(self) -> None:
        self.obs_history.zero_()
        self.last_action.zero_()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_obs(self) -> torch.Tensor:
        robot = self.controller.robot
        q = robot.data.root_quat_w      # (4,) [w,x,y,z]
        base_pos = robot.data.root_pos_w  # (3,)

        base_lin_vel = robot.data.root_lin_vel_b   # already body frame
        base_ang_vel = robot.data.root_ang_vel_b

        joint_pos_rel = robot.data.joint_pos - self.default_pos
        joint_vel = robot.data.joint_vel

        # Ball position and velocity in robot body frame
        # Apply visibility mask: zeros when ball is in warmup / out of view / random vanish
        ball_pos_w = self.controller.ball_pos_w
        ball_vel_w = self.controller.ball_vel_w
        vis = 1.0 if getattr(self.controller, 'ball_visible', True) else 0.0
        ball_pos_b = _quat_rot_inv(q, ball_pos_w - base_pos) * vis
        ball_vel_b = _quat_rot_inv(q, ball_vel_w) * vis

        # Hand positions in robot body frame
        lh_pos_b = _quat_rot_inv(q, self.controller.left_hand_pos_w - base_pos)
        rh_pos_b = _quat_rot_inv(q, self.controller.right_hand_pos_w - base_pos)

        return torch.cat([
            base_lin_vel,    # 3
            base_ang_vel,    # 3
            joint_pos_rel,   # 23
            joint_vel,       # 23
            self.last_action,  # 23
            ball_pos_b,      # 3
            ball_vel_b,      # 3
            lh_pos_b,        # 3
            rh_pos_b,        # 3
        ])  # total 87

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(self) -> torch.Tensor:
        obs = self._compute_obs().clamp(-100.0, 100.0)

        # Roll history and append current observation
        self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
        self.obs_history[-1] = obs

        with torch.no_grad():
            action = self._model(self.obs_history.flatten())
            action = action.clamp(-100.0, 100.0)

        self.last_action = action.clone()

        # Joint position targets
        targets = self.default_pos + action * self.action_scale
        return targets


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

@configclass
class GoalkeeperPolicyCfg(PolicyCfg):
    constructor = GoalkeeperPolicy
    checkpoint_path: str = "models/goalkeeper_t1_latest.pt"
    enable_safety_fallback: bool = True


@configclass
class GoalkeeperT1RobotCfg(RobotCfg):
    name: str = "Booster_T1_Goalkeeper"
    joint_names: list = JOINT_NAMES
    body_names: list = [
        "Trunk", "H1", "H2",
        "AL1", "AL2", "AL3", "left_hand_link",
        "AR1", "AR2", "AR3", "right_hand_link",
        "Waist",
        "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left",
        "Shank_Left", "Ankle_Cross_Left", "left_foot_link",
        "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right",
        "Shank_Right", "Ankle_Cross_Right", "right_foot_link",
    ]
    sim_joint_names: list = JOINT_NAMES   # identity mapping (training order = deploy order)
    sim_body_names: list = []
    joint_stiffness: list = KP
    joint_damping: list = KD
    effort_limit: list = EFFORT
    default_joint_pos: list = TRAINING_DEFAULT_POS
    mjcf_path: str = ""   # unused – GoalkeeperMujocoController builds scene internally
    prepare_state: PrepareStateCfg = PrepareStateCfg(
        stiffness=_PREPARE_KP,
        damping=_PREPARE_KD,
        joint_pos=_PREPARE_POS,
    )


@configclass
class GoalkeeperT1ControllerCfg(ControllerCfg):
    """Task configuration for T1 goalkeeper sim2sim deployment."""
    robot: GoalkeeperT1RobotCfg = GoalkeeperT1RobotCfg()
    policy: GoalkeeperPolicyCfg = GoalkeeperPolicyCfg()
    vel_command = None
    # policy_dt=0.02 (50 Hz), decimation=10 → physics_dt=0.002 s (500 Hz)
