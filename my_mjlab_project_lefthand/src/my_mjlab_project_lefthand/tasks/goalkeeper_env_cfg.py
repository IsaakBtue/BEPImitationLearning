"""Goalkeeper environment configuration for MuJoCo Lab — left-hand specialization."""
from __future__ import annotations

import mujoco
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from my_mjlab_project_lefthand.mdp import MultiMotionCommandCfg
import my_mjlab_project_lefthand.mdp.observations as gk_obs
import my_mjlab_project_lefthand.mdp.rewards as gk_rew
import my_mjlab_project_lefthand.mdp.resets as gk_resets

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"))
_FEET_CFG = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))

import my_mjlab_project as _base_pkg
_MOTION_FILE = str(Path(_base_pkg.__file__).parent / "motions" / "data" / "lefthand.npz")


def get_ball_spec(radius: float = 0.11, mass: float = 0.42) -> mujoco.MjSpec:
    """Soccer ball as a free-floating sphere with proper bounciness."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius, 0.0, 0.0),
        mass=mass,
        rgba=(1.0, 0.5, 0.0, 1.0),
    )
    return spec


def goalkeeper_env_cfg(play: bool = False):
    """Goalkeeper task: G1 humanoid blocks a soccer ball using pose-tracking rewards."""
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    # Remove motion tracking obs — not used by this policy
    for _obs in ("command", "motion_anchor_pos_b", "motion_anchor_ori_b"):
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    cfg.sim.nconmax = 100
    cfg.sim.njmax = 500

    cfg.scene.entities = {
        "robot": get_g1_robot_cfg(),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }
    cfg.scene.num_envs = 1020

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    cfg.commands["motion"] = MultiMotionCommandCfg(
        entity_name="robot",
        motion_files=(_MOTION_FILE,),
        ball_name="ball",
        anchor_body_name="torso_link",
        body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={} if play else {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={} if play else {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(0.0, 0.0) if play else (-0.1, 0.1),
        sampling_mode="start" if play else "uniform",
    )

    actor_extra = {
        "ball_pos_b": ObservationTermCfg(
            func=gk_obs.ball_pos_b,
            params={"ball_name": "ball"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "ball_vel_b": ObservationTermCfg(
            func=gk_obs.ball_vel_b,
            params={"ball_name": "ball"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "left_hand_pos_b": ObservationTermCfg(
            func=gk_obs.left_hand_pos_b,
            params={"asset_cfg": _HAND_CFG},
        ),
        "right_hand_pos_b": ObservationTermCfg(
            func=gk_obs.right_hand_pos_b,
            params={"asset_cfg": _HAND_CFG},
        ),
    }
    cfg.observations["actor"].terms.update(actor_extra)
    cfg.observations["critic"].terms.update(actor_extra)

    cfg.rewards.update({
        "eereach": RewardTermCfg(
            func=gk_rew.eereach,
            weight=10.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "reach_th": 0.3},
        ),
        "catch_success": RewardTermCfg(
            func=gk_rew.catch_success,
            weight=5.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "catch_th": 0.3},
        ),
        "stopball": RewardTermCfg(
            func=gk_rew.stopball,
            weight=2.0,
            params={"ball_name": "ball"},
        ),
        "stayonline": RewardTermCfg(
            func=gk_rew.stayonline,
            weight=-2.0,
        ),
        "noretreat": RewardTermCfg(
            func=gk_rew.noretreat,
            weight=-2.0,
        ),
        "feetorientation": RewardTermCfg(
            func=gk_rew.feetorientation,
            weight=3.0,
            params={"asset_cfg": _FEET_CFG},
        ),
        "postorientation": RewardTermCfg(
            func=gk_rew.postorientation,
            weight=3.0,
            params={"ball_name": "ball"},
        ),
        "postangvel": RewardTermCfg(
            func=gk_rew.postangvel,
            weight=3.0,
            params={"ball_name": "ball"},
        ),
        "postlinvel": RewardTermCfg(
            func=gk_rew.postlinvel,
            weight=1.0,
            params={"ball_name": "ball"},
        ),
    })

    cfg.episode_length_s = 5.0 if not play else 1.0e9

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot[1-7]_collision$"
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)

    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def _make_play_base(*, keep_motion_command: bool):
    """Shared setup for both play variants."""
    from mjlab.managers.event_manager import EventTermCfg

    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0

    if not keep_motion_command:
        cfg.commands.pop("motion", None)

    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",
        params={"ball_name": "ball"},
    )

    for _obs in ("robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
                 "robot_body_ang_vel_b", "body_pos", "body_ori"):
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    for _rew in ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                 "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"):
        cfg.rewards.pop(_rew, None)

    for _term in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        cfg.terminations.pop(_term, None)

    return cfg


def goalkeeper_play_env_cfg():
    """Play env: autonomous ball reset, motion command removed."""
    return _make_play_base(keep_motion_command=False)


def goalkeeper_play_withoverlay_env_cfg():
    """Play mode config that KEEPS motion command for overlay visualization."""
    return _make_play_base(keep_motion_command=True)
