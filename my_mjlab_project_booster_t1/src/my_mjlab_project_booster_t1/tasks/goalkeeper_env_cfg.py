"""Goalkeeper environment configuration for Booster T1."""
from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
import mjlab.envs.mdp.observations as mjlab_obs
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from my_mjlab_project_booster_t1.mdp import MultiMotionCommandCfg
import my_mjlab_project_booster_t1.mdp.observations as gk_obs
import my_mjlab_project_booster_t1.mdp.rewards as gk_rew
import my_mjlab_project_booster_t1.mdp.resets as gk_resets
from my_mjlab_project_booster_t1.robots.t1_constants import get_t1_robot_cfg, T1_ACTION_SCALE

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_hand_link", "right_hand_link"))
_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def get_ball_spec(radius: float = 0.11, mass: float = 0.42) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius, 0.0, 0.0),
        mass=mass,
        rgba=(1.0, 1.0, 0.0, 1.0),
    )
    geom.friction = (0.4, 0.005, 0.0001)
    geom.solref = [0.002, 0.0001]
    geom.solimp = [0.0001, 0.001, 0.0001, 0.5, 2.0]
    geom.margin = 0.001
    geom.gap = 0.0001
    return spec


def goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {
        "robot": get_t1_robot_cfg(),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }
    cfg.scene.num_envs = 1020

    cfg.scene.sensors = (ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    ),)

    cfg.sim.nconmax = 100
    cfg.sim.njmax = 500

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE

    from pathlib import Path as _Path
    _data_dir = _Path(__file__).parent.parent / "motions" / "data"
    motion_files = (str(_data_dir / "lefthand_t1.npz"),)

    cfg.commands["motion"] = MultiMotionCommandCfg(
        entity_name="robot",
        motion_files=motion_files,
        ball_name="ball",
        anchor_body_name="Trunk",
        body_names=(
            "Trunk",
            "Hip_Roll_Left",
            "Shank_Left",
            "Ankle_Cross_Left",
            "Hip_Roll_Right",
            "Shank_Right",
            "Ankle_Cross_Right",
            "Waist",
            "AL2",
            "AL3",
            "left_hand_link",
            "AR2",
            "AR3",
            "right_hand_link",
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

    # Remove tracking observations: they're not available at play time,
    # so remove them from training to avoid distribution shift
    _tracking_obs = ["robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
                     "robot_body_ang_vel_b", "body_pos", "body_ori"]
    for _obs in _tracking_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    # Replace IMU-sensor-based observations with direct state reads.
    # The base tracking config uses builtin_sensor (needs robot/imu_lin_vel sensor),
    # but T1 has no IMU sensor — use direct simulation state instead.
    _robot_cfg = SceneEntityCfg("robot")
    for group in cfg.observations.values():
        group.terms["base_lin_vel"] = ObservationTermCfg(
            func=mjlab_obs.base_lin_vel, params={"asset_cfg": _robot_cfg}
        )
        group.terms["base_ang_vel"] = ObservationTermCfg(
            func=mjlab_obs.base_ang_vel, params={"asset_cfg": _robot_cfg}
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
        # ================================================================
        # Task rewards (from original isaacgym Humanoid-Goalkeeper)
        # ================================================================
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
            weight=100.0,  # ← CRITICAL FIX: was 2.0, now matches isaacgym's 100.0
            params={"ball_name": "ball"}
        ),

        # ================================================================
        # Stability/Balance rewards (from original isaacgym)
        # ================================================================
        "stayonline": RewardTermCfg(func=gk_rew.stayonline, weight=-2.0),
        "noretreat": RewardTermCfg(func=gk_rew.noretreat, weight=-2.0),
        "feetorientation": RewardTermCfg(
            func=gk_rew.feetorientation, weight=3.0, params={"asset_cfg": _FEET_CFG}
        ),
        "postorientation": RewardTermCfg(
            func=gk_rew.postorientation, weight=3.0, params={"ball_name": "ball"}
        ),
        "postangvel": RewardTermCfg(
            func=gk_rew.postangvel, weight=3.0, params={"ball_name": "ball"}
        ),
        "postlinvel": RewardTermCfg(
            func=gk_rew.postlinvel, weight=1.0, params={"ball_name": "ball"}
        ),
    })

    # ====================================================================
    # Override motion tracking weights from base config
    # (These come from make_tracking_env_cfg() in mjlab)
    # Replacing AMP with explicit motion tracking to learn the lefthand save
    # ====================================================================
    cfg.rewards["motion_global_root_pos"].weight = 10.0   # was 0.5 → Force sideways dive
    cfg.rewards["motion_global_root_ori"].weight = 5.0    # was 0.5 → Force rotation
    cfg.rewards["motion_body_pos"].weight = 10.0          # was 1.0 → Force body pose
    cfg.rewards["motion_body_ori"].weight = 3.0           # was 1.0
    cfg.rewards["motion_body_lin_vel"].weight = 3.0       # was 1.0
    cfg.rewards["motion_body_ang_vel"].weight = 3.0       # was 1.0

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_[12]$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot_link",
        "right_foot_link",
        "left_hand_link",
        "right_hand_link",
    )

    cfg.episode_length_s = 5.0 if not play else 1.0e9
    cfg.viewer.body_name = "Trunk"

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def goalkeeper_play_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    cfg.commands.pop("motion", None)

    from mjlab.managers.event_manager import EventTermCfg
    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",
        params={"ball_name": "ball"},
    )

    _tracking_obs = ["robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
                     "robot_body_ang_vel_b", "body_pos", "body_ori"]
    for _obs in _tracking_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    for _rew in ["motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                 "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"]:
        cfg.rewards.pop(_rew, None)

    for _term in ["anchor_pos", "anchor_ori", "ee_body_pos"]:
        cfg.terminations.pop(_term, None)

    return cfg


def goalkeeper_play_withoverlay_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0

    from mjlab.managers.event_manager import EventTermCfg
    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",
        params={"ball_name": "ball"},
    )

    _tracking_obs = ["robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
                     "robot_body_ang_vel_b", "body_pos", "body_ori"]
    for _obs in _tracking_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    # NOTE: Keep motion rewards in play_withoverlay so the overlay visualization works correctly.
    # The policy was trained with these rewards, and they're needed for RSI to initialize
    # the robot at the correct motion reference position each episode.
    # (Removing rewards but keeping motion command breaks consistency)

    for _term in ["anchor_pos", "anchor_ori", "ee_body_pos"]:
        cfg.terminations.pop(_term, None)

    return cfg
