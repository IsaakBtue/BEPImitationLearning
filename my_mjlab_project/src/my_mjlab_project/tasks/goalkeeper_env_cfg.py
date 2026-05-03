"""Goalkeeper environment configuration for MuJoCo Lab."""
from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from my_mjlab_project.mdp import MultiMotionCommandCfg
import my_mjlab_project.mdp.observations as gk_obs
import my_mjlab_project.mdp.rewards as gk_rew
import my_mjlab_project.mdp.resets as gk_resets

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"))
_FEET_CFG = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))


def get_ball_spec(radius: float = 0.11, mass: float = 0.42) -> mujoco.MjSpec:
    """Soccer ball as a free-floating sphere with proper bounciness."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius, 0.0, 0.0),
        mass=mass,
        rgba=(1.0, 1.0, 0.0, 1.0),  # Yellow
    )
    geom.friction = (0.4, 0.005, 0.0001)  # slide, roll, spin friction
    # solref = [timeconst, dampratio]: stiff contact with zero damping for maximum elasticity
    geom.solref = [0.002, 0.0001]  # Stiff contact (0.002s), near-zero damping (0.0001) for extreme bounce
    # solimp = [dmin, dmax, dref, width, midpoint]: implicit contact damping
    geom.solimp = [0.0001, 0.001, 0.0001, 0.5, 2.0]
    # margin/gap control when contacts are generated and included
    geom.margin = 0.001  # Detect contacts at 1mm distance
    geom.gap = 0.0001   # Include contacts when dist < margin - gap
    return spec


def goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Goalkeeper task: G1 humanoid blocks a soccer ball using pose-tracking rewards."""
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    # Remove motion-reference observations so the policy trains and infers
    # from ball + joint state only — no motion file needed at play time.
    _motion_obs = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"]
    for _obs in _motion_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    # ------------------------------------------------------------------ #
    # MuJoCo model parameters (for ball collisions)
    # ------------------------------------------------------------------ #
    cfg.sim.nconmax = 100  # Contact buffer
    cfg.sim.njmax = 500    # Joint buffer

    # ------------------------------------------------------------------ #
    # Scene: add ball entity
    # ------------------------------------------------------------------ #
    cfg.scene.entities = {
        "robot": get_g1_robot_cfg(),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }
    cfg.scene.num_envs = 1020

    # ------------------------------------------------------------------ #
    # Action scale
    # ------------------------------------------------------------------ #
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    # ------------------------------------------------------------------ #
    # Command: replace MotionCommandCfg with MultiMotionCommandCfg
    # ------------------------------------------------------------------ #
    from pathlib import Path as _Path
    _data_dir = _Path(__file__).parent.parent / "motions" / "data"
    motion_files = tuple(
        str(_data_dir / f"{n}.npz")
        for n in ("lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep")
    )

    cfg.commands["motion"] = MultiMotionCommandCfg(
        entity_name="robot",
        motion_files=motion_files,
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

    # ------------------------------------------------------------------ #
    # Observations: add goalkeeper-specific ball / hand terms
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Rewards: add goalkeeper task + regularization terms
    # ------------------------------------------------------------------ #
    goalkeeper_rewards = {
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
    }
    cfg.rewards.update(goalkeeper_rewards)

    # ------------------------------------------------------------------ #
    # Episode length: ~5 s covers a full ball trajectory
    # ------------------------------------------------------------------ #
    cfg.episode_length_s = 5.0 if not play else 1.0e9

    # ------------------------------------------------------------------ #
    # DR / termination: inherit from tracking cfg
    # ------------------------------------------------------------------ #
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
    # Remove anchor terminations (incompatible with -0.39m z-correction applied to motion data)
    # Rely on goalkeeper rewards instead
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def goalkeeper_play_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1

    # Play mode settings: allow viewer to reset and re-randomize ball
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0  # Reset every 10 seconds for new ball trajectory

    # Remove motion command entirely — policy is 100% autonomous
    cfg.commands.pop("motion", None)

    # Add autonomous ball reset (runs on every episode reset)
    from mjlab.managers.event_manager import EventTermCfg
    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",
        params={"ball_name": "ball"},
    )

    # Remove all tracking observations (depend on motion data)
    _tracking_obs = ["robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b", "robot_body_ang_vel_b",
                     "body_pos", "body_ori"]
    for _obs in _tracking_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)
    # Remove all motion-dependent rewards (require command attributes that won't exist)
    _motion_rewards = ["motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                       "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"]
    for _rew in _motion_rewards:
        cfg.rewards.pop(_rew, None)
    # Remove all motion-dependent terminations (anchor tracking, body position bounds, etc)
    _motion_terminations = ["anchor_pos", "anchor_ori", "ee_body_pos"]
    for _term in _motion_terminations:
        cfg.terminations.pop(_term, None)
    return cfg
