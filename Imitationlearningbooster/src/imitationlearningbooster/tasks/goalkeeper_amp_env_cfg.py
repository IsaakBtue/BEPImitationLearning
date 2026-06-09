"""Goalkeeper AMP environment config for Booster T1 (mjlab).

This config removes motion tracking rewards (not available at inference) and adds AMP
discriminator observations. The task focuses on ball interception while the AMP module
ensures motion naturalness via adversarial motion priors.
"""
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import mjlab.envs.mdp as mjlab_mdp
import mjlab.envs.mdp.observations as mjlab_obs

from imitationlearningbooster.mdp import MultiMotionCommandCfg
import imitationlearningbooster.mdp.observations as gk_obs
import imitationlearningbooster.mdp.rewards as gk_rew
import imitationlearningbooster.mdp.resets as gk_resets
from imitationlearningbooster.mdp.resets import adaptive_curriculum_update
from imitationlearningbooster.robots.t1_constants import get_t1_robot_cfg, T1_ACTION_SCALE
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

_DATA_DIR = Path(__file__).parent.parent / "motions" / "data"
_MOTION_FILES = tuple(
    str(_DATA_DIR / f"{name}_t1.npz")
    for name in ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
)

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_hand_link", "right_hand_link"))
_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_KNEE_BODY_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
_ARM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_WAIST_JOINT_CFG = SceneEntityCfg("robot", joint_names=("Waist",))
_ALL_JOINT_CFG = SceneEntityCfg("robot")


def get_ball_spec(radius: float = 0.11, mass: float = 0.42) -> mujoco.MjSpec:
    """Create ball entity specification with physics properties."""
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


def goalkeeper_amp_env_cfg(play: bool = False, num_steps_per_env: int = 100) -> ManagerBasedRlEnvCfg:
    """Create AMP-augmented goalkeeper environment config.

    Args:
        play: If True, disable corruption, remove events, extend episode length.
        num_steps_per_env: Steps per environment per rollout call.

    Returns:
        ManagerBasedRlEnvCfg configured for AMP training.
    """
    cfg = make_tracking_env_cfg()

    # Remove all domain randomization events.
    # The base tracking config injects: base_com (COM offset), encoder_bias (joint bias),
    # foot_friction (friction randomization), and push_robot (velocity perturbations).
    # None of these are needed for the goalkeeper task; removing them keeps training
    # stable and avoids confounding DR effects during early policy learning.
    _dr_events = ["base_com", "encoder_bias", "foot_friction", "push_robot"]
    for _ev in _dr_events:
        cfg.events.pop(_ev, None)

    cfg.scene.entities["robot"] = get_t1_robot_cfg()
    cfg.scene.entities["ball"] = EntityCfg(spec_fn=get_ball_spec)

    # Fix 1.6: set num_envs to match baseline
    cfg.scene.num_envs = 6144

    # Fix 1.5 + Fix 1.7: contact sensors and sim contact capacity
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="feet_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left|right)_foot_[12]$",
                entity="robot",
            ),
            secondary=None,
            fields=("found", "force"),
            reduce="netforce",
            history_length=0,
        ),
    )

    # Fix 1.7: sim contact capacity for ball contact simulation
    cfg.sim.nconmax = 100
    cfg.sim.njmax = 500

    # Fix 1.8: viewer body tracking
    cfg.viewer.body_name = "Trunk"

    # Motion command: 6 motions for AMP discriminator variety
    cfg.commands["motion"] = MultiMotionCommandCfg(
        entity_name="robot",
        motion_files=_MOTION_FILES,
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
        debug_vis=False,
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
        sampling_mode="start",
        static_partition=True,
    )

    # Action scale
    joint_pos_action = cfg.actions["joint_pos"]
    joint_pos_action.scale = T1_ACTION_SCALE

    # Remove motion tracking observations (not available at inference)
    _tracking_obs = [
        "robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
        "robot_body_ang_vel_b", "body_pos", "body_ori",
        "command", "motion_anchor_pos_b", "motion_anchor_ori_b"
    ]
    for obs_key in _tracking_obs:
        cfg.observations["actor"].terms.pop(obs_key, None)
        cfg.observations["critic"].terms.pop(obs_key, None)

    # Replace IMU-based obs with direct state reads (T1 has no IMU)
    _robot_cfg = SceneEntityCfg("robot")
    for group in cfg.observations.values():
        group.terms["base_lin_vel"] = ObservationTermCfg(
            func=mjlab_obs.base_lin_vel, params={"asset_cfg": _robot_cfg},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        group.terms["base_ang_vel"] = ObservationTermCfg(
            func=mjlab_obs.base_ang_vel, params={"asset_cfg": _robot_cfg},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        group.terms["joint_vel"] = ObservationTermCfg(
            func=mjlab_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

    # Remove all motion tracking rewards — AMP handles posture and arm naturalness.
    # AMP discriminator sees all 23 joint positions (including arms) across 2 consecutive
    # frames, so it learns to detect unnatural joint velocity patterns. This mirrors the
    # upstream G1 design (num_obs=29*2=58) where AMP is the sole arm guidance mechanism.
    motion_reward_keys = [
        "motion_global_root_pos", "motion_global_root_ori",
        "motion_body_ori",
        "motion_body_lin_vel", "motion_body_ang_vel",
        "motion_body_pos",
    ]
    for key in motion_reward_keys:
        cfg.rewards.pop(key, None)

    _robot_cfg_all = SceneEntityCfg("robot")
    cfg.rewards.update({
        "eereach": RewardTermCfg(
            func=gk_rew.eereach, weight=20.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "reach_th": 0.2},
        ),
        "hand_proximity_strict": RewardTermCfg(
            func=gk_rew.hand_proximity_strict, weight=10.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "strict_th": 0.15},
        ),
        "stopball": RewardTermCfg(
            func=gk_rew.stopball, weight=100.0,
            params={"ball_name": "ball"},
        ),
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
        # Fix 1.3: missing reward terms from baseline goalkeeper_env_cfg.py
        "successland": RewardTermCfg(
            func=gk_rew.successland, weight=4.0,
            params={"ball_name": "ball", "asset_cfg": _FEET_CFG},
        ),
        "penalize_sharpcontact": RewardTermCfg(
            func=gk_rew.penalize_sharpcontact, weight=-100.0,
        ),
        "penalize_kneeheight": RewardTermCfg(
            func=gk_rew.penalize_kneeheight, weight=-100.0,
            params={"asset_cfg": _KNEE_BODY_CFG},
        ),
        "penalize_self_collision": RewardTermCfg(
            func=gk_rew.penalize_self_collision, weight=-50.0,
        ),
        "feet_slippage": RewardTermCfg(
            func=gk_rew.feet_slippage, weight=3.0,
            params={"asset_cfg": _FEET_CFG},
        ),
        "postupperdofpos": RewardTermCfg(
            func=gk_rew.postupperdofpos, weight=1.0,
            params={"ball_name": "ball", "asset_cfg": _ARM_JOINT_CFG},
        ),
        "postwaistdofpos": RewardTermCfg(
            func=gk_rew.postwaistdofpos, weight=1.0,
            params={"ball_name": "ball", "asset_cfg": _WAIST_JOINT_CFG},
        ),
        "dof_acc": RewardTermCfg(
            func=mjlab_mdp.joint_acc_l2, weight=-2.5e-7,
            params={"asset_cfg": _robot_cfg_all},
        ),
        "action_rate_l2": RewardTermCfg(func=mjlab_mdp.action_acc_l2, weight=-0.1),
        "torques": RewardTermCfg(
            func=gk_rew.torques_normalized_l2, weight=-1e-5,
            params={"asset_cfg": _robot_cfg_all},
        ),
        "dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2, weight=-5e-4,
            params={"asset_cfg": _robot_cfg_all},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mjlab_mdp.joint_pos_limits, weight=-3.0,
            params={"asset_cfg": _ALL_JOINT_CFG},
        ),
        "dof_vel_limits": RewardTermCfg(
            func=gk_rew.dof_vel_limits, weight=-2.0,
            params={"asset_cfg": _ALL_JOINT_CFG},
        ),
        "torque_limits": RewardTermCfg(
            func=gk_rew.torque_limits, weight=-3.0,
            params={"asset_cfg": _ALL_JOINT_CFG},
        ),
        "deviation_waist_joint": RewardTermCfg(
            func=gk_rew.deviation_waist_joint, weight=-0.001,
            params={"asset_cfg": _WAIST_JOINT_CFG},
        ),
        "ang_vel_xy": RewardTermCfg(func=gk_rew.base_ang_vel_xy_l2, weight=-0.1),
    })

    # Goalkeeper observations: ball and hands
    # Fix 1.1: add left_hand_pos_b and right_hand_pos_b (required for 870-dim compatibility)
    _gk_extra_obs = {
        "ball_pos_b": ObservationTermCfg(
            func=gk_obs.ball_pos_b,
            noise=Unoise(-0.05, 0.05),
            params={"ball_name": "ball"}
        ),
        "ball_vel_b": ObservationTermCfg(
            func=gk_obs.ball_vel_b,
            noise=Unoise(-0.1, 0.1),
            params={"ball_name": "ball"}
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
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms.update(_gk_extra_obs)

    # Observation history: 10 frames (matches original G1 AMP)
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 10

    # AMP observation group: single frame of raw joint positions per step, no corruption.
    # history_length=1 → 23 dims per step. The runner (him_amp_runner.py) concatenates
    # current + previous step AMP obs to produce the 46-dim (23*2) discriminator input,
    # matching upstream G1's num_obs=29*2=58 design. Setting history_length=2 here would
    # double-stack the frames (obs_group gives 46, runner adds another 46 → 92 dims),
    # breaking the normalizer which expects 46.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "amp_obs": ObservationTermCfg(func=gk_obs.joint_pos_amp, noise=None)
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )

    # Observation delay in training (0–2 steps = 0–40 ms at 50 Hz)
    if not play:
        for term_cfg in cfg.observations["actor"].terms.values():
            term_cfg.delay_min_lag = 0
            term_cfg.delay_max_lag = 2
            term_cfg.delay_per_env = True

    # Terminations: remove motion-tracking-specific terminations
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("ee_body_pos", None)

    cfg.terminations.update({
        "bad_orientation": TerminationTermCfg(
            func=mjlab_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
            time_out=False,
        ),
        "base_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.4},
            time_out=False,
        ),
        # Fix 1.4: sharpforce_termination (mirrors upstream sharpforce_buf > 1500 N at feet)
        "sharpforce": TerminationTermCfg(
            func=gk_resets.sharpforce_termination,
            params={"max_contact_force": 1500.0},
            time_out=False,
        ),
    })

    if play:
        cfg.curriculum = {}
    else:
        stage1_step = 600 * num_steps_per_env
        stage2_step = 1200 * num_steps_per_env
        cfg.curriculum = {
            "stopball_curriculum": CurriculumTermCfg(
                func=mjlab_mdp.reward_curriculum,
                params={
                    "reward_name": "stopball",
                    # G1 peak: 100 * (1 + 0.5*3) = 250 at curriculumupdate=3.
                    # Three stages approximate the continuous G1 scaling.
                    "stages": [
                        {"step": 0,           "weight": 100.0},
                        {"step": stage1_step, "weight": 175.0},
                        {"step": stage2_step, "weight": 250.0},
                    ],
                },
            ),
            "eereach_curriculum": CurriculumTermCfg(
                func=mjlab_mdp.reward_curriculum,
                params={
                    "reward_name": "eereach",
                    # Doubled from (10/15/20): at 10 the arm signal was overwhelmed
                    # by AMP + regularisation terms before 2k iterations.
                    "stages": [
                        {"step": 0,           "weight": 20.0},
                        {"step": stage1_step, "weight": 28.0},
                        {"step": stage2_step, "weight": 36.0},
                    ],
                },
            ),
            "hand_proximity_strict_curriculum": CurriculumTermCfg(
                func=mjlab_mdp.reward_curriculum,
                params={
                    "reward_name": "hand_proximity_strict",
                    # Doubled from (5/7.5/10) for same reason as eereach.
                    "stages": [
                        {"step": 0,           "weight": 10.0},
                        {"step": stage1_step, "weight": 15.0},
                        {"step": stage2_step, "weight": 20.0},
                    ],
                },
            ),
            "adaptive_curriculum": CurriculumTermCfg(
                # Mirrors G1 legged_robot.py reset_idx curriculum driver exactly:
                #   curriculumupdate = int(mean(episode_length_buf[env_ids]) / 50)
                # Updates every ~500 sim steps (min_gate). Sets env._curriculumupdate
                # (0→3) and env._ball_difficulty (0→1) used by _reset_ball and eereach.
                func=adaptive_curriculum_update,
                params={"min_gate": 500},
            ),
            "dof_pos_limits_curriculum": CurriculumTermCfg(
                func=mjlab_mdp.reward_curriculum,
                params={
                    "reward_name": "dof_pos_limits",
                    "stages": [
                        {"step": 0,           "weight": -3.0},
                        {"step": stage1_step, "weight": -6.0},
                        {"step": stage2_step, "weight": -9.0},
                    ],
                },
            ),
            "torque_limits_curriculum": CurriculumTermCfg(
                func=mjlab_mdp.reward_curriculum,
                params={
                    "reward_name": "torque_limits",
                    "stages": [
                        {"step": 0,           "weight": -3.0},
                        {"step": stage1_step, "weight": -6.0},
                        {"step": stage2_step, "weight": -9.0},
                    ],
                },
            ),
        }
    cfg.episode_length_s = 1e9 if play else 3.0
    if play:
        cfg.observations["actor"].enable_corruption = False

    return cfg


def goalkeeper_amp_play_env_cfg(num_steps_per_env: int = 100) -> ManagerBasedRlEnvCfg:
    """Play config for AMP goalkeeper: 1 env, auto-reset, no motion file required."""
    from imitationlearningbooster.tasks.goalkeeper_env_cfg import get_axes_spec

    cfg = goalkeeper_amp_env_cfg(play=True, num_steps_per_env=num_steps_per_env)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    cfg.scene.entities["axes"] = EntityCfg(spec_fn=get_axes_spec)

    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",
        params={"ball_name": "ball"},
    )

    for _term in ["anchor_pos", "anchor_ori", "ee_body_pos"]:
        cfg.terminations.pop(_term, None)

    return cfg


def goalkeeper_amp_play_withoverlay_env_cfg(num_steps_per_env: int = 100) -> ManagerBasedRlEnvCfg:
    """Play config for AMP goalkeeper with ghost-robot overlay cycling through all 6 motions."""
    cfg = goalkeeper_amp_play_env_cfg(num_steps_per_env=num_steps_per_env)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    motion_cmd.debug_vis = True
    motion_cmd.cycle_motions = True

    return cfg
