"""Goalkeeper environment configuration for Booster T1."""
from __future__ import annotations

import math
import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
import mjlab.envs.mdp as mjlab_mdp
import mjlab.envs.mdp.dr as mjlab_dr
import mjlab.envs.mdp.observations as mjlab_obs
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
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


def get_axes_spec() -> mujoco.MjSpec:
    """World-frame axis markers for play-mode orientation reference.

    Red  = +X (robot slides laterally in this direction to intercept)
    Green = +Y (ball approaches FROM here — robot faces +Y)
    Blue  = +Z (up)

    Purely visual: contype=0, conaffinity=0, no physics.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="world_axes")
    body.pos = [0.0, 0.0, 0.0]

    def _arrow(name: str, end: list[float], rgba: tuple) -> None:
        g = body.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.fromto = [0.0, 0.0, 0.02] + end
        g.size = [0.03, 0.0, 0.0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0
        g.group = 1

    _arrow("axis_x", [1.5, 0.0, 0.02], (1.0, 0.2, 0.2, 0.9))   # red  → +X
    _arrow("axis_y", [0.0, 2.0, 0.02], (0.2, 1.0, 0.2, 0.9))   # green → +Y (ball)
    _arrow("axis_z", [0.0, 0.0, 0.52], (0.2, 0.4, 1.0, 0.9))   # blue  → +Z

    return spec


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


def goalkeeper_env_cfg(play: bool = False, num_steps_per_env: int = 24) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {
        "robot": get_t1_robot_cfg(),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }
    cfg.scene.num_envs = 6144

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
        sampling_mode="start",
    )

    # Remove tracking observations: they're not available at play time,
    # so remove them from training to avoid distribution shift.
    # Also remove motion-command reference obs (command, motion_anchor_pos/ori_b)
    # since the real robot won't have motion reference data at inference time.
    _tracking_obs = ["robot_body_pos_b", "robot_body_ori_b", "robot_body_lin_vel_b",
                     "robot_body_ang_vel_b", "body_pos", "body_ori",
                     "command", "motion_anchor_pos_b", "motion_anchor_ori_b"]
    for _obs in _tracking_obs:
        cfg.observations["actor"].terms.pop(_obs, None)
        cfg.observations["critic"].terms.pop(_obs, None)

    # Replace IMU-sensor-based observations with direct state reads.
    # base_ang_vel: actor + critic (G1 includes in actor obs, before num_one_step_obs cutoff).
    # base_lin_vel: critic-only (G1 places after cutoff — privileged_obs_buf only).
    _robot_cfg = SceneEntityCfg("robot")
    for group in cfg.observations.values():
        group.terms["base_ang_vel"] = ObservationTermCfg(
            func=mjlab_obs.base_ang_vel, params={"asset_cfg": _robot_cfg}
        )
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mjlab_obs.base_lin_vel, params={"asset_cfg": _robot_cfg}
    )

    # Actor gets ball_pos only — must infer trajectory from 10-step position history.
    # Matches G1: actor slice ends after actions (96 dims), ball_vel is after cutoff.
    actor_extra = {
        "ball_pos_b": ObservationTermCfg(
            func=gk_obs.ball_pos_b,
            params={"ball_name": "ball"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }
    # Critic-only privileged observations — mirrors G1's privileged_obs_buf.
    critic_extra = {
        "ball_vel_b": ObservationTermCfg(
            func=gk_obs.ball_vel_b,
            params={"ball_name": "ball"},
        ),
        "left_hand_pos_b": ObservationTermCfg(
            func=gk_obs.left_hand_pos_b,
            params={"asset_cfg": _HAND_CFG},
        ),
        "right_hand_pos_b": ObservationTermCfg(
            func=gk_obs.right_hand_pos_b,
            params={"asset_cfg": _HAND_CFG},
        ),
        "ball_intercept_pos_b": ObservationTermCfg(
            func=gk_obs.ball_intercept_pos_b,
            params={"ball_name": "ball"},
        ),
        "reach_dist_to_intercept": ObservationTermCfg(
            func=gk_obs.reach_dist_to_intercept,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG},
        ),
    }
    cfg.observations["actor"].terms.update(actor_extra)
    cfg.observations["critic"].terms.update(actor_extra)
    cfg.observations["critic"].terms.update(critic_extra)

    # Proprioceptive obs noise — from KaydenKnapik's deployed T1 running task.
    # Only applied to actor (critic gets clean state for better value estimates).
    _actor_noise = {
        "base_ang_vel":      Unoise(n_min=-0.2,  n_max=0.2),
        "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
        "joint_pos":         Unoise(n_min=-0.01, n_max=0.01),
        "joint_vel":         Unoise(n_min=-1.5,  n_max=1.5),
    }
    for _term, _noise in _actor_noise.items():
        if _term in cfg.observations["actor"].terms:
            cfg.observations["actor"].terms[_term].noise = _noise

    cfg.rewards.update({
        # ================================================================
        # Task rewards (from original isaacgym Humanoid-Goalkeeper)
        # ================================================================
        "eereach": RewardTermCfg(
            func=gk_rew.eereach,
            weight=10.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "reach_th": 0.2},
        ),
        "eereach_velmod": RewardTermCfg(
            func=gk_rew.eereach_velmod,
            weight=10.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "reach_th": 0.2},
        ),
        "hand_proximity_strict": RewardTermCfg(
            func=gk_rew.hand_proximity_strict,
            weight=5.0,
            params={"ball_name": "ball", "asset_cfg": _HAND_CFG, "strict_th": 0.15},
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

    # Second-order smoothness (jerk) penalty — matches original's _reward_smoothness.
    # Original weight is -0.1; our earlier -0.01 was 10x too weak, allowing jerk to balloon.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mjlab_mdp.action_acc_l2, weight=-0.1)

    # Regularization terms ported from original Humanoid-Goalkeeper.
    # These were all absent in our port; original used all of them for smooth stable motion.
    _robot_cfg_all = SceneEntityCfg("robot")
    cfg.rewards["dof_acc"] = RewardTermCfg(
        func=mjlab_mdp.joint_acc_l2, weight=-2.5e-7,
        params={"asset_cfg": _robot_cfg_all},
    )
    cfg.rewards["torques"] = RewardTermCfg(
        func=gk_rew.torques_normalized_l2, weight=-1e-5,
        params={"asset_cfg": _robot_cfg_all},
    )
    cfg.rewards["dof_vel"] = RewardTermCfg(
        func=mjlab_mdp.joint_vel_l2, weight=-5e-4,
        params={"asset_cfg": _robot_cfg_all},
    )
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
        func=gk_rew.base_ang_vel_xy_l2, weight=-0.1,
    )

    # ====================================================================
    # Gap fixes: rewards present in upstream Humanoid-Goalkeeper but
    # previously missing from this port.
    # ====================================================================

    # Gap 1 — Soft joint/torque/velocity limit penalties (-3.0 / -2.0 / -3.0)
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=mjlab_mdp.joint_pos_limits, weight=-3.0,
        params={"asset_cfg": _ALL_JOINT_CFG},
    )
    cfg.rewards["dof_vel_limits"] = RewardTermCfg(
        func=gk_rew.dof_vel_limits, weight=-2.0,
        params={"asset_cfg": _ALL_JOINT_CFG},
    )
    cfg.rewards["torque_limits"] = RewardTermCfg(
        func=gk_rew.torque_limits, weight=-3.0,
        params={"asset_cfg": _ALL_JOINT_CFG},
    )

    # Gap 2 — Safety: hard-contact / kneeling penalties (-100 each)
    cfg.rewards["penalize_sharpcontact"] = RewardTermCfg(
        func=gk_rew.penalize_sharpcontact, weight=-100.0,
    )
    cfg.rewards["penalize_kneeheight"] = RewardTermCfg(
        func=gk_rew.penalize_kneeheight, weight=-100.0,
        params={"asset_cfg": _KNEE_BODY_CFG},
    )

    # Gap 3 — Successland: reward safe landing after save (4.0)
    cfg.rewards["successland"] = RewardTermCfg(
        func=gk_rew.successland, weight=4.0,
        params={"ball_name": "ball", "asset_cfg": _FEET_CFG},
    )

    # Gap 5 — Post-save recovery: arms and waist return to neutral (1.0 each)
    cfg.rewards["postupperdofpos"] = RewardTermCfg(
        func=gk_rew.postupperdofpos, weight=1.0,
        params={"ball_name": "ball", "asset_cfg": _ARM_JOINT_CFG},
    )
    cfg.rewards["postwaistdofpos"] = RewardTermCfg(
        func=gk_rew.postwaistdofpos, weight=1.0,
        params={"ball_name": "ball", "asset_cfg": _WAIST_JOINT_CFG},
    )
    cfg.rewards["deviation_waist_joint"] = RewardTermCfg(
        func=gk_rew.deviation_waist_joint, weight=-0.001,
        params={"asset_cfg": _WAIST_JOINT_CFG},
    )

    # Gap 6 — Feet slippage: reward exp(-10*contactvel) matching upstream weight +3.0
    cfg.rewards["feet_slippage"] = RewardTermCfg(
        func=gk_rew.feet_slippage, weight=3.0,
        params={"asset_cfg": _FEET_CFG},
    )

    # ====================================================================
    # Domain randomization — matches KaydenKnapik's deployed T1 running task
    # (proven sim2real) plus goalkeeper-specific ball randomisation.
    # ====================================================================

    # Ball reset is handled by MultiMotionCommand._reset_ball — no separate event needed.

    # --- Robot DR (KaydenKnapik-validated, proven on real T1 hardware) ---

    # Encoder bias: simulates per-joint encoder offsets in real T1 hardware.
    cfg.events["encoder_bias"] = EventTermCfg(
        mode="startup",
        func=mjlab_dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.015, 0.015),
        },
    )

    # PD gain scaling: simulates actuator calibration uncertainty (±20%).
    # Matches G1's randomize_kp/kd [0.8, 1.2].
    cfg.events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=mjlab_dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # Push robot: upgraded to 6-DOF matching KaydenKnapik's deployed config.
    # Interval (3–8 s) fires occasionally within the 3 s episode.
    cfg.events["push_robot"].interval_range_s = (3.0, 8.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
    }

    # Foot friction: keep KaydenKnapik-validated [0.3, 1.2] — geom_names set below.

    # --- Ball DR (goalkeeper-specific) ---

    # Ball mass: ±20% around FIFA standard 0.42 kg, resampled each episode.
    cfg.events["ball_mass"] = EventTermCfg(
        mode="reset",
        func=mjlab_dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "ranges": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # Ball friction: randomizes how ball slides off hands/body (0.2–0.8).
    # Covers wet/dry ball, gloves vs bare skin, different ball coatings.
    cfg.events["ball_friction"] = EventTermCfg(
        mode="startup",
        func=mjlab_dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
            "operation": "abs",
            "ranges": (0.2, 0.8),
        },
    )

    # ====================================================================
    # Observation delay: 2–8 steps = 40–160 ms at 50 Hz.
    # KaydenKnapik deployed T1 with DELAY_MIN=2, DELAY_MAX=8 — matches real
    # hardware sensor latency. Not applied in play mode (hardware already has
    # its own latency; simulating it on top would double-count).
    # ====================================================================
    if not play:
        for term_cfg in cfg.observations["actor"].terms.values():
            term_cfg.delay_min_lag = 2
            term_cfg.delay_max_lag = 8
            term_cfg.delay_per_env = True

    # Scale task rewards up as training progresses, matching G1 curriculum strategy.
    # common_step_counter increments by 1 per env.step() call.
    # Curriculum thresholds scale with num_steps_per_env:
    # - stage 1 at iteration 600: 600 * num_steps_per_env
    # - stage 2 at iteration 1200: 1200 * num_steps_per_env
    stage1_step = 600 * num_steps_per_env
    stage2_step = 1200 * num_steps_per_env
    cfg.curriculum = {
        "stopball_curriculum": CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "stopball",
                "stages": [
                    {"step": 0,            "weight": 100.0},
                    {"step": stage1_step,  "weight": 150.0},
                    {"step": stage2_step,  "weight": 200.0},
                ],
            },
        ),
        "eereach_curriculum": CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "eereach",
                "stages": [
                    {"step": 0,            "weight": 10.0},
                    {"step": stage1_step,  "weight": 15.0},
                    {"step": stage2_step,  "weight": 20.0},
                ],
            },
        ),
        "eereach_velmod_curriculum": CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "eereach_velmod",
                "stages": [
                    {"step": 0,            "weight": 10.0},
                    {"step": stage1_step,  "weight": 15.0},
                    {"step": stage2_step,  "weight": 20.0},
                ],
            },
        ),
    }

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_[12]$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    # Remove all motion-tracking-specific terminations from the base config.
    # These check deviation from reference motion pose, not relevant for goalkeeper.
    # Observation history: G1 stacks 10 timesteps (num_actor_history=10).
    # mjlab applies per-group history_length uniformly to all terms.
    # Applied unconditionally (train + play) so the policy always sees the same input format.
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 10

    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("ee_body_pos", None)

    # Replace with the two G1-equivalent fall terminations:
    # 1. Tipping: matches G1's gravity_termination_buf (proj_grav XY norm > 0.8).
    #    XY norm > 0.8  →  |proj_grav_z| < 0.6  →  acos(0.6) ≈ 0.927 rad ≈ 53°.
    #    Use 1.0 rad (57°) to give the diving motion a small extra margin.
    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=mjlab_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    # 2. Collapse: terminate if trunk drops below 0.4 m (half standing height).
    #    Catches kneeling/folding collapses not caught by tilt alone.
    cfg.terminations["base_height"] = TerminationTermCfg(
        func=mjlab_mdp.root_height_below_minimum,
        params={"minimum_height": 0.4},
        time_out=False,
    )

    cfg.episode_length_s = 3.0 if not play else 1.0e9
    cfg.viewer.body_name = "Trunk"

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def goalkeeper_play_env_cfg(num_steps_per_env: int = 24) -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True, num_steps_per_env=num_steps_per_env)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    cfg.scene.entities["axes"] = EntityCfg(spec_fn=get_axes_spec)

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

    for _term in ["anchor_pos", "anchor_ori", "ee_body_pos"]:
        cfg.terminations.pop(_term, None)

    return cfg


def goalkeeper_play_withoverlay_env_cfg(num_steps_per_env: int = 24) -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True, num_steps_per_env=num_steps_per_env)
    cfg.scene.num_envs = 1
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    cfg.scene.entities["axes"] = EntityCfg(spec_fn=get_axes_spec)

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
