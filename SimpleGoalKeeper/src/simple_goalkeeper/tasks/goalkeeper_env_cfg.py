"""Goalkeeper environment configuration for Booster T1 (Phase 1 — feet only)."""
from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
import mjlab.envs.mdp as mjlab_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

from simple_goalkeeper.robots.t1_constants import (
    T1_ACTION_SCALE_HEADLESS,
    get_t1_headless_robot_cfg,
)
import simple_goalkeeper.mdp as gk_mdp

_BALL_XML = Path(__file__).parents[1] / "robots" / "xmls" / "ball.xml"
assert _BALL_XML.exists(), f"ball.xml not found at {_BALL_XML}"

BALL_NAME = "ball"

_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_ALL_JOINTS_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_WAIST_JOINT_CFG = SceneEntityCfg("robot", joint_names=("Waist",))
_ROBOT_CFG = SceneEntityCfg("robot")
_KNEE_BODY_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
_RECOVERY_ARM_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_RECOVERY_WAIST_CFG = SceneEntityCfg("robot", joint_names=("Waist",))


def _make_ball_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_BALL_XML)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.11),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
        ),
    )


def goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Booster T1 goalkeeper environment — Phase 1 (feet only, beyondAMP).

    Flat terrain, no commands, minimal DR. Ball always spawns in robot
    local +X frame so goalkeeper behavior is world-orientation-independent.
    """
    cfg = make_velocity_env_cfg()

    # Fix native viewer camera: ASSET_BODY origin requires entity_name + body_name.
    cfg.viewer.body_name = "Trunk"

    # ------------------------------------------------------------------
    # Scene: flat terrain, headless T1, add ball
    # ------------------------------------------------------------------
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove terrain/height sensors (flat terrain — nothing to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    )
    # Add contact sensors needed for sharpcontact/self-collision/slippage rewards.
    cfg.scene.sensors = cfg.scene.sensors + (
        ContactSensorCfg(
            name="feet_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left|right)_foot[1-4]_collision$",
                entity="robot",
            ),
            secondary=None,
            fields=("found", "force"),
            reduce="netforce",
            history_length=0,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )

    cfg.scene.entities["robot"] = get_t1_headless_robot_cfg()
    cfg.scene.entities[BALL_NAME] = _make_ball_entity_cfg()

    # Action scale: per-joint 0.25 * effort / stiffness.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE_HEADLESS

    # ------------------------------------------------------------------
    # Commands: none (no velocity command)
    # ------------------------------------------------------------------
    cfg.commands.clear()

    # ------------------------------------------------------------------
    # Curriculum: ramp ball difficulty 0→1 over training
    # Stages match upstream: stage1 at 600 iters, stage2 at 1200 iters
    # (num_steps_per_env=24 by default → same thresholds as Imitationlearningbooster).
    # ------------------------------------------------------------------
    _num_steps = 24
    # Reward-weight curriculum stages (fixed schedule, independent of ball difficulty).
    _stage1 = 1000 * _num_steps   # step 24000 → iter 1000
    _stage2 = 2000 * _num_steps   # step 48000 → iter 2000
    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["ball_difficulty"] = CurriculumTermCfg(
            func=gk_mdp.ball_difficulty_curriculum,
            params={
                # Adaptive curriculum: mirrors Humanoid-Goalkeeper G1 (legged_robot.py:325-336).
                # difficulty += step_size × int(mean_ep_len / ep_len_divisor)
                # every update_interval per-env steps. Longer episodes → faster advance.
                "update_interval": 500,   # per-env steps between updates (same as G1)
                "ep_len_divisor":  50,    # same divisor as G1
                "step_size":       0.01,  # difficulty units per curriculumupdate per check
                #   → at ep_len=144 curriculumupdate=2: reaches 1.0 in ~50 checks ≈ 1000 iters
            },
        )
        cfg.curriculum["stopball_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "stopball",
                "stages": [
                    {"step": 0,        "weight": 50.0},
                    {"step": _stage1,  "weight": 87.0},
                    {"step": _stage2,  "weight": 125.0},
                ],
            },
        )
        cfg.curriculum["softstop_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "softstop",
                "stages": [
                    {"step": 0,        "weight": 100.0},
                    {"step": _stage1,  "weight": 150.0},
                    {"step": _stage2,  "weight": 200.0},
                ],
            },
        )
        # NOTE: torque_limits and dof_pos_limits intentionally NOT in curriculum.
        # Tripling the torque penalty at the same time balls get harder (wider y range)
        # forces the robot into low-torque yaw strategies instead of lateral steps.
        # Keep both fixed at -3.0 throughout so the robot can use full joint effort
        # for aggressive saves at hard difficulty without extra penalty.
        cfg.curriculum["footreach_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "footreach",
                "stages": [
                    {"step": 0,        "weight": 10.0},
                    {"step": _stage1,  "weight": 15.0},
                    {"step": _stage2,  "weight": 20.0},
                ],
            },
        )

    # ------------------------------------------------------------------
    # Observations
    # Ball is always visible during Phase 1 training so the policy has
    # a clean signal from the start. Visibility gating (warmup + vanish)
    # is left for play/sim2real evaluation.
    # ------------------------------------------------------------------
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=gk_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mjlab_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mjlab_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=mjlab_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mjlab_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mjlab_mdp.last_action),
        "ball_pos_b": ObservationTermCfg(
            func=gk_mdp.ball_pos_b,
            params={"ball_name": BALL_NAME, "always_visible": not play},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "ball_vel_b": ObservationTermCfg(
            func=gk_mdp.ball_vel_b,
            params={"ball_name": BALL_NAME, "always_visible": not play},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "left_foot_pos_b": ObservationTermCfg(
            func=gk_mdp.left_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
        "right_foot_pos_b": ObservationTermCfg(
            func=gk_mdp.right_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
    }
    # Critic: same terms without noise.
    critic_terms = {
        k: ObservationTermCfg(func=v.func, params=v.params)
        for k, v in actor_terms.items()
    }

    cfg.observations["actor"] = ObservationGroupCfg(terms=actor_terms, enable_corruption=True)
    cfg.observations["critic"] = ObservationGroupCfg(terms=critic_terms, enable_corruption=False)

    # AMP discriminator uses absolute joint_pos + joint_vel to match NPZ convention.
    # NPZ stores raw dof_pos (absolute), so the robot obs must also be absolute.
    # Mirrors Imitationlearningbooster's joint_pos_amp approach.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "joint_pos": ObservationTermCfg(func=gk_mdp.joint_pos_abs, noise=None),
            "joint_vel": ObservationTermCfg(func=gk_mdp.joint_vel_abs, noise=None),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    # Observation history: 10 frames (matches G1 upstream num_actor_history=10).
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 10

    # Observation delay (training only): 0–2 steps = 0–40 ms at 50 Hz.
    if not play:
        for term_cfg in cfg.observations["actor"].terms.values():
            term_cfg.delay_min_lag = 0
            term_cfg.delay_max_lag = 2
            term_cfg.delay_per_env = True

    # ------------------------------------------------------------------
    # Rewards — aligned with Imitationlearningbooster proven structure.
    # stopball must come first: initialises env._sb_init_vx used by multiple terms.
    # ------------------------------------------------------------------
    cfg.rewards = {
        # --- primary task signal ---
        "stopball": RewardTermCfg(
            func=gk_mdp.stopball,
            weight=50.0,
            params={"ball_name": BALL_NAME, "delta_vel_threshold": 2.0},
        ),
        # --- partial deflection signal (fires before stopball; gates _ball_is_behind) ---
        "softstop": RewardTermCfg(
            func=gk_mdp.softstop,
            weight=100.0,
            params={"ball_name": BALL_NAME, "velocity_threshold": 0.2},
        ),
        # --- clean-trap bonus: ball nearly dead after deflection ---
        "cleanstop": RewardTermCfg(
            func=gk_mdp.cleanstop,
            weight=25.0,
            params={"ball_name": BALL_NAME, "speed_threshold": 0.25},
        ),
        # --- ball interception (feet-only) ---
        "footreach": RewardTermCfg(
            func=gk_mdp.footreach,
            weight=10.0,
            params={"ball_name": BALL_NAME, "reach_th": 0.3, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        "foot_proximity": RewardTermCfg(
            func=gk_mdp.foot_proximity,
            weight=5.0,
            params={"ball_name": BALL_NAME, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        # --- active stepping: reward lifting feet during approach ---
        "foot_clearance": RewardTermCfg(
            func=gk_mdp.foot_clearance,
            weight=2.0,
            params={"ball_name": BALL_NAME, "target_height": 0.10, "asset_cfg": _FEET_CFG},
        ),
        # --- goalkeeper stance ---
        "stayonline": RewardTermCfg(
            func=gk_mdp.stayonline,
            weight=-2.0,
        ),
        "noretreat": RewardTermCfg(
            func=gk_mdp.noretreat,
            weight=-2.0,
        ),
        "feetorientation": RewardTermCfg(
            func=gk_mdp.feetorientation,
            weight=3.0,
            params={"asset_cfg": _FEET_CFG},
        ),
        # --- post-save recovery (active only when ball is behind) ---
        "postorientation": RewardTermCfg(
            func=gk_mdp.postorientation,
            weight=3.0,
            params={"ball_name": BALL_NAME},
        ),
        "postangvel": RewardTermCfg(
            func=gk_mdp.postangvel,
            weight=3.0,
            params={"ball_name": BALL_NAME},
        ),
        "postlinvel": RewardTermCfg(
            func=gk_mdp.postlinvel,
            weight=1.0,
            params={"ball_name": BALL_NAME},
        ),
        "postupperdofpos": RewardTermCfg(
            func=gk_mdp.postupperdofpos,
            weight=1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_ARM_CFG},
        ),
        "postwaistdofpos": RewardTermCfg(
            func=gk_mdp.postwaistdofpos,
            weight=1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_WAIST_CFG},
        ),
        # --- full-body T-pose recovery (all 21 joints → home keyframe default) ---
        "post_default_pose": RewardTermCfg(
            func=gk_mdp.post_default_pose,
            weight=5.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _ALL_JOINTS_CFG, "std": 0.5},
        ),
        # --- hardware safety ---
        "penalize_kneeheight": RewardTermCfg(
            func=gk_mdp.penalize_kneeheight,
            weight=-100.0,
            params={"min_height": 0.15, "asset_cfg": _KNEE_BODY_CFG},
        ),
        "penalize_sharpcontact": RewardTermCfg(
            func=gk_mdp.penalize_sharpcontact,
            weight=-100.0,
            params={"force_threshold": 1200.0},
        ),
        "penalize_self_collision": RewardTermCfg(
            func=gk_mdp.penalize_self_collision,
            weight=-50.0,
        ),
        "feet_slippage": RewardTermCfg(
            func=gk_mdp.feet_slippage,
            weight=3.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # --- joint limits ---
        "dof_pos_limits": RewardTermCfg(
            func=mjlab_mdp.joint_pos_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "dof_vel_limits": RewardTermCfg(
            func=gk_mdp.dof_vel_limits,
            weight=-2.0,
            params={"vel_threshold": 10.0, "asset_cfg": _ALL_JOINTS_CFG},
        ),
        "torque_limits": RewardTermCfg(
            func=gk_mdp.torque_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        # --- stability ---
        "ang_vel_xy": RewardTermCfg(
            func=gk_mdp.ang_vel_xy_l2,
            weight=-0.1,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "ang_vel_z": RewardTermCfg(
            func=gk_mdp.ang_vel_z_l2,
            weight=-2.0,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "deviation_waist_joint": RewardTermCfg(
            func=gk_mdp.deviation_waist_joint,
            weight=-0.001,
            params={"asset_cfg": _WAIST_JOINT_CFG},
        ),
        # --- regularisation ---
        "torques": RewardTermCfg(
            func=gk_mdp.torques_normalized_l2,
            weight=-1e-5,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2,
            weight=-0.3,
        ),
        "action_acc_l2": RewardTermCfg(
            func=mjlab_mdp.action_acc_l2,
            weight=-0.1,
        ),
        "dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2,
            weight=-5e-4,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "dof_acc": RewardTermCfg(
            func=mjlab_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
    }

    # ------------------------------------------------------------------
    # Events — Domain Randomisation (mirrors BoosterT1mjlab kick task)
    # ------------------------------------------------------------------
    # foot_friction: randomise foot-ground friction per episode (startup).
    # Same geom names as kick task: (left|right)_foot{1-4}_collision.
    _foot_geoms = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 5)
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = _foot_geoms

    # encoder_bias: per-joint position sensor offset at startup (±0.015 rad).
    # Already configured correctly by base make_velocity_env_cfg — no changes.

    # base_com: randomise trunk CoM position at startup (±2.5 cm XY, ±3 cm Z).
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    # push_robot: random velocity impulse every 1-3 s during training.
    # Required for sim2real robustness — goalkeeper must maintain stance
    # under external disturbances. Kept at the same magnitude as kick task.

    # Reset robot root to HOME position at the goal line, robot facing +X.
    # CRITICAL: without this, the root position and yaw never reset between episodes.
    # Drifting yaw breaks all world-X reward terms (stopball, ball_exit_termination,
    # stayonline) because ball spawns in robot-local +X but checks use world +X.
    # Keep x=(0,0) and y=(0,0): goalkeeper stays centred on the goal line.
    # Small yaw=(−0.1, 0.1) adds ±5° robustness without breaking world-X assumptions.
    cfg.events["reset_base"] = EventTermCfg(
        func=mjlab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {},
        },
    )

    # RSI: after reset_robot_joints sets joints to default, overwrite with a random
    # NPZ frame. This exposes the policy to mid-motion dynamics from episode start,
    # preventing the standing-still local optimum seen without RSI.
    cfg.events["init_motion_loader"] = EventTermCfg(
        func=gk_mdp.init_motion_loader,
        mode="startup",
        params={},
    )
    # Ball reset fires BEFORE RSI so MotionResetManager can read the new ball
    # velocity and select the matching motion tier (single/double/triple step).
    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_mdp.reset_ball_rolling,
        mode="reset",
        params={
            "ball_name":      BALL_NAME,
            "dist_range":     (1.5, 2.5),
            "y_start_range":  (-0.3, 0.3),
            "y_end_range":    (-1.0, 1.0),
            "speed_range":    (2.0, 3.5),
            "spawn_z":        0.12,
        },
    )

    cfg.events["reset_from_motion_data"] = EventTermCfg(
        func=gk_mdp.reset_from_motion_data,
        mode="reset",
        params={},
    )

    # Per-step catchstep decrement for ball visibility warmup.
    cfg.events["tick_catchstep"] = EventTermCfg(
        func=gk_mdp.tick_catchstep,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=mjlab_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": _ROBOT_CFG},
        ),
        "base_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.4, "asset_cfg": _ROBOT_CFG},
        ),
        "ball_exit": TerminationTermCfg(
            func=gk_mdp.ball_exit_termination,
            params={"ball_name": BALL_NAME, "behind_threshold": -0.5},
            time_out=False,
        ),
        "sharpforce": TerminationTermCfg(
            func=gk_mdp.sharpforce_termination,
            params={"max_contact_force": 1500.0},
            time_out=False,
        ),
    }

    # ------------------------------------------------------------------
    # Episode length
    # ------------------------------------------------------------------
    # Play: 10 s for evaluation. Training: 3 s matches ILB (goalkeeper_amp_env_cfg.py:422).
    # 6 s was wasted compute: post-save steps all have footreach=0 (behind gate) and
    # dilute the stopball per-step signal by 2× vs 3 s (max stopball/step 0.33→0.67).
    cfg.episode_length_s = 10.0 if play else 3.0

    # ------------------------------------------------------------------
    # Play-mode overrides
    # ------------------------------------------------------------------
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("out_of_terrain_bounds", None)
        # No disturbance pushes during play/eval — mirrors kick task play mode.
        cfg.events.pop("push_robot", None)
        # RSI is kept in play mode so each episode starts from a random motion frame,
        # matching the training distribution. init_motion_loader is registered
        # unconditionally above; reset_from_motion_data is NOT popped here.
        # Play: rolling ball — same function as training so visualisation matches
        # the distribution the policy was trained on. vz=0 keeps ball at foot level.
        cfg.events["reset_ball"] = EventTermCfg(
            func=gk_mdp.reset_ball_rolling,
            mode="reset",
            params={
                "ball_name":     BALL_NAME,
                "dist_range":    (1.5, 2.5),
                "y_start_range": (-0.3, 0.3),
                "y_end_range":   (-1.0, 1.0),
                "speed_range":   (2.0, 3.5),
                "spawn_z":       0.12,
            },
        )

    return cfg


# Body names in T1 headless entity-local order (Trunk = index 0, no world body).
# Must match NPZ body_pos_w second dimension: body_pos_w[:, 0, :] = Trunk, etc.
_T1_HEADLESS_BODY_NAMES: tuple[str, ...] = (
    "Trunk", "H1", "H2",
    "AL1", "AL2", "AL3", "left_hand_link",
    "AR1", "AR2", "AR3", "right_hand_link",
    "Waist",
    "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left",
    "Shank_Left", "Ankle_Cross_Left", "left_foot_link",
    "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right",
    "Shank_Right", "Ankle_Cross_Right", "right_foot_link",
)

_MOTIONS_DATA_DIR = Path(__file__).parents[1] / "motions" / "data"


def goalkeeper_env_cfg_withoverlay(
    motion_file: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Play-mode config with ghost-robot overlay cycling through all reference motions.

    The ghost follows the NPZ motion while the policy runs normally — no RSI
    teleportation. Cycles through all NPZ files in motions/data/ in order.

    Args:
        motion_file: Path to a specific NPZ file to pin to that motion only.
            If None, cycles through all files in motions/data/.
    """
    from simple_goalkeeper.mdp.commands import (
        CyclingGhostMotionCommandCfg,
        GhostMotionCommandCfg,
    )

    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1

    if motion_file is not None:
        cfg.commands["motion_ghost"] = GhostMotionCommandCfg(
            motion_file=motion_file,
            anchor_body_name="Trunk",
            body_names=_T1_HEADLESS_BODY_NAMES,
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
        )
    else:
        npz_files = sorted(_MOTIONS_DATA_DIR.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files in {_MOTIONS_DATA_DIR}")
        cmd = CyclingGhostMotionCommandCfg(
            motion_file=str(npz_files[0]),  # required by parent cfg; overridden at build
            anchor_body_name="Trunk",
            body_names=_T1_HEADLESS_BODY_NAMES,
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
        )
        cmd.motion_files = [str(f) for f in npz_files]
        cfg.commands["motion_ghost"] = cmd

    return cfg
