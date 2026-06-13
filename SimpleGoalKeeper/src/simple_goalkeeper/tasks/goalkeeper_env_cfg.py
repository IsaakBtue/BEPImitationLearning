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
    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["ball_difficulty"] = CurriculumTermCfg(
            func=gk_mdp.ball_difficulty_curriculum,
            params={
                "stages": [
                    {"step": 0,                    "difficulty": 0.0},
                    {"step": 600  * _num_steps,    "difficulty": 0.5},
                    {"step": 1200 * _num_steps,    "difficulty": 1.0},
                ],
            },
        )
        cfg.curriculum["stopball_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "stopball",
                "stages": [
                    {"step": 0,                    "weight": 100.0},
                    {"step": 600  * _num_steps,    "weight": 175.0},
                    {"step": 1200 * _num_steps,    "weight": 250.0},
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

    # beyondAMP discriminator observes basic joint_pos + joint_vel (single frame).
    cfg.observations["amp"] = amp_obs_basic_group()

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
    # Rewards
    # Design principles (matching Imitationlearningbooster proven structure):
    #   - footreach (w=10): dense phase1 lateral alignment + phase2 proximity×vel_sigma
    #   - stopball (w=100): one-time bonus when ball is deflected (primary task signal)
    #   - ball_positive_vx (w=10): continuous reward for sustained deflection
    #   - posture removed: AMP handles motion naturalness; posture created stand-still optimum
    #   - ball_vx_reduction removed: it peaked when ball stopped naturally (do-nothing reward)
    # ------------------------------------------------------------------
    cfg.rewards = {
        # --- ball interception (feet-only) ---
        # stopball must come before footreach: footreach calls _ball_is_behind which reads
        # env._sb_init_vx set by stopball. On the first step of every episode stopball must
        # run first so _sb_init_vx is populated before footreach queries it.
        "stopball": RewardTermCfg(
            func=gk_mdp.stopball,
            weight=100.0,
            params={"ball_name": BALL_NAME, "delta_vel_threshold": 1.0},
        ),
        "footreach": RewardTermCfg(
            func=gk_mdp.footreach,
            weight=10.0,
            params={"ball_name": BALL_NAME, "reach_th": 0.3, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        "ball_positive_vx": RewardTermCfg(
            func=gk_mdp.ball_positive_vx,
            weight=10.0,
            params={"ball_name": BALL_NAME, "target_speed": 5.0},
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
            weight=1.5,
            params={"asset_cfg": _FEET_CFG},
        ),
        # --- stability ---
        "ang_vel_xy": RewardTermCfg(
            func=gk_mdp.ang_vel_xy_l2,
            weight=-0.1,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "deviation_waist_joint": RewardTermCfg(
            func=gk_mdp.deviation_waist_joint,
            weight=-0.001,
            params={"asset_cfg": _WAIST_JOINT_CFG},
        ),
        # --- joint safety ---
        "dof_pos_limits": RewardTermCfg(
            func=mjlab_mdp.joint_pos_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        # --- regularisation ---
        "action_rate_l2": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2,
            weight=-0.3,
        ),
        "dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2,
            weight=-0.001,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
    }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    # Remove DR events not needed for Phase 1 (no domain randomization yet).
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    # Remove push_robot: random impulses disrupt the goalkeeper stance and prevent
    # the policy from learning to react to the ball during early training.
    cfg.events.pop("push_robot", None)

    # Goalkeeper spawn: robot starts at/near the goal line, centered, facing +X.
    # The velocity-base default (±0.5 m in X and Y) is for locomotion generalization
    # and is wrong here — it puts the robot up to 0.5 m off-center before the ball
    # even spawns, immediately triggering stayonline penalties.
    cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.15, 0.15)

    # Ball reset: spawn in robot local +X frame with parabolic arc to foot level.
    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_mdp.reset_ball_local_frame,
        mode="reset",
        params={
            "ball_name": BALL_NAME,
            "dist_range": (2.0, 4.0),
            "lateral_range": (-0.5, 0.5),
            "spawn_height_range": (0.2, 0.7),   # m above floor at spawn
            "arrive_height_range": (0.1, 0.4),  # m above floor at robot — feet/shins
            "speed_range": (3.0, 7.0),
        },
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
    }

    # ------------------------------------------------------------------
    # Episode length
    # ------------------------------------------------------------------
    cfg.episode_length_s = 10.0 if play else 4.0

    # ------------------------------------------------------------------
    # Play-mode overrides
    # ------------------------------------------------------------------
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("out_of_terrain_bounds", None)

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
    """Play-mode config with ghost-robot overlay showing a reference motion.

    The ghost follows the NPZ motion while the policy runs normally — no RSI
    teleportation. Requires the NPZ files to be regenerated after the converter
    fix (world body excluded from body_pos_w).

    Args:
        motion_file: Path to a specific NPZ file, or None to use the first one
            found in motions/data/.
    """
    from simple_goalkeeper.mdp.commands import GhostMotionCommandCfg

    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1

    if motion_file is None:
        npz_files = sorted(_MOTIONS_DATA_DIR.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files in {_MOTIONS_DATA_DIR}")
        motion_file = str(npz_files[0])

    cfg.commands["motion_ghost"] = GhostMotionCommandCfg(
        motion_file=motion_file,
        anchor_body_name="Trunk",
        body_names=_T1_HEADLESS_BODY_NAMES,
        entity_name="robot",
        debug_vis=True,
        resampling_time_range=(10.0, 10.0),  # ghost resamples every 10s (one full episode)
        viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
    )

    return cfg
