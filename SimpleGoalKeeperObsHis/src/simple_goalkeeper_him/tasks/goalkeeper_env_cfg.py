"""Goalkeeper environment config for SimpleGoalKeeperHim (HIM-PPO, 2-disc AMP, 21-DOF)."""
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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from simple_goalkeeper_him.robots.t1_constants import (
    T1_ACTION_SCALE_HEADLESS,
    get_t1_headless_robot_cfg,
)
import simple_goalkeeper_him.mdp as gk_mdp

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
    """Booster T1 goalkeeper -- HIM-PPO, 2-disc AMP, 21-DOF headless T1, feet only.

    Architecture: GoalkeeperAmpRunner (HIM-PPO) + 2 AMP discriminators (left/right).
    Rewards/curriculum/ball-spawn from SimpleGoalKeeper (proven foot-only design).
    Ball spawn routes y_end sign by discriminator side via env._motion_type_ids.
    """
    cfg = make_velocity_env_cfg()

    cfg.viewer.body_name = "Trunk"

    # Scene: flat plane, headless T1, ball
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    )
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
    cfg.scene.num_envs = 4096

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE_HEADLESS

    # Contact/constraint budget tuned for 8192 envs on 32 GB GPU.
    # nconmax: observed peak ~30 contacts/env (ball + 8 foot geoms + floor + self-collision).
    # njmax: overflow warnings peaked at 354 during random-policy init; 400 adds headroom.
    # Spikes occur when robot falls with all foot geoms + ball + self-collision active
    # simultaneously. Peak drops once the policy stabilises and falls become rarer.
    cfg.sim.nconmax = 40
    cfg.sim.njmax = 400

    # Commands: none
    cfg.commands.clear()

    # Curriculum
    _num_steps = 50   # matches GoalkeeperAmpRunner num_steps_per_env (50 to fit 8192 envs)
    _stage1 = 2000 * _num_steps   # step 100k  = 2000 iters @ 50 steps
    _stage2 = 4000 * _num_steps   # step 200k  = 4000 iters @ 50 steps
    cfg.curriculum.clear()
    if not play:
        # Episode-length-adaptive difficulty — mirrors G1's curriculum driver.
        # Ball difficulty advances when the policy survives longer, not on a clock.
        cfg.curriculum["ball_difficulty"] = CurriculumTermCfg(
            func=gk_mdp.ball_difficulty_curriculum,
            params={"min_gate": 500},
        )
        # Stopball/footreach weight curriculum is still step-based since it only
        # ramps rewards, not physics difficulty.
        cfg.curriculum["stopball_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "stopball",
                "stages": [
                    {"step": 0,          "weight": 100.0},
                    {"step": 2_000_000,  "weight": 175.0},
                    {"step": 4_000_000,  "weight": 250.0},
                ],
            },
        )
        cfg.curriculum["footreach_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "footreach",
                "stages": [
                    {"step": 0,          "weight": 10.0},
                    {"step": 2_000_000,  "weight": 15.0},
                    {"step": 4_000_000,  "weight": 20.0},
                ],
            },
        )
        cfg.curriculum["softstop_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "softstop",
                "stages": [
                    {"step": 0,          "weight": 50.0},
                    {"step": 2_000_000,  "weight": 75.0},
                    {"step": 4_000_000,  "weight": 100.0},
                ],
            },
        )
        # Torque limits: ramp penalty as policy stabilises so early random flailing
        # doesn't dominate training with a huge penalty from the start.
        cfg.curriculum["torque_limits_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "torque_limits",
                "stages": [
                    {"step": 0,          "weight": -3.0},
                    {"step": 2_000_000,  "weight": -6.0},
                    {"step": 4_000_000,  "weight": -9.0},
                ],
            },
        )

    # Observations
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
        # Zone ID: tells the policy which AMP discriminator it's assigned to (0=left, 1=right).
        # Mirrors G1's end_regions/3 observation term.  No noise — discrete signal.
        "motion_type_id": ObservationTermCfg(func=gk_mdp.motion_type_id),
        # Predicted ball Y at goal-line crossing: tells the policy WHERE to place its foot.
        # Mirrors G1's end_target_local.  Noise ±0.05 m simulates tracking error.
        "ball_landing_y": ObservationTermCfg(
            func=gk_mdp.ball_landing_y_b,
            params={"ball_name": BALL_NAME},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }
    critic_terms = {
        k: ObservationTermCfg(func=v.func, params=v.params)
        for k, v in actor_terms.items()
    }

    cfg.observations["actor"] = ObservationGroupCfg(terms=actor_terms, enable_corruption=True)
    cfg.observations["critic"] = ObservationGroupCfg(terms=critic_terms, enable_corruption=False)

    # Privileged obs for HIM supervision (ball estimator + region estimator GT).
    # Not stacked (history_length=1): always current step only.
    # Layout: [ball_pos_b (3), ball_vel_b (3), motion_type_id (1)] = 7D.
    # Read by runner in _update_with_amp; never fed to actor/critic models.
    cfg.observations["privileged"] = ObservationGroupCfg(
        terms={
            "ball_pos_b_true": ObservationTermCfg(
                func=gk_mdp.ball_pos_b,
                params={"ball_name": BALL_NAME, "always_visible": True},
            ),
            "ball_vel_b_true": ObservationTermCfg(
                func=gk_mdp.ball_vel_b,
                params={"ball_name": BALL_NAME, "always_visible": True},
            ),
            "motion_type_id_priv": ObservationTermCfg(func=gk_mdp.motion_type_id),
        },
        enable_corruption=False,
        history_length=1,
    )

    # AMP obs: joint_pos only (21-dim) to match motion loader convention.
    # history_length=1 -- runner concatenates current+prev to form 42-dim discriminator input.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "amp_obs": ObservationTermCfg(func=gk_mdp.joint_pos_abs, noise=None),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )

    # 10-frame observation history for actor/critic (HIM-PPO design).
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 10

    if not play:
        for term_cfg in cfg.observations["actor"].terms.values():
            term_cfg.delay_min_lag = 0
            term_cfg.delay_max_lag = 2
            term_cfg.delay_per_env = True

    # Rewards (all from SimpleGoalKeeper foot-only design)
    cfg.rewards = {
        "stopball": RewardTermCfg(
            func=gk_mdp.stopball,
            weight=100.0,
            params={"ball_name": BALL_NAME, "delta_vel_threshold": 1.0},
        ),
        "softstop": RewardTermCfg(
            func=gk_mdp.softstop,
            weight=50.0,   # curriculum ramps: 50 → 75 → 100 (softstop_curriculum)
            params={"ball_name": BALL_NAME, "velocity_threshold": 0.2},
        ),
        "footreach": RewardTermCfg(
            func=gk_mdp.footreach,
            weight=10.0,
            params={"ball_name": BALL_NAME, "reach_th": 0.3, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        "stayonline": RewardTermCfg(func=gk_mdp.stayonline, weight=-2.0),
        "noretreat": RewardTermCfg(func=gk_mdp.noretreat, weight=-2.0),
        "feetorientation": RewardTermCfg(
            func=gk_mdp.feetorientation,
            weight=3.0,
            params={"asset_cfg": _FEET_CFG},
        ),
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
        "penalize_kneeheight": RewardTermCfg(
            func=gk_mdp.penalize_kneeheight,
            weight=-100.0,
            params={"min_height": 0.15, "asset_cfg": _KNEE_BODY_CFG},
        ),
        "penalize_sharpcontact": RewardTermCfg(
            func=gk_mdp.penalize_sharpcontact,
            weight=-100.0,
            params={"force_threshold": 1000.0},
        ),
        "penalize_self_collision": RewardTermCfg(
            func=gk_mdp.penalize_self_collision,
            weight=-50.0,
        ),
        "feet_slippage": RewardTermCfg(
            func=gk_mdp.feet_slippage,
            weight=3.0,
            params={"asset_cfg": _FEET_CFG},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mjlab_mdp.joint_pos_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "dof_vel_limits": RewardTermCfg(
            func=gk_mdp.dof_vel_limits,
            weight=-2.0,
            params={"vel_limit": 20.0, "soft_factor": 0.9, "asset_cfg": _ALL_JOINTS_CFG},
        ),
        "torque_limits": RewardTermCfg(
            func=gk_mdp.torque_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "ang_vel_xy": RewardTermCfg(
            func=gk_mdp.ang_vel_xy_l2,
            weight=-0.1,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "ang_vel_z": RewardTermCfg(
            func=gk_mdp.ang_vel_z_l2,
            weight=-0.5,
            params={"asset_cfg": _ROBOT_CFG},
        ),
        "deviation_waist_joint": RewardTermCfg(
            func=gk_mdp.deviation_waist_joint,
            weight=-0.001,
            params={"asset_cfg": _WAIST_JOINT_CFG},
        ),
        "torques": RewardTermCfg(
            func=gk_mdp.torques_normalized_l2,
            weight=-1e-5,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "action_rate_l2": RewardTermCfg(func=mjlab_mdp.action_rate_l2, weight=-0.3),
        "action_acc_l2": RewardTermCfg(func=mjlab_mdp.action_acc_l2, weight=-0.1),
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

    # Events
    cfg.events.pop("foot_friction",   None)
    cfg.events.pop("encoder_bias",    None)
    cfg.events.pop("base_com",        None)
    cfg.events.pop("push_robot",      None)
    # RSI (reset_from_motion_data) handles root state; reset_base is redundant.
    cfg.events.pop("reset_base",      None)

    # Startup: assign static motion type partition (left/right disc).
    cfg.events["assign_motion_types"] = EventTermCfg(
        func=gk_mdp.assign_motion_types,
        mode="startup",
        params={},
    )

    cfg.events["init_motion_loader"] = EventTermCfg(
        func=gk_mdp.init_motion_loader,
        mode="startup",
        params={},
    )
    cfg.events["reset_from_motion_data"] = EventTermCfg(
        func=gk_mdp.reset_from_motion_data,
        mode="reset",
        params={},
    )

    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_mdp.reset_ball_rolling,
        mode="reset",
        params={
            "ball_name":     BALL_NAME,
            "dist_range":    (1.5, 2.5),
            "y_start_range": (-0.5, 0.5),
            "y_end_range":   (-0.5, 0.5),
            "speed_range":   (2.0, 3.5),
            "spawn_z":       0.12,
        },
    )

    cfg.events["tick_catchstep"] = EventTermCfg(
        func=gk_mdp.tick_catchstep,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    # Terminations
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
        # Terminate 80 steps (~1.6 s at 50 Hz) after ball is deflected or crosses
        # the goal line.  Prevents the robot from slowly falling after a save without
        # triggering bad_orientation (57°) or base_height (0.4 m).
        "post_save_timeout": TerminationTermCfg(
            func=gk_mdp.post_save_timeout_termination,
            params={"ball_name": BALL_NAME, "timeout_steps": 80},
            time_out=False,
        ),
    }

    cfg.episode_length_s = 10.0 if play else 4.0

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.events.pop("reset_from_motion_data", None)
        cfg.events["reset_ball"] = EventTermCfg(
            func=gk_mdp.reset_ball_rolling,
            mode="reset",
            params={
                "ball_name":     BALL_NAME,
                "dist_range":    (1.5, 2.5),
                "y_start_range": (-0.5, 0.5),
                "y_end_range":   (-0.5, 0.5),
                "speed_range":   (2.0, 3.5),
                "spawn_z":       0.12,
            },
        )

    return cfg


# Body names for headless T1 in entity-local order (Trunk = index 0, no world body).
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

    The ghost follows the NPZ motion at its world-space position while the policy
    runs normally — no RSI teleportation. Cycles through all NPZ files in order.

    Args:
        motion_file: Path to a specific NPZ file. If None, cycles through all files.
    """
    from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
    from simple_goalkeeper_him.mdp.commands import (
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
        # Filter to only our own stepping NPZ files (skip old ILB files in the same dir)
        npz_files = sorted(f for f in _MOTIONS_DATA_DIR.glob("*_booster_t1.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files in {_MOTIONS_DATA_DIR}")
        cmd = CyclingGhostMotionCommandCfg(
            motion_file=str(npz_files[0]),
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
