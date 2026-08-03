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
# NEW 2026-07-30 (user request): AL2/AR2 (not AL1/AR1) chosen as "the
# shoulder" because they carry left/right_shoulder_collision, the same body
# already used as the shoulder reference elsewhere in this task. NOTE:
# resolved SceneEntityCfg.body_ids order does NOT match this declared
# body_names order (verified live -- see penalize_arm_above_shoulder's
# docstring, rewards.py) -- the reward function resolves each name
# explicitly via robot.find_bodies(...)'s returned names list rather than
# assuming a fixed position, so this declaration order is documentation
# only, not load-bearing.
_ARM_HEIGHT_CFG = SceneEntityCfg(
    "robot",
    body_names=("AL2", "AR2", "left_hand_link", "right_hand_link"),
)
_RECOVERY_WAIST_CFG = SceneEntityCfg("robot", joint_names=("Waist",))
# FIX 2026-07-22: see postlegdofpos's docstring (rewards.py) -- G1 has no
# leg-recovery reward to port because its legs aren't the catching limb;
# SGK's legs are, so this fills the gap that left them with no post-save
# return-to-default incentive.
_RECOVERY_LEG_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Hip_Pitch", "Left_Knee_Pitch",
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Hip_Pitch", "Right_Knee_Pitch",
        "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
)


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
        # FIX 2026-07-15: "feet_contact" above has secondary=None -- it fires
        # whenever a foot touches ANYTHING (in practice, mostly the ground),
        # not specifically the ball. stopball/softstop's "correct foot"
        # gate used it to check whether the assigned foot caused the
        # deflection, but a foot standing normally on the ground already
        # satisfies it -- the gate was nearly vacuous. This sensor is
        # ball-specific: primary=foot geoms, secondary=ball_geom, so
        # "found" only fires on genuine foot-ball contact.
        ContactSensorCfg(
            name="ball_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left|right)_foot[1-4]_collision$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="geom", pattern="ball_geom", entity=BALL_NAME),
            fields=("found", "force"),
            reduce="netforce",
            history_length=0,
        ),
        # DEBUG 2026-07-28: "ball_contact" above only matches foot geoms, so a
        # ball touch against the head/chin (head_collision, t1_headless.xml --
        # a static sphere on H2, fixed since head joints were removed) is
        # invisible to both feet_contact and ball_contact and therefore to
        # penalize_wrong_foot_ball_contact and its viewer plot too. User
        # observed the wrong-foot plot never reacting and asked whether the
        # ball might actually be hitting the chin instead -- this sensor lets
        # play.py's viewer confirm/deny that independently. Was purely
        # additive/display-only at introduction; FIX 2026-07-30 wired it into
        # penalize_wrong_foot_ball_contact (rewards.py) as a real training
        # penalty. See docs/BugFixes.md.
        ContactSensorCfg(
            name="head_ball_contact",
            primary=ContactMatch(mode="geom", pattern="head_collision", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="ball_geom", entity=BALL_NAME),
            fields=("found", "force"),
            reduce="netforce",
            history_length=0,
        ),
        # DEBUG 2026-07-28: same gap class as head_ball_contact above, found
        # via the user directly observing MuJoCo's native contact-point
        # overlay (orange dot) between the trailing leg and the ball while
        # penalize_wrong_foot_ball_contact stayed 0. Confirmed via a scripted
        # replay of the real trained checkpoint: the shin (left_shin_collision/
        # right_shin_collision) and knee geoms are NOT foot[1-4]_collision, so
        # they were invisible to ball_contact/feet_contact -- the policy makes
        # genuine foot-ball saves via the shin routinely, and confirmed at
        # least one case of the WRONG side's shin/knee touching the ball
        # while penalize_wrong_foot_ball_contact read 0. Primary resolves in
        # geom-index order [left_shin, left_knee, right_knee, right_shin] --
        # verified directly (env.scene["leg_ball_contact"].primary_names) --
        # so found[:, :2]=left, found[:, 2:]=right, matching the existing
        # feet_contact/ball_contact left/right split convention. See
        # penalize_wrong_foot_ball_contact (rewards.py) and docs/BugFixes.md.
        ContactSensorCfg(
            name="leg_ball_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(left|right)_(shin|knee)_collision$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="geom", pattern="ball_geom", entity=BALL_NAME),
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
    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["ball_difficulty"] = CurriculumTermCfg(
            func=gk_mdp.ball_difficulty_curriculum,
            params={
                # Adaptive curriculum: mirrors Humanoid-Goalkeeper G1 (legged_robot.py:325-336).
                # difficulty += step_size × int(mean_ep_len / ep_len_divisor)
                # every update_interval per-env steps. Longer episodes → faster advance.
                "update_interval": 500,   # per-env steps between updates (same as G1)
                "ep_len_divisor":  50,    # same as G1 (matches reward curricula divisor exactly
                                          # so ball difficulty and reward weights advance at the
                                          # same episode-length boundary)
                "step_size":       0.01,  # difficulty units per curriculumupdate per check
            },
        )
        # Episode-length-driven weight curriculum — mirrors G1 compute_reward() lines 359-364:
        #   weight = base * (1 + 0.5 * curriculumupdate)  where cu = int(mean_ep_len / 50)
        # All three terms share env._curriculumupdate (set by whichever runs first each window).
        # G1 max (cu=3, ep_len=150): softstop 105→262.5, footreach 10→25.
        #
        # FIX 2026-07-20 (reward audit item 8, peak-magnitude cap): the six
        # terms that can all fire around a single save event -- stopball,
        # softstop, single_foot_save, cleanstop, inner_face_orientation_save,
        # foot_inner_face_continuous -- summed to a combined max-curriculum
        # peak of 18.75+131.25+100+50+50+10 = 360 even after the same-day
        # halving pass above, still 1.44x G1's own ceiling for the equivalent
        # single "did you stop it" event (_reward_stopball, curriculum-scaled
        # 100->250, no per-foot/orientation/cleanliness sub-bonuses). Scaled
        # this whole group down by a further 25/36 (0.69444) so the combined
        # peak lands at exactly 250, preserving each term's relative share of
        # the group (SGK's feet-specific quality bonuses stay individually
        # reasoned, just resized to fit under G1's ceiling as a group):
        #   stopball                    7.5   -> 5.21   (max 18.75 -> 13.02)
        #   softstop                   52.5   -> 36.46  (max 131.25 -> 91.15)
        #   single_foot_save           50.0   -> 34.72  (max 100 -> 69.44)
        #   cleanstop                  25.0   -> 17.36  (max 50 -> 34.72)
        #   inner_face_orientation_save 25.0  -> 17.36  (max 50 -> 34.72)
        #   foot_inner_face_continuous  5.0   -> 3.47   (max 10 -> 6.94)
        # New combined peak: 13.02+91.15+69.44+34.72+34.72+6.94 = 250.0 (exact,
        # 25/36 chosen precisely so the sum lands on G1's ceiling). See
        # docs/BugFixes.md for the full derivation. Static `weight=` values in
        # cfg.rewards below (stopball/softstop) are also updated to match
        # these new base weights -- they were previously stale pre-halving
        # values only used in play mode (no curriculum there); this also
        # fixes that pre-existing static/curriculum mismatch as a side effect.
        cfg.curriculum["softstop_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "softstop",
                "base_weight": 36.46,   # 52.5 * 25/36 -- see item-8 cap comment above
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # NOTE: torque_limits and dof_pos_limits intentionally NOT in curriculum.
        # Tripling the torque penalty at the same time balls get harder (wider y range)
        # forces the robot into low-torque yaw strategies instead of lateral steps.
        # Keep both fixed at -3.0 throughout so the robot can use full joint effort
        # for aggressive saves at hard difficulty without extra penalty.
        cfg.curriculum["footreach_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "footreach",
                "base_weight": 10.0,     # G1 eereach_init=10 → max 25 at cu=3
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # FIX 2026-07-26 (near-region oscillation): near-region analog of
        # blue_stick_landing_curriculum below -- same base_weight (8.0),
        # ported for the same reason (dense "close AND slow" anti-
        # oscillation signal near-region footreach has no equivalent of).
        # Narrow-region-only by construction (rewards.py:near_stick_reach's
        # own env._blue_wide gate) -- this curriculum entry only ever scales
        # a reward value that is already zero for wide/far envs, so it
        # cannot influence far-region training despite living in this same
        # shared config file.
        cfg.curriculum["near_stick_reach_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "near_stick_reach",
                "base_weight": 8.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # G1 lines 363-364: stopball weight also grows with curriculum (same formula as eereach).
        cfg.curriculum["stopball_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "stopball",
                "base_weight": 5.21,    # 7.5 * 25/36 -- see item-8 cap comment above
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch's
        # curriculum for its own 3 reward terms -- same reward_curriculum_ep_len
        # mechanism already used above, base weights taken from the branch's own
        # tuning (unretuned for this reimplementation). Without these, the "must
        # land at blue first" incentive would stay fixed while footreach/stopball/
        # softstop all grow via curriculum -- an asymmetry the branch's own
        # 2026-07-09 fix found caused genuine landing rate to collapse as the
        # cheap, immediate overshoot penalty and growing downstream save payoff
        # both outpaced it.
        cfg.curriculum["blue_ball_landed_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "blue_ball_landed",
                "base_weight": 10.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        cfg.curriculum["blue_overshoot_penalty_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "blue_overshoot_penalty",
                "base_weight": -60.0,  # FIX 2026-07-23: was -30.0, too lenient
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        cfg.curriculum["blue_stick_landing_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "blue_stick_landing",
                "base_weight": 8.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # NEW 2026-07-24: same curriculum shape as the other blue_* terms --
        # without this, the trunk-drive incentive would stay fixed while
        # footreach/stopball/softstop/blue_ball_landed all grow via
        # curriculum (see the blue_ball_landed_curriculum comment above for
        # why an asymmetric curriculum among these terms caused problems
        # before). base_weight=5.0 is an initial, unvalidated guess --
        # matches this term's static play-mode weight (cu=0 baseline).
        cfg.curriculum["blue_trunk_drive_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "blue_trunk_drive",
                "base_weight": 5.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        # Correct-foot-save quality bonuses: weights double at cu >= 3 (ep_len ≈ 144 steps).
        # Only makes sense once the robot already saves reliably (cu=3 = footreach fully ramped).
        # One entry per reward, same pattern as reward_curriculum_ep_len.
        # Base weights scaled by 25/36 (item-8 peak-magnitude cap, see comment above).
        for _name, _base in (
            ("single_foot_save",             34.72),
            ("cleanstop",                    17.36),
            ("inner_face_orientation_save",  17.36),
            ("foot_inner_face_continuous",    3.47),
        ):
            cfg.curriculum[f"{_name}_curriculum"] = CurriculumTermCfg(
                func=gk_mdp.correct_foot_save_curriculum,
                params={"reward_name": _name, "base_weight": _base, "activate_at_cu": 3},
            )
        # G1's own `success` reward (legged_robot.py:1402-1403): continuing,
        # doubled-after-save, close-to-target signal -- NOT part of the item-8
        # group above (G1 itself keeps it outside _reward_stopball's own
        # weight ceiling). base=5.0 -> max 12.5 at cu=3, matching G1's
        # success_init=5.0 (g1_29_config.py:300) exactly under the same
        # weight=base*(1+0.5*cu) formula. See rewards.py:success docstring.
        cfg.curriculum["success_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "success",
                "base_weight": 5.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )

    # ------------------------------------------------------------------
    # Observations
    # Ball is fully visible during the entire approach and save (always_visible
    # =True — the full G1 visibility port was reverted, see CLAUDE.md). The
    # post-save release is the v2 gate (ported from SimpleGoalKeeper, ce69f36):
    # obs zeroes once the ball is behind the torso (x_body < 0.05, G1
    # flying-mask front edge) or the 75-step window since launch closes (G1
    # catchstep analog; > max 1.3 s flight so it never blinds a ball still en
    # route). The policy learns to disengage and recover to the default pose
    # in the guaranteed blind tail of every episode.
    # ------------------------------------------------------------------
    # Actor: only terms available at deployment on real hardware.
    # base_lin_vel, ball_vel_b, foot_pos_b removed — not measurable at deployment.
    # ball_pos_b kept as 3D (XY from camera + Z height for ball height awareness).
    # FIX 2026-07-20 (item 21, obs-scaling audit): G1's compute_observations()
    # (legged_robot.py:405-419) bakes fixed per-term scales directly into the
    # obs vector before it ever reaches the network: obs_scales.ang_vel=0.25,
    # dof_pos=1.0 (no-op), dof_vel=0.05, lin_vel=2.0, ball_vel=0.2
    # (g1_29_config.py:191-201). SGK set no scale= on any term, leaving raw
    # magnitudes (e.g. dof_vel in rad/s, O(1-10)) unscaled going into the
    # network — a real divergence from G1's proven recipe. Added scale= to
    # every term with a direct G1 analog, term-for-term.
    #
    # NOT scaled, by design, matching G1's ACTUAL (not merely configured)
    # behavior: G1's obs_scales.ball_pos=0.3 is defined in config
    # (g1_29_config.py:198) but is dead — grep of legged_robot.py confirms it
    # is never applied to end_target_local or any other position term; G1's
    # ball position, hand position, and dist terms are all left as raw
    # meters. So ball_pos_b (position-type) and the SGK-only foot-position
    # terms below correctly stay unscaled to match G1's real mechanism, not
    # its aspirational-but-unused config value. projected_gravity and
    # actions are likewise left raw in G1 (no obs_scales entry touches them
    # at all).
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mjlab_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,  # matches G1 obs_scales.ang_vel
        ),
        "projected_gravity": ObservationTermCfg(
            func=mjlab_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=mjlab_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,  # matches G1 obs_scales.dof_pos (no-op, kept explicit)
        ),
        "joint_vel": ObservationTermCfg(
            func=mjlab_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,  # matches G1 obs_scales.dof_vel
        ),
        "actions": ObservationTermCfg(func=mjlab_mdp.last_action),
        # XY only — matches BoosterT1mjlab kick task for deployment compatibility.
        # Noise lives INSIDE the term (noise_scale), not on the manager: mjlab
        # applies manager noise after the term returns, which would re-noise the
        # gated zeros into a phantom ball. G1 noises first, then masks
        # (legged_robot.py:425-426) — noise_scale reproduces that ordering.
        # Unscaled — see item-21 note above (G1's ball_pos scale is dead code).
        "ball_pos_b": ObservationTermCfg(
            func=gk_mdp.ball_pos_xy_b,
            params={
                "ball_name": BALL_NAME,
                "always_visible": True,
                "hide_behind_torso": True,
                "hide_after_steps": 75,
                "noise_scale": 0.05,
            },
        ),
    }
    # Critic: actor terms + privileged info not available at deployment.
    # base_lin_vel, ball_vel_b, foot_pos_b help the value function during training
    # but are thrown away at deployment — only the actor network is used.
    # Critic also uses full 3D ball_pos_b for richer value estimation.
    # `scale=v.scale` propagated so shared terms (ang_vel/dof_pos/dof_vel) get
    # the same scaling on the critic path as the actor path — matches G1,
    # whose compute_termination_observations() applies the identical
    # obs_scales to the privileged/critic obs vector (legged_robot.py:447-455).
    critic_terms = {
        k: ObservationTermCfg(func=v.func, params=v.params, scale=v.scale)
        for k, v in actor_terms.items()
    }
    critic_terms["ball_pos_b"] = ObservationTermCfg(
        func=gk_mdp.ball_pos_b,
        params={"ball_name": BALL_NAME, "always_visible": True},
    )
    critic_terms.update({
        "base_lin_vel": ObservationTermCfg(
            func=gk_mdp.base_lin_vel,
            scale=2.0,  # matches G1 obs_scales.lin_vel
        ),
        "ball_vel_b": ObservationTermCfg(
            func=gk_mdp.ball_vel_b,
            params={"ball_name": BALL_NAME, "always_visible": True},
            scale=0.2,  # matches G1 obs_scales.ball_vel (applied at legged_robot.py:415)
        ),
        # No G1 analog (feet-only design). Natural magnitude relative to root
        # is already O(0.1-0.8) m (leg length), i.e. within G1's design intent
        # of O(0.1-1) scaled inputs — left unscaled rather than inventing an
        # arbitrary factor, consistent with G1 leaving its own unscaled
        # position-type terms (hand_pos, end_target_local, dist) raw.
        "left_foot_pos_b": ObservationTermCfg(
            func=gk_mdp.left_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
        "right_foot_pos_b": ObservationTermCfg(
            func=gk_mdp.right_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
    })

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

    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 1

    # FIX 2026-07-20 (item 22, obs-scaling audit): removed. G1 only delays
    # actions (domain_rand.delay=True, g1_29_config.py:292; applied in
    # step()'s action-interpolation loop, legged_robot.py:130-134) -- there is
    # no per-observation delay mechanism anywhere in G1's proven recipe. This
    # per-term delay_min_lag/delay_max_lag/delay_per_env block was a second,
    # compounding source of temporal staleness on top of G1's action delay
    # with no G1 analog. SGK's action-delay equivalent (BuiltinPositionActuatorCfg's
    # own delay_min_lag/delay_max_lag=1-3, t1_constants.py:47,53 etc.) is a
    # separate mechanism (per-actuator PD-target delay) and is untouched by
    # this removal.

    # ------------------------------------------------------------------
    # Rewards — aligned with Imitationlearningbooster proven structure.
    # stopball must come first: initialises env._sb_init_vx used by multiple terms.
    # ------------------------------------------------------------------
    cfg.rewards = {
        # --- primary task signal ---
        # FIX 2026-07-20 (reward audit item 8): static weight= values below
        # for stopball/softstop/single_foot_save/cleanstop/
        # inner_face_orientation_save/foot_inner_face_continuous now match
        # their curriculum's (cu=0) base_weight -- see the item-8 comment
        # above cfg.curriculum["softstop_curriculum"] for the full peak-
        # magnitude-cap derivation. These static values are what play mode
        # actually uses (curriculum is only registered `if not play`); before
        # this fix, stopball/softstop's static values were stale pre-halving
        # leftovers (15.0/105.0) that no longer matched their own curriculum's
        # base_weight (7.5/52.5 pre-cap) at all.
        # FIX 2026-07-23: added asset_cfg=_FEET_CFG to both stopball's and
        # softstop's params. Without it, their internal _get_reach_target_y
        # call fell back to that function's own default SceneEntityCfg --
        # a *different* object than _FEET_CFG that never appears in any
        # term's params dict, so mjlab's manager setup (which only resolves
        # SceneEntityCfg.body_ids for objects it finds inside a term's own
        # params -- manager_base.py:_resolve_common_term_cfg) never resolved
        # it. body_ids silently stayed the un-resolved default slice(None)
        # (all bodies) for the whole run. Since stopball is registered
        # first each tick -- the one call allowed to update the blue-landing
        # settle counter -- this made dist_to_blue permanently garbage on
        # exactly the call that mattered, blocking blue_ball_landed from
        # ever firing off a genuine plant even though every other
        # properly-resolved term (footreach etc.) saw the correct distance
        # the same tick. See rewards.py's stopball/softstop docstrings.
        "stopball": RewardTermCfg(
            func=gk_mdp.stopball,
            weight=5.21,
            params={"ball_name": BALL_NAME, "delta_vel_threshold": 0.6, "asset_cfg": _FEET_CFG},
        ),
        # --- partial deflection signal (fires before stopball; gates _ball_is_behind) ---
        "softstop": RewardTermCfg(
            func=gk_mdp.softstop,
            weight=36.46,
            params={"ball_name": BALL_NAME, "velocity_threshold": 0.05, "asset_cfg": _FEET_CFG},
        ),
        # --- single-foot save bonus: same foot first contacted AND caused the reversal ---
        "single_foot_save": RewardTermCfg(
            func=gk_mdp.single_foot_save,
            weight=34.72,
            params={"ball_name": BALL_NAME},
        ),
        # --- clean-trap bonus: continuous payout by how dead the ball is after deflection ---
        # FIX 2026-07-28 (user request): speed_threshold widened 0.10->1.0 (fire
        # condition, unchanged one-time latch) now that the payout itself is a
        # continuous scale (best_speed/decay_rate, see cleanstop()'s docstring)
        # instead of a binary bonus -- see rewards.py:cleanstop.
        "cleanstop": RewardTermCfg(
            func=gk_mdp.cleanstop,
            weight=17.36,
            params={"ball_name": BALL_NAME, "speed_threshold": 1.0, "best_speed": 0.2, "decay_rate": 3.75},
        ),
        # --- continuing close-to-target signal, tiered 1.0x/2.0x/3.0x by softstop/cleanstop
        # (ports G1 _reward_success; FIX 2026-07-27 retiered off stopball -- see rewards.py).
        # MUST stay registered after "cleanstop" above: success() reads env._cleanstop_flag,
        # which cleanstop() only sets when IT runs -- registering success first would read a
        # one-tick-stale value on the exact step cleanstop first fires. ---
        "success": RewardTermCfg(
            func=gk_mdp.success,
            weight=5.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG, "strict_th": 0.15},
        ),
        # --- save quality bonuses (fire on top of softstop, not as a gate) ---
        "inner_face_orientation_save": RewardTermCfg(
            func=gk_mdp.inner_face_orientation_save,
            weight=17.36,
            params={"ball_name": BALL_NAME, "alignment_threshold": 0.7, "asset_cfg": _FEET_CFG},
        ),
        "foot_inner_face_continuous": RewardTermCfg(
            func=gk_mdp.foot_inner_face_continuous,
            weight=3.47,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # --- NEW 2026-07-27 (user request): trailing foot has no orientation
        # shaping anywhere else in this table (the two terms above only ever
        # touch the leading/assigned foot) -- always active, no ~behind gate.
        "trailing_foot_forward_continuous": RewardTermCfg(
            func=gk_mdp.trailing_foot_forward_continuous,
            weight=3.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # --- NEW 2026-07-27 (user request): leading foot rotates back toward
        # forward once the save has happened (behind-gated) -- complementary
        # to foot_inner_face_continuous's (~behind)-gated sideways target, so
        # the two can never be active on the same step.
        "postleadfootorientation": RewardTermCfg(
            func=gk_mdp.postleadfootorientation,
            weight=2.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # NEW 2026-08-01 (user request): flat, time-boxed bonus (~0.2s) for
        # the assigned foot staying airborne right after the save, so
        # postleadfootorientation actually has hangtime to work with instead
        # of relying on whatever the dive physics happens to leave it. See
        # rewards.py:postsave_foot_airtime docstring.
        "postsave_foot_airtime": RewardTermCfg(
            func=gk_mdp.postsave_foot_airtime,
            weight=1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # NEW 2026-07-28 (user request): whole-body yaw heading recovery --
        # user observed the robot's hips/whole body drifting into a
        # left/right yaw post-save, not just a foot. postorientation (below)
        # is roll/pitch-only and yaw-invariant; ang_vel_z only penalizes yaw
        # rate, not final heading. No G1 equivalent (static catch task never
        # needs to reface). See rewards.py:postheadingorientation.
        "postheadingorientation": RewardTermCfg(
            func=gk_mdp.postheadingorientation,
            weight=2.5,
            params={"ball_name": BALL_NAME},
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
        # FIX 2026-07-26 (near-region oscillation): dense "close AND slow"
        # reward, ported from blue_stick_landing -- see rewards.py's
        # near_stick_reach docstring and docs/BugFixes.md. Zero for
        # wide/far-region envs by construction (near_stick_reach's own
        # env._blue_wide gate), so this entry cannot influence far-region
        # behavior despite sharing this config file with the blue_* terms.
        "near_stick_reach": RewardTermCfg(
            func=gk_mdp.near_stick_reach,
            weight=8.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # --- two-stage blue->green waypoint bonus, v2 reimplementation
        # (2026-07-23) of the blue-ball-waypoint branch mechanism, removed
        # from this project's lineage 2026-07-10. See rewards.py's
        # _get_reach_target_y/blue_ball_landed/blue_overshoot_penalty/
        # blue_stick_landing docstrings and docs/BugFixes.md. ---
        "blue_ball_landed": RewardTermCfg(
            func=gk_mdp.blue_ball_landed,
            weight=10.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # Without this, ignoring the blue waypoint entirely on a wide crossing
        # earns the same reward (zero, from the landing-gated terms above) as
        # attempting and failing -- no gradient discourages skipping it.
        # FIX 2026-07-23: -30.0 -> -60.0 (user request: "too lenient for the
        # blue ball"). Doubled to match base_weight below (curriculum-scaled
        # in step with blue_ball_landed/blue_stick_landing's own 2x growth).
        "blue_overshoot_penalty": RewardTermCfg(
            func=gk_mdp.blue_overshoot_penalty,
            weight=-60.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # Dense reward for "close AND slow" near blue -- the exact joint
        # condition the settle-window landing check requires.
        "blue_stick_landing": RewardTermCfg(
            func=gk_mdp.blue_stick_landing,
            weight=8.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        # NEW 2026-07-24 (user request): trunk (whole-body), not foot-specific,
        # velocity incentive toward the current two-stage target -- fills the
        # gap where footreach's own vel_sigma (foot velocity) only reactivates
        # once ball_x_local <= 1.5m, leaving no locomotion incentive for the
        # (often much longer) window between a genuine blue landing and the
        # ball finally closing in. See rewards.py:blue_trunk_drive docstring.
        "blue_trunk_drive": RewardTermCfg(
            func=gk_mdp.blue_trunk_drive,
            weight=5.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
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
        "foot_ang_vel_xy": RewardTermCfg(
            func=gk_mdp.foot_ang_vel_xy,
            # FIX 2026-07-20 (reward-weight audit): was -0.5. No G1
            # equivalent; live-measured real per-step magnitude (-1.80) was
            # meaningfully inflating raw task-reward scale and diluting
            # AMP's real proportional influence on the blended objective.
            # Halved rather than removed -- likely still doing real work.
            weight=-0.25,
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
        # FIX 2026-07-23: 1.0 -> 5.0, deliberate G1 divergence (G1 uses 1.0).
        # See postupperdofpos's docstring (rewards.py) -- stuck near its
        # floor on both master's last run and the v2 branch, unlike
        # postlegdofpos/postwaistdofpos at the same weight tier.
        "postupperdofpos": RewardTermCfg(
            func=gk_mdp.postupperdofpos,
            weight=5.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_ARM_CFG},
        ),
        # NEW 2026-07-30 (user request): supplementary to postupperdofpos --
        # specifically targets "hand above its own shoulder" rather than the
        # aggregate 8-joint pose error. Gated to the STEADY post-save window
        # only (see penalize_arm_above_shoulder's docstring, rewards.py) so
        # it can't fight legitimate dive/balance-recovery arm motion, the
        # exact failure mode that got arm_torque_limits/arm_action_rate_l2/
        # arm_action_acc_l2 reverted 2026-07-29. Weight modest/supplementary,
        # not the -100 "bad technique" tier. See docs/BugFixes.md.
        "penalize_arm_above_shoulder": RewardTermCfg(
            func=gk_mdp.penalize_arm_above_shoulder,
            weight=-2.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _ARM_HEIGHT_CFG},
        ),
        # NEW 2026-08-03 (user request): ported near-verbatim from sibling
        # project BoosterT1mjlab (tasks/velocity/mdp/rewards.py, weight -0.02
        # there), which uses this exact mechanism to "encourage natural arm
        # swing" for its own T1 locomotion policy. Reads the native MuJoCo
        # subtreeangmom sensor (root_angmom, already present in this
        # project's t1.xml/t1_headless.xml but never read by any code here
        # before this fix) -- whole-body angular momentum about the Trunk.
        # Physically-adaptive complement to postupperdofpos: doesn't fight
        # a genuine counterbalance swing (which legitimately produces high
        # momentum), only penalizes momentum that's still nonzero once the
        # body should already be stable. Small/gentle by design, matching
        # BoosterT1mjlab's own weight -- not meant to replace postupperdofpos.
        # See rewards.py:angular_momentum_penalty and docs/BugFixes.md.
        "angular_momentum_penalty": RewardTermCfg(
            func=gk_mdp.angular_momentum_penalty,
            weight=-0.02,
            params={},
        ),
        # FIX 2026-07-27: 1.0 -> 3.0. User reported the waist visibly
        # rotating post-save; training logs confirmed this reward stuck
        # near its floor (~0.31 mean episode reward), the same "stuck
        # near its floor relative to siblings" symptom -- and now
        # magnitude -- that motivated postupperdofpos's 2026-07-23 bump
        # (that fix's own comment above named postwaistdofpos as one of
        # the terms postupperdofpos was compared against; that comparison
        # is now stale). See postwaistdofpos's docstring (rewards.py) and
        # docs/BugFixes.md.
        "postwaistdofpos": RewardTermCfg(
            func=gk_mdp.postwaistdofpos,
            weight=3.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_WAIST_CFG},
        ),
        # FIX 2026-07-22: new leg-recovery term, no G1 equivalent to copy a
        # weight from -- matched to postupperdofpos's weight/shape since
        # both use the same exp(-1*err) exponent and play the analogous
        # role for this task's catching limb. See postlegdofpos's docstring
        # (rewards.py) and docs/BugFixes.md.
        "postlegdofpos": RewardTermCfg(
            func=gk_mdp.postlegdofpos,
            weight=1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_LEG_CFG},
        ),
        # --- hardware safety ---
        # FIX 2026-07-22: penalize_kneeheight (shank-based) unregistered,
        # replaced by penalize_baseheight (root/torso-based) -- user found
        # via play.py's new per-episode min-height + terminated_by
        # reporting (docs/BugFixes.md, same date) that shank height reads
        # low on legitimate deep lunges (a real athletic motion for this
        # foot-only task), producing false-positive gating; root height
        # stays reliably higher during an intentional lunge and only drops
        # on a genuine fall. Thresholds (0.59 here, 0.57 on the
        # base_height termination below) are the user's own values from
        # watching real play-session data, mirroring the prior
        # kneeheight(0.295)/shank_height(0.27) pair's ~2.5cm graded-
        # warning-before-hard-cutoff gap. penalize_kneeheight/
        # shank_height_termination functions kept (not deleted) in case
        # shank-based gating is revisited.
        "penalize_baseheight": RewardTermCfg(
            func=gk_mdp.penalize_baseheight,
            weight=-100.0,
            params={"min_height": 0.59},
        ),
        "penalize_sharpcontact": RewardTermCfg(
            func=gk_mdp.penalize_sharpcontact,
            weight=-100.0,
            params={"force_threshold": 1700.0},
        ),
        "penalize_self_collision": RewardTermCfg(
            func=gk_mdp.penalize_self_collision,
            weight=-50.0,
        ),
        # FIX 2026-07-22: new penalty, ball touching the non-assigned
        # ("non-leading") foot. See penalize_wrong_foot_ball_contact's
        # docstring (rewards.py) -- the existing correct-foot gates on
        # stopball/softstop/cleanstop use a ground-contact sensor that can't
        # tell "assigned foot happens to be on the ground" from "assigned
        # foot actually touched the ball", so a wrong-foot save could still
        # farm those rewards. This uses the ball-specific "ball_contact"
        # sensor directly, independent of those gates, to discourage it.
        # FIX 2026-07-23: -30 -> -100. Confirmed via training logs on both
        # master's last run and the v2 branch that both-feet catching is
        # still happening at essentially the same rate on both -- -30 wasn't
        # outweighing the save-quality benefit. See rewards.py docstring.
        "penalize_wrong_foot_ball_contact": RewardTermCfg(
            func=gk_mdp.penalize_wrong_foot_ball_contact,
            weight=-100.0,
            params={"ball_name": BALL_NAME},
        ),
        "feet_slippage": RewardTermCfg(
            func=gk_mdp.feet_slippage,
            # FIX 2026-07-20 (reward-weight audit): was 5.0 -- reverted to
            # G1's literal matched weight (feet_slippage=3.0,
            # g1_29_config.py). Real per-step magnitude (+2.38) was one of
            # the larger positive contributors, part of the raw task-reward
            # inflation diluting AMP's real share of the blended objective.
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
            # FIX 2026-07-20 (reward audit item 7): was a single flat 10 rad/s
            # cap for every joint (`vel_threshold=10.0`) -- T1's real per-joint
            # motor velocity limits (t1_constants.py ElectricActuator specs)
            # span ~7.3-16.4 rad/s, so the flat 10 silently never fired for
            # the slower joints (arms ~9.3, waist/hip-roll/hip-yaw ~7.3).
            # T1's MJCF defines no joint velocity limit at all (only a
            # position `range`), so there is no XML value to source instead --
            # confirmed by grepping xmls/t1_headless.xml and t1.xml for
            # <actuator>/<general>/velocity attributes (none exist). Switched
            # to per-joint limits x0.9 (soft_factor, matching G1's
            # soft_dof_vel_limit=0.9, g1_29_config.py:349) sourced from the
            # same real motor specs already used for action_scale/kp/kd --
            # see _T1_VEL_LIMIT_MAP in rewards.py and docs/BugFixes.md.
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "torque_limits": RewardTermCfg(
            func=gk_mdp.torque_limits,
            weight=-3.0,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        # REVERTED 2026-07-29 (user request): arm_torque_limits penalized
        # actual torque/effort regardless of resulting pose -- diagnosed
        # (alongside arm_action_rate_l2/arm_action_acc_l2 below) as fighting
        # legitimate counterbalance motion during a dive, dragging down
        # footreach/ball_exit/episode-length (confirmed via matched-iteration
        # comparison against the pre-2026-07-28 run, docs/BugFixes.md). User
        # wants arm posture controlled by postupperdofpos (a pose-matching
        # term, not a movement/effort penalty) instead -- see that term's
        # during_scale bump below. arm_dof_vel (further down) is kept: same
        # movement-penalty family, but a real G1-matched mechanism (dof_vel)
        # at a small, mostly-inert weight, not one of these three speculative
        # SGK-only additions.
        # --- stability ---
        "ang_vel_xy": RewardTermCfg(
            func=gk_mdp.ang_vel_xy_l2,
            # FIX 2026-07-20 (reward audit item 5): was -0.5, drifted 5x from
            # G1's -0.1 (g1_29_config.py:323). Commit ee7eb04 ("stronger body
            # roll/pitch penalty") changed -0.1 -> -0.5 as an ad hoc training
            # tweak with no G1-based justification (not "read G1 and found a
            # reason to diverge" -- just "stronger penalty felt needed" at
            # the time) -- CLAUDE.md's own Reward Design table kept -0.1 the
            # whole time, confirming this was accidental drift the doc never
            # caught up with, not a documented decision. Reverted to G1's
            # literal value.
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
        # --- regularisation ---
        "torques": RewardTermCfg(
            func=gk_mdp.torques_normalized_l2,
            weight=-1e-5,
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2,
            # FIX 2026-07-20 (reward audit item 4): was -0.1, mistakenly
            # carrying G1's smoothness=-0.1 match (g1_29_config.py:325) here.
            # mjlab's action_rate_l2 is `action - prev_action` -- FIRST order
            # (mjlab/envs/mdp/rewards.py) -- so it is NOT the structural
            # analog of G1's `_reward_smoothness`, which is SECOND order
            # (`actions - 2*last_actions + last_last_actions`,
            # legged_robot.py:1532-1534). That match belongs to action_acc_l2
            # (below), which mjlab implements as
            # `action - 2*prev_action + prev_prev_action` -- the identical
            # formula. Swapped: action_acc_l2 now takes G1's -0.1; this term
            # (genuinely no G1 equivalent) gets a reasoned, halved value
            # following the same day's halving treatment of other
            # no-G1-equivalent terms found to be inflating raw task-reward
            # magnitude (foot_ang_vel_xy -0.5->-0.25, and the old,
            # mislabeled action_acc_l2 -0.1->-0.05).
            weight=-0.05,
        ),
        "action_acc_l2": RewardTermCfg(
            func=mjlab_mdp.action_acc_l2,
            # FIX 2026-07-20 (reward audit item 4): was -0.05, mislabeled "no
            # G1 equivalent". mjlab's action_acc_l2
            # (`action - 2*prev_action + prev_prev_action`) IS the exact
            # structural match for G1's `_reward_smoothness`
            # (legged_robot.py:1532-1534: `actions - last_actions -
            # last_actions + last_last_actions`, algebraically identical) --
            # weight=-0.1 in g1_29_config.py:325. Reverted to G1's literal
            # matched weight; see action_rate_l2's comment for the full swap
            # rationale.
            weight=-0.1,
        ),
        # REVERTED 2026-07-29 (user request): arm_action_rate_l2/
        # arm_action_acc_l2 penalized how fast the commanded arm target
        # changed, again independent of resulting pose -- same "wrong
        # problem class" diagnosis as arm_torque_limits above. Reverted
        # alongside it; see that entry's comment and docs/BugFixes.md for
        # the full regression evidence. The underlying custom functions
        # (_resolve_arm_action_indices helper included) were removed from
        # rewards.py since nothing else calls them.
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
        # NEW 2026-07-28 (user request): dof_vel above is diluted across all
        # ~21 joints, giving any single arm joint's swing only a tiny share
        # of the gradient. User reported the arms swinging noticeably for
        # counterbalance -- this scopes the same joint_vel_l2 penalty to
        # just the 8 arm joints (_RECOVERY_ARM_CFG, already defined above)
        # for a concentrated, undiluted signal. No G1 equivalent to match --
        # G1's smoothness/action-rate terms are flat, non-curriculum, and
        # whole-body the entire run (checked g1_29_config.py -- no
        # arm-specific or curriculum-scaled smoothness mechanism exists;
        # g1_29_config.py's unrelated `curriculum_joints` list touches the
        # same joint group but for actuator-noise-injection curriculum, not
        # a reward). Weight is a first empirical guess (10x dof_vel's
        # diluted per-joint share, since scoped to 8 joints instead of ~21),
        # not a G1-matched value -- not yet validated against a live run.
        "arm_dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2,
            weight=-5e-3,
            params={"asset_cfg": _RECOVERY_ARM_CFG},
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
            "dist_range":     (1.5, 3.5),
            "y_start_range":  (-0.3, 0.3),
            "y_end_range":    (-1.1, 1.1),
            "t_flight_range": (0.7, 1.1),
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
        # FIX 2026-07-22 (user judgment call from real play-session data,
        # via play.py's new min-height/terminated_by reporting): was 0.4,
        # an untuned starting value. Retuned to 0.57, paired with
        # penalize_baseheight's 0.59 above (~2.5cm graded-warning gap,
        # matching the prior kneeheight/shank_height pair's design). The
        # shank_height termination below is REMOVED (not just loosened) --
        # user found it false-positives on legitimate deep lunges; root
        # height is the more reliable fall-vs-lunge signal for this task.
        "base_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.57, "asset_cfg": _ROBOT_CFG},
        ),
        "ball_exit": TerminationTermCfg(
            func=gk_mdp.ball_exit_termination,
            params={"ball_name": BALL_NAME, "behind_threshold": -0.5},
            time_out=False,
        ),
        "sharpforce": TerminationTermCfg(
            func=gk_mdp.sharpforce_termination,
            params={"max_contact_force": 2500.0},
            time_out=False,
        ),
        # shank_height REMOVED 2026-07-22 -- see base_height's FIX comment
        # above and docs/BugFixes.md. gk_mdp.shank_height_termination kept
        # defined (events.py) in case shank-based gating is revisited.
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
        # In-term ball noise must be zeroed explicitly — enable_corruption only
        # disables manager-level noise, and ball_pos_b noises inside the term
        # (G1 noise-before-mask ordering).
        cfg.observations["actor"].terms["ball_pos_b"].params["noise_scale"] = 0.0
        cfg.terminations.pop("out_of_terrain_bounds", None)
        # No disturbance pushes during play/eval — mirrors kick task play mode.
        cfg.events.pop("push_robot", None)
        # RSI disabled in play mode by default — mirrors BoosterT1mjlab kicking task
        # which never registers reset_from_motion_data in play. Always starting from
        # standing makes play behaviour deterministic and consistent across episodes.
        # Use --no-rsi False in the play script to re-enable RSI if needed.
        cfg.events.pop("reset_from_motion_data", None)
        # Play: rolling ball — same function as training so visualisation matches
        # the distribution the policy was trained on. vz=0 keeps ball at foot level.
        cfg.events["reset_ball"] = EventTermCfg(
            func=gk_mdp.reset_ball_rolling,
            mode="reset",
            params={
                "ball_name":     BALL_NAME,
                "dist_range":    (1.5, 3.5),
                "y_start_range": (-0.3, 0.3),
                "y_end_range":   (-1.1, 1.1),
                "t_flight_range": (0.7, 1.1),
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
