"""Goalkeeper reward terms for SimpleGoalKeeper (Phase 1 — feet only)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv

from simple_goalkeeper.robots.t1_constants import (
    ANKLE_ACTUATOR,
    ARM_ACTUATOR,
    HIP_PITCH_ACTUATOR,
    KNEE_ACTUATOR,
    WAIST_HIP_ROLL_YAW_ACTUATOR,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")

# FIX 2026-08-01 (user request): leading-foot "block posture" target angle,
# used by inner_face_orientation_save/foot_inner_face_continuous. Was a pure
# axis target (foot long axis parallel to Y = 90 deg off forward, i.e. fully
# sideways). Now 60 deg off forward (30 deg off full-sideways Y) -- a
# shallower, more forward-leaning block stance. expected_sign flips the Y
# component: +sin for the left foot's +Y-leaning target, -sin for the right
# foot's -Y-leaning target.
#
# FIX 2026-08-07 (user request): 60 -> 45 deg off forward (45 deg off Y,
# i.e. a diagonal target, exactly halfway between fully-forward and
# fully-sideways). Root cause this targets: T1's ankle has no yaw DOF, so
# ANY commanded foot-heading rotation goes entirely through Hip_Yaw, which
# reaction-torques the trunk into a whole-body yaw spin (root-caused
# 2026-08-03, mirror-symmetric +11.3/-5.1 deg drift by save side; a live
# far-left-region checkpoint replay this session showed the resulting
# ang_vel_z NOT decaying -- sustained near-constant yaw rate for 20+
# consecutive steps well past every gating window, i.e. nothing was
# actively arresting it once the reward windows closed). The 60 deg target
# demanded more Hip_Yaw excursion than necessary; 45 deg reduces the total
# commanded rotation (and therefore the reaction torque) while still
# asking for a genuinely turned, not flat-forward, block posture -- user's
# own framing: "the landing 90 degrees off is really unstable." Paired
# with `foot_ang_vel_z` (same session, damps rotation SPEED) -- this
# reduces the rotation ANGLE being demanded in the first place, a
# complementary lever on the same root cause.
# FIX 2026-08-08 (user request): reverted 45->60 ("back to the 30 degrees
# from the y axis again" -- this project's off-forward/off-Y convention
# maps 60 off-forward to 30 off-Y, i.e. the exact 2026-08-01 value, not a
# new one). alignment_threshold deliberately left at 0.85 (not reverted to
# 0.7) per explicit user request -- a narrower cone than the original 60deg
# calibration used, not yet re-validated against a live training run.
#
# FIX 2026-08-10 (user request, model_11250.pt replay): 60->75 ("revert
# back to 15 degrees from the y axis") -- user observed the leading foot
# almost not rotating in Z toward the commanded angle at all. This makes
# the target MORE extreme (closer to full-sideways), the opposite direction
# of every prior yaw-spin-motivated fix (60->45->60) -- deliberate: a
# shallower target gives a high "do nothing" reward even at zero rotation
# (cos(60deg)=0.5), supplying little gradient to actually turn the foot;
# 75deg (cos(75deg)=0.26 at zero rotation) punishes staying put harder.
# alignment_threshold (0.85) left unchanged, not re-examined as part of
# this fix. Not yet validated against a live training run.
#
# FIX 2026-08-13 (user request): reverted 75->60 -- back to the exact value
# that trained model_17250.pt (the checkpoint the user confirmed correctly
# rotated the foot; verified via git history this was the actual value in
# effect when that run launched, commit 65d6852). Applied together with
# foot_ang_vel_z's new post-save-only gate (see that function's docstring)
# and alignment_threshold's matching revert (goalkeeper_env_cfg.py) -- the
# combined hypothesis is the steeper/tighter 75deg+0.85 target plus an
# unconditional yaw-rate penalty made "don't rotate" locally optimal where
# the baseline run's easier target + no pre-save yaw penalty made rotating
# clearly worth it. Not yet validated against a live training run.
#
# FIX 2026-08-15 (user request): 60 -> 50 deg off forward. Root cause:
# rendered-image + FK analysis (docs/superpowers -- see the 2026-08-15
# "Inner-Face Reach Limit" artifact) found T1's Hip_Yaw -- the ONLY DOF that
# can rotate the foot's heading, since the ankle has no yaw axis -- physically
# tops out at +/-57.30 deg (hard joint range) and only sweeps the foot
# +/-54.17 deg off forward from a neutral stance (kinematic coupling loses a
# further ~3 deg beyond the raw joint limit). 60 deg was never fully
# reachable; the foot could get to within ~5.8 deg of it at best (neutral
# stance) and off a real save-landing pose the gap measured 10.84 deg. 45 deg
# was considered and rejected: it's the same direction as the 2026-08-10 fix
# above (a shallower target raises the do-nothing reward -- cos(45deg)=0.71
# vs cos(60deg)=0.50 -- which is exactly what made the foot barely rotate
# before 75deg was tried). 50 deg sits inside the ~54.17 deg physical ceiling
# with margin (not pinned at the hard limit) while staying steeper than 45deg
# (cos(50deg)=0.64) so the do-nothing gradient doesn't regress as far. Not yet
# validated against a live training run.
# FIX 2026-08-16 (user request, "i want the target of the foot to be at 80
# degrees, because left near literally shows it is possible"): 50 -> 80.
# The 54.17deg "physical ceiling" reasoning above is now known to be
# incomplete -- it was derived from Hip_Yaw's own joint range alone, before
# the same-day investigation found (a) Waist contributes a second, larger
# (+/-90deg) yaw source between Trunk and the legs, and (b) the angle this
# reward actually measures (now the true toe-axis root-local azimuth, see
# foot_inner_face_continuous's own 2026-08-16 fix history) also picks up a
# real Ankle_Pitch gimbal-coupling contribution at large yaw. Live-checked
# against model_13000.pt (a checkpoint trained under the OLD 50deg target,
# so not itself evidence of what 80deg trains to): left_near's real
# achieved toe-axis azimuth averaged ~89deg world-frame / ~77deg
# robot-local (after subtracting ~12deg average Trunk world spin) at the
# softstop-firing moment -- 80deg sits within that observed range, not
# past it. Not yet validated against a live training run under this target.
# FIX 2026-08-22 (user request): retargeted 80->70 -- see docs/BugFixes.md.
# FIX 2026-08-23 (user request): retargeted 70->55.
_FOOT_TARGET_ANGLE_DEG = 55.0
_FOOT_TARGET_COS = math.cos(math.radians(_FOOT_TARGET_ANGLE_DEG))
_FOOT_TARGET_SIN = math.sin(math.radians(_FOOT_TARGET_ANGLE_DEG))

# FIX 2026-08-05 (user request): overshoot-side steepening for
# foot_inner_face_continuous. Was a plain cos(angle-target) on both sides of
# the target -- symmetric in shape but very flat near its own peak (a 30deg
# overshoot, i.e. drifting back to the OLD 90deg value, only cost ~13% of
# peak value: cos(30deg)=0.87). A live checkpoint replay (model_39750,
# 2026-08-04_16-43-37_6144_airtimerampfix run) found exactly this drift:
# far-region saves converged to ~69deg (near target), near-region saves
# stayed at ~83-85deg (near the OLD target) -- the weak overshoot gradient
# was too easily overridden by stopball/softstop's much heavier pull toward
# a near-perpendicular block angle for near-region's more head-on ball
# trajectories. Same steepness convention as foot_clearance's
# clearance_sigma=300 (exp(-clearance_sigma*(height-target)^2)): chosen so
# error = target_magnitude (foot_clearance: 0.10m; here: the 30deg gap back
# to the old 90deg value) scores ~0.05, half that error scores ~0.47.
# Applied ONLY on the overshoot side (foot rotated past the target, on the
# correct side) -- the undershoot/wrong-side branch keeps the original
# cos(angle-target) formula unchanged, preserving the deliberate 2026-07-10
# fix (a plain zero-floor reward can't distinguish "never rotated" from
# "rotated the wrong way"; cos's asymmetric shape around a non-90deg target
# already does distinguish them, and a symmetric Gaussian rescale would
# have erased that distinction).
#
# FIX 2026-08-15 (user request): 0.00333 -> 0.03, following the target
# retarget 60->50 (see _FOOT_TARGET_ANGLE_DEG above). The original value was
# calibrated so error = target_magnitude (the 30deg gap from the then-target
# 60deg back to the old 90deg value) scores ~0.05 -- at the NEW 50deg target,
# reusing 0.00333 would score the equivalent "one full step back" overshoot
# (50->60deg, a 10deg gap -- 60 being the OLD target, the exact drift this
# term exists to punish) at exp(-0.00333*10^2)=0.72, only a ~28% drop: far
# too gentle to read as "drastic." Same steepness convention, rescaled to the
# new, smaller reference gap: error=10deg scores ~0.05 (exp(-0.03*10^2)=0.05),
# half that error (5deg) scores ~0.47 -- identical calibration philosophy to
# the 2026-08-05 fix above, just re-anchored to the new target's own "back to
# the old target" distance instead of the old target's.
# FIX 2026-08-16 (user request, "i want sharper drop off above 80
# degrees"): 0.03 -> 0.10 (chosen via AskUserQuestion from
# moderate(0.06)/sharp(0.10)/very sharp(0.15) previews). At the max
# curriculum weight (6.94): 5deg overshoot now scores 0.57 (8% of peak,
# was 3.28/47%), 8deg scores 0.01 (was 1.02/15%). Same reference-gap
# calibration philosophy as the 2026-08-05/08-15 fixes above, just a
# steeper value picked directly by the user rather than re-derived from a
# specific reference gap. Not yet validated against a live training run.
_FOOT_OVERSHOOT_SIGMA = 0.10

# FIX 2026-08-16 (user request, "the difference between 45 deg and the
# wanted 80 deg is only 1.3 of reward is this good enough?"): at the
# current max curriculum weight (6.94), cos(err) gave 45deg-off-target
# (35deg short of the 80deg _FOOT_TARGET_ANGLE_DEG) 82% of peak reward --
# a 1.3-point gap, confirmed live-observed and verified analytically. Too
# flat to strongly incentivize full commitment given this term's own
# modest peak next to much larger terms elsewhere (stopball/softstop up to
# 90+). First tried replacing cos(err) with a plain Gaussian
# (sigma=0.0015, chosen via AskUserQuestion from gentle/moderate/steep) --
# reverted before shipping: a plain Gaussian is floored at 0, silently
# erasing the deliberate 2026-07-10 fix where cos(err) goes NEGATIVE for a
# wrong-direction rotation (distinguishing it from "never rotated," which
# a zero-floored shape can't do -- see foot_inner_face_continuous's own
# fix history). Real fix: sign-preserving power of cosine,
# `sign(cos(err)) * |cos(err)|^_FOOT_UNDERSHOOT_POWER` -- keeps cos's
# natural sign (restores the 2026-07-10 distinguishing behavior) while the
# power steepens the falloff near the peak. Power=9 (odd, to keep the sign
# flip) chosen by matching the approved Gaussian's own curve numerically:
# cos(35deg)^n=0.16 -> n=9.18, rounded to 9, verified within ~0.05 of the
# Gaussian's values across the full undershoot range. 45deg-off now scores
# ~17% of peak (was 82% under plain cosine), fades out below ~30deg,
# genuinely negative past +/-90deg from target.
#
# FIX 2026-08-17 (user request, "it didn't learn to rotate the foot"):
# 9 -> 5. At power=9 the curve was apparently too steep to supply usable
# gradient from a mostly-unrotated start -- 45deg-off-target (35deg short
# of the 80deg target) scored only ~17% of peak, and points further out
# score even less, giving little incentive to begin rotating at all. At
# power=5, 45deg-off now scores ~38% of peak -- flatter near the start of
# the climb while keeping the same sign-preserving shape (still negative
# past +/-90deg from target, still distinguishes "wrong direction" from
# "not yet rotated"). Chosen via AskUserQuestion (5/3/7 offered). Not yet
# validated against a live training run.
_FOOT_UNDERSHOOT_POWER = 5
_DEFAULT_KNEE_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
# FIX 2026-08-03 (user request, G1-comparison finding): postupperdofpos's ONLY
# consumer. Was Shoulder_Pitch/Roll + Elbow_Pitch/Yaw x2 sides (8 joints) --
# now Elbow_Pitch/Yaw only (4 joints), matching G1's actual scope exactly.
# G1's own post-catch upper-body recovery reward (_reward_postupperdofpos,
# Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:1502-1507)
# never includes shoulder: `upper_body_joint_indices = cat(elbow_joint_indices,
# wrist_joint_indices)` (legged_robot.py:1315), built from cfg.control.elbow_joints
# (2 total) + wrist_joints (6 total) (g1_29_config.py:175,177) -- shoulder is
# never in this specific reward's target set. SGK's version included shoulder,
# the joint with by far the largest dive-reach excursion, into the SAME
# exp(-kernel_scale*sum_sq_err) kernel as the small-range joints -- confirmed
# via live checkpoint replay (rewards.py's own postupperdofpos docstring
# history) that post-save error is ~50x larger in far regions (7.81) than
# near (0.155), saturating the kernel to near-zero gradient regardless of
# kernel_scale tuning. Once a joint pins at a static extreme under that
# saturated kernel, nothing else in the table pulls it back (arm_dof_vel/
# action_rate_l2/action_acc_l2 penalize MOVEMENT, not position -- a still
# joint reads ~0 on all of them). G1 avoids this entirely by never asking
# this reward to recover the joint that legitimately needs to travel far for
# a reach/dive. Dropping shoulder mirrors that -- postupperdofpos no longer
# fights genuine counterbalance need, matching G1's actual design rather
# than a literal-but-wrong joint-count port. Not yet validated against a
# live training run. See docs/BugFixes.md.
_ARM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
# NEW 2026-08-06 (user request): postshoulderdofpos's ONLY consumer. Deliberately
# a SEPARATE SceneEntityCfg/reward from _ARM_JOINT_CFG/postupperdofpos rather
# than adding shoulder back into that term's scope -- the whole reason shoulder
# was removed from postupperdofpos (2026-08-03 fix, see that function's
# docstring) was that mixing shoulder's large dive-reach excursion into the
# same sum-of-squares as elbow's much smaller range saturated the shared
# exp(-kernel_scale*err) kernel for BOTH joints at once. Keeping shoulder in
# its own kernel means its (much larger) error magnitude can be tuned
# independently without dragging elbow's sensitivity along with it.
_SHOULDER_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll",
    ),
)
_WAIST_JOINT_CFG_RECOVERY = SceneEntityCfg("robot", joint_names=("Waist",))
# NEW 2026-07-30: local default mirroring goalkeeper_env_cfg.py's _ARM_HEIGHT_CFG
# (same body order requirement -- see penalize_arm_above_shoulder's docstring).
_ARM_HEIGHT_CFG = SceneEntityCfg(
    "robot",
    body_names=("AL2", "AR2", "left_hand_link", "right_hand_link"),
)
# FIX 2026-07-22: added alongside postupperdofpos/postwaistdofpos -- see
# postlegdofpos's docstring below for why G1 has no equivalent reward to
# port for these joints, and why that's a task-structure fact, not a gap
# in the existing G1-parity port.
_LEG_JOINT_CFG_RECOVERY = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Hip_Pitch", "Left_Knee_Pitch",
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Hip_Pitch", "Right_Knee_Pitch",
        "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
)
_ALL_JOINTS_CFG = SceneEntityCfg("robot")

# Per-joint stiffness (kp) for torque normalisation. Source: ILB _T1_KP_MAP,
# which was validated on real T1 hardware via KaydenKnapik/BoosterT1mjlab.
_T1_KP_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": 15.0, "Left_Shoulder_Roll": 15.0,
    "Left_Elbow_Pitch":    15.0, "Left_Elbow_Yaw":     15.0,
    "Right_Shoulder_Pitch": 15.0, "Right_Shoulder_Roll": 15.0,
    "Right_Elbow_Pitch":   15.0, "Right_Elbow_Yaw":    15.0,
    "Waist":               80.0,
    "Left_Hip_Pitch":     120.0, "Right_Hip_Pitch":    120.0,
    "Left_Hip_Roll":       80.0, "Left_Hip_Yaw":        80.0,
    "Right_Hip_Roll":      80.0, "Right_Hip_Yaw":       80.0,
    "Left_Knee_Pitch":    200.0, "Right_Knee_Pitch":   200.0,
    "Left_Ankle_Pitch":    50.0, "Right_Ankle_Pitch":   50.0,
    "Left_Ankle_Roll":     40.0, "Right_Ankle_Roll":    40.0,
}

# Per-joint effort limits from T1_serial_clean.xml actuatorfrcrange.
# Source: ILB _T1_EFFORT_MAP (same hardware-validated values).
_T1_EFFORT_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": 36.0, "Left_Shoulder_Roll": 36.0,
    "Left_Elbow_Pitch":    36.0, "Left_Elbow_Yaw":     36.0,
    "Right_Shoulder_Pitch": 36.0, "Right_Shoulder_Roll": 36.0,
    "Right_Elbow_Pitch":   36.0, "Right_Elbow_Yaw":    36.0,
    "Waist":               40.0,
    "Left_Hip_Pitch":      55.0, "Right_Hip_Pitch":     55.0,
    "Left_Hip_Roll":       40.0, "Left_Hip_Yaw":        40.0,
    "Right_Hip_Roll":      40.0, "Right_Hip_Yaw":       40.0,
    "Left_Knee_Pitch":     65.0, "Right_Knee_Pitch":    65.0,
    "Left_Ankle_Pitch":    50.0, "Right_Ankle_Pitch":   50.0,
    "Left_Ankle_Roll":     50.0, "Right_Ankle_Roll":    50.0,
}

# Per-joint velocity limits (rad/s), from real T1 motor specs already defined
# in t1_constants.py (ElectricActuator.velocity_limit, rated RPM converted to
# rad/s -- t1_constants._rpm()). The T1 MJCF (xmls/t1_headless.xml) defines no
# <actuator>/<general> block and no per-joint velocity attribute at all --
# joints only carry a position `range` -- so there is no XML-sourced velocity
# limit to read; MuJoCo does not require or enforce one on its own. These
# values are the correct, confidently-sourced alternative: real hardware motor
# ratings already used elsewhere in this codebase (ARM/WAIST_HIP_ROLL_YAW/
# HIP_PITCH/KNEE/ANKLE actuators feed action_scale and kp/kd), just never
# wired into a joint-velocity limit before now. Source of item 7's 2026-07-20
# reward audit fix -- see docs/BugFixes.md.
_T1_VEL_LIMIT_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": ARM_ACTUATOR.velocity_limit, "Left_Shoulder_Roll": ARM_ACTUATOR.velocity_limit,
    "Left_Elbow_Pitch":    ARM_ACTUATOR.velocity_limit, "Left_Elbow_Yaw":     ARM_ACTUATOR.velocity_limit,
    "Right_Shoulder_Pitch": ARM_ACTUATOR.velocity_limit, "Right_Shoulder_Roll": ARM_ACTUATOR.velocity_limit,
    "Right_Elbow_Pitch":   ARM_ACTUATOR.velocity_limit, "Right_Elbow_Yaw":    ARM_ACTUATOR.velocity_limit,
    "Waist":               WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Left_Hip_Pitch":      HIP_PITCH_ACTUATOR.velocity_limit, "Right_Hip_Pitch": HIP_PITCH_ACTUATOR.velocity_limit,
    "Left_Hip_Roll":       WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, "Left_Hip_Yaw": WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Right_Hip_Roll":      WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, "Right_Hip_Yaw": WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Left_Knee_Pitch":     KNEE_ACTUATOR.velocity_limit, "Right_Knee_Pitch": KNEE_ACTUATOR.velocity_limit,
    "Left_Ankle_Pitch":    ANKLE_ACTUATOR.velocity_limit, "Right_Ankle_Pitch": ANKLE_ACTUATOR.velocity_limit,
    "Left_Ankle_Roll":     ANKLE_ACTUATOR.velocity_limit, "Right_Ankle_Roll":  ANKLE_ACTUATOR.velocity_limit,
}


# Post-save recovery stance: straight legs (unchanged rationale below), arms
# matching HOME_KEYFRAME's exact pose, waist centered.
#
# FIX 2026-07-27 (user request, "look more soccer-like"): arms were
# originally a flat 45-deg pure-roll T-pose (Shoulder_Pitch/Elbow_Pitch/
# Elbow_Yaw all 0.0) -- user found this looked like the arms were "really
# far out" compared to HOME_KEYFRAME's own arm pose. Root cause wasn't just
# the roll angle (45 deg vs HOME_KEYFRAME's 23.5 deg) -- HOME_KEYFRAME also
# bends Shoulder_Pitch/Elbow_Pitch/Elbow_Yaw, which visually tucks the arm
# down and in; zeroing those (as the original T-pose did) reads as a much
# more extreme, fully-extended pose regardless of roll angle. Considered
# copying HOME_KEYFRAME wholesale (arms AND legs) but rejected: HOME_KEYFRAME
# includes the crouched legs (Hip_Pitch=-0.3/Knee_Pitch=0.6/Ankle_Pitch=-0.3)
# that motivated this whole stance-retarget fix in the first place -- a
# bent-knee crouch requires continuous active torque to hold and training
# logs showed the policy wasn't reliably settling into it post-save
# (postlegdofpos stuck ~0.07). Reverting the legs would risk reintroducing
# that exact measured problem. Kept straight legs, replaced ONLY the arm
# values with HOME_KEYFRAME's exact numbers (t1_constants.py:98-103).
#
# FIX 2026-07-27 (CORRECTED, then fine-tuned, THEN matched to Booster's own
# official T1 walk-policy default pose): earlier same-day edits chased
# Shoulder_Roll toward 0.0 believing that was "hanging down" -- confirmed
# via render it was actually this joint's T-POSE reference; corrected
# toward its range limit, then hand-tuned to 20deg of flare (1.2217 rad).
# User then asked to check Booster Robotics' own official T1 walking
# controller's default pose (booster_deploy repo,
# tasks/locomotion/locomotion.py:205-214, T1WalkControllerCfg) --
# Shoulder_Pitch=0.2, Shoulder_Roll=∓1.3, Elbow_Pitch=0.0, Elbow_Yaw=∓0.5
# (same Left-negative/Right-positive sign convention already used here).
# All four arm values below now match that official pose exactly, mirroring
# HOME_KEYFRAME's own value (t1_constants.py) -- keep them in sync if
# either changes again. See docs/BugFixes.md for the misdiagnosis writeup,
# reference images, and this comparison. Every joint in _RECOVERY_ARM_CFG/
# _RECOVERY_LEG_CFG/_RECOVERY_WAIST_CFG has an explicit entry below --
# including ones that don't change from default_joint_pos (Hip_Roll,
# Hip_Yaw, Ankle_Roll, Waist, all already 0.0) -- so the map is a complete,
# explicit stance definition, not a partial override.
_POST_SAVE_STANCE_MAP: dict[str, float] = {
    "Left_Hip_Roll": 0.0,          "Right_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,           "Right_Hip_Yaw": 0.0,
    "Left_Hip_Pitch": 0.0,         "Right_Hip_Pitch": 0.0,
    "Left_Knee_Pitch": 0.0,        "Right_Knee_Pitch": 0.0,
    "Left_Ankle_Pitch": 0.0,       "Right_Ankle_Pitch": 0.0,
    "Left_Ankle_Roll": 0.0,        "Right_Ankle_Roll": 0.0,
    "Left_Shoulder_Pitch": 0.2,    "Right_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.3,    "Right_Shoulder_Roll": 1.3,
    "Left_Elbow_Pitch": 0.0,       "Right_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,        "Right_Elbow_Yaw": 0.5,
    "Waist": 0.0,
}


def _get_correct_foot_idx(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Determine which foot (left=0, right=1) should contact the ball.

    Ball crossing Y > env origin Y → left foot (0) should reach.
    Ball crossing Y <= env origin Y → right foot (1) should reach.

    Returns: (N,) tensor of dtype int64, values 0 or 1.
    """
    crossing_y = _get_ball_crossing_y(env, ball_name)
    is_left_ball = crossing_y > env.scene.env_origins[:, 1]
    foot_idx = torch.where(is_left_ball,
                           torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                           torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    return foot_idx


_SOLE_ZONE_LOCAL_POS = (0.0125, 0.0, -0.032)
"""Must match left_sole_vis/right_sole_vis's `pos` in t1_headless.xml exactly
(both feet use the identical local offset). Same "duplicated, must-match-XML
constant" class as scripts/play.py's own copy of these two values -- keep
both in sync if the XML geom ever moves."""

_SOLE_ZONE_HALF_SIZE = (0.065, 0.03, 0.005)
"""Must match left_sole_vis/right_sole_vis's `size` (box half-extents) in
t1_headless.xml exactly. See _SOLE_ZONE_LOCAL_POS above."""

_SOLE_ZONE_BALL_RADIUS = 0.10
"""Ball radius for the sphere-vs-box check below. Matches the existing
`_BALL_GEOM_RADIUS`/`_BALL_GEOM_RADIUS` convention already used elsewhere in
this file and in scripts/play.py's own copy of this constant."""


def _sole_ball_contact_per_foot(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """(N, 2) bool: does the ball's sphere genuinely overlap [left, right]
    `_sole_vis` marker box's EXACT geometry (pos/size), vectorized across all
    envs.

    NEW 2026-08-15 (user request): "if the contact is detected between the
    sole of the foot and the ball that the reward for cleanstop is not
    given, because i only want to save with the side of the feet." Reuses
    the identical literal sphere-vs-box check `scripts/play.py`'s
    `_compute_sole_ball_contact` uses for its P-panel diagnostic (added the
    same session, per the user's explicit "the red marker is not cosmetic i
    wanted to use that geom quite literally") -- same geometry, same
    reasoning, now also driving a training-affecting gate, not just a
    viewer plot. Deliberately NOT read from a `ball_contact` ContactSensorCfg
    (the real, larger `foot[1-4]_collision` capsules) -- that sensor's
    footprint is bigger than the marker box the user explicitly wants used
    literally; reusing it here would silently gate on a different, larger
    zone than what the user actually specified.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]    # (N, 2, 3)
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)
    ball_pos_w = ball.data.root_link_pos_w  # (N, 3)

    device = foot_pos_w.device
    box_center = torch.tensor(_SOLE_ZONE_LOCAL_POS, device=device, dtype=torch.float32)
    box_half = torch.tensor(_SOLE_ZONE_HALF_SIZE, device=device, dtype=torch.float32)

    rel_w = ball_pos_w.unsqueeze(1) - foot_pos_w  # (N, 2, 3)
    n = env.num_envs
    local = quat_apply_inverse(
        foot_quat_w.reshape(n * 2, 4), rel_w.reshape(n * 2, 3)
    ).reshape(n, 2, 3)
    offset = local - box_center
    closest_offset = torch.clamp(offset, -box_half, box_half)
    dist = (offset - closest_offset).norm(dim=-1)  # (N, 2)
    return dist <= _SOLE_ZONE_BALL_RADIUS


def _ball_is_behind(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Bool mask (N,): ball has been deflected OR has crossed the goal line.

    Fires when ANY of:
      - ball_x < 0 (crossed goal line)
      - _softstop_flag is set (ball reversed to positive X velocity)
      - _sb_flag is set (delta_vx > 1.0 — hard contact, even without full reversal)

    Both flags needed: softstop alone misses hard partial deflections; stopball alone
    misses soft full reversals. Together they prevent footreach from running on
    after any meaningful contact, stopping the farm-footreach-post-stopball exploit.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    softstop_fired = getattr(env, "_softstop_flag", None)
    stopball_fired = getattr(env, "_sb_flag", None)
    already_deflected = (
        (softstop_fired if softstop_fired is not None
         else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        | (stopball_fired if stopball_fired is not None
           else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    )

    return (ball_x_local < 0.0) | already_deflected


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_local = torch.zeros(env.num_envs, 3, device=env.device)
    x_local[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_local)


def _get_ball_crossing_y(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Frozen Y coordinate where the ball will cross the goal line (x_local = 0).

    Uses _rsi_cross_y set analytically by reset_ball_rolling at spawn time (no buffer
    lag). Converts from env-relative to world frame by adding env_origin_y.
    Falls back to velocity-based estimate only when _rsi_cross_y is unavailable.
    """
    just_reset = env.episode_length_buf <= 1
    if not hasattr(env, "_ball_crossing_y"):
        env._ball_crossing_y = env.scene.env_origins[:, 1].clone()

    if just_reset.any():
        rsi_cross_y = getattr(env, "_rsi_cross_y", None)
        if rsi_cross_y is not None:
            # Analytically computed at spawn: no buffer lag, exact match with RSI pool.
            env._ball_crossing_y[just_reset] = (
                env.scene.env_origins[just_reset, 1] + rsi_cross_y[just_reset]
            )
        else:
            ball: Entity = env.scene[ball_name]
            ball_pos_w = ball.data.root_link_pos_w
            ball_vel_w = ball.data.root_link_lin_vel_w
            ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]
            bvx = ball_vel_w[just_reset, 0].clamp(max=-0.1)
            bvy = ball_vel_w[just_reset, 1]
            t_cross = ball_x_local[just_reset] / (-bvx)
            env._ball_crossing_y[just_reset] = ball_pos_w[just_reset, 1] + bvy * t_cross
    return env._ball_crossing_y


_INNER_FOOT_TARGET_OFFSET = 0.05
"""NEW 2026-08-15 (user request). User wants to save with the INNER part of
the leading foot rather than dead center, which means slightly overshooting
outward (away from center) relative to the ball's true calculated crossing
point. Applied ONLY to the GREEN target -- the full/final crossing point
footreach and foot_proximity converge on, both the frozen far-field value
(_get_reach_target_y's phase1_active=False branch) and the live-ball value
they switch to inside the last 0.5m (both would otherwise silently erase the
offset right at the moment it matters most for contact). Deliberately NOT
applied to: the blue midpoint (a pre-position waypoint, not the save point),
orange/red (separate trailing-foot waypoint system), _get_ball_crossing_y
itself (the true ball prediction -- success/stopball/stayonline/region
ground truth/the play.py rod marker all correctly keep reading the real,
unmodified value), or any direction-only usage (_get_correct_foot_idx,
footreach's own overshoot-kill-switch, red_overshoot_penalty's direction
sign). Must match play.py's _INNER_FOOT_TARGET_OFFSET (viewer-only green
sphere marker) -- verify both if this ever changes."""


def _get_foot_block_offset(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Signed Y offset pushing the GREEN foot-aim target outward (away from
    center) by _INNER_FOOT_TARGET_OFFSET. Add this to whatever base Y value
    a caller is already using (frozen crossing point or live ball position)
    -- see _INNER_FOOT_TARGET_OFFSET's own docstring for exactly which
    callers should and shouldn't use this.
    """
    crossing_y = _get_ball_crossing_y(env, ball_name)
    start_y = env.scene.env_origins[:, 1]
    sign = torch.sign(crossing_y - start_y)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # dead-center: default outward
    return sign * _INNER_FOOT_TARGET_OFFSET


def _get_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    wide_threshold: float = 0.5,  # FIX 2026-08-01: was 0.65, reverted to 0.5, kept in sync with regions.py's near/far boundary
    landing_radius: float = 0.15,  # FIX 2026-07-24: was 0.08 (user request: too strict at full difficulty)
    landing_speed_threshold: float = 1.0,  # FIX 2026-07-24: reverted to the pre-2026-07-23 value (was 0.15); see below
) -> torch.Tensor:
    """Two-stage reach target for wide crossings: v2 reimplementation of the
    "blue-ball-waypoint" branch mechanism (removed from this project's
    lineage 2026-07-10, "green-ball-baseline"), reimplemented 2026-07-23 on
    top of current master rather than merged/cherry-picked (master had
    diverged 74+ commits since the branch split -- AMP fixes, reward
    audits, region routing, wrong-foot gating -- none of which the old
    branch has).

    For crossings where |crossing_y - start_y| > wide_threshold OR the env's
    permanent region assignment (env._region_id) is a far region (1 or 3 --
    the authoritative label for THIS region-conditioned task, which the
    original branch predates), the target is the MIDPOINT between the
    robot's stance and the true crossing point until the assigned foot
    (_get_correct_foot_idx) physically lands there (airborne then in ground
    contact via the feet_contact sensor, within landing_radius of the
    midpoint AND slower than landing_speed_threshold), then switches to the
    full crossing point. Narrow crossings always target the full point.
    Hard gate, no elapsed-time fallback -- a crossing that never lands stays
    targeting the midpoint for the whole episode.

    Ported mechanisms from the original branch's investigation (see
    docs/BugFixes.md and the research-mapping subagent report in
    conversation for the full history this condenses):

    - Settle window (not instantaneous): requires the assigned foot to stay
      in contact AND within landing_radius for _BLUE_SETTLE_STEPS(3)
      consecutive REAL steps (memoized via env._blue_last_settle_step since
      this function is called up to 7x/step by different reward terms
      sharing one post-physics snapshot) before checking speed -- a genuine
      footstrike carries residual swing velocity for a few steps after
      first contact, decaying as weight transfers.
    - LEAKY decrement (branch mechanism #9, the single best-evidenced fix
      in the whole prior investigation -- 44.8% genuine landing rate at
      iter 2000 of blue_leaky_settle_count_2026-07-13, vs. near-0% under
      the original hard reset-to-zero): a single dropped frame decrements
      the settle counter by 1 (floored at 0) instead of zeroing it outright,
      so an isolated contact-sensor glitch or brief overshoot doesn't erase
      an otherwise-genuine plant in progress. Root cause this addresses:
      the natural "plant" phase between two swings in the DoubleStep/
      TripleStep reference clips is only ~11 frames wide even at 1.0x
      pace -- a stochastic policy has almost no margin for a single missed
      frame under a hard-reset counter.
    - Curriculum-eased landing radius AND speed (branch mechanism #1): lerps
      landing_radius from loose (0.20m at ball_difficulty=0, i.e. an easy
      target even a rough approach can hit) to the caller's strict default
      (0.15m at difficulty=1, FIX 2026-07-24 -- was 0.08m, widened on user
      report that the strict end was too tight to reliably land in even
      for an approaching-competent policy), tied to the existing
      env._ball_difficulty curriculum scalar -- a policy that has never
      landed once has no gradient trace of what success feels like; this
      gives it an easy version to discover first. landing_speed_threshold
      eases from 2.0 m/s at d=0 down to 1.0 m/s at d=1 (FIX 2026-07-24:
      reverted to this looser band -- see below).

      History: the 2.0/1.0 m/s band was flattened to a hardcoded 0.15 m/s
      on 2026-07-23 after live evidence that it let a foot merely stride
      through the (large, eased) radius zone register as "landed" without
      ever stopping (genuine plants measured 0.002-0.05 m/s, nowhere near
      even 1.0 m/s). That was then re-curriculum'd the same day to
      0.30->0.15 m/s to restore an easy-difficulty gradient. REVERTED
      2026-07-24 (user request) back to 2.0->1.0 m/s: with the 0.30/0.15
      band, blue_ball_landed kept declining over iterations 0-6700 of
      blue_v2_landinggatefix_2026-07-23 (0.041 peak -> 0.007) while
      blue_stick_landing and blue_overshoot_penalty both kept moving the
      wrong direction, suggesting the strict end was now hard enough to
      actively suppress the landing gradient rather than just filter
      false positives. This reintroduces the documented pass-through
      leak risk as a deliberate trade -- if blue_ball_landed recovers
      without a corresponding new failure mode, keep it; if the leak
      reappears (foot striding through without stopping), the fix is to
      re-tighten with a smoother curriculum shape (e.g. more difficulty
      steps between the eased and strict ends) rather than snapping
      straight back to 0.15.
    - _blue_landed_was_free classification: a landing achieved suspiciously
      early (episode_length_buf < 10) can't reflect genuine approach work
      (RSI donor poses are mid-motion by construction) -- flagged so
      downstream consumers (stopball/softstop's landing_ok gate) don't pay
      out for a landing the policy didn't actually earn this episode.

    NOT ported from the original branch (see docs/BugFixes.md for the full
    per-mechanism assessment): RSI practice-fraction curriculum and
    RSI teleport-seeding (both unvalidated, superseded within the branch
    itself), landing-payoff scaling (entangled with a reward leak for part
    of its life), AMP clip retiming (a confirmed NEGATIVE result -- genuine
    landing rate got worse as retiming pace increased), self-imitation
    learning (produced a headline 96.6% number but the branch's own final
    finding cast real doubt on whether that reflected genuinely learning
    the task, via a separate unrelated wrong-foot-credit bug).

    Root XY is pinned to env.scene.env_origins at reset (reset_base event),
    so "robot start Y" can be read live with no new cache.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y

    rel = getattr(env, "_rsi_cross_y", None)
    lateral = rel if rel is not None else (full_y - start_y)
    wide = lateral.abs() > wide_threshold

    # This region-conditioned task's own far/near label (env._region_id) is
    # authoritative over the threshold check above, same rationale as the
    # original branch's _REGION_IS_FAR (predates the region system here --
    # reimplemented against env._region_id directly, matching the far-region
    # convention already used throughout this codebase, e.g. regions.py's
    # _FAR_REGION_IDS = (1, 3)).
    region_id = getattr(env, "_region_id", None)
    if region_id is not None:
        wide = wide | (region_id == 1) | (region_id == 3)

    # Cached so footreach/foot_proximity/stopball/softstop can gate on
    # "wide AND not yet landed" without recomputing the crossing geometry.
    env._blue_wide = wide

    half_y = start_y + (full_y - start_y) / 2.0

    # DEBUG 2026-07-23 (TEMPORARY, remove after landing-gate investigation):
    # expose the Y-offsets (relative to robot start) that make up this gate's
    # target math, plus whether "wide" was forced by the region label despite
    # a small actual lateral offset, so we can see directly whether a landing
    # happens near the true midpoint or near the full/green target instead.
    env._blue_dbg_half_off = half_y - start_y
    env._blue_dbg_full_off = full_y - start_y
    env._blue_dbg_wide_by_dist = lateral.abs() > wide_threshold

    n = env.num_envs
    if not hasattr(env, "_blue_was_airborne"):
        env._blue_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_settle_count = torch.zeros(n, dtype=torch.int64, device=env.device)
        env._blue_landed_was_free = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed_genuine = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_last_settle_step = torch.full((n,), -1, dtype=torch.int64, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_was_airborne[just_reset] = False
    env._blue_landed[just_reset] = False
    env._blue_settle_count[just_reset] = 0
    env._blue_landed_was_free[just_reset] = False
    env._blue_landed_genuine[just_reset] = False

    # Curriculum-eased landing radius (branch mechanism #1) -- eases
    # 0.20m -> 0.15m with difficulty, giving an early policy a bigger
    # target. FIX 2026-07-24: strict end was 0.08m; widened to 0.15m
    # (user request -- too strict to reliably land in at full difficulty).
    d = float(min(max(getattr(env, "_ball_difficulty", 1.0), 0.0), 1.0))
    landing_radius = 0.20 + (landing_radius - 0.20) * d
    env._blue_landing_radius_current = landing_radius

    # FIX 2026-07-24: reverted to the pre-2026-07-23 band (2.0 m/s at d=0
    # -> 1.0 m/s at d=1), per user request, after the narrower 0.30->0.15
    # m/s band (see git history same day) coincided with blue_ball_landed
    # declining over iterations 0-6700 of blue_v2_landinggatefix_2026-07-23
    # (0.041 peak -> 0.007) while blue_stick_landing/blue_overshoot_penalty
    # both moved the wrong direction -- the strict end looked hard enough
    # to be suppressing the landing gradient outright, not just filtering
    # false positives. This reintroduces the documented pass-through leak
    # (a foot striding through the loose radius zone at up to 1.0 m/s can
    # register as "landed" without a genuine stop, vs. 0.002-0.05 m/s for
    # real plants) as a deliberate trade against the widened landing_radius
    # above -- if the leak reappears, prefer a smoother curriculum shape
    # over snapping straight back to 0.15.
    _EASY_LANDING_SPEED_THRESHOLD = 2.0
    landing_speed_threshold = _EASY_LANDING_SPEED_THRESHOLD + (landing_speed_threshold - _EASY_LANDING_SPEED_THRESHOLD) * d
    # Cached unconditionally (mirrors env._blue_landing_radius_current above)
    # so callers/tests can read the resolved threshold without needing the
    # full robot/feet_contact scene mocked.
    env._blue_landing_speed_threshold_current = landing_speed_threshold

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
        foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                    # (N,)
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                  # (N, 3)
        assigned_foot_vel = foot_vel_w[arange_n, foot_idx]                  # (N, 3)

        found = feet_contact.data.found                                    # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(foot_idx == 0, left_in_contact, right_in_contact)  # (N,)

        # REVERTED 2026-07-23: tried gating foot_in_contact on foot height
        # above floor_z (env.scene.env_origins[:, 2]) to exclude a foot
        # suspended by ball-contact alone (see git history same day for the
        # attempt + rationale). Live-tested: settle_count stopped
        # accumulating AT ALL afterward, even for cases that previously
        # accumulated fine (up to 10/3) -- i.e. the height check was false
        # in the real sim even while the foot was visibly grounded and the
        # contact sensor read True, for reasons not reproduced by an
        # isolated mock with idealized (zero-offset) inputs. Reverted to the
        # plain contact-sensor signal (matches this function's behavior
        # before today's ball-contact investigation) to restore basic
        # settle/landing function. The ball-touching-counts-as-landed issue
        # (feet_contact fires on ball contact too, not just ground -- see
        # goalkeeper_env_cfg.py's own comment on this sensor) is real and
        # still open, but needs a cleaner fix than a height gate -- revisit
        # separately rather than block basic landing detection on it.
        currently_airborne = ~foot_in_contact
        env._blue_was_airborne |= currently_airborne

        goal_x_w = env.scene.env_origins[:, 0]
        target_point_xy = torch.stack([goal_x_w, half_y], dim=-1)  # (N, 2)
        # Horizontal-only distance -- foot_in_contact already guarantees ground
        # height whenever this fires, so a Z term would be redundant.
        dist_to_blue = torch.norm(assigned_foot_pos[:, :2] - target_point_xy, dim=-1)
        # Horizontal speed gate -- a foot merely sweeping through blue's
        # coordinate at speed (mid-swing toward green) must not count as
        # landing there. Only a foot that has actually decelerated (genuine
        # plant) satisfies this alongside the position check.
        foot_speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

        candidate = wide & env._blue_was_airborne & foot_in_contact & (dist_to_blue < landing_radius)
        # Memoization guard: this function is called up to 7x/step by
        # different reward terms sharing one unchanged post-physics
        # snapshot -- without this, the settle count would increment/decay
        # once PER CALL instead of once per real physics tick.
        is_first_call_this_tick = env.episode_length_buf != env._blue_last_settle_step
        # DEBUG 2026-07-23 (TEMPORARY): snapshot the PRE-update stored value
        # so we can see what episode_length_buf was actually compared
        # against, not just the post-update value (which trivially always
        # matches episode_length_buf after the line below runs).
        env._blue_dbg_last_settle_step_before = env._blue_last_settle_step.clone()
        env._blue_last_settle_step = env.episode_length_buf.clone()
        # FIX (branch mechanism #9, "leaky" decrement): on a miss, decrement
        # by 1 (floored at 0) instead of hard-reset to 0 -- see docstring.
        _settle_before = env._blue_settle_count[0].item()
        env._blue_settle_count = torch.where(
            candidate,
            torch.where(is_first_call_this_tick, env._blue_settle_count + 1, env._blue_settle_count),
            torch.where(is_first_call_this_tick, (env._blue_settle_count - 1).clamp(min=0), env._blue_settle_count),
        )
        # DEBUG 2026-07-23 (TEMPORARY): raw, unconditional, per-CALL (not
        # per-tick) print of every single call to this function for env 0 --
        # settle appears stuck at 0 despite candidate=True for many
        # consecutive real ticks in play, which the display (only showing
        # the last of ~7 same-tick calls) can't fully explain. This prints
        # every call so we can see the actual call-by-call sequence.
        if bool(wide[0].item()) and dist_to_blue[0].item() < 0.3:
            import sys as _sys
            import inspect as _inspect
            _caller = _inspect.currentframe().f_back.f_code.co_name
            # DEBUG 2026-07-24: dist/radius/foot_pos/target_xy trimmed to 2
            # decimals (were 4 decimals / raw float64 .tolist(), e.g.
            # "0.0071395160630345345" -- unreadable noise for a live debug
            # trace). asset_cfg_id shortened to id(asset_cfg) % 1_000_000 --
            # the full id() is a full memory address (naturally a huge,
            # arbitrary number on 64-bit systems), but this print only ever
            # needs it as a same-vs-different fingerprint across calls (see
            # the FIX 2026-07-23 comment above), not a real identifier.
            _foot_pos_r = [round(v, 2) for v in assigned_foot_pos[0].tolist()]
            _target_xy_r = [round(v, 2) for v in target_point_xy[0].tolist()]
            print(
                f"[RAWSETTLE] ep_len={env.episode_length_buf[0].item()} "
                f"caller={_caller} "
                f"cand={bool(candidate[0].item())} "
                f"[wide={bool(wide[0].item())} "
                f"airborne={bool(env._blue_was_airborne[0].item())} "
                f"contact={bool(foot_in_contact[0].item())} "
                f"distOk={bool((dist_to_blue[0] < landing_radius).item())} "
                f"dist={dist_to_blue[0].item():.2f} radius={landing_radius:.2f}] "
                f"first_call={bool(is_first_call_this_tick[0].item())} "
                f"settle_before={_settle_before} settle_after={env._blue_settle_count[0].item()} "
                f"foot_idx={foot_idx[0].item()} body_ids={list(asset_cfg.body_ids)} "
                f"asset_cfg_id={id(asset_cfg) % 1_000_000} "
                f"foot_pos={_foot_pos_r} target_xy={_target_xy_r}",
                file=_sys.stderr,
            )
        _BLUE_SETTLE_STEPS = 3
        newly_landed = (
            (env._blue_settle_count >= _BLUE_SETTLE_STEPS)
            & (foot_speed < landing_speed_threshold)
            & ~env._blue_landed
        )
        env._blue_landed |= newly_landed

        # Per-landing-event free classification: a landing achieved
        # suspiciously early (few steps since reset) can't reflect real
        # approach work; RSI donor poses are mid-motion by construction.
        _BLUE_LANDING_FREE_STEP_THRESHOLD = 10
        env._blue_landed_was_free = torch.where(
            newly_landed,
            env.episode_length_buf < _BLUE_LANDING_FREE_STEP_THRESHOLD,
            env._blue_landed_was_free,
        )

        # DEBUG 2026-07-23 (TEMPORARY, remove after landing-gate investigation):
        # expose per-step internals for play.py's analytics printer so the
        # failing condition can be read directly instead of guessed at.
        env._blue_dbg_dist = dist_to_blue
        env._blue_dbg_speed = foot_speed
        env._blue_dbg_contact = foot_in_contact
        env._blue_dbg_candidate = candidate
        env._blue_dbg_first_call = is_first_call_this_tick
        env._blue_dbg_wide = wide
        env._blue_dbg_radius = landing_radius
        env._blue_dbg_speed_th = landing_speed_threshold
        env._blue_dbg_settle = env._blue_settle_count.clone()
        env._blue_dbg_foot_idx = foot_idx
        env._blue_dbg_ep_len = env.episode_length_buf.clone()
        env._blue_dbg_was_airborne = env._blue_was_airborne.clone()
        ball_contact: ContactSensor = env.scene["ball_contact"]
        ball_found = ball_contact.data.found                                # (N, 8)
        left_touching_ball = (ball_found[:, :4] > 0).any(dim=-1)
        right_touching_ball = (ball_found[:, 4:] > 0).any(dim=-1)
        env._blue_dbg_touching_ball = torch.where(foot_idx == 0, left_touching_ball, right_touching_ball)
        env._blue_dbg_foot_off = assigned_foot_pos[:, 1] - start_y

    # FIX 2026-07-24: phase1_active (and every other consumer of raw
    # env._blue_landed below, see grep for "_blue_landed" across this file)
    # previously trusted env._blue_landed unconditionally -- but a "free"
    # landing (env._blue_landed_was_free, episode_length_buf < 10 -- almost
    # certainly RSI seeding the reset with a donor foot trajectory already
    # near the target, not genuine approach work) still set env._blue_landed
    # True permanently for the rest of the episode. Only the one-shot save
    # bonus (stopball/softstop/success's landing_ok) excluded free landings;
    # phase1_active/blue_overshoot_penalty/blue_stick_landing/footreach's
    # ball_close+blue_approach did not, so a free landing would silently
    # switch the ENTIRE episode's targeting straight to the real crossing
    # point (skipping blue) while only the save bonus correctly still
    # refused to pay out -- observed behavior: some episodes cleanly
    # double-step through blue, others appear to "ignore blue and go
    # straight for the real ball" with SB·/SS· never firing, consistent
    # with which outcome depends on whether that episode's reset happened
    # to trip a free landing. env._blue_landed_genuine (used everywhere
    # else in this file that previously read raw env._blue_landed, see
    # docs/BugFixes.md 2026-07-24) is the single source of truth for "landed
    # AND that landing wasn't free" -- computed once here since this is the
    # only place env._blue_landed/env._blue_landed_was_free are updated,
    # and every downstream consumer already calls this function first each
    # tick to stay fresh.
    env._blue_landed_genuine = env._blue_landed & ~env._blue_landed_was_free

    phase1_active = wide & ~env._blue_landed_genuine
    # NEW 2026-08-15 (user request): GREEN (full-crossing, phase1_active=False)
    # branch only -- half_y (blue midpoint) deliberately untouched. See
    # _INNER_FOOT_TARGET_OFFSET's docstring.
    return torch.where(phase1_active, half_y, full_y + _get_foot_block_offset(env, ball_name))


def _get_orange_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.15,
    landing_speed_threshold: float = 1.0,
) -> torch.Tensor:
    """Trailing-foot ("orange") mirror of _get_reach_target_y -- see that
    function's docstring for the full blue-ball-waypoint mechanism history
    this reuses (leaky settle-count decrement, curriculum-eased landing
    radius/speed, free-landing classification).

    Target formula (confirmed with user via worked examples, 2026-08-08
    design spec; constant raised 0.30->0.60 same day per user request --
    "the orange ball needs to be way further than the blue ball" -- then
    adjusted 0.60->0.50 same day -- see docs/BugFixes.md):
        delta = full_y - start_y                          (signed)
        shrunk = sign(delta) * max(|delta| - 0.50, 0.0)    (50cm off the top, sign-safe)
        orange_y = start_y + shrunk / 2.0

    Equivalently: blue's own midpoint, 25cm short of it in delta-magnitude
    terms -- NOT a plain `delta - 0.50` (that would push the target further
    OUT, not in, for right-side/negative-delta crossings).

    Reuses env._blue_wide (set by _get_reach_target_y, which every existing
    wide-gated reward already calls first in the reward-manager's term
    order) directly as the wide/narrow gate -- "wide" is a property of the
    ball's crossing geometry, not which foot is being tracked, so no
    separate wide/region computation is added here. Defensive fallback to
    all-False if _blue_wide isn't set yet (should not happen in the real
    term order).

    Keyed to the TRAILING foot (1 - _get_correct_foot_idx), not the leading
    one. Unlike blue, this function does NOT graduate to a second/live-ball
    target once landed -- the landing-focused subset gives the trailing foot
    one target for the whole wide-crossing window, no live-ball-tracking
    phase (see design spec's "Explicitly out of scope" section).

    Maintains its own state namespace (env._orange_*), entirely separate
    from env._blue_*'s -- never reads or writes any env._blue_* field except
    the read-only env._blue_wide gate above.

    See docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.

    Not yet validated against a live training run.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    delta = full_y - start_y
    wide = getattr(env, "_blue_wide", torch.zeros_like(delta, dtype=torch.bool))
    env._orange_wide = wide

    shrunk = torch.sign(delta) * (delta.abs() - 0.50).clamp(min=0.0)
    orange_y = start_y + shrunk / 2.0

    n = env.num_envs
    if not hasattr(env, "_orange_was_airborne"):
        env._orange_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_settle_count = torch.zeros(n, dtype=torch.int64, device=env.device)
        env._orange_landed_was_free = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_landed_genuine = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_last_settle_step = torch.full((n,), -1, dtype=torch.int64, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._orange_was_airborne[just_reset] = False
    env._orange_landed[just_reset] = False
    env._orange_settle_count[just_reset] = 0
    env._orange_landed_was_free[just_reset] = False
    env._orange_landed_genuine[just_reset] = False

    d = float(min(max(getattr(env, "_ball_difficulty", 1.0), 0.0), 1.0))
    landing_radius = 0.20 + (landing_radius - 0.20) * d
    env._orange_landing_radius_current = landing_radius

    _EASY_LANDING_SPEED_THRESHOLD = 2.0
    landing_speed_threshold = (
        _EASY_LANDING_SPEED_THRESHOLD + (landing_speed_threshold - _EASY_LANDING_SPEED_THRESHOLD) * d
    )
    env._orange_landing_speed_threshold_current = landing_speed_threshold

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
        foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                      # (N,)
        trailing_idx = 1 - foot_idx
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]                # (N, 3)
        assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]                # (N, 3)

        found = feet_contact.data.found                                      # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(trailing_idx == 0, left_in_contact, right_in_contact)

        currently_airborne = ~foot_in_contact
        env._orange_was_airborne |= currently_airborne

        goal_x_w = env.scene.env_origins[:, 0]
        target_point_xy = torch.stack([goal_x_w, orange_y], dim=-1)          # (N, 2)
        dist_to_orange = torch.norm(assigned_foot_pos[:, :2] - target_point_xy, dim=-1)
        foot_speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

        candidate = wide & env._orange_was_airborne & foot_in_contact & (dist_to_orange < landing_radius)
        is_first_call_this_tick = env.episode_length_buf != env._orange_last_settle_step
        env._orange_last_settle_step = env.episode_length_buf.clone()
        _ORANGE_SETTLE_STEPS = 3
        env._orange_settle_count = torch.where(
            candidate,
            torch.where(is_first_call_this_tick, env._orange_settle_count + 1, env._orange_settle_count),
            torch.where(is_first_call_this_tick, (env._orange_settle_count - 1).clamp(min=0), env._orange_settle_count),
        )
        newly_landed = (
            (env._orange_settle_count >= _ORANGE_SETTLE_STEPS)
            & (foot_speed < landing_speed_threshold)
            & ~env._orange_landed
        )
        env._orange_landed |= newly_landed

        # NEW 2026-08-08 (user request, final-review follow-up): minimal live
        # diagnostics -- mirrors blue's own env._blue_dbg_dist/_speed/_contact/
        # _settle/_foot_idx (rewards.py's _get_reach_target_y), scoped to just
        # the fields play.py's per-episode accumulator needs (not blue's full
        # temporary per-step debug dump, which was explicitly marked
        # TEMPORARY/for-removal cruft, not part of this function's permanent
        # design). Lets a real training/play run explain a low orange landing
        # rate (distance vs. speed vs. contact vs. settle-count) instead of
        # only seeing the final _orange_landed flag.
        env._orange_dbg_dist = dist_to_orange
        env._orange_dbg_speed = foot_speed
        env._orange_dbg_contact = foot_in_contact
        env._orange_dbg_settle = env._orange_settle_count.clone()
        env._orange_dbg_foot_idx = trailing_idx

        _ORANGE_LANDING_FREE_STEP_THRESHOLD = 10
        env._orange_landed_was_free = torch.where(
            newly_landed,
            env.episode_length_buf < _ORANGE_LANDING_FREE_STEP_THRESHOLD,
            env._orange_landed_was_free,
        )

    env._orange_landed_genuine = env._orange_landed & ~env._orange_landed_was_free

    return orange_y


def _get_red_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.15,
    landing_speed_threshold: float = 1.0,
) -> torch.Tensor:
    """Trailing-foot ("red") second-stage mirror of _get_orange_reach_target_y --
    see that function's docstring for the shared leaky-settle-count/curriculum/
    free-landing mechanism this reuses verbatim.

    NEW 2026-08-15 (user request): closes the gap orange's own docstring
    explicitly flags ("does NOT graduate to a second/live-ball target once
    landed") -- red is that second target for the trailing foot, active only
    once BOTH the leading foot has genuinely landed at blue AND the trailing
    foot has genuinely landed at orange (`env._blue_landed_genuine &
    env._orange_landed_genuine`, cached as `env._red_active`) -- the
    literal two-condition AND the user asked for.

    Target formula -- anchored at `full_y` (green), NOT `start_y` like
    orange: a literal copy of orange's start_y-anchored shrink formula tops
    out AT blue as its constant shrinks to zero, so it can never place a
    point past blue toward green. Anchoring at full_y and shrinking backward
    instead was the first attempt, but that formula's distance from green
    (`max(|delta|-0.50,0)/2`) scales with the total crossing distance --
    live-checked (2026-08-15, real `model_39750.pt` rollout) it collapsed to
    as little as 0.029m from green for crossings just barely over the 0.5m
    wide threshold, nowhere near the intended "behind green" gap. FIX
    (same day, user request, "just like -0.4 away from green ball full_Y"):
    replaced with a flat offset, CLAMPED to never exceed blue's own distance
    from green (`|delta|/2`):
        red_offset = min(RED_OFFSET_FROM_GREEN, |delta|/2)   (0.4m cap)
        red_y = full_y - sign(delta) * red_offset
    Live-checked again (2026-08-15) with the plain uncapped 0.4m offset: for
    a real crossing with |delta|=0.711m (a common "just barely wide"
    crossing -- blue's own distance from green there is only 0.356m), the
    uncapped version placed red BEFORE blue (closer to start), inverting the
    intended start<orange<blue<red<green order. User confirmed via
    AskUserQuestion to clamp rather than accept the inversion -- for
    |delta| &gt;= 0.8m (blue's distance from green &gt;= 0.4m) this clamp never
    engages and red gets the full 0.4m gap requested; for smaller wide
    crossings red collapses onto blue's own position instead of passing it
    (a degenerate "red==blue" case, same class as orange's own
    collapse-to-start_y degenerate case above, not a new failure mode).
    User-confirmed via AskUserQuestion (2026-08-15): weights equal to orange
    (not a further quarter-of-blue halving -- red's extra landing-gate
    already limits false credit relative to orange's ungated single stage),
    and capped at red for the rest of the episode (no further graduation to
    live green, mirroring orange's own design exactly).

    Not yet validated against a live training run.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    delta = full_y - start_y
    wide = getattr(env, "_blue_wide", torch.zeros_like(delta, dtype=torch.bool))
    env._red_wide = wide

    # FIX 2026-08-17 (user request, "standard position of 20cm to 30cm on
    # further ranges"): 0.4 -> 0.25. Same clamp formula/mechanism, only the
    # far-range cap value changed -- for |delta|>=0.5m (2*0.25) red now sits
    # a flat 25cm from green instead of 40cm; narrower wide crossings still
    # scale down below that via the unchanged |delta|/2 term.
    RED_OFFSET_FROM_GREEN = 0.25
    red_offset = torch.clamp(delta.abs() / 2.0, max=RED_OFFSET_FROM_GREEN)
    red_y = full_y - torch.sign(delta) * red_offset

    # Gate: settle progress can only accrue once BOTH upstream landings are
    # genuine. Defensive getattr fallback (all-False) mirrors orange's own
    # defensive read of env._blue_wide -- should not trigger in the real
    # term order (red's RewardTermCfg entries are registered after both
    # blue's and orange's, so both flags are already fresh this tick).
    zeros = torch.zeros_like(delta, dtype=torch.bool)
    red_active = getattr(env, "_blue_landed_genuine", zeros) & getattr(env, "_orange_landed_genuine", zeros)
    env._red_active = red_active

    n = env.num_envs
    if not hasattr(env, "_red_was_airborne"):
        env._red_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._red_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._red_settle_count = torch.zeros(n, dtype=torch.int64, device=env.device)
        env._red_landed_was_free = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._red_landed_genuine = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._red_last_settle_step = torch.full((n,), -1, dtype=torch.int64, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._red_was_airborne[just_reset] = False
    env._red_landed[just_reset] = False
    env._red_settle_count[just_reset] = 0
    env._red_landed_was_free[just_reset] = False
    env._red_landed_genuine[just_reset] = False

    d = float(min(max(getattr(env, "_ball_difficulty", 1.0), 0.0), 1.0))
    landing_radius = 0.20 + (landing_radius - 0.20) * d
    env._red_landing_radius_current = landing_radius

    _EASY_LANDING_SPEED_THRESHOLD = 2.0
    landing_speed_threshold = (
        _EASY_LANDING_SPEED_THRESHOLD + (landing_speed_threshold - _EASY_LANDING_SPEED_THRESHOLD) * d
    )
    env._red_landing_speed_threshold_current = landing_speed_threshold

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
        foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                      # (N,)
        trailing_idx = 1 - foot_idx
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]                # (N, 3)
        assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]                # (N, 3)

        found = feet_contact.data.found                                      # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(trailing_idx == 0, left_in_contact, right_in_contact)

        currently_airborne = ~foot_in_contact
        env._red_was_airborne |= currently_airborne

        goal_x_w = env.scene.env_origins[:, 0]
        target_point_xy = torch.stack([goal_x_w, red_y], dim=-1)             # (N, 2)
        dist_to_red = torch.norm(assigned_foot_pos[:, :2] - target_point_xy, dim=-1)
        foot_speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

        candidate = (
            wide & red_active & env._red_was_airborne & foot_in_contact
            & (dist_to_red < landing_radius)
        )
        is_first_call_this_tick = env.episode_length_buf != env._red_last_settle_step
        env._red_last_settle_step = env.episode_length_buf.clone()
        _RED_SETTLE_STEPS = 3
        env._red_settle_count = torch.where(
            candidate,
            torch.where(is_first_call_this_tick, env._red_settle_count + 1, env._red_settle_count),
            torch.where(is_first_call_this_tick, (env._red_settle_count - 1).clamp(min=0), env._red_settle_count),
        )
        newly_landed = (
            (env._red_settle_count >= _RED_SETTLE_STEPS)
            & (foot_speed < landing_speed_threshold)
            & ~env._red_landed
        )
        env._red_landed |= newly_landed

        env._red_dbg_dist = dist_to_red
        env._red_dbg_speed = foot_speed
        env._red_dbg_contact = foot_in_contact
        env._red_dbg_settle = env._red_settle_count.clone()
        env._red_dbg_foot_idx = trailing_idx

        _RED_LANDING_FREE_STEP_THRESHOLD = 10
        env._red_landed_was_free = torch.where(
            newly_landed,
            env.episode_length_buf < _RED_LANDING_FREE_STEP_THRESHOLD,
            env._red_landed_was_free,
        )

    env._red_landed_genuine = env._red_landed & ~env._red_landed_was_free

    return red_y


def footreach(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    reach_th: float = 0.3,
    sigma: float = 5.0,
    phase2_threshold: float = 1.5,
) -> torch.Tensor:
    """Reach reward adapted from Imitationlearningbooster eereach, for feet instead of hands.

    Phase 1 (ball x_local > phase2_threshold, default 1.5 m): reward lateral alignment
    with the FROZEN crossing Y (where the ball will cross the goal line), not the live
    ball Y. This gives a stable pre-positioning target even for angled shots where live
    ball Y != arrival Y.

    Phase 2 (ball x_local <= phase2_threshold): sigmoid reach reward × lateral vel_sigma
    so actively diving/stepping toward the ball gives up to 10× the static reach reward.

    vel_sigma = 1 + 3 * clamp(vel_toward_crossing_side, 0, 3).

    Upright gate suppresses reward when falling (mirrors eereach upright gate).

    v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch's
    two-stage mechanism, removed from this project's lineage 2026-07-10
    ("green-ball-baseline"). See _get_reach_target_y's docstring for the
    full mechanism history and what was/wasn't ported.

    Assigned-foot targeting (ported from the branch's 2026-07-11 fix):
    phase1_rew's lateral_error and vel_sigma's vel_toward are keyed off the
    task-ASSIGNED foot's (_get_correct_foot_idx) own Y position/velocity,
    not root -- for a foot (unlike G1's hand-reach, which is root-relative
    since a hand can reach independently of torso position), the root can
    sit near the target purely from stance width/weight-shift while the
    assigned foot itself overshoots or lags, implicitly crediting the wrong
    foot on wide two-stage crossings.

    Blue decel-zone (ported from the branch's 2026-07-11/07-12 fix): decays
    vel_sigma's speed bonus toward neutral (1.0x) as the assigned foot
    closes within 0.30m of blue on a wide, unlanded crossing, reaching full
    neutral at the SAME curriculum-eased landing_radius _get_reach_target_y
    used this call (env._blue_landing_radius_current) -- without this nothing
    disincentivizes carrying speed through the exact zone the foot should be
    decelerating into for a genuine plant.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)

    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)

    # Fixed, task-assigned foot (not the geometrically-closest foot) --
    # computed once up front so phase1/vel_sigma/phase2 all credit the same foot.
    foot_idx = _get_correct_foot_idx(env, ball_name)                   # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, foot_idx, 1]                # (N,)
    assigned_foot_vel_y = foot_vel_w[arange_n, foot_idx, 1]            # (N,)

    # Reach target: full crossing point, or the two-stage midpoint-then-full
    # schedule on wide crossings. See _get_reach_target_y.
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,)

    # FIX 2026-07-24: overshoot-kill flag. footreach's vel_sigma (below) picks
    # up the foot's raw post-physics velocity, which includes the impulse
    # from a ball collision, not just deliberate policy motion -- confirmed
    # live via play.py traces: footreach spiked 5->13->21->28 across 3 ticks
    # while dist_to_blue stayed flat (~0.37-0.38, not converging), then
    # crashed back to ~8 the same tick foot_ang_vel_xy hit -10.58 and
    # feet_slippage crashed to 0.13 -- both independent signatures of a hard
    # impact -- confirming the peak was the contact impulse, not genuine
    # approach. No G1 equivalent exists for this gate (checked every reward
    # function in legged_robot.py -- G1's eereach never zeros on distance,
    # its `behind` mask only sets a fixed vel_sigma=2.0; see docs/BugFixes.md).
    # Uses the SAME direction-aware signed_progress blue_overshoot_penalty
    # already computes, so "overshot" means the same thing in both places.
    # Sticky per episode (not per-tick) so a single clear miss stops footreach
    # from paying out on repeat dives at the same spot; foot_proximity/
    # blue_stick_landing/blue_ball_landed are untouched, so a genuine
    # recovery-and-land afterward still earns real reward through those.
    full_y_ov = _get_ball_crossing_y(env, ball_name)                       # (N,)
    start_y_ov = env.scene.env_origins[:, 1]                               # (N,)
    half_y_ov = start_y_ov + (full_y_ov - start_y_ov) / 2.0
    direction_ov = torch.sign(full_y_ov - start_y_ov)
    signed_progress = direction_ov * (assigned_foot_y - half_y_ov)
    if not hasattr(env, "_footreach_overshot_flag"):
        env._footreach_overshot_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._footreach_overshot_flag[env.episode_length_buf <= 1] = False
    # FIX 2026-07-25: was a fixed 0.20 (matching the LOOSEST/easy-curriculum
    # end of the landing radius, per the 2026-07-24 comment above). But
    # env._blue_landing_radius_current (set fresh by _get_reach_target_y,
    # called above) tightens with curriculum down to 0.15 -- at high
    # difficulty a foot could overshoot up to 0.20m, MORE than the actual
    # landing tolerance (0.15m), before this kill-switch ever fired. User
    # reported footreach still spiking after the 2026-07-24 fix; tracking
    # the same curriculum-eased distance used everywhere else for "at blue"
    # (_get_reach_target_y/blue_ball_landed/blue_stick_landing) instead of a
    # stale fixed constant closes that gap. See docs/BugFixes.md.
    _FOOTREACH_OVERSHOOT_KILL = float(getattr(env, "_blue_landing_radius_current", 0.20))
    overshot_now = env._blue_wide & ~env._blue_landed_genuine & (signed_progress > _FOOTREACH_OVERSHOOT_KILL)
    env._footreach_overshot_flag |= overshot_now

    # Phase 1: pre-position laterally when ball is far (> 1.5 m in front).
    # Uses the ASSIGNED FOOT's Y, not root -- see docstring above.
    lateral_error = reach_target_y - assigned_foot_y                    # positive → target right
    asidegoal = lateral_error.clamp(-1.0, 1.0)
    asidegoal = torch.where(asidegoal.abs() < 0.3, torch.zeros_like(asidegoal), asidegoal)
    phase1_rew = 1.0 - asidegoal.abs()                                  # 1=aligned, 0=1 m off

    # Phase 2 crossing point: frozen (two-stage) when far, live ball when close (mirrors
    # G1 end_target update). G1 post_physics_step lines 203-206: end_target switches to
    # ball_states[:, :3] when balllocal < 0.5 — robot converges on the actual ball in the
    # final 0.5 m. The X coordinate stays at goal_x_w (stayonline constraint keeps the
    # foot at the goal line).
    goal_x_w  = env.scene.env_origins[:, 0]       # (N,)
    floor_z_w = env.scene.env_origins[:, 2]       # (N,)
    live_y = ball_pos_w[:, 1]
    live_z = ball_pos_w[:, 2]
    # On wide crossings this switch is ALSO gated on having genuinely landed
    # at the blue midpoint: otherwise the policy can skip the landing gate
    # entirely and still converge on the live ball once it closes to 0.5 m.
    # FIX 2026-07-24: env._blue_landed -> env._blue_landed_genuine (excludes
    # "free"/suspiciously-early landings, see _get_reach_target_y). Without
    # this, a free landing let this switch to live-ball tracking too.
    ball_close = (ball_x_local < 0.5) & (~env._blue_wide | env._blue_landed_genuine)

    # Switch from frozen (two-stage) target to live ball Y/Z when ball is within 0.5 m.
    # NEW 2026-08-15 (user request): offset the live-ball target too, or the
    # green offset silently vanishes right at the final approach into
    # contact -- see _INNER_FOOT_TARGET_OFFSET's docstring.
    target_y = torch.where(ball_close, live_y + _get_foot_block_offset(env, ball_name), reach_target_y)
    target_z = torch.where(ball_close, live_z, floor_z_w + 0.10)

    crossing_point = torch.stack(
        [goal_x_w, target_y, target_z], dim=-1
    )                                                                     # (N, 3)
    foot_pos_active = foot_pos_w[arange_n, foot_idx]
    dist_to_crossing = torch.norm(foot_pos_active - crossing_point, dim=-1)  # (N,)
    reach_rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (dist_to_crossing - reach_th)))

    # Lateral velocity toward crossing side amplifies the reach reward.
    # Uses the ASSIGNED FOOT's Y velocity, not root -- see docstring above.
    vel_toward = torch.where(lateral_error > 0, assigned_foot_vel_y, -assigned_foot_vel_y)
    # Flat 1-10x for every region, matching G1's hand/step formula -- see
    # docs/BugFixes.md 2026-07-18 for why a region-conditional escalation
    # was tried and reverted (wrong G1 analog: that's jump's mechanism, not
    # step's, and jump pairs it with airborne-specific safety rewards this
    # always-grounded task has no equivalent of).
    vel_sigma = 1.0 + 3.0 * vel_toward.clamp(0.0, 3.0)

    # REMOVED 2026-08-29 (user request), was: "Blue decel-zone" -- decayed
    # the speed bonus toward neutral as the assigned foot closed on blue
    # specifically (wide, unlanded crossing), added 2026-07-24 so vel_sigma
    # wouldn't keep rewarding carrying speed straight through the zone the
    # foot should be planting in.
    #
    # Root cause of the removal: this is the wide-crossing sibling of the
    # `near_decel` block just removed above (narrow crossings) -- and
    # `blue_stick_landing`'s own docstring already states, for this exact
    # wide-crossing case, that IT is "the real reason the [oscillation]
    # problem doesn't surface [here], not the decay" (see near_stick_reach's
    # docstring, which quotes this). Same redundancy, same fix: rely on
    # `blue_stick_landing` alone (rewards.py, "close AND slow" near blue) as
    # the anti-oscillation/plant mechanism instead of a blanket vel_sigma
    # kill-switch. This one wasn't caught by the narrow-crossing fix earlier
    # today because it's a SEPARATE code block gated on the opposite
    # condition (`env._blue_wide` here vs `~env._blue_wide` for the removed
    # `near_decel`) -- both lived in this same function but only ever apply
    # to disjoint crossing types, so fixing one had zero effect on the
    # other.
    #
    # `blue_trunk_drive`'s own matching decel-zone (its own `blue_approach`
    # block) removed the same day, same reasoning -- see that function.
    #
    # Not yet validated against a live training run -- this is a genuine
    # experiment (user: "im curious if that will work"), not a proven fix
    # like near_decel's removal was. Watch for the wide-crossing oscillation
    # exploit's return (footreach spiking while the assigned foot swings
    # back and forth across blue without landing) -- if it reappears,
    # strengthen blue_stick_landing rather than reinstating this block.

    # REMOVED 2026-08-29 (user request), was: FIX 2026-07-26 near-region
    # velocity neutralization -- held vel_sigma flat at neutral (1.0x, no
    # speed boost) for the ENTIRE pre-arrival window on narrow crossings
    # (~blue_wide & ~ball_close), added to kill an oscillation-farming
    # exploit (policy swinging back and forth across the target to farm
    # repeated velocity-toward-target bursts, model_13750,
    # blue_v2_overshootradius_2026-07-25).
    #
    # Root cause of the removal: `near_stick_reach` (added the SAME DAY,
    # same window, see that function below) already fixes this exploit
    # class directly -- it rewards "close AND slow" (a fast oscillating
    # pass scores near zero at the distance peak because the speed term
    # collapses), which is a strictly more targeted mechanism than a blanket
    # "no speed credit at all" gate. near_stick_reach's own docstring
    # already said as much for the analogous wide-crossing case
    # ("blue_stick_landing is the real reason the problem doesn't surface
    # there, not the decay") -- this removal is applying that same
    # conclusion to the narrow case it was written about.
    #
    # Side effect this blanket gate was ALSO causing (found investigating a
    # 2026-08-29 user report that a fully-trained near-region checkpoint
    # commits to the reach target much later than an early checkpoint does,
    # "stops ~75% of the way, then does the last 25%" once the ball closes
    # to <0.5m): zeroing vel_sigma for the whole ~2m-to-0.5m approach window
    # removes ANY reward benefit to moving fast until the very end, so a
    # policy with enough training time to fully exploit the reward landscape
    # learns to coast to "close enough" (reach_rew's own sigmoid already
    # near-saturates well before exact alignment) and only burst in the
    # final 0.5m where vel_sigma is live again -- worse the more it
    # converges, matching the reported early-vs-late-checkpoint difference.
    # near_stick_reach does not have this failure mode: it's a continuous
    # exp(-dist)*exp(-speed) reward, so it naturally gives ~zero credit for
    # "far and slow" too, rather than removing all speed credit outright.
    #
    # Not yet validated against a live training run -- watch for the
    # oscillation exploit's return (footreach spiking while ball_x_local
    # sits mid-range, same signature as the original 2026-07-25 report) now
    # that near_stick_reach is the sole guard; if it reappears, the fix is a
    # stronger near_stick_reach (higher weight/sharper speed_sigma), not
    # reintroducing this blanket neutralization.

    # Combine: phase1 when ball is far, phase2 sigmoid when close.
    # FIX 2026-08-25 (user request): phase2_threshold param added (was a
    # hardcoded 1.5) so this term's registration can widen the vel_sigma
    # speed-urgency window without affecting other callers' default. See
    # docs/BugFixes.md, 2026-08-25 "faster blue/green approach" entry.
    phase1_mask = ball_x_local > phase2_threshold
    taskrew = torch.where(phase1_mask, phase1_rew, reach_rew * vel_sigma)

    # Upright gate: suppress reward when robot is falling.
    projected_grav = robot.data.projected_gravity_b
    upright = 1.0 - torch.clamp(torch.sum(projected_grav[:, :2] ** 2, dim=1), 0.0, 1.0)
    behind = _ball_is_behind(env, ball_name)
    return taskrew * upright * (~behind).float() * (~env._footreach_overshot_flag).float()


def near_stick_reach(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 8.0,
    speed_sigma: float = 1.5,
) -> torch.Tensor:
    """Dense reward for the assigned foot being simultaneously CLOSE to and
    SLOW near the near-region reach target, before the ball has closed in.

    FIX 2026-07-26 (near-region oscillation): ported from `blue_stick_landing`
    (wide/far crossings' own anti-oscillation mechanism -- see that
    function's docstring and docs/BugFixes.md). Two footreach-only attempts
    to fix near-region oscillation (a distance-gated vel_sigma decay, then a
    flat vel_sigma neutralization gated on ball_close) both left the exploit
    intact somewhere, because neither ever PENALIZES speed near the target
    -- they only ever stop AMPLIFYING it. `blue_stick_landing` does something
    structurally different for wide crossings: it directly rewards the joint
    condition "close AND slow", so a fast oscillation pass scores near ZERO
    at the exact moment dist is near zero, because the speed term collapses
    at high velocity even though the distance term is near its peak. This is
    what actually opposes oscillation for wide crossings; footreach's own
    vel_sigma decay (blue_approach) has the identical structural gap this
    function's docstring describes and would very likely be exploitable
    the same way on its own -- blue_stick_landing is the real reason the
    problem doesn't surface there, not the decay.

    exp(-dist_sigma * dist) * exp(-speed_sigma * speed), peaking at "close
    AND stopped". Same sigma values as blue_stick_landing (unretuned).

    Narrow-region-only by construction (`~env._blue_wide` gate) -- returns
    exactly 0.0 for every wide/far-region env, so it cannot affect far-region
    training signal even though it shares the same reward-config file.
    Active only during the pre-arrival window (`~ball_close`, same formula
    footreach uses) -- zero once the ball closes to within 0.5m, so the
    genuine last-moment interception dive is never penalized for being fast.
    """
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,) -- full crossing point for narrow
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, reach_target_y], dim=-1)   # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]  # (N,)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                       # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                     # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, foot_idx]                     # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    # Same ball_close formula footreach/foot_proximity use.
    ball_close = (ball_x_local < 0.5) & (~env._blue_wide | env._blue_landed_genuine)
    near_active = (~env._blue_wide) & (~ball_close)

    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * near_active.float()


def foot_proximity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Dense continuous reward for foot near the ball crossing point.

    Uses the frozen goal-line crossing point (env._ball_crossing_y, set by
    _get_ball_crossing_y) as the target when the ball is far (>= 0.5 m from goal line).
    When the ball is within 0.5 m, switches to the live ball Y/Z position (mirrors G1
    end_target update in post_physics_step lines 203-206). Rewards exp(-sigma * dist)
    so there is always a gradient pulling the foot toward the arrival point, unlike the
    one-shot stopball. Deactivates once the ball is behind (same gate as footreach).
    Weight: +5.0.

    v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch's
    two-stage mechanism, removed from this project's lineage 2026-07-10
    ("green-ball-baseline"). See footreach's/_get_reach_target_y's
    docstrings for the full mechanism history.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)

    # Frozen crossing Y — used for foot-side selection.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    # Reach target: full crossing point, or the two-stage midpoint-then-full
    # schedule on wide crossings. See _get_reach_target_y.
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]

    # Live ball switch: when ball is within 0.5 m of goal line, track live position.
    # Mirrors G1 post_physics_step: end_target = ball_states[:, :3] when balllocal < 0.5.
    # On wide crossings this switch is ALSO gated on having genuinely landed
    # at the blue midpoint (mirrors footreach) — see _get_reach_target_y.
    # FIX 2026-07-24: env._blue_landed -> env._blue_landed_genuine (excludes
    # free/suspiciously-early landings) -- see _get_reach_target_y.
    ball_close = (ball_x_local < 0.5) & (~env._blue_wide | env._blue_landed_genuine)
    live_y = ball_pos_w[:, 1]
    live_z = ball_pos_w[:, 2]
    # NEW 2026-08-15 (user request): see footreach's matching fix and
    # _INNER_FOOT_TARGET_OFFSET's docstring.
    target_y = torch.where(ball_close, live_y + _get_foot_block_offset(env, ball_name), reach_target_y)
    target_z = torch.where(ball_close, live_z, env_z + 0.10)

    crossing_point = torch.stack(
        [goal_x_w, target_y, target_z], dim=-1                         # (N, 3)
    )

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
    is_left_ball = crossing_y > env.scene.env_origins[:, 1]
    foot_idx = torch.where(is_left_ball,
                           torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                           torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), foot_idx]
    min_dist = torch.norm(foot_pos_active - crossing_point, dim=-1)     # (N,)

    behind = _ball_is_behind(env, ball_name)
    return torch.exp(-sigma * min_dist) * (~behind).float()


def blue_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """One-shot bonus when the assigned foot lands at the blue (midpoint) target.

    v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch
    mechanism. Fires the first step the two-stage gate (env._blue_landed)
    becomes true -- the assigned foot was airborne at some point after
    reset, then came into ground contact within landing_radius of the
    midpoint target (settle-window + speed-gated, see _get_reach_target_y).
    Narrow crossings never fire this (env._blue_wide is always false for
    them).
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_landed is fresh this step

    if not hasattr(env, "_blue_landed_bonus_flag"):
        env._blue_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._blue_landed_bonus_flag[just_reset] = False

    # FIX 2026-07-24: env._blue_landed -> env._blue_landed_genuine -- a free
    # landing (RSI luck, not genuine approach work) must not earn this bonus
    # either. See _get_reach_target_y.
    fired = env._blue_landed_genuine & ~env._blue_landed_bonus_flag
    env._blue_landed_bonus_flag |= fired
    return fired.float()


def blue_overshoot_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.08,
    max_overshoot: float = 0.5,
) -> torch.Tensor:
    """Penalize the assigned foot for advancing past the blue midpoint,
    toward green, on a wide crossing that hasn't been genuinely landed yet.

    v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch
    mechanism. Without this term, a wide episode where the policy ignores
    the blue waypoint entirely earns exactly the same reward (zero, from
    the landing-gated stopball/softstop/blue_ball_landed) as one where it
    tried and failed -- with no cost differential, nothing pushes the
    policy away from ignoring wide episodes altogether. This makes
    overshooting past blue while unlanded worse than stopping short of it
    or reaching it, on every step.

    Deliberately does NOT touch footreach's vel_sigma (which already
    amplifies reward for velocity toward whichever point is the CURRENT
    reach target, and separately decays that bonus in the blue decel-zone
    -- see footreach). The combination is intentional: vel_sigma still
    rewards approaching blue fast, while this term makes carrying that
    speed past blue without stopping costly.

    landing_radius matches _get_reach_target_y's own arrival radius (0.08)
    as a deadband -- being within it counts as "at blue", not overshooting.
    max_overshoot bounds the per-step penalty so a single step can't
    dominate the return.

    Zero on narrow crossings and once genuinely landed.

    FIX 2026-08-12 (user request): added an episode-freshness guard,
    `env.episode_length_buf >= _BLUE_OVERSHOOT_FREE_STEP_THRESHOLD`. Unlike
    every landing-BONUS consumer (`blue_ball_landed`, `stopball`/`softstop`'s
    `landing_ok`), this penalty had no exclusion for a landing-adjacent
    position that's an artifact of where training's RSI (`reset_from_motion_
    data`, 80% of training resets, mid-motion donor poses by construction --
    see _get_reach_target_y's own `_blue_landed_was_free` classification,
    same `episode_length_buf<10` threshold, same rationale) happens to drop
    the assigned foot, rather than the policy's own approach. Without this,
    an episode whose RSI donor frame starts the foot already past the blue
    hinge earns a negative reward on step 0/1, before any action has been
    taken -- exactly the false-positive class `_blue_landed_was_free` exists
    to exclude for the landing bonuses, just never mirrored here for the
    penalty. (Play mode disables RSI by default -- `goalkeeper_env_cfg.py`,
    "Always starting from standing" -- so this guard has no visible effect
    when watching a policy play; it only matters during training.) Not yet
    validated against a live training run.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed fresh

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    half_y = start_y + (full_y - start_y) / 2.0
    direction = torch.sign(full_y - start_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                   # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, foot_idx, 1]                # (N,)

    signed_progress = direction * (assigned_foot_y - half_y)
    overshoot = torch.clamp(signed_progress - landing_radius, min=0.0, max=max_overshoot)

    # FIX 2026-07-24: env._blue_landed -> env._blue_landed_genuine -- a free
    # landing must not silence this penalty either. See _get_reach_target_y.
    # FIX 2026-08-12: added the episode-freshness term -- mirrors
    # _get_reach_target_y's own `_blue_landed_was_free` threshold (10 steps),
    # duplicated here (not imported) since that constant is locally scoped
    # inside that function; kept numerically identical deliberately.
    _BLUE_OVERSHOOT_FREE_STEP_THRESHOLD = 10
    fresh_enough = env.episode_length_buf >= _BLUE_OVERSHOOT_FREE_STEP_THRESHOLD
    phase1_active = env._blue_wide & ~env._blue_landed_genuine & fresh_enough
    return overshoot * phase1_active.float()


def blue_stick_landing(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 8.0,
    speed_sigma: float = 1.5,
) -> torch.Tensor:
    """Dense reward for the assigned foot being simultaneously CLOSE to and
    SLOW near the blue midpoint on a wide, unlanded crossing --
    exp(-dist_sigma * dist_to_blue) * exp(-speed_sigma * foot_speed), peaking
    exactly at "close AND stopped" (a genuine plant), zero elsewhere.

    v2 reimplementation (2026-07-23) of the blue-ball-waypoint branch
    mechanism. Rationale: nothing else in the reward stack gives dense
    credit for the EXACT joint condition (close + slow) the settle-window
    check requires -- footreach/foot_proximity reward proximity to blue,
    and footreach's vel_sigma rewards speed toward blue, but nothing
    rewards decelerating once close. Without a dense basin peaking at the
    target joint condition, a policy has to discover that exact
    combination by chance before any gradient reinforces it. Mirrors this
    project's own `cleanstop` (rewards low BALL speed near a target after
    softstop) -- same "reward stillness near a point" pattern, applied to
    the FOOT approaching blue instead of the ball after a save.

    dist_sigma=8/speed_sigma=1.5 (not the original branch's initial
    dist_sigma=15/speed_sigma=3, widened after those made this a near-cliff
    basin that vanished as soon as stochastic exploration pushed the policy
    just past it): decays to ~0.5 by ~9cm / ~0.46 m/s, ~0.1 by ~29cm /
    ~1.5 m/s. Neither independently retuned for this reimplementation.

    Zero on narrow crossings and once genuinely landed -- mirrors
    blue_overshoot_penalty's phase1_active gate exactly.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed fresh

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    half_y = start_y + (full_y - start_y) / 2.0
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, half_y], dim=-1)           # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                   # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                 # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, foot_idx]                 # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    # FIX 2026-07-24: env._blue_landed -> env._blue_landed_genuine -- a free
    # landing must not silence this reward either. See _get_reach_target_y.
    phase1_active = env._blue_wide & ~env._blue_landed_genuine
    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * phase1_active.float()


def blue_trunk_drive(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward the robot's TRUNK (root) lateral velocity toward the current
    two-stage reach target -- complements footreach, which only rewards the
    task-ASSIGNED FOOT's velocity (see footreach's own docstring on why foot,
    not root, is used there). A foot can extend toward blue/green through leg
    motion alone without the trunk (whole body) actually translating --
    exactly the failure mode a genuine multi-step far-region approach needs
    to avoid, since a standing robot can "reach" with a leg swing but not
    genuinely walk to the target.

    Two phases, both keyed off _get_reach_target_y's own live state (the same
    single source of truth footreach/foot_proximity/stopball/softstop share,
    called here too so this term's _blue_wide/_blue_landed_genuine reads are
    fresh even if this term happens to run before those others this tick):

      - Approaching blue (wide, not genuinely landed): reward trunk velocity
        toward blue, decaying to ZERO (not footreach's neutral-1x floor --
        this term has no separate proximity reward underneath it to protect)
        within the SAME curriculum-eased decel-zone footreach's vel_sigma
        uses (env._blue_landing_radius_current, 0.30m outer edge) -- so this
        doesn't fight the landing gate's need for a genuine stop-and-plant.
      - After a genuine landing (wide, landed): reward trunk velocity toward
        green, undecayed -- mirrors footreach's own phase-2 vel_sigma, which
        also doesn't decay approaching the final target (carrying speed into
        the actual save is correct; only the artificial blue pause point
        needs a deliberate stop).

    FIX 2026-07-24 (user request: "two accelerations -- toward blue, then
    decelerate, then accelerate again after blue has landed"). Root cause:
    footreach's own phase-2 velocity bonus (vel_sigma) only applies once
    ball_x_local <= 1.5m. On a wide crossing, a genuine landing at blue
    typically happens while the ball is still much farther out than that --
    blue exists precisely so the assigned foot has time to plant early, then
    travel further to green -- so between "landed at blue" and "ball finally
    within 1.5m", nothing rewarded moving the TRUNK toward green at all (the
    near-region/already-close-ball case was always covered by footreach's
    existing vel_sigma; this term fills the specifically-far-region,
    specifically-post-landing gap, using trunk velocity per user request
    since footreach's own incentive is foot-specific, not whole-body).

    Zero on narrow crossings (env._blue_wide always False there -- no
    two-stage target exists to drive the trunk toward; footreach's own
    vel_sigma already covers narrow/near-region locomotion) and once the
    ball is behind (mirrors footreach's own behind-gate).
    """
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,) blue or green Y, phase-aware

    robot: Entity = env.scene["robot"]
    trunk_y = robot.data.root_link_pos_w[:, 1]
    trunk_vel_y = robot.data.root_link_lin_vel_w[:, 1]

    lateral_error = reach_target_y - trunk_y                            # positive → target right
    vel_toward = torch.where(lateral_error > 0, trunk_vel_y, -trunk_vel_y)
    drive = vel_toward.clamp(0.0, 3.0)                                  # 0..3 m/s, same clamp range as footreach's vel_sigma

    # REMOVED 2026-08-29 (user request), was: Blue decel-zone -- same
    # mechanism/constants as footreach's own (just-removed) blue decel-zone,
    # but decaying all the way to 0 (not a 1x floor) since this term has no
    # separate proximity reward underneath it to protect. Removed for the
    # same reason as footreach's: redundant with `blue_stick_landing`, which
    # already handles anti-oscillation/plant credit for wide crossings (see
    # footreach's own removal comment above for the full evidence). Not yet
    # validated against a live training run -- same watch-list as footreach's
    # removal (wide-crossing oscillation exploit returning).

    behind = _ball_is_behind(env, ball_name)
    active = env._blue_wide & (~behind)
    return drive * active.float()


def orange_foot_proximity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Trailing-foot mirror of foot_proximity -- dense exp(-sigma*dist) pull of
    the TRAILING (non-assigned) foot toward the orange target. Wide-crossings
    only (env._blue_wide) -- unlike foot_proximity, which has no such gate
    because _get_reach_target_y itself switches targets on narrow crossings;
    _get_orange_reach_target_y never switches, so this function gates
    explicitly instead. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.

    Not yet validated against a live training run.
    """
    robot: Entity = env.scene[asset_cfg.name]
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z = env.scene.env_origins[:, 2]
    target_point = torch.stack([goal_x_w, orange_y, env_z + 0.10], dim=-1)   # (N, 3)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), trailing_idx]
    dist = torch.norm(foot_pos_active - target_point, dim=-1)

    behind = _ball_is_behind(env, ball_name)
    return torch.exp(-sigma * dist) * env._orange_wide.float() * (~behind).float()


def orange_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_ball_landed -- one-shot bonus when the
    trailing foot genuinely lands at the orange target. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.

    Not yet validated against a live training run.
    """
    _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _orange_landed_genuine is fresh

    if not hasattr(env, "_orange_landed_bonus_flag"):
        env._orange_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._orange_landed_bonus_flag[just_reset] = False

    fired = env._orange_landed_genuine & ~env._orange_landed_bonus_flag
    env._orange_landed_bonus_flag |= fired
    return fired.float()


def orange_overshoot_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.08,
    max_overshoot: float = 0.5,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_overshoot_penalty -- penalizes the
    trailing foot for advancing past the orange target, toward the true
    crossing point, before landing there. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.

    Not yet validated against a live training run.
    """
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    direction = torch.sign(full_y - start_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, trailing_idx, 1]            # (N,)

    signed_progress = direction * (assigned_foot_y - orange_y)
    overshoot = torch.clamp(signed_progress - landing_radius, min=0.0, max=max_overshoot)

    phase1_active = env._orange_wide & ~env._orange_landed_genuine
    return overshoot * phase1_active.float()


def orange_stick_landing(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 8.0,
    speed_sigma: float = 1.5,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_stick_landing -- dense reward for the
    trailing foot being simultaneously CLOSE to and SLOW near the orange
    target on a wide, unlanded crossing. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.

    Not yet validated against a live training run.
    """
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, orange_y], dim=-1)              # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]              # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]              # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    phase1_active = env._orange_wide & ~env._orange_landed_genuine
    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * phase1_active.float()


def trailing_foot_reach(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    reach_th: float = 0.3,
    sigma: float = 5.0,
) -> torch.Tensor:
    """General foot-target reach reward for the trailing foot's orange->red
    waypoint sequence -- gives the trailing foot the same urgency mechanism
    footreach already gives the leading foot (sigmoid reach reward x a
    velocity-toward-target multiplier, up to 10x), closing the gap that made
    the double-step sequence slow: orange_foot_proximity/red_foot_proximity
    are flat exp(-sigma*dist) pulls with no speed incentive at all, and
    blue_trunk_drive only rewards whole-body trunk velocity, not the
    trailing foot specifically.

    NEW 2026-08-15 (user request, "make a general foot target reward, so
    not footreach alike because it is ball focused but the fixes implement
    for the trailing foot here"). Deliberately NOT a literal footreach port:

    - No phase1 (ball_x_local > 1.5m) / phase2 split. footreach's phase1
      exists to withhold the speed multiplier until the ball is close
      enough to matter -- a ball-position concept. This reward's target
      (orange or red) is relevant for the whole wide-crossing window
      regardless of ball position, so it's always sigmoid-reach x vel_sigma,
      no phase gate.
    - No live-ball tracking switch. footreach switches its target to the
      live ball once ball_x_local < 0.5m (the leading foot's actual
      interception job). The trailing foot's job is capped at its own
      waypoint by design (orange's own docstring: "does NOT graduate to a
      second/live-ball target"; red mirrors this) -- this reward never
      reads live ball position at all.
    - Auto-switches target orange_y -> red_y via env._red_active (mirrors
      _get_reach_target_y's own blue_y -> full_y switch on
      env._blue_landed_genuine), so ONE function covers the whole
      double-step sequence rather than two separate ball-specific rewards.
    - Decel-zone near whichever target is currently active (same shape as
      footreach's own blue_decel_zone), using the orange/red landing radius
      (numerically identical -- both curriculum-ease from the same 0.15m
      default) as the floor, so vel_sigma decays toward neutral right at
      the target instead of rewarding carrying speed through it.

    Deliberately NOT ported (per explicit user request -- only port fixes
    if the same failure mode is actually observed on the trailing foot,
    not preemptively): footreach's overshoot-kill flag (a fix for a
    ball-impact-impulse false-positive specific to footreach's own
    live-training history) and its near-region-oscillation vel_sigma
    neutralization (narrow crossings have no orange/red concept at all --
    env._orange_wide is always False there, so this reward is already zero
    for narrow crossings by construction, unlike footreach which needed an
    explicit narrow-region guard).

    Zero once the trailing foot's job is fully done (env._red_active &
    env._red_landed_genuine) -- no reward for lingering at red after
    genuinely landing there.

    Not yet validated against a live training run.
    """
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    red_y = _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    red_active = env._red_active
    target_y = torch.where(red_active, red_y, orange_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]                 # (N, 3)
    assigned_foot_vel_y = foot_vel_w[arange_n, trailing_idx, 1]            # (N,)

    goal_x_w = env.scene.env_origins[:, 0]
    floor_z_w = env.scene.env_origins[:, 2]
    target_point = torch.stack([goal_x_w, target_y, floor_z_w + 0.10], dim=-1)  # (N, 3)
    dist_to_target = torch.norm(assigned_foot_pos - target_point, dim=-1)       # (N,)
    reach_rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (dist_to_target - reach_th)))

    lateral_error = target_y - assigned_foot_pos[:, 1]
    vel_toward = torch.where(lateral_error > 0, assigned_foot_vel_y, -assigned_foot_vel_y)
    vel_sigma = 1.0 + 3.0 * vel_toward.clamp(0.0, 3.0)

    current_landed_genuine = torch.where(red_active, env._red_landed_genuine, env._orange_landed_genuine)
    approaching = env._orange_wide & ~current_landed_genuine
    _DECEL_ZONE = 0.30
    _DECEL_FLOOR = float(getattr(env, "_orange_landing_radius_current", 0.08))
    decay_frac = ((dist_to_target - _DECEL_FLOOR) / (_DECEL_ZONE - _DECEL_FLOOR)).clamp(0.0, 1.0)
    vel_sigma = torch.where(approaching, 1.0 + (vel_sigma - 1.0) * decay_frac, vel_sigma)

    done = red_active & env._red_landed_genuine
    behind = _ball_is_behind(env, ball_name)
    return reach_rew * vel_sigma * env._orange_wide.float() * (~behind).float() * (~done).float()


def sequence_promptness(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    promptness_ref: float = 1.5,
) -> torch.Tensor:
    """One-shot bonus rewarding a WIDE crossing's full blue->orange->red->save
    relay happening with margin to spare, not just barely in time -- direct
    answer to the user's "make the whole blue ball etc earlier" request.

    NEW 2026-08-15 (user request). Caches each stage's "spare distance"
    (ball_x_local remaining, clamped to [0,1] against `promptness_ref`) the
    instant it genuinely completes -- blue landing, orange landing, red
    landing, AND the save itself (env._sb_flag first firing) -- but only
    PAYS OUT once, at the exact tick the save genuinely happens, as the
    average of whichever stages actually occurred (0 for a stage that never
    happened, e.g. red never activating, or orange never landing).

    Deliberately deferred-payout, not 3-4 separate immediate one-shot
    bonuses (the original proposal) -- user's own reasoning: paying out
    promptness at each stage independently would reward a policy that
    blitzes blue/orange/red fast but then fails to actually save; only a
    genuinely completed save should earn credit for having been prompt
    along the way. This also means blue/orange/red's own promptness values
    are computed and cached long before payout, but never returned as
    reward on their own -- only folded into this term's single payout tick.

    Cap: each cached component is already clamp(ball_x_local/promptness_ref,
    0, 1), and the average of up to 4 such values is inherently bounded in
    [0,1] -- no separate cap logic needed (user's own "could we cap it too"
    concern is satisfied structurally, not via an extra clamp on the sum).

    promptness_ref=1.5 reuses footreach's own existing "ball still far"
    phase boundary rather than introducing a new magic constant.

    Wide crossings only (env._blue_wide) -- narrow crossings have no
    blue/orange/red concept, unrelated to the double-step slowness problem
    this addresses; this term is exactly 0 there.

    landing_ok (stopball's own gate, `~env._blue_wide | env._blue_landed_
    genuine`) already guarantees blue is genuinely landed by the time a wide
    crossing's save can fire at all -- so the blue component is always
    captured by save-time on a wide crossing; orange/red are optional,
    scoring 0 if skipped, which is the intended "reward completing the
    WHOLE relay" shaping.

    Not yet validated against a live training run.
    """
    # Defensive freshness calls -- registration order already guarantees
    # these are fresh (this term is registered after stopball/blue/orange/
    # red's own terms in goalkeeper_env_cfg.py), but every other multi-stage
    # reader in this file (e.g. blue_overshoot_penalty) makes the same
    # belt-and-suspenders call rather than relying on registration order
    # alone.
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    promptness_now = (ball_x_local / promptness_ref).clamp(0.0, 1.0)

    n = env.num_envs
    if not hasattr(env, "_seq_blue_promptness"):
        env._seq_blue_promptness = torch.zeros(n, device=env.device)
        env._seq_orange_promptness = torch.zeros(n, device=env.device)
        env._seq_red_promptness = torch.zeros(n, device=env.device)
        env._seq_save_promptness = torch.zeros(n, device=env.device)
        env._seq_blue_captured = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._seq_orange_captured = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._seq_red_captured = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._seq_paid = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._seq_blue_promptness[just_reset] = 0.0
    env._seq_orange_promptness[just_reset] = 0.0
    env._seq_red_promptness[just_reset] = 0.0
    env._seq_save_promptness[just_reset] = 0.0
    env._seq_blue_captured[just_reset] = False
    env._seq_orange_captured[just_reset] = False
    env._seq_red_captured[just_reset] = False
    env._seq_paid[just_reset] = False

    newly_blue = env._blue_landed_genuine & ~env._seq_blue_captured
    env._seq_blue_promptness = torch.where(newly_blue, promptness_now, env._seq_blue_promptness)
    env._seq_blue_captured |= newly_blue

    newly_orange = env._orange_landed_genuine & ~env._seq_orange_captured
    env._seq_orange_promptness = torch.where(newly_orange, promptness_now, env._seq_orange_promptness)
    env._seq_orange_captured |= newly_orange

    newly_red = env._red_landed_genuine & ~env._seq_red_captured
    env._seq_red_promptness = torch.where(newly_red, promptness_now, env._seq_red_promptness)
    env._seq_red_captured |= newly_red

    sb_flag = getattr(env, "_sb_flag", torch.zeros(n, dtype=torch.bool, device=env.device))
    newly_saved = sb_flag & ~env._seq_paid
    env._seq_save_promptness = torch.where(newly_saved, promptness_now, env._seq_save_promptness)

    total = (
        env._seq_blue_promptness + env._seq_orange_promptness
        + env._seq_red_promptness + env._seq_save_promptness
    ) / 4.0

    fired = newly_saved & env._blue_wide
    env._seq_paid |= fired
    return fired.float() * total


def red_foot_proximity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Trailing-foot mirror of orange_foot_proximity, for the "red" second
    waypoint -- dense exp(-sigma*dist) pull toward red, active only once
    env._red_active (both blue and orange genuinely landed). See
    _get_red_reach_target_y's docstring for the full mechanism/formula.

    FIX 2026-08-15 (user request): gate changed from `(~behind)` to
    `(~env._red_landed_genuine)`, matching red_overshoot_penalty/
    red_stick_landing's `phase1_active` pattern (their gate has always been
    `_red_wide & _red_active & ~_red_landed_genuine`, no `behind` involved).
    `_red_active` itself requires blue AND orange to have already genuinely
    landed, which structurally only happens once the ball has already
    deflected -- so `behind` is very likely already True by the time
    `_red_active` first turns True, making the old `(~behind)` gate
    dead-on-arrival: this term could almost never fire a nonzero reward.
    Root cause of the user's complaint that the trailing foot doesn't
    visibly stay planted near the ball after a save even though this term
    exists to reward exactly that. Not yet validated against a live
    training run.
    """
    robot: Entity = env.scene[asset_cfg.name]
    red_y = _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z = env.scene.env_origins[:, 2]
    target_point = torch.stack([goal_x_w, red_y, env_z + 0.10], dim=-1)   # (N, 3)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), trailing_idx]
    dist = torch.norm(foot_pos_active - target_point, dim=-1)

    return torch.exp(-sigma * dist) * env._red_wide.float() * env._red_active.float() * (~env._red_landed_genuine).float()


def red_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Trailing-foot mirror of orange_ball_landed -- one-shot bonus when the
    trailing foot genuinely lands at red. No extra gate needed here beyond
    env._red_landed_genuine itself -- that flag can only become true after
    env._red_active by construction (see _get_red_reach_target_y's
    `candidate` line).

    Not yet validated against a live training run.
    """
    _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _red_landed_genuine is fresh

    if not hasattr(env, "_red_landed_bonus_flag"):
        env._red_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._red_landed_bonus_flag[just_reset] = False

    fired = env._red_landed_genuine & ~env._red_landed_bonus_flag
    env._red_landed_bonus_flag |= fired
    return fired.float()


def red_overshoot_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.08,
    max_overshoot: float = 0.5,
) -> torch.Tensor:
    """Trailing-foot mirror of orange_overshoot_penalty -- penalizes the
    trailing foot for advancing past red, toward the true crossing point,
    before landing there. Gated additionally on env._red_active (unlike
    orange_overshoot_penalty, which has no upstream landing gate) so this
    can't fire before the trailing foot is even allowed to target red.

    Not yet validated against a live training run.
    """
    red_y = _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    direction = torch.sign(full_y - start_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, trailing_idx, 1]            # (N,)

    signed_progress = direction * (assigned_foot_y - red_y)
    overshoot = torch.clamp(signed_progress - landing_radius, min=0.0, max=max_overshoot)

    phase1_active = env._red_wide & env._red_active & ~env._red_landed_genuine
    return overshoot * phase1_active.float()


def red_stick_landing(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 8.0,
    speed_sigma: float = 1.5,
) -> torch.Tensor:
    """Trailing-foot mirror of orange_stick_landing -- dense reward for the
    trailing foot being simultaneously CLOSE to and SLOW near red, gated
    additionally on env._red_active (see red_overshoot_penalty's docstring).

    Not yet validated against a live training run.
    """
    red_y = _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, red_y], dim=-1)              # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]              # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]              # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    phase1_active = env._red_wide & env._red_active & ~env._red_landed_genuine
    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * phase1_active.float()


def stopball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    delta_vel_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """One-time reward when ball X velocity increases by >= delta_vel_threshold (m/s).

    Ball approaches with negative X velocity; foot contact reverses or decelerates it.
    Fires exactly once per episode when delta_vx > threshold, providing the primary
    training signal for a successful save. Mirrors Imitationlearningbooster stopball.

    v2 landing gate (2026-07-23, reimplemented from the blue-ball-waypoint
    branch, removed 2026-07-10): on wide crossings (env._blue_wide), this
    can only fire once the assigned foot has genuinely landed at the blue
    midpoint (env._blue_landed & ~env._blue_landed_was_free -- excludes
    landings achieved suspiciously early, which can't reflect real approach
    work) -- otherwise the policy can skip the two-stage waypoint entirely
    with one continuous reach and still collect the full save reward. Layered
    ON TOP OF the correct-foot-contact gating below (added independently,
    2026-07-14/07-15, after this landing gate had already been removed) --
    both must hold, not either/or. Narrow crossings are unaffected
    (env._blue_wide always false for them).

    FIX 2026-07-14: was missing any check on WHICH foot deflected the ball --
    a deflection off the wrong (non-assigned) foot fired full reward exactly
    like a correct-foot save. Now requires the task-assigned foot
    (_get_correct_foot_idx) to actually be in contact, matching softstop's
    own correct-foot gate (which already existed but only fed downstream
    quality bonuses, never gated softstop itself either -- see that fix).

    FIX 2026-07-15: that check used "feet_contact", a GROUND-contact sensor
    (secondary=None -- fires whenever a foot touches anything, mostly the
    ground) -- not ball-specific. Tried switching to a dedicated
    "ball_contact" foot-vs-ball ContactSensorCfg instead, but that made
    stopball/softstop stop firing almost entirely in practice (MuJoCo
    contact detection for a small rolling ball vs. foot geoms is
    apparently too narrow a window to reliably catch at the exact step
    the velocity threshold trips). REVERTED to "feet_contact".

    FIX 2026-07-15 (second pass): instead, requires stopball's own raw
    deflection condition to also be true the SAME step softstop's condition
    fires (env._sb_deflection_now, set below, read by softstop) -- a
    genuine single-contact event should trip both stopball's delta-vx
    check and softstop's absolute-reversal check at essentially the same
    physical instant; a coincidental "wrong foot happens to be standing
    near the ball as it settles" scenario should not.
    """
    # FIX 2026-07-23: must pass asset_cfg through -- calling with the bare
    # 2-arg form fell back to this function's own default parameter, which
    # is a *different* SceneEntityCfg object than the one registered in
    # goalkeeper_env_cfg.py's params for footreach/blue_ball_landed/etc.
    # mjlab only resolves SceneEntityCfg.body_ids (str names -> int ids) for
    # objects it finds inside a term's OWN registered params dict
    # (manager_base.py:_resolve_common_term_cfg) -- an object that's never
    # in any params dict never gets resolved, so its body_ids silently stays
    # the un-resolved default `slice(None)` (all bodies) for the entire run.
    # Since stopball is registered first each tick and is therefore the one
    # call allowed to update _get_reach_target_y's settle counter, indexing
    # body_link_pos_w with body_ids=slice(None) then picking index 0/1 read
    # some arbitrary early body (root/pelvis), not a foot -- garbage
    # dist_to_blue, permanently keeping candidate False on the only call
    # that mattered, even though every OTHER (properly asset_cfg-resolved)
    # term saw candidate=True the same tick. Confirmed live via a raw
    # per-call print: stopball's call read dist=0.29 while footreach's call
    # (same tick) read dist=0.08 for what should be the identical foot.
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed are fresh this step

    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_sb_init_vx"):
        env._sb_init_vx = ball_x_vel.clone()
        env._sb_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset] = False
    env._sb_init_vx[just_reset] = ball_x_vel[just_reset].clone()

    delta_vx = ball_x_vel - env._sb_init_vx
    in_front = ball_x_local > -0.3  # allow 0.3 m past goal line: deflection accumulates gradually

    foot_idx = _get_correct_foot_idx(env, ball_name)
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_in_contact = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)
    correct_foot_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]

    # FIX 2026-07-24: reuses env._blue_landed_genuine (== env._blue_landed &
    # ~env._blue_landed_was_free, computed once in _get_reach_target_y,
    # already called above this line) instead of recomputing the same
    # expression locally -- pure dedup, same value as before.
    landing_ok = ~env._blue_wide | env._blue_landed_genuine

    # Raw condition (not gated by the one-shot ~env._sb_flag latch) -- this
    # is "is a deflection happening right now", read by softstop below.
    # Registered before "softstop" in goalkeeper_env_cfg.py's rewards dict,
    # so this is fresh (this step, not stale) by the time softstop runs.
    deflection_now = (delta_vx > delta_vel_threshold) & in_front & correct_foot_contact & landing_ok
    env._sb_deflection_now = deflection_now

    fired = deflection_now & ~env._sb_flag
    env._sb_flag |= fired
    return fired.float()


def softstop(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    velocity_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """One-time reward when ball world-X velocity exceeds velocity_threshold (m/s).

    The ball must actually reverse direction and be rolling back into the field (+X).
    This is a genuine save: foot contact deflected the ball away from the goal.
    Uses its own _softstop_flag, independent of stopball.

    Also gates _ball_is_behind and tracks which foot was in contact when softstop fires
    (used by single_foot_save, inner_face_orientation_save, cleanstop, airborne_at_save).

    v2 landing gate (2026-07-23, reimplemented from the blue-ball-waypoint
    branch, removed 2026-07-10): same wide-crossing landing gate as
    stopball -- see that function's docstring.

    FIX 2026-07-14: correct-foot contact was computed AFTER `fired`, only to
    record env._softstop_correct_foot for downstream quality bonuses -- it
    never gated `fired` itself, so a deflection off the wrong (non-assigned)
    foot earned this reward's full weight (up to 262.5, the single largest
    term in the whole reward table) exactly like a correct-foot save. Now
    computed unconditionally and required in `fired`. env._softstop_correct_foot
    is consequently always True whenever _softstop_flag is True -- downstream
    gates on it become no-ops, which is fine (correct-foot is now guaranteed
    at this point rather than merely tracked).

    FIX 2026-07-15: that check used "feet_contact", a GROUND-contact sensor
    (secondary=None) -- not ball-specific, so a foot merely standing on the
    ground satisfied it regardless of which foot actually deflected the
    ball. Tried a dedicated "ball_contact" foot-vs-ball sensor instead, but
    that made stopball/softstop stop firing almost entirely in practice.
    REVERTED to "feet_contact".

    FIX 2026-07-15 (second pass): instead, now additionally requires
    env._sb_deflection_now (stopball's own raw deflection condition, set
    THIS SAME step -- stopball is registered before softstop in
    goalkeeper_env_cfg.py's rewards dict) to also be true. A genuine
    single-contact event should trip both stopball's delta-vx check and
    softstop's absolute-reversal check at essentially the same physical
    instant; requiring both prevents a wrong-foot-standing-nearby
    coincidence from firing this on its own. See stopball's docstring.
    """
    # FIX 2026-07-23: must pass asset_cfg through -- calling with the bare
    # 2-arg form fell back to this function's own default parameter, which
    # is a *different* SceneEntityCfg object than the one registered in
    # goalkeeper_env_cfg.py's params for footreach/blue_ball_landed/etc.
    # mjlab only resolves SceneEntityCfg.body_ids (str names -> int ids) for
    # objects it finds inside a term's OWN registered params dict
    # (manager_base.py:_resolve_common_term_cfg) -- an object that's never
    # in any params dict never gets resolved, so its body_ids silently stays
    # the un-resolved default `slice(None)` (all bodies) for the entire run.
    # Since stopball is registered first each tick and is therefore the one
    # call allowed to update _get_reach_target_y's settle counter, indexing
    # body_link_pos_w with body_ids=slice(None) then picking index 0/1 read
    # some arbitrary early body (root/pelvis), not a foot -- garbage
    # dist_to_blue, permanently keeping candidate False on the only call
    # that mattered, even though every OTHER (properly asset_cfg-resolved)
    # term saw candidate=True the same tick. Confirmed live via a raw
    # per-call print: stopball's call read dist=0.29 while footreach's call
    # (same tick) read dist=0.08 for what should be the identical foot.
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed are fresh this step

    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_softstop_flag"):
        env._softstop_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._softstop_correct_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # NEW 2026-08-15 (user request): "if the contact is detected between
        # the sole of the foot and the ball that the reward for cleanstop is
        # not given... only want to save with the side of the feet."
        # Computed HERE (not inside cleanstop/single_foot_save themselves)
        # specifically because `asset_cfg` is only genuinely resolved on
        # THIS function's own registered params (goalkeeper_env_cfg.py
        # passes asset_cfg=_FEET_CFG for softstop; cleanstop/single_foot_save
        # take no asset_cfg param at all) -- see
        # .claude/skills/reward-shaping-scene-entity-cfg. Passing softstop's
        # own already-resolved `asset_cfg` through to
        # _sole_ball_contact_per_foot avoids ever needing an unresolved
        # default SceneEntityCfg.
        #
        # FIX 2026-08-15 (same day, user report: "when sole ball contact is
        # hit i still see a spike in cleanstop how come"): the first version
        # only captured sole contact at the EXACT tick softstop fired --
        # missed sole contact happening a step or two before/after that
        # instant, including anywhere in cleanstop's own settle window
        # (multiple ticks of low ball speed AFTER softstop fires, before
        # cleanstop actually pays out). Replaced with a per-EPISODE latch,
        # updated every call (not just when `fired`) and OR'd in, matching
        # the user's confirmed intent (`AskUserQuestion`: whole-episode latch
        # vs. softstop-to-cleanstop window only vs. custom -- user picked
        # whole-episode) -- any sole touch by the assigned foot at ANY point
        # in the episode permanently blocks cleanstop/single_foot_save for
        # the rest of it.
        env._episode_sole_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._softstop_flag[just_reset] = False
    env._softstop_correct_foot[just_reset] = False
    env._episode_sole_contact[just_reset] = False

    in_front = ball_x_local > -0.3

    foot_idx = _get_correct_foot_idx(env, ball_name)
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_in_contact = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)
    correct_foot_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]

    same_step_as_stopball = getattr(
        env, "_sb_deflection_now", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    # FIX 2026-07-24: reuses env._blue_landed_genuine (== env._blue_landed &
    # ~env._blue_landed_was_free, computed once in _get_reach_target_y,
    # already called above this line) instead of recomputing the same
    # expression locally -- pure dedup, same value as before.
    landing_ok = ~env._blue_wide | env._blue_landed_genuine

    fired = (
        (ball_x_vel > velocity_threshold) & in_front & correct_foot_contact
        & same_step_as_stopball & landing_ok & ~env._softstop_flag
    )
    env._softstop_correct_foot[fired] = correct_foot_contact[fired]

    # Computed and latched EVERY call, unconditional on `fired` -- see the
    # 2026-08-15 FIX comment above for why (episode-wide latch, not a
    # single-instant capture).
    sole_per_foot = _sole_ball_contact_per_foot(env, ball_name, asset_cfg=asset_cfg)  # (N, 2)
    assigned_sole_contact_now = sole_per_foot[torch.arange(env.num_envs, device=env.device), foot_idx]
    env._episode_sole_contact |= assigned_sole_contact_now

    env._softstop_flag |= fired
    return fired.float()


def success(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    strict_th: float = 0.15,
) -> torch.Tensor:
    """Continuing, tiered-after-save, close-to-target reward. Originally ported
    from G1 _reward_success; FIX 2026-07-27 (user request) retiered the
    multiplier off softstop/cleanstop instead of stopball -- see that fix's
    comment further down this function for the current 1.0x/2.0x/3.0x ladder
    and why. The `dist`/`landing_ok` mechanics below are unchanged from the
    original G1 port.

    G1 (legged_robot.py:1402-1403): `(success_flag + 1.0) * (dist < strict_th)`.
    success_flag is set once inside _reward_stopball, on the exact same
    delta-vx deflection event that fires stopball itself (legged_robot.py:
    1411-1413: `success_ids = (stop_flag==0) & changevel; success_flag[success_ids]
    = 1.0`), and persists for the rest of the episode (only cleared on reset,
    legged_robot.py:721). `dist` is a continuously-updated hand-to-end_target
    distance recomputed every physics step for every env regardless of phase
    (post_physics_step lines 203-223) -- crucially NOT gated by "ball behind":
    the reward keeps firing at 2x (once success_flag is set) for as long as the
    hand stays within strict_th of the target, including after the save.
    strict_th=0.15 (g1_29_config.py:342), tighter than eereach's own reach_th=0.2
    -- success requires closer proximity than the reach reward's own threshold.

    SGK port:
      - success_flag -> originally env._sb_flag (stopball's flag, matching G1's
        own event exactly). FIX 2026-07-27 changed this to a 3-tier ladder off
        env._softstop_flag/env._cleanstop_flag instead -- a deliberate SGK
        divergence beyond the literal G1 port, not a parity fix. See the
        ordering/staleness comment further down this function.
      - dist -> distance from the task-assigned foot (_get_correct_foot_idx) to
        the same frozen-far/live-close crossing_point used by footreach and
        foot_proximity (see those docstrings: frozen crossing_y when the ball
        is >= 0.5 m from the goal line, live ball Y/Z when closer -- mirrors
        G1's own end_target update, post_physics_step lines 203-206). Not
        gated by _ball_is_behind, matching G1 exactly: the foot is rewarded for
        staying planted at the save spot through the rest of the episode, not
        only during the approach.

    Weight/curriculum: G1 uses 5 -> 12.5 (success_init=5.0, g1_29_config.py:300,
    scaled by the same weight = base*(1+0.5*curriculumupdate) formula as every
    other curriculum-scaled reward here, max at curriculumupdate=3). Ported
    verbatim as `success_curriculum` in goalkeeper_env_cfg.py. Not folded into
    the item-8 stopball/softstop/quality-bonus peak-magnitude cap: G1 itself
    keeps `success` as its own separate-scale term, outside `_reward_stopball`'s
    own weight ceiling -- this port preserves that same separation.

    FIX 2026-07-24 (v2 landing gate, user request): this term was ported
    from G1 without stopball/softstop's wide-crossing landing_ok gate (see
    stopball's docstring) -- unlike stopball/softstop, `dist` here targets
    the true/final crossing_point directly, never the blue midpoint, so the
    un-gated base term `1.0 * (dist < strict_th)` could fire continuously
    just from beelining straight to the final target on a wide crossing,
    completely bypassing the two-stage waypoint the rest of the mechanism
    exists to enforce. Added the same landing_ok gate stopball/softstop use
    (wide crossings must have a genuine, non-suspiciously-early blue
    landing first); narrow crossings are unaffected (landing_ok is always
    True there). success_flag (_sb_flag) itself is already gated
    transitively -- stopball only sets it once landing_ok holds -- so this
    closes the same loophole for the un-doubled base term too.
    """
    # Ensures _blue_wide/_blue_landed are fresh this step, mirroring
    # stopball/softstop's own call (see stopball's asset_cfg docstring for
    # why the asset_cfg must be threaded through explicitly). Memoized
    # per real physics tick, so calling this again here (stopball/softstop
    # already called it earlier this tick) is a cheap no-op on the settle
    # counter, not a double-increment.
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                                # (N, 3)
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]        # (N,)
    crossing_y = _get_ball_crossing_y(env, ball_name)                     # (N,)

    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]
    ball_close = ball_x_local < 0.5
    target_y = torch.where(ball_close, ball_pos_w[:, 1], crossing_y)
    target_z = torch.where(ball_close, ball_pos_w[:, 2], env_z + 0.10)
    crossing_point = torch.stack([goal_x_w, target_y, target_z], dim=-1)  # (N, 3)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]     # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                       # (N,)
    arange = torch.arange(env.num_envs, device=env.device)
    foot_pos_active = foot_pos_w[arange, foot_idx]
    dist = torch.norm(foot_pos_active - crossing_point, dim=-1)            # (N,)

    # FIX 2026-07-27 (user request): retiered off stopball (env._sb_flag) --
    # stopball is just the initial deflection, the easiest/loosest event in
    # this reward table, not the actual outcome the user wants "success" to
    # track. Now a 3-tier ladder off softstop/cleanstop instead: 1.0x before
    # either fires, 2.0x once softstop fires (full velocity reversal --
    # "a little easier"), 3.0x once cleanstop fires (ball nearly dead +
    # correct foot -- the genuinely hard, final outcome). cleanstop's own
    # gate already requires softstop_fired first (cleanstop() reads
    # env._softstop_flag), so cleanstop_flag=True implies softstop_flag=True
    # -- the two terms can never double-count, this is a clean 1/2/3 ladder
    # with no extra branching needed.
    #
    # IMPORTANT ORDERING NOTE: this term is registered in
    # goalkeeper_env_cfg.py AFTER "cleanstop" (moved there specifically for
    # this fix -- previously "success" was registered 3rd and "cleanstop"
    # 5th) so that env._cleanstop_flag is guaranteed fresh THIS tick, not
    # last tick's stale value -- the same class of staleness bug this
    # project has hit before (see the 2026-07-23 asset_cfg staleness fix
    # elsewhere in this file). env._softstop_flag is unaffected by ordering
    # either way since softstop was already registered before success.
    softstop_flag = getattr(
        env, "_softstop_flag", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    cleanstop_flag = getattr(
        env, "_cleanstop_flag", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    multiplier = 1.0 + softstop_flag.float() + cleanstop_flag.float()
    # FIX 2026-07-24: reuses env._blue_landed_genuine (== env._blue_landed &
    # ~env._blue_landed_was_free, computed once in _get_reach_target_y,
    # already called above this line) instead of recomputing the same
    # expression locally -- pure dedup, same value as before.
    landing_ok = ~env._blue_wide | env._blue_landed_genuine
    return multiplier * (dist < strict_th).float() * landing_ok.float()


def single_foot_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    window: int = 3,
) -> torch.Tensor:
    """One-time bonus when stopball and softstop both fire within `window` steps AND correct foot made contact.

    Same-step coincidence was too strict: MuJoCo soft contacts accumulate over
    several steps, so stopball (delta_vx > threshold) and softstop (ball_x_vel > threshold)
    can fire 1-3 steps apart on a clean single-foot save. A two-touch exploit still
    fires them many steps apart (first foot slows, second foot taps back).

    Tracks the episode step when each flag first rose, fires when both are set,
    within window, and the correct foot was in contact at softstop moment.
    Two int32 tensors, negligible GPU cost.
    """
    sb_flag = getattr(env, "_sb_flag", None)
    ss_flag = getattr(env, "_softstop_flag", None)

    if sb_flag is None or ss_flag is None:
        return torch.zeros(env.num_envs, device=env.device)

    _UNSET = -999
    if not hasattr(env, "_sfs_flag"):
        env._sfs_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._sfs_sb_step = torch.full((env.num_envs,), _UNSET, dtype=torch.int32, device=env.device)
        env._sfs_ss_step = torch.full((env.num_envs,), _UNSET, dtype=torch.int32, device=env.device)
        env._sfs_sb_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._sfs_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._sfs_flag[just_reset]    = False
    env._sfs_sb_step[just_reset] = _UNSET
    env._sfs_ss_step[just_reset] = _UNSET
    env._sfs_sb_prev[just_reset] = False
    env._sfs_ss_prev[just_reset] = False

    step = env.episode_length_buf.to(torch.int32)

    sb_just_fired = sb_flag & ~env._sfs_sb_prev
    ss_just_fired = ss_flag & ~env._sfs_ss_prev
    env._sfs_sb_step[sb_just_fired] = step[sb_just_fired]
    env._sfs_ss_step[ss_just_fired] = step[ss_just_fired]
    env._sfs_sb_prev[:] = sb_flag
    env._sfs_ss_prev[:] = ss_flag

    both_recorded = (env._sfs_sb_step >= 0) & (env._sfs_ss_step >= 0)
    within_window = (env._sfs_sb_step - env._sfs_ss_step).abs() <= window
    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    # NEW 2026-08-15 (user request, "also gate single_foot_save"): same
    # episode-wide sole-contact gate as cleanstop -- see that function's own
    # comment and softstop()'s latching of env._episode_sole_contact.
    sole_contact = getattr(env, "_episode_sole_contact", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = both_recorded & within_window & correct_foot & ~sole_contact & ~env._sfs_flag
    env._sfs_flag |= fired
    return fired.float()


def ball_vx_reduction(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    max_speed: float = 8.0,
) -> torch.Tensor:
    """Reward for stopping the ball's incoming velocity along robot +X axis. Shape (N,).

    vx_local < 0 means ball coming toward robot. Reward peaks when vx_local = 0
    (ball stopped). Clamped to [0, max_speed] before normalisation so the reward
    scale is stable regardless of spawn speed.
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)  # (N, 3)

    vx_local = (ball.data.root_link_lin_vel_w * x_axis_w).sum(dim=-1)  # (N,)
    incoming = (-vx_local).clamp(0.0, max_speed)
    return torch.exp(-(incoming ** 2) / (4.0 ** 2))


def ball_positive_vx(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    target_speed: float = 5.0,
) -> torch.Tensor:
    """Reward ball deflected back along robot local +X axis. Shape (N,).

    Uses robot-local +X (not world +X) to stay consistent with local-frame ball
    spawning: a save means the ball travels back in the direction it came from,
    which is the robot's local +X.
    Returns clamp(vx_local / target_speed, 0, 1) — normalised to [0, 1].
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)
    vx_local = (ball.data.root_link_lin_vel_w * x_axis_w).sum(dim=-1)
    return (vx_local / target_speed).clamp(0.0, 1.0)


def posture(
    env: "ManagerBasedRlEnv",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Gaussian reward for staying near default joint pose. Shape (N,)."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_sq = torch.square(joint_pos - default_pos)
    return torch.exp(-torch.mean(error_sq, dim=1) / (std ** 2))


def ang_vel_xy_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Sum of squared base roll+pitch angular velocity. Shape (N,)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def ang_vel_z_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Squared yaw angular velocity. Shape (N,).

    A goalkeeper should face the field; spinning wastes reach radius and
    delays re-positioning. Penalised separately from roll/pitch so the weight
    can be tuned independently.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])


def stayonline(
    env: "ManagerBasedRlEnv",
    line_offset: float = 0.2,
    max_offset: float = 1.2,
) -> torch.Tensor:
    """Penalty for robot drifting away from the goal line along global X.

    Ball approaches from local +X so the goal line is at X = env_origin_X.
    Returns the X deviation above `line_offset`, clamped to `max_offset`.
    """
    robot: Entity = env.scene["robot"]
    x_local = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    dist = torch.clamp(x_local.abs(), line_offset, max_offset) - line_offset
    return dist


def noretreat(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalty for retreating backward (negative body-frame forward velocity).

    Uses body-frame X so the penalty stays correct when the robot yaws during a dive.
    """
    robot: Entity = env.scene["robot"]
    fwd_vel = robot.data.root_link_lin_vel_b[:, 0]
    return -torch.clamp(fwd_vel, -1.0, 0.0)


def feetorientation(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reward for keeping feet flat (gravity aligned with foot z-axis).

    Flat feet produce better ball deflections during a save.
    """
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    robot: Entity = env.scene[asset_cfg.name]
    feet_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w_exp = gravity_w[:, None, :].expand(-1, 2, -1)
    gravity_foot = quat_apply(quat_inv(feet_quat_w), gravity_w_exp)
    err = torch.sum(gravity_foot[..., :2] ** 2, dim=-1).sum(dim=-1)
    return torch.exp(-sigma * err)


# REMOVED 2026-08-15 (user request): foot_ang_vel_xy (whole-body world-frame
# foot roll+pitch angular velocity, sum across both feet) deleted outright --
# superseded by ankle_pitch_vel/ankle_roll_vel (goalkeeper_env_cfg.py), which
# read each ankle joint's own LOCAL velocity instead of the world-frame body
# rate this function measured. The world-frame read conflated genuine
# ankle-driven rotation with hip/knee-driven leg-swing propagation -- live-
# confirmed via the scripted_yaw motion-sample tooling (this term reacted
# even to a "pure" Hip_Yaw sweep, which is not tilt at all) and previously
# the reason its own -3.0 (6x) attempt on 2026-08-10 had to be reverted
# (killed legitimate reach/dive motion too, see docs/BugFixes.md 2026-08-13).
# Its only unique remaining contribution -- penalizing hip/knee-driven foot
# rotation -- was exactly that failure mode, not a benefit, once the
# joint-local terms cover the genuinely-wanted signal cleanly. See
# docs/BugFixes.md, 2026-08-15.
#
# REMOVED 2026-08-13 (user request): foot_ang_vel_z (foot YAW angular-
# velocity penalty, NEW 2026-08-07, gated post-save-only 2026-08-13 earlier
# this same session) deleted outright -- had no equivalent in
# model_17250.pt's (airbornelatchfix) training code at all. Removed
# alongside foot_ang_vel_xy's weight revert (-3.0 -> -0.25,
# goalkeeper_env_cfg.py) and postleadfootorientation's gate revert (below)
# to isolate whether these reward-config differences, not training
# maturity, explain the leading-foot rotation gap vs baseline. See
# docs/BugFixes.md.


# FIX 2026-08-15 (user request): target changed from dead-vertical (0,0) to a
# 15 deg forward lean. User observed the robot habitually leaning forward
# post-save and judged that stance more stable than the fully upright target
# this reward previously enforced -- baking the lean into the target instead
# of fighting it. Sign derived from this project's own Frame Convention (body
# X = forward, robot faces world +X): grav_b = R_world_to_body . (0,0,-1), a
# forward pitch by theta gives grav_b[:,0] = sin(theta) > 0. Not yet
# render-verified against a live checkpoint -- confirm via sgk_play that the
# resulting lean reads as forward, not backward, before trusting this sign.
_HOME_LEAN_TARGET = math.sin(math.radians(15.0))  # ~0.259


def postorientation(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Posture reward targeting a slight forward lean — always active.

    AMP only sees joint_pos/joint_vel, not root orientation, so it cannot push
    the root toward this target. Gating on ball-is-behind means no signal during
    ball approach (80% of episode), causing the policy to drift into backward lean.
    """
    grav_b = env.scene["robot"].data.projected_gravity_b
    err = (grav_b[:, 0] - _HOME_LEAN_TARGET) ** 2 + grav_b[:, 1] ** 2
    return torch.exp(-3.0 * err)


def postangvel(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Low XY angular velocity reward — active when ball is behind. Mirrors ILB postangvel."""
    behind = _ball_is_behind(env, ball_name)
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b
    err = torch.sum(ang_vel[:, :2] ** 2, dim=1)
    return torch.exp(-3.0 * err) * behind.float()


def postlinvel(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Low forward velocity reward — active when ball is behind. Mirrors ILB postlinvel."""
    behind = _ball_is_behind(env, ball_name)
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b
    err = lin_vel[:, 0] ** 2
    return torch.exp(-3.0 * err) * behind.float()


def deviation_waist_joint(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalty for waist joint deviation from default pose (always active)."""
    robot: Entity = env.scene[asset_cfg.name]
    delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(delta), dim=-1)


def penalize_kneeheight(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_KNEE_CFG,
) -> torch.Tensor:
    """Penalise shank bodies dropping below min_height above floor.

    Detects kneeling/falling states that would damage real hardware.
    Returns sum of excess below threshold across both shanks.

    FIX 2026-07-22: unregistered from goalkeeper_env_cfg.py (along with the
    shank_height termination) -- user found shank height false-positives on
    legitimate deep lunges (research via play.py's new per-episode min-height
    + terminated_by reporting, docs/BugFixes.md same date). Replaced by
    penalize_baseheight + the retuned base_height termination below.
    Function kept (not deleted) in case shank-based gating is revisited.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                # (N,)
    shank_z_local = shank_pos_w[:, :, 2] - floor_z[:, None]             # (N, 2)
    violation = torch.clamp(min_height - shank_z_local, min=0.0)        # (N, 2)
    return violation.sum(dim=-1)


def penalize_baseheight(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.59,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalise root (Trunk) height dropping below min_height above floor.

    FIX 2026-07-22: replaces penalize_kneeheight as the graded warning half
    of this project's fall-prevention pair (paired with the base_height
    termination, goalkeeper_env_cfg.py, both user-tuned from real play
    session data via the min_base/min_Lsh/min_Rsh + terminated_by reporting
    added to play.py this same date). Shank height fired on legitimate deep
    lunges (a real athletic motion for a foot-only save) because a lunge
    inherently brings one knee close to the floor even while balanced;
    root/torso height stays much higher during an intentional lunge and
    only drops when the robot is actually falling, matching the user's own
    read of the play-session data. 0.59 / termination's 0.57 mirrors the
    existing kneeheight(0.295)/shank_height(0.27) pair's ~2.5cm graded-
    warning-before-hard-cutoff gap.

    Mirrors penalize_kneeheight's shape exactly (clamp(min_height - z, 0)),
    just on the single root body instead of two shank bodies.
    """
    robot: Entity = env.scene[asset_cfg.name]
    base_z_w = robot.data.root_link_pos_w[:, 2]           # (N,)
    floor_z = env.scene.env_origins[:, 2]                  # (N,)
    base_z_local = base_z_w - floor_z                       # (N,)
    return torch.clamp(min_height - base_z_local, min=0.0)  # (N,)


def dof_vel_limits(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    soft_factor: float = 0.9,
) -> torch.Tensor:
    """Penalise joint velocities above per-joint T1 motor velocity limits.

    Mirrors G1 _reward_dof_vel_limits exactly (legged_robot.py:1550-1553):
        sum(clip(|dof_vel| - dof_vel_limits * soft_dof_vel_limit, min=0))
    with soft_dof_vel_limit=0.9 (g1_29_config.py:349) -- both the per-joint
    sourcing and the 0.9 safety margin, and the LINEAR (not squared) excess.

    FIX 2026-07-20 (reward audit item 7): previously a single flat 10 rad/s
    cap for every joint regardless of actual per-joint limit, with a squared
    excess kernel -- both diverged from G1. The flat 10 rad/s cap silently
    never fired for T1's slower joints (e.g. arms at ~9.3 rad/s, waist/hip-
    roll/hip-yaw at ~7.3 rad/s -- see _T1_VEL_LIMIT_MAP): a joint already
    past ITS real limit could sit under the flat 10 and be judged compliant
    by this reward. Real per-joint limits sourced from t1_constants.py motor
    specs (ElectricActuator.velocity_limit); the T1 MJCF itself defines no
    joint velocity limit at all (only a position `range`), so there is no
    XML value to use instead -- see docs/BugFixes.md for the full check.
    """
    robot: Entity = env.scene[asset_cfg.name]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]                  # (N, J)

    if not hasattr(env, "_t1_vel_limits"):
        all_names = robot.joint_names
        # Fallback for any unmapped joint (should not occur -- all 21 headless
        # T1 DOF are covered by _T1_VEL_LIMIT_MAP): use the slowest-rated
        # group (waist/hip-roll/hip-yaw) as a conservative default.
        vel_all = torch.full(
            (len(all_names),), WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, device=env.device
        )
        for i, name in enumerate(all_names):
            if name in _T1_VEL_LIMIT_MAP:
                vel_all[i] = _T1_VEL_LIMIT_MAP[name]
        env._t1_vel_limits = vel_all

    vel_limits = env._t1_vel_limits[asset_cfg.joint_ids] * soft_factor   # (J,)
    excess = torch.clamp(vel.abs() - vel_limits, min=0.0)                # (N, J)
    return excess.sum(dim=-1)


def torques_normalized_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ALL_JOINTS_CFG,
) -> torch.Tensor:
    """Penalize torques normalized by per-joint stiffness (kp). Mirrors ILB.

    sum(square(torque / kp)) — dimensionless position-error proxy comparable
    across joints. Without normalization, high-stiffness leg joints (kp=200)
    dominate the penalty over soft arm joints (kp=15).
    """
    robot: Entity = env.scene[asset_cfg.name]
    torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids]

    if not hasattr(env, "_t1_kp_inv"):
        all_names = robot.joint_names
        kp_all = torch.ones(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _T1_KP_MAP:
                kp_all[i] = _T1_KP_MAP[name]
        env._t1_kp_inv = 1.0 / kp_all

    kp_inv = env._t1_kp_inv[asset_cfg.joint_ids]
    return torch.sum(torch.square(torques * kp_inv), dim=-1)


def torque_limits(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ALL_JOINTS_CFG,
    soft_factor: float = 0.95,
) -> torch.Tensor:
    """Penalize actuator torques exceeding per-joint soft torque limit. Mirrors ILB.

    Uses _T1_EFFORT_MAP instead of a universal cap so arms (36 Nm) and
    hip-yaw joints (40 Nm) are penalised at their actual limits.
    """
    robot: Entity = env.scene[asset_cfg.name]
    torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids].abs()

    if not hasattr(env, "_t1_effort_limits"):
        all_names = robot.joint_names
        effort_all = torch.ones(len(all_names), device=env.device) * 50.0
        for i, name in enumerate(all_names):
            if name in _T1_EFFORT_MAP:
                effort_all[i] = _T1_EFFORT_MAP[name]
        env._t1_effort_limits = effort_all

    effort_limits = env._t1_effort_limits[asset_cfg.joint_ids]
    out_of_limit = (torques - effort_limits * soft_factor).clamp(min=0.0)
    return out_of_limit.sum(dim=-1)


def postupperdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _ARM_JOINT_CFG,
    during_scale: float = 1.0,
    kernel_scale: float = 0.15,
) -> torch.Tensor:
    """Reward arm joints returning to default pose. Always active, same strength throughout.

    exp(-kernel_scale * sum_sq_err) x (1.0 if behind else during_scale) -- bounded [0, 1].
    With during_scale=1.0 (current default), the `behind` branch is a no-op --
    this term now pulls toward the target pose at full, constant strength for
    the entire episode, pre- and post-save alike.

    FIX 2026-07-28 (user request, simplification): was `x behind.float()`
    (zero during the approach). User reported arm_torque_limits/
    arm_action_rate_l2/arm_action_acc_l2 (same-day, dynamics-based terms)
    all read ~0 on an episode where the arms were visibly frozen pointing
    backward -- correctly diagnosed as the wrong problem class (those
    penalize movement/torque; a still, wrong-static-pose arm trips none of
    them). This term is the one that's actually supposed to target static
    pose, but gating it to zero for the ~80% of the episode before the save
    meant a badly-drifted arm had already accumulated during approach with
    no pull back until `behind` turned on. Rather than add yet another new
    term, user asked to simplify: keep this one mechanism but make it always
    active, at reduced strength before the save so it doesn't fight
    legitimate counterbalance motion during a dive, full strength (1.0)
    after. Does NOT fix the underlying exp(-err) vanishing-gradient-far-
    from-target issue (still ~0 gradient once err is large) -- this is a
    scope-limited fix per explicit user request, not a claim the mechanism
    is now complete.

    FIX 2026-07-29 (user request): `during_scale` 0.3 -> 0.5, alongside
    fully reverting arm_torque_limits/arm_action_rate_l2/arm_action_acc_l2
    (goalkeeper_env_cfg.py, functions removed from this file). Matched-
    iteration comparison against the pre-2026-07-28 run confirmed those
    three -- and this term's own 0.3 pre-save scale -- were collectively
    over-constraining the arms: footreach/ball_exit/episode-length all
    regressed, and this term's OWN reward collapsed ~10x (1.64->0.16 at
    matched iterations), consistent with "always-on toward a pose, plus
    three separate movement/effort penalties" fighting the arm motion
    needed for dives. User's actual goal was narrower than the full
    penalty stack suggested: prevent the arm ending up in a bad static
    position (e.g. behind the body), not dampen all arm motion generally.
    Keeping only this pose-matching mechanism (raised toward, not all the
    way to, full pre-save strength) targets that directly without the
    movement-agnostic side penalties. arm_dof_vel (goalkeeper_env_cfg.py)
    is kept unchanged -- same movement-penalty family as the three
    reverted terms, but a real G1-matched mechanism (dof_vel) at a small
    weight, judged a minor contributor to the regression, not the driver.

    FIX 2026-07-29 (2nd same-day change, user request): user reported the
    2026-07-29 during_scale fix only seemed to help near-region episodes --
    far-region arm movement still looked "really weird" both during and
    after the save. Root-caused via direct checkpoint replay (--force-region
    probe, both pre-save and post-save error tracked separately): far-region
    dives require a much larger arm excursion than near-region ones, and
    the arm error was NOT recovering after the save either -- measured
    mean sum_sq_err was 0.155 (near, post-save) vs 7.810 (far, post-save),
    a ~50x difference. At kernel_scale's old implicit value of 1.0,
    exp(-1.0*7.81) = 0.0004 -- already fully saturated to the exp(-err)
    kernel's zero-gradient floor, so this term provided essentially NO
    learning signal for far-region post-save recovery, regardless of
    during_scale (which only affects the PRE-save multiplier and can't fix
    a post-save-only saturation). Near-region's much smaller error
    (0.155) sits well within the kernel's sensitive range (exp(-1.0*0.155)
    = 0.856), which is why only near-region episodes looked improved by
    the during_scale change. This is not new/introduced by that change --
    it's the pre-existing "exp(-err) vanishing-gradient-far-from-target"
    limitation this docstring already flagged, just now empirically
    localized to far regions specifically.

    Fix: kernel_scale lowered from an implicit 1.0 to 0.15, flattening the
    exponential's falloff so it retains real, non-vanishing gradient at
    the error magnitudes far-region dives actually produce. Checked against
    the measured near/far error values above: exp(-0.15*0.155)=0.977 (near,
    barely changed from before) vs exp(-0.15*7.81)=0.310 (far, was
    0.0004 -- no longer saturated, real signal restored) at post-save.
    0.15 is an empirical choice balancing "far regions get meaningful
    gradient" against "near regions don't lose their existing fine-grained
    pose discrimination" -- not G1-matched (no G1 equivalent recovery-
    quality kernel exists to size against). Deliberately kept the same
    exp(-k*err) family already used throughout this reward table (postorientation/
    postangvel/postlinvel/postlegdofpos/postwaistdofpos/foot_clearance all
    use this shape) rather than switching to a different kernel family,
    to stay consistent with the codebase's established convention. Only
    postupperdofpos was touched -- postlegdofpos/postwaistdofpos likely
    have the same latent far-region saturation risk (unverified, out of
    scope for this fix, worth a follow-up check). Not yet validated
    against a live training run. See docs/BugFixes.md.

    FIX 2026-07-30: `during_scale` 0.5 -> 0.8 (user request). Hypothesis:
    the post-save arm-pose target is only reachable with real gradient once
    the ball goes behind (scale=1.0) -- if the pre-save arm position drifts
    too far from the target during the ~0.5-strength approach window, the
    post-save error may START from a worse position than the reward's
    exp(-kernel_scale*err) kernel can climb out of in the remaining episode
    time, especially compounding with the far-region saturation this
    function already has a documented history of. Raising the pre-save pull
    closer to full strength (0.8, still short of 1.0 to avoid re-fighting
    legitimate dive counterbalance motion, the exact regression that
    motivated introducing during_scale in the first place) is a direct,
    testable lever on the STARTING error post-save, distinct from
    kernel_scale (which reshapes the reward's sensitivity to a given error,
    not the error itself). Not yet validated against a live training run --
    compare mean post-save postupperdofpos and the raw joint-space error
    (see debugging methodology this session established, e.g.
    `.claude/skills/debugging-mujoco-contact-sensors/probe_template.py`'s
    "load real checkpoint, measure real state" pattern) against this same
    checkpoint lineage once retrained. See docs/BugFixes.md.

    FIX 2026-08-01: `during_scale` 0.8 -> 1.0 (user request) -- explicitly
    to remove the pre-/post-save distinction entirely, so this term pulls
    toward the target pose at the SAME strength for the whole episode,
    not just during the recovery window. This makes the `torch.where(behind,
    1.0, during_scale)` branch a literal no-op (both branches now evaluate
    to 1.0) -- kept as a parameter rather than hardcoded, since during_scale
    has been retuned three times already (0.3->0.5->0.8->1.0) and may be
    lowered again if this reintroduces the original 2026-07-28 regression
    (arms fighting legitimate dive/counterbalance motion, which is why
    during_scale existed below full strength in the first place). Not yet
    validated against a live training run.

    FIX 2026-07-23: weight raised 1.0 -> 5.0 (deliberate G1 divergence, not
    a parity fix -- G1 also uses 1.0, g1_29_config.py:317). User reported
    arm pose still looking "really weird" post-save on both master's last
    checkpoint and the v2 branch; confirmed via training logs this reward
    was stuck near its floor on both (~0.016-0.066 out of a possible max of
    1.0), while postlegdofpos and postwaistdofpos (same weight tier, same
    exp(-err) shape) were noticeably higher. G1's arms are the active,
    catching limb with the whole task structured around them settling into
    a natural rest pose; SGK's arms have no such structural pull in this
    foot-only task, so the same G1-matched weight plausibly isn't enough
    relative pressure here even though it was sufficient for G1's own task.
    Not yet validated against a live run.

    CORRECTION 2026-08-03: the "G1's arms have structural pull from
    catching" explanation above was never actually checked against G1's
    code (no file/line citation accompanied it) and turned out to be wrong
    -- see the FIX 2026-08-03 entry below and `_ARM_JOINT_CFG`'s comment for
    the real, cited difference (G1 never asks this reward to recover
    shoulder at all, avoiding the exact saturation this port's inclusion of
    shoulder was causing). Weight is left at 5.0 unchanged by that fix --
    only the joint scope narrowed -- so this entry's weight history remains
    accurate, just not its causal story.

    FIX 2026-07-27: retargeted from robot.data.default_joint_pos (the
    crouched HOME_KEYFRAME pose) to _POST_SAVE_STANCE_MAP's straight-leg,
    45-deg-arms-out stance -- a bent-knee crouch requires continuous active
    torque to hold, and the policy was not reliably settling into it
    post-save (see docs/superpowers/specs/2026-07-25-post-save-stance-
    design.md). Cache shared with postlegdofpos/postwaistdofpos (see
    env._post_save_stance_target below) -- first time a single lazy cache
    in this file is shared across more than one reward function; the
    hasattr guard makes this safe regardless of which of the three runs
    first each tick. Weight/shape/curriculum unchanged -- only the
    comparison target moved. See docs/BugFixes.md.

    FIX 2026-08-03 (user request, G1-comparison finding): `asset_cfg`'s
    default (`_ARM_JOINT_CFG`) narrowed from 8 joints (shoulder+elbow x2) to
    4 (elbow only x2), matching G1's actual upper-body-recovery joint scope
    exactly -- G1's `_reward_postupperdofpos` (`legged_robot.py:1502-1507`)
    only ever targets `upper_body_joint_indices = cat(elbow_joint_indices,
    wrist_joint_indices)` (`legged_robot.py:1315`, from `g1_29_config.py:
    175,177`) -- shoulder is never in G1's version of this reward. Root-
    caused via live checkpoint replay comparing two sequential training
    runs: post-save arm error was ~50x larger in far regions (7.81) than
    near (0.155) -- shoulder's own large dive-reach excursion dominating the
    shared sum-of-squares -- saturating `exp(-kernel_scale*err)` to near-
    zero regardless of `kernel_scale` tuning, and once a joint pinned at a
    static extreme under that saturated kernel nothing else in the reward
    table pulled it back (`arm_dof_vel`/`action_rate_l2`/`action_acc_l2` all
    penalize MOVEMENT, not position -- a still joint reads ~0 on all of
    them). G1 avoids this by design, never asking this reward to recover
    the one joint that legitimately needs to travel far for a reach/dive.
    See `_ARM_JOINT_CFG`'s own comment and docs/BugFixes.md for the full
    investigation trail (during_scale and the foot-orientation retarget
    were both investigated and ruled out/downgraded as the primary cause
    first). Not yet validated against a live training run.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, "_post_save_stance_target"):
        all_names = robot.joint_names
        stance_all = torch.zeros(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _POST_SAVE_STANCE_MAP:
                stance_all[i] = _POST_SAVE_STANCE_MAP[name]
        env._post_save_stance_target = stance_all

    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - env._post_save_stance_target[asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    scale = torch.where(behind, 1.0, during_scale)
    return torch.exp(-kernel_scale * err) * scale


def postshoulderdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _SHOULDER_JOINT_CFG,
    during_scale: float = 1.0,
    kernel_scale: float = 0.15,
) -> torch.Tensor:
    """Reward shoulder joints returning to default pose. Same exp(-kernel_scale
    * sum_sq_err) x (1.0 if behind else during_scale) shape as `postupperdofpos`,
    but a SEPARATE reward/kernel targeting `_SHOULDER_JOINT_CFG` (Shoulder_Pitch/
    Roll x2 sides) instead of `_ARM_JOINT_CFG` (Elbow_Pitch/Yaw x2 sides).

    NEW 2026-08-06 (user request), root-caused by a live-checkpoint replay of
    `model_12500.pt` (6144_shoulderscopewiring run, force-region probe, left_far/
    right_far/left_near): after the 2026-08-03/08-06 fixes correctly narrowed
    `postupperdofpos` to elbow-only (matching G1's real scope, see that
    function's docstring), NOTHING in the reward table pulls the shoulder
    toward a rest pose at any point in the episode -- `penalize_arm_above_shoulder`
    only penalizes the specific "hand above its own shoulder" geometry, not
    general shoulder deviation; `arm_dof_vel` penalizes shoulder VELOCITY only
    (and is diluted across all 8 arm joints, weight -5e-3); `angular_momentum_
    penalty` is whole-body and gentle (-0.02). Measured directly: far-region
    `shoulder_err` (this function's own error term, unrewarded before this fix)
    climbs monotonically through the post-save window instead of recovering --
    left_far: 0.74 (pre-save) -> 1.76 (early-post, steps 0-20) -> 9.57 (steady-
    post, >20 steps); right_far: 0.79 -> 0.95 -> 2.03. This is a genuine
    unsupervised drift (gets WORSE the longer the episode continues after the
    save), not a slow-convergence artifact -- exactly the kind of gap
    `postupperdofpos`'s own history (this file, FIX 2026-08-03) already
    predicted once shoulder was correctly dropped from that term's scope but
    never given a replacement.

    Why a NEW reward instead of adding shoulder back into `postupperdofpos`'s
    `asset_cfg` (user's explicit request, and the right call): that's exactly
    the state the 2026-08-03 fix moved AWAY from -- shoulder's error magnitude
    (0.7-9.6 in the data above) is roughly an order of magnitude larger than
    elbow's own post-save error family at comparable kernel_scale, so sharing
    one sum-of-squares/one kernel_scale between them re-creates the original
    saturation risk for whichever joint has the smaller natural range (elbow).
    A separate kernel lets each joint group's `kernel_scale` be tuned to its
    own error distribution independently.

    Parameter choices (both first empirical guesses, NOT yet validated against
    a live training run):
    - `kernel_scale=0.15`: reused from `postupperdofpos` rather than re-derived,
      since the measured shoulder-only error range above (0.7 pre-save to 9.6
      steady-post-far) is close to the range `kernel_scale=0.15` was originally
      tuned against (the OLD 8-joint combined error, up to 7.81 far) -- shoulder
      was in fact the dominant contributor to that old combined error, so this
      is a reasonable starting point, not an arbitrary copy.
    - `during_scale=1.0` (FIX 2026-08-07, user request; was 0.3): matches
      `postupperdofpos`'s current, fully-tuned value directly rather than
      re-running its 0.3->0.5->0.8->1.0 ramp-up history separately for
      shoulder. Original reasoning for starting conservative (risk of
      reintroducing the `arm_torque_limits`/`arm_action_rate_l2`/
      `arm_action_acc_l2` over-suppression reverted 2026-07-29) still applies
      in principle, but not yet tested at full strength for shoulder
      specifically -- not yet validated against a live training run.

    Weight 5.0, matching `postupperdofpos`'s tier (same "large-excursion arm
    joint needs an undiluted, single-purpose pose-recovery pull" reasoning).
    Not yet validated against a live training run. See docs/BugFixes.md.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, "_post_save_stance_target"):
        all_names = robot.joint_names
        stance_all = torch.zeros(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _POST_SAVE_STANCE_MAP:
                stance_all[i] = _POST_SAVE_STANCE_MAP[name]
        env._post_save_stance_target = stance_all

    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - env._post_save_stance_target[asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    scale = torch.where(behind, 1.0, during_scale)
    return torch.exp(-kernel_scale * err) * scale


def penalize_arm_above_shoulder(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _ARM_HEIGHT_CFG,
) -> torch.Tensor:
    """Penalty for a hand rising above its own shoulder height, active for the
    entire post-save window (any step `_ball_is_behind` is true).

    NEW 2026-07-30 (user request). `postupperdofpos` already pulls the arms
    toward a fixed target pose, but its `exp(-kernel_scale*err)` shape gives
    no special weight to any one axis of the error -- it doesn't specifically
    discourage the visually worst failure mode the user reported ("arms
    flying above the shoulders"), just the aggregate 8-joint deviation. This
    term targets that specific geometry directly.

    Geometry: compares each hand body's world Z against its OWN shoulder's
    world Z (not a fixed height) -- `AL2`/`AR2` (t1_headless.xml: the
    shoulder-roll link, carries `left/right_shoulder_collision`) and
    `left_hand_link`/`right_hand_link` (elbow-yaw link). Per-side excess =
    clamp(hand_z - shoulder_z, min=0) -- zero unless the hand is literally
    above the shoulder, squared for a smooth gradient. Using each side's OWN
    shoulder (not a robot-frame constant) stays correct through any base
    pitch/roll/height, unlike a fixed world-Z threshold.

    FIX 2026-07-30 (caught live before shipping, per this session's own
    debugging-mujoco-contact-sensors skill -- "never assume order, verify
    against the live object"): `SceneEntityCfg.body_ids` does NOT preserve
    declared `body_names` order -- it resolves in MODEL body-index order.
    Verified live: `_ARM_HEIGHT_CFG`'s declared (AL2, AR2, left_hand_link,
    right_hand_link) resolves as (AL2, left_hand_link, AR2, right_hand_link)
    instead, because the model's kinematic tree fully declares the left-arm
    chain (AL1->AL2->AL3->left_hand_link) before the right-arm chain begins
    -- so AL2 and left_hand_link end up adjacent in index order, ahead of
    AR2. A naive positional 0/1/2/3 index into `asset_cfg.body_ids` would
    have silently paired the LEFT hand against the RIGHT shoulder. Fixed by
    resolving each body's actual position via `robot.find_bodies(...)`'s
    returned (ids, names) pair and looking up by NAME, not position (cached
    on `env._arm_above_shoulder_body_idx` after first call).

    Gating: A live-checkpoint balance-correlation probe this session (see
    docs/BugFixes.md, 2026-07-30) found arm-pose error genuinely correlates
    with body tilt during far-region dives (r=+0.22 to +0.59 across two
    checkpoints) -- i.e. raised arms during an active dive are plausibly
    real balance-recovery motion, not pure bad habit. This term stays gated
    OFF for the dive itself (`~behind`) to avoid repeating the exact mistake
    `arm_torque_limits`/`arm_action_rate_l2`/`arm_action_acc_l2` made (added
    2026-07-28, reverted 2026-07-29 for over-suppressing legitimate dive
    counterbalance motion and regressing real save metrics).

    FIX 2026-08-06 (user request): was additionally gated to only fire once
    `_ball_is_behind` had held for > 20 consecutive steps (the "steady"
    window), via a dedicated `env._arm_above_shoulder_run_len` counter --
    i.e. completely silent for the first ~0.4s after every save. A live
    checkpoint replay of `model_12500.pt` (6144_shoulderscopewiring run)
    found this is exactly the window where the visible "violent arm swing"
    happens: elbow/shoulder joint speed roughly doubles-to-quadruples in
    that first 20 steps versus the dive itself (far-region, right side
    worst: elbow speed mean 4.32 rad/s vs 1.38 rad/s pre-save), and with
    this gate at 20 steps nothing was penalizing "hand above shoulder"
    anywhere in that window. Removed the run-length counter and its
    steady_steps parameter entirely -- now fires for the whole post-save
    window (`behind` alone), matching `postupperdofpos`'s own
    always-full-strength-post-save convention (during_scale=1.0). Not yet
    validated against a live training run -- if this reintroduces fighting
    against genuine post-save balance-recovery motion (the risk the
    original 20-step gate existed to avoid), the gate should come back,
    narrower (e.g. a short window right after `behind` flips, not the full
    steady window this fix removes). See docs/BugFixes.md.

    Weight -2.0 (modest, supplementary -- comparable to `postleadfootorientation`
    at 2.0 in magnitude, not in the same -100 "bad technique" tier as
    `penalize_wrong_foot_ball_contact`/`penalize_self_collision`; this is a
    posture refinement on top of `postupperdofpos`, not a hard technique
    violation). Not yet validated against a live training run. See
    docs/BugFixes.md.
    """
    robot: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, "_arm_above_shoulder_body_idx"):
        ids, names = robot.find_bodies(list(asset_cfg.body_names))
        pos_in_ids = {name: i for i, name in enumerate(names)}
        env._arm_above_shoulder_body_idx = {
            "left_shoulder": pos_in_ids["AL2"],
            "right_shoulder": pos_in_ids["AR2"],
            "left_hand": pos_in_ids["left_hand_link"],
            "right_hand": pos_in_ids["right_hand_link"],
        }
    idx = env._arm_above_shoulder_body_idx

    body_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 4, 3)
    left_shoulder_z = body_pos_w[:, idx["left_shoulder"], 2]
    right_shoulder_z = body_pos_w[:, idx["right_shoulder"], 2]
    left_hand_z = body_pos_w[:, idx["left_hand"], 2]
    right_hand_z = body_pos_w[:, idx["right_hand"], 2]

    left_excess = torch.clamp(left_hand_z - left_shoulder_z, min=0.0)
    right_excess = torch.clamp(right_hand_z - right_shoulder_z, min=0.0)
    excess = torch.square(left_excess) + torch.square(right_excess)

    behind = _ball_is_behind(env, ball_name)

    return excess * behind.float()


def angular_momentum_penalty(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "robot/root_angmom",
) -> torch.Tensor:
    """Penalize whole-body angular momentum -- a physically-adaptive
    alternative/complement to postupperdofpos's fixed-pose arm pull.

    NEW 2026-08-03 (user request), ported near-verbatim from the sibling
    BoosterT1mjlab project (`tasks/velocity/mdp/rewards.py:angular_momentum_
    penalty`, weight -0.02 there), which uses this exact mechanism
    ("Penalize whole-body angular momentum to encourage natural arm swing")
    for its own T1 locomotion policy. Reads MuJoCo's native `subtreeangmom`
    sensor (`<subtreeangmom name="root_angmom" body="Trunk"/>`, already
    present in this project's own t1.xml/t1_headless.xml since they share
    the same robot asset lineage -- confirmed via grep this sensor existed
    in both XMLs already but was never read by any Python code here before
    this fix) -- whole-body angular momentum about the Trunk subtree,
    shape (N, 3).

    Unlike postupperdofpos (pulls arms toward a fixed target pose,
    regardless of whether the body actually needs counterbalancing right
    now -- the mechanism repeatedly found fighting legitimate dive/
    counterbalance motion this session, see that function's FIX 2026-08-03
    entry), this is physically adaptive: a real dive/save naturally
    produces high whole-body angular momentum (that's what counterbalancing
    IS), so this term only meaningfully penalizes once the body is already
    stable and momentum is needlessly nonzero -- it can never tell a
    genuine counterbalance swing from gratuitous flailing on its own, so it
    is intentionally a small, gentle regularizer (matching BoosterT1mjlab's
    own -0.02) layered alongside postupperdofpos, not a replacement for it.

    No G1 equivalent (G1 catches with hands, no equivalent whole-body-
    momentum reward exists in legged_robot.py -- grepped, none found).
    Also grounded in external literature (arXiv 2507.04140, "Learning
    Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent
    RL") which independently arrives at the same technique (a "CAM damping
    reward") for the same underlying problem (natural arm motion during
    dynamic whole-body balance). Not yet validated against a live training
    run. See docs/BugFixes.md.
    """
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    return torch.sum(torch.square(angmom_sensor.data), dim=-1)


def postwaistdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _WAIST_JOINT_CFG_RECOVERY,
) -> torch.Tensor:
    """Reward waist joint returning to default pose after ball is behind. Mirrors ILB.

    exp(-3 * sum_sq_err) × behind — bounded [0, 1], reward peaks at default pose.

    FIX 2026-07-27: retargeted from robot.data.default_joint_pos to
    _POST_SAVE_STANCE_MAP (see postupperdofpos's docstring for the shared-
    cache mechanism and full rationale). The Waist joint's target value is
    unchanged (0.0, matches its existing default_joint_pos) -- this alone
    changes nothing behaviorally, it's purely the unification move so all
    three joint-position post-save terms share one stance definition.

    Weight raised 1.0 -> 3.0 in the same commit (goalkeeper_env_cfg.py):
    user reported the waist visibly rotating post-save; training logs
    confirmed this reward stuck near its floor (~0.31 mean episode reward,
    the currently-running 15k-iteration run), the same "stuck near its
    floor relative to siblings" symptom that motivated postupperdofpos's
    2026-07-23 bump -- that fix's own docstring named postwaistdofpos as
    one of the terms postupperdofpos was compared against at the time
    ("noticeably higher"); that comparison is now stale, postwaistdofpos
    has since fallen to the same floor. Waist (t1_headless.xml) is a pure
    yaw joint (axis="0 0 1", range +/-1.57 rad) -- exactly the joint that
    would visibly present as "the robot rotating" if under-converged. 3.0
    is a deliberate, evidence-based tuning choice (no G1 equivalent to
    size against -- G1 has no comparable single-joint post-save waist term
    at a different weight to port from), below postupperdofpos's 5.0 since
    that term controls 8 joints across 2 limbs vs. this term's 1. See
    docs/BugFixes.md.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, "_post_save_stance_target"):
        all_names = robot.joint_names
        stance_all = torch.zeros(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _POST_SAVE_STANCE_MAP:
                stance_all[i] = _POST_SAVE_STANCE_MAP[name]
        env._post_save_stance_target = stance_all

    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - env._post_save_stance_target[asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-3.0 * err) * behind.float()


def postlegdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _LEG_JOINT_CFG_RECOVERY,
) -> torch.Tensor:
    """Reward leg joints returning to default (standing) pose after ball is behind.

    FIX 2026-07-22: user reported the policy settles into a "weird pose" after
    a save rather than a sensible standing pose. G1's post-save pose recovery
    (postupperdofpos/postwaistdofpos, both above) only covers elbow+wrist
    ("upper_body_joint_indices", legged_robot.py:1315) and waist
    (legged_robot.py:1306) -- G1's own leg_joint_indices (all 12 hip/knee/
    ankle joints, legged_robot.py:1276-1284) have NO post-save reward at
    all. That's not a gap in this port -- it's correct for G1's own task:
    G1 catches with its HANDS, so the catching limb (arms, via elbow+wrist)
    needs an explicit pull back to neutral after use, while the legs are
    mostly just standing/braced throughout and don't drift far from default
    on their own.

    This project's task structurally inverts which limb does the catching:
    SGK saves with its FEET, so the legs are the limb that gets thrown into
    an extreme, off-default configuration during a dive/step/reach -- and
    until this fix, nothing pulled them back afterward, matching the
    reported "weird pose" symptom exactly (only arms/waist had a return-to-
    default incentive; legs, wherever the save left them, had none). This
    mirrors postupperdofpos's exact shape (exp(-1*sum_sq_err), same gentle
    exponent, same _ball_is_behind gate) applied to the role-equivalent
    limb group for this task rather than G1's literal joint list --
    consistent with the top-level CLAUDE.md instruction to check whether a
    G1 decision "was designed for hands and needs adaptation for feet."

    exp(-1 * sum_sq_err) x behind -- bounded [0, 1], reward peaks at default pose.

    FIX 2026-07-27: retargeted from robot.data.default_joint_pos (the
    crouched HOME_KEYFRAME pose) to _POST_SAVE_STANCE_MAP's straight-leg
    stance -- confirmed via training logs this was the single worst-
    converging post-save term (~0.07 mean episode reward vs.
    postupperdofpos's ~2.0 in the currently-running 15k-iteration run),
    consistent with a bent-knee crouch requiring continuous active torque
    to hold. Cache shared with postupperdofpos/postwaistdofpos -- see that
    function's docstring for the shared-cache mechanism. Weight/shape/
    curriculum unchanged -- only the comparison target moved. See
    docs/superpowers/specs/2026-07-25-post-save-stance-design.md and
    docs/BugFixes.md.

    FIX 2026-08-06 (user request): added the same leading-foot airborne
    gate postleadfootorientation uses (its FIX 2026-08-01 entry) -- was
    unconditional `behind` only, so this term (all 12 leg joints, straight-
    leg stance target) pulled even while the leading foot was already
    planted, same "active pull while grounded = slippage against the
    floor" issue postleadfootorientation's own gate was added to prevent.
    Now zero unless the leading (assigned) foot is genuinely airborne,
    using the identical feet_contact/ball_contact exclusion pattern
    (feet_slippage/postleadfootorientation) to tell a lingering foot-ball
    touch apart from real ground contact. No window_steps cap added here
    (unlike postleadfootorientation's 2026-08-03 follow-up) -- not
    requested; this only adds the airborne condition. See docs/BugFixes.md.

    FIX 2026-08-06 (later same day, user request): airborne now sourced from
    `_leading_foot_airborne_latched` instead of a raw per-step contact read
    computed locally -- the raw version could flicker back to "airborne"
    after a bounce/sensor dropout right at touchdown, re-opening this term's
    pull on a foot that had already genuinely landed. See that helper's
    docstring for the full mechanism and `postleadfootorientation`'s FIX
    2026-08-06 entry for the user-observed symptom that caused this.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]

    airborne = _leading_foot_airborne_latched(env, ball_name)

    if not hasattr(env, "_post_save_stance_target"):
        all_names = robot.joint_names
        stance_all = torch.zeros(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _POST_SAVE_STANCE_MAP:
                stance_all[i] = _POST_SAVE_STANCE_MAP[name]
        env._post_save_stance_target = stance_all

    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - env._post_save_stance_target[asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-1.0 * err) * behind.float() * airborne.float()



def penalize_sharpcontact(
    env: "ManagerBasedRlEnv",
    force_threshold: float = 1200.0,
) -> torch.Tensor:
    """Binary penalty when mean foot contact force exceeds force_threshold.

    Mirrors ILB / upstream Humanoid-Goalkeeper _reward_penalize_sharpcontact.
    Geom layout in feet_contact sensor (sorted by name):
        0: left_foot_1, 1: left_foot_2  → left foot
        2: right_foot_1, 3: right_foot_2 → right foot
    Per-foot max over geoms, then mean — matches upstream 2-body mean.
    Weight: -100.0.
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)          # [B, 8]
    left_max  = force_per_geom[:, :4].max(dim=-1).values     # left_foot1-4
    right_max = force_per_geom[:, 4:].max(dim=-1).values     # right_foot1-4
    mean_force = (left_max + right_max) / 2.0
    return (mean_force > force_threshold).float()


def penalize_self_collision(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Binary penalty when any self-collision is detected in the Trunk subtree.

    Reads from the self_collision ContactSensor. data.found shape: [B, 1].
    Weight: -50.0.
    """
    sensor: ContactSensor = env.scene["self_collision"]
    return (sensor.data.found > 0).any(dim=-1).float()


def penalize_wrong_foot_ball_contact(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    knee_proximity_margin: float = 0.05,
) -> torch.Tensor:
    """Binary penalty when: the WRONG (trailing) leg touches the ball with
    ANY part (sole, shin, or knee), OR the CORRECT (leading) leg touches the
    ball with its KNEE specifically (sole = legitimate save, shin = excluded,
    see FIX 2026-07-30 below for why). Chin was tried and reverted the same
    day (see below).

    FIX 2026-07-22: user reported (from reviewing checkpoints around
    iteration ~10000 of green_gradpen10_2026-07-21) that the policy learned
    to catch/stop the ball with its non-leading foot -- i.e. whichever foot
    happens to be nearby, not the task-assigned one (_get_correct_foot_idx,
    based on which side the ball is crossing on) -- and still farm cleanstop/
    softstop/single_foot_save because their existing "correct foot" gate
    uses "feet_contact", a GROUND-contact sensor (see softstop's FIX
    2026-07-15 docstring): the assigned foot merely standing on the ground
    at the same instant satisfies that gate even when a DIFFERENT foot is
    the one actually touching the ball. That gate is intentionally left as
    is (a prior attempt to tighten it via a ball-specific sensor made those
    rewards stop firing almost entirely -- same docstring), so this adds an
    independent, direct penalty instead: uses "ball_contact", the dedicated
    foot-vs-ball ContactSensorCfg (primary=foot geoms, secondary=ball_geom,
    goalkeeper_env_cfg.py) that only fires on genuine foot-ball contact, and
    fires whenever the foot NOT matching _get_correct_foot_idx is in that
    contact set -- independent of whatever the existing correct-foot gates
    believe. Not gated by _ball_is_behind: any wrong-foot ball touch, before
    or after a save, is the behavior being discouraged.

    Geom layout in ball_contact sensor (sorted by name, matches feet_contact):
        0-3: left_foot1-4 -> left foot, 4-7: right_foot1-4 -> right foot.
    Weight: -100.0 (was -30.0).

    FIX 2026-07-23: user reviewed checkpoints from both master's last run
    (green_baseheight_postleg_2026-07-22) and the v2 blue-ball-waypoint run
    and reported the policy still catches the ball with both feet in
    practice. Confirmed via training logs: this reward's logged value is
    essentially identical between the two runs (~-0.020 to -0.022 on
    master, ~-0.020 to -0.021 on v2) -- a persistent, non-trivial firing
    rate on both, not something introduced by the v2 reimplementation.
    The mechanism itself has no bug (re-verified line by line) -- at -30
    the penalty simply isn't outweighing whatever save-quality benefit the
    policy gets from planting both feet. No G1 equivalent exists to check
    parity against (already documented as SGK-only), so raising the
    magnitude is a plain tuning call: -100 puts it in the same severity
    tier as the other "bad technique" penalties (penalize_self_collision
    -50, penalize_sharpcontact/penalize_baseheight -100) rather than the
    much weaker tier it was in before. Not yet validated against a live run.

    FIX 2026-07-28: "ball_contact" above only matches foot[1-4]_collision
    geoms -- it never saw genuine ball contact against the SHIN or KNEE
    (left_shin_collision/right_shin_collision, left/right_knee_collision,
    t1_headless.xml), which have no collision-sensor coverage anywhere in
    this task. User spotted this live via MuJoCo's native contact-point
    overlay (an orange dot between the trailing leg and the ball while this
    reward read 0) and it was confirmed by replaying the real trained
    checkpoint with a probe sensor: the policy makes genuine foot-ball
    contact via the shin routinely, and at least one sampled event was the
    WRONG side's shin touching the ball while this function still returned
    0 -- a real, exploitable blind spot, not just a display artifact (see
    docs/BugFixes.md). Added "leg_ball_contact" (goalkeeper_env_cfg.py,
    same primary/secondary shape as ball_contact) and OR it into the
    wrong-touch check below. Deliberately narrow: only the WRONG side's
    leg counts (matches this function's existing scope -- wrong SIDE, not
    "any non-foot body part"); the correct side's shin/knee touching is not
    penalized here, since that's a save-quality question, not a wrong-foot
    one. Same -100 weight, no new tuning.

    Geom layout in leg_ball_contact sensor (geom-index order, verified via
    primary_names, NOT alphabetical): 0-1: left_shin,left_knee -> left leg,
    2-3: right_knee,right_shin -> right leg. Sub-order within each side
    (shin-then-knee vs knee-then-shin) doesn't matter since only the
    left/right split is used.

    FIX 2026-07-30 (user request, revised same day after two follow-ups):
    widened beyond "wrong SIDE sole only" to also cover:
    (1) WRONG side: the whole leg now counts, not just the sole -- sole OR
    shin OR knee on the trailing/non-assigned side all penalized the same as
    before (the trailing leg touching the ball with ANY body part is bad
    technique, regardless of which part).
    (2) CORRECT/leading side: KNEE ONLY (not shin) is also penalized -- the
    leading foot deflecting the ball with its knee specifically is still bad
    technique (uncontrolled deflection vs a controlled sole/toe save). The
    leading side's SHIN is deliberately left unpenalized: a live-checkpoint
    probe (docs/BugFixes.md) found the leg_ball_contact sensor's "knee/shin"
    label was misleading in practice -- nearly every real firing was
    genuinely the SHIN (confirmed via surface-gap math: ~0.001m at the shin,
    ~0.11m at the knee, i.e. the knee sphere was never actually touching),
    which fires on essentially every save given the shin's position relative
    to a rolling ball's height -- an unavoidable-contact problem in the same
    class as chin (see (3) below), not a useful "bad technique" signal on
    the leading side specifically.
    (3) Chin/head contact was tried and reverted (same day): the chin is
    essentially unavoidable contact given this task's ball trajectories, so
    penalizing it produces an inescapable penalty rather than a useful
    signal. `head_ball_contact` stayed a viewer-only diagnostic sensor
    (goalkeeper_env_cfg.py, play.py), not read by this or any other reward
    -- UNTIL FIX 2026-08-07 below.

    FIX 2026-08-07 (user request, re-added after explicit confirmation):
    chin/head contact re-wired back in. History of this specific flip-flop,
    all same day: (a) added once on a direct-sounding user request without
    first listing the change and waiting for approval -- a violation of
    this project's Change Approval Workflow (see CLAUDE.md), caught by the
    user, reverted; (b) extensive same-day investigation followed (a
    1,280-episode real-checkpoint replay probe, a forced-contact unit test,
    the "HD" console indicator, a P-panel visibility fix) all confirming
    the sensor and detection pipeline work correctly but genuine chin/head-
    ball contact essentially never happens with the real trained policy;
    (c) user then explicitly asked to re-add it anyway, and this time was
    asked to confirm given the direct contradiction with their own earlier
    "we deliberately dont choose the chin because that one doesn't work" --
    confirmed via AskUserQuestion. Given (b), this is expected to have
    near-zero practical effect on training (the event essentially never
    fires) -- it's a defensive close of the exploit path, not a response to
    an active, measured problem.

    FIX 2026-08-07 (same day, immediate follow-up, user request: "make it
    really strict the chin, but keep the threshold of the knee"): switched
    from the raw `head_ball_contact` "found" contact flag to a distance-
    based proximity check instead, mirroring the EXACT fix already proven
    for the leading knee (FIX 2026-07-30 2nd follow-up below) -- "found"
    requires real geometric penetration and under-fires relative to visible
    near-misses (the same problem that motivated the knee's own distance-
    based rewrite). New `head_proximity_margin` param (default 0.20m,
    independent of and NOT reusing `knee_proximity_margin` -- user
    explicitly asked to keep the knee's own threshold unchanged). Head
    sphere: radius 0.08m (`_HEAD_GEOM_RADIUS`, t1_headless.xml
    `head_collision`), center offset (0.01, 0, 0.11) from the H2 body's
    local origin (NOT zero, unlike the knee spheres -- requires rotating
    the local offset into world frame via quat_apply before computing
    distance). Total trigger radius from the offset head-sphere center:
    0.08 + 0.10 (ball) + 0.20 = 0.38m. Unconditional (no side split, single
    head geom), same -100 weight tier as every other sub-condition here.

    FIX 2026-08-07 (same day, immediate follow-up, user request: "increase
    the threshold of the chin again it is not working properly"):
    `head_proximity_margin` 0.20m -> 0.65m (total trigger radius 0.38m ->
    0.83m). At 0.38m the check could never fire in practice -- the
    1,280-episode real-checkpoint replay probe done earlier this session
    (see docs/BugFixes.md) found the CLOSEST the ball ever got to the head
    across every sampled episode was 0.736m, already outside a 0.38m
    trigger radius. 0.65m pushes the total radius (0.83m) past that
    measured closest-approach data point, so the check can now actually
    fire against real observed behavior instead of being geometrically
    unreachable. Still meaningfully smaller than the knee's own diagnostic-
    only 1.0m excursion earlier today (never intended as a real value).

    Net effect: penalty fires on (wrong-side sole OR wrong-side shin OR
    wrong-side knee) OR (leading-side knee only) OR (head/chin proximity,
    distance-based). Same -100 weight except for the leading-knee term (see
    FIX 2026-07-30 2nd follow-up below).

    leg_ball_contact geom-index order (verified via primary_names): 0=left_shin,
    1=left_knee, 2=right_knee, 3=right_shin.

    FIX 2026-07-30 (2nd follow-up, user request): the leading-knee check above
    used the "found" contact flag (`leg_ball_contact`), which requires actual
    geometric overlap within MuJoCo's contact margin -- user reported this
    essentially never fires even when the ball visibly passes right by the
    knee, because contact resolution needs real penetration and the knee
    sphere (r=0.06m) is small relative to a single physics step's ball
    displacement at typical approach speeds. Replaced with a direct
    ball-to-knee distance check instead of the contact sensor: fires whenever
    the ball's surface comes within `knee_proximity_margin` of the knee
    sphere's surface, not just literal contact. Knee sphere center = the
    Shank_Left/Shank_Right body origin (t1_headless.xml: `left/right_knee_
    collision` has no `pos` offset, so it's exactly at the body origin);
    radius 0.06m (`_KNEE_GEOM_RADIUS`). Ball radius 0.10m (`_BALL_GEOM_
    RADIUS`, ball.xml). `knee_proximity_margin` widens the trigger zone
    beyond literal contact -- tunable via this function's param
    (goalkeeper_env_cfg.py). The wrong-side whole-leg check is left on the
    contact sensor (unchanged) -- only the leading-knee case had a
    reported under-firing problem.

    FIX 2026-08-07 (user request): `knee_proximity_margin` 0.05m -> 0.10m
    (total leading-knee trigger radius from the knee sphere's center:
    0.21m -> 0.26m). User reviewed `model_10250.pt` and found the policy
    still using the chin/head to make saves. Chin/head contact itself is
    NOT re-penalized here (that attempt was made and reverted the same
    session per this project's Change Approval Workflow -- see
    docs/BugFixes.md); this widens the EXISTING leading-knee proximity
    penalty instead, as a lower-risk first step toward discouraging
    any-body-part-except-the-sole saves more broadly. First doubling,
    not data-driven -- not yet validated against a live training run.

    FIX 2026-08-07 (2nd same-day increase, user request): `knee_proximity_
    margin` 0.10m -> 0.15m (total trigger radius 0.26m -> 0.31m). Same-day
    follow-up increase, same reasoning as above -- chin/head deliberately
    stays un-penalized (explicit user confirmation this session: "we
    deliberately dont choose the chin because that one doesn't work"), so
    the knee-proximity lever remains the sole tuning knob for this
    exploit-family concern for now. Not data-driven, not yet validated
    against a live training run.

    FIX 2026-08-07 (3rd same-day increase, DIAGNOSTIC-ONLY, user request,
    REVERTED same day): `knee_proximity_margin` 0.15m -> 1.0m (total trigger
    radius 0.31m -> 1.16m), purely so the user could visually confirm the
    "wrong_foot_ball_contact"/"knee_distance_contact" P-panel plots actually
    react in the live viewer. Confirmed working -- at 1.0m the plots
    correctly spiked whenever the correct foot merely approached/overshot
    the footreach aim point (ball broadly nearby, nowhere close to literal
    knee contact) and maxed out right at episode start (ball spawns within
    1.16m of the knee from step one) -- both expected artifacts of the
    oversized radius, not bugs, and both went away once reverted.

    FIX 2026-08-07 (4th same-day change, REVERT, user-confirmed diagnostic
    complete): `knee_proximity_margin` back to 0.15m (total trigger radius
    0.31m) -- the last genuinely-tuned value from earlier today. Restores
    the signal's meaning: fires on genuine near-knee proximity, not "ball
    is somewhere near the robot."

    FIX 2026-08-07 (5th same-day change, user request): `knee_proximity_
    margin` 0.15m -> 0.05m (total trigger radius 0.31m -> 0.21m) -- user
    wanted to watch `model_10250.pt` (run `6144_footyawspinfix_2026-08-07`)
    under the EXACT threshold it was actually trained/checkpointed under.
    Before any change made in this session, `goalkeeper_env_cfg.py` had no
    explicit `knee_proximity_margin` override at all -- the registration
    silently relied on this function's own original default, 0.05m, which
    is therefore the true value in effect for every checkpoint from this
    run including model_10250.pt. This is the original, pre-session value,
    not a new tuning decision.

    FIX 2026-08-07 (6th same-day change, user request: "i only want the
    treshold of the chin so revert back again, and ignore the whole head
    thing, even remove the whole head touch penalty for now"): removed the
    head/chin proximity sub-condition entirely (`head_proximity_margin`
    param, `_HEAD_GEOM_RADIUS`/`_HEAD_LOCAL_OFFSET`/`head_center_w`/
    `dist_to_head`/`head_threshold`/`head_near`, and its OR into the
    return) -- back to exactly the pre-2026-08-07-session form (wrong-side
    sole/leg OR leading-knee proximity only). The chin/head mechanism had
    gone through five threshold changes this session (0.20->0.65->1.0) plus
    a viewer-visualization detour without ever being confirmed working
    against real play; rather than continue tuning a param the user no
    longer wants active, it's removed outright. `knee_proximity_margin`
    (this function's remaining tunable) is unaffected. `_compute_wrong_foot_
    contact_flash` (play.py) and its P-panel/console head-diagnostics were
    reverted the same way in the same change -- see docs/BugFixes.md.

    FIX 2026-08-07 (7th same-day change, user request: "implement it in the
    wrong foot ball contact" -- referring to the new `left_shin`/`right_shin`
    MuJoCo sites added to t1_headless.xml/t1.xml the same day): added a
    leading-shin proximity sub-condition, the same distance-based pattern as
    the existing leading-knee check just above, but against the shin's own
    mid-point (t1_headless.xml's `left_shin`/`right_shin` sites, roughly
    midway between the knee joint and the ankle -- see the XML's own
    comment) instead of the knee joint. Read via `robot.find_sites(...)` +
    `robot.data.site_pos_w` -- the actual compiled site position, not a
    hand-duplicated offset constant (the exact "3-file-duplicated-constant"
    problem the head/chin mechanism had earlier this session). Reuses
    `knee_proximity_margin` rather than adding a new param -- both are the
    same "leading leg proximity" concept, just checked at two different
    points along the leg (knee joint vs mid-shin), and the user's own
    "keep the threshold of the knee" instruction earlier this session
    treated the leg's proximity margin as a single tunable. `_SHIN_GEOM_
    RADIUS` (0.05) matches `left_shin_collision`'s/`left_shin_vis`'s own
    radius in the XML (t1_headless.xml).
    """
    _KNEE_GEOM_RADIUS = 0.06
    _SHIN_GEOM_RADIUS = 0.035  # FIX 2026-08-07: 0.05->0.035, matches left/right_shin_vis's tightened radius (t1_headless.xml)
    _BALL_GEOM_RADIUS = 0.10

    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) 0=left, 1=right
    wrong_foot_idx = 1 - foot_idx
    env_ar = torch.arange(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene["ball_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_sole = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_sole = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    sole_touch = torch.stack([left_sole, right_sole], dim=-1)  # (B, 2)
    wrong_sole_touch = sole_touch[env_ar, wrong_foot_idx]

    leg_sensor: ContactSensor = env.scene["leg_ball_contact"]
    leg_found = leg_sensor.data.found  # [B, 4]: 0=left_shin,1=left_knee,2=right_knee,3=right_shin
    left_leg_touch = (leg_found[:, :2] > 0).any(dim=-1)   # left shin OR knee
    right_leg_touch = (leg_found[:, 2:] > 0).any(dim=-1)  # right knee OR shin
    leg_touch = torch.stack([left_leg_touch, right_leg_touch], dim=-1)  # (B, 2)
    wrong_leg_touch = leg_touch[env_ar, wrong_foot_idx]  # wrong side: whole leg

    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w  # (B, 3)
    robot: Entity = env.scene["robot"]
    shank_ids = robot.find_bodies(["Shank_Left", "Shank_Right"])[0]
    knee_pos_w = robot.data.body_link_pos_w[:, shank_ids, :]  # (B, 2, 3): 0=left,1=right
    dist_to_knee = (ball_pos_w.unsqueeze(1) - knee_pos_w).norm(dim=-1)  # (B, 2)
    knee_threshold = _KNEE_GEOM_RADIUS + _BALL_GEOM_RADIUS + knee_proximity_margin
    knee_near = dist_to_knee < knee_threshold  # (B, 2)
    leading_knee_touch = knee_near[env_ar, foot_idx]  # correct side: knee proximity only

    shin_site_ids = robot.find_sites(["left_shin", "right_shin"])[0]
    shin_pos_w = robot.data.site_pos_w[:, shin_site_ids, :]  # (B, 2, 3): 0=left,1=right
    dist_to_shin = (ball_pos_w.unsqueeze(1) - shin_pos_w).norm(dim=-1)  # (B, 2)
    shin_threshold = _SHIN_GEOM_RADIUS + _BALL_GEOM_RADIUS + knee_proximity_margin
    shin_near = dist_to_shin < shin_threshold  # (B, 2)
    leading_shin_touch = shin_near[env_ar, foot_idx]  # correct side: shin proximity only

    return (wrong_sole_touch | wrong_leg_touch | leading_knee_touch | leading_shin_touch).float()


def airborne_at_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """One-time bonus when softstop fires with correct foot airborne.

    softstop fires unconditionally on ball velocity; this fires on top of it
    as a quality bonus when the robot had the correct foot in the air at the save moment,
    rewarding the committed step/dive motion over a flat-footed shuffle.
    Weight: +15.0.
    """
    if not hasattr(env, "_aas_ss_prev"):
        env._aas_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._aas_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._aas_ss_prev[just_reset] = False
    env._aas_flag[just_reset]    = False

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    fc = env.scene["feet_contact"].data.found                      # (B, 8)
    left_in_contact = (fc[:, :4] > 0).any(dim=-1)
    right_in_contact = (fc[:, 4:] > 0).any(dim=-1)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)

    foot_idx = _get_correct_foot_idx(env, ball_name)
    correct_foot_in_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]
    correct_foot_airborne = ~correct_foot_in_contact

    just_fired = softstop_fired & ~env._aas_ss_prev
    env._aas_ss_prev[:] = softstop_fired

    correct_foot_contact = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = just_fired & correct_foot_airborne & correct_foot_contact & ~env._aas_flag
    env._aas_flag |= fired
    return fired.float()


def inner_face_orientation_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    tolerance_deg: float = 5.0,
) -> torch.Tensor:
    """One-time bonus when softstop fires and the assigned foot is turned sideways
    (block posture), rotated toward the side it's actually saving on.

    Checks that the foot's long axis (toe-heel, local X) world-frame signed yaw is
    within tolerance_deg of _FOOT_TARGET_ANGLE_DEG (the block-posture target angle,
    off forward, mirrored by side) at the save moment.

    FIX 2026-08-22 (user request, "only 5 above or below"): was a cosine-threshold
    check (`cos(target-yaw) > alignment_threshold`), which is symmetric but NOT a
    clean degree window -- alignment_threshold=0.8 meant a ~36.87° cone on EACH side
    of the target, i.e. this fired anywhere in [target-36.87°, target+36.87°], much
    wider than the user expected from watching training. Replaced with a plain
    `abs(signed_progress - target) < tolerance_deg` window -- tolerance_deg=5.0 means
    exactly [target-5°, target+5°], not derived from any cosine/dot-product math.
    signed_progress = expected_sign * yaw_deg (same construction foot_inner_face_
    continuous already uses for its own overshoot check below) converts the
    left/right-mirrored yaw into a single canonical "how far rotated toward the
    block posture" scale, so the same tolerance_deg applies symmetrically to both
    feet without a sign special-case.

    FIX 2026-08-01 (user request): target retargeted from full-sideways (90° off forward,
    parallel to world Y) to 60° off forward (30° off Y) — see _FOOT_TARGET_ANGLE_DEG.
    alignment_threshold left unchanged; it now measures a ~45° cone around the new target
    instead of the old one.

    FIX 2026-08-07 (user request): target further retargeted 60°→45° off forward (see
    _FOOT_TARGET_ANGLE_DEG's own docstring for the yaw-spin root cause this addresses).
    alignment_threshold tightened 0.7→0.85 (cone ~45.6°→~31.8°) IN THE SAME FIX — left
    unchanged, a 45.6° cone around a 45° target would span roughly [-0.6°, 90.6°] off
    forward, i.e. almost the ENTIRE possible foot-rotation range would satisfy this
    check regardless of actual orientation, gutting its discriminative power. The
    tightened cone keeps the check meaningfully selective around the new, shallower
    target.

    FIX 2026-07-10: was `.abs()` on the alignment ("either +Y or -Y direction is a
    valid sideways save"). That reasoning was wrong -- a left-side save needs the
    foot rotated toward +Y, a right-side save needs -Y; they are mirror images, not
    interchangeable. abs() rewarded either direction equally, so the policy had no
    pressure to learn the correct, side-matched rotation -- it could satisfy this
    reward on every save by consistently rotating one way, even when that's
    backwards for half the saves. Confirmed live: a trained checkpoint's right foot
    (right-side saves) learned a genuine ~38 degree rotation toward -Y, while the
    left foot (left-side saves) stayed near-neutral (~3 degrees) -- it never
    discovered the mirror-image +Y rotation, because nothing demanded it
    specifically. See docs/BugFixes.md.

    FIX 2026-07-10 (assigned foot, not geometrically-closest foot): was
    `left_closer = dist[:,0] <= dist[:,1]` -- the GEOMETRICALLY closest foot to the
    ball's live position at the save moment, not the FIXED, task-assigned foot
    (_get_correct_foot_idx, based on which side the ball crossed on). If the
    assigned foot overshoots past the ball, the stationary lagging (wrong) foot can
    become geometrically closer at that instant, so this orientation check would
    silently evaluate the WRONG foot's quaternion -- while `correct_foot` below
    (from _softstop_correct_foot, itself built from _get_correct_foot_idx) still
    correctly requires the ASSIGNED foot to have made contact. That mismatch let
    the assigned foot's genuinely-correct orientation go unrewarded (checking the
    lagging foot instead, usually not correctly oriented -- false negative) or, in
    the reverse case, let a coincidentally-oriented lagging foot fire the reward
    despite the real save foot doing nothing to earn it (false positive). Now uses
    _get_correct_foot_idx directly, consistent with `correct_foot`'s own gate.
    expected_sign is +1 for the left foot (should rotate toward +Y), -1 for the
    right foot (should rotate toward -Y).

    This is orthogonal to feetorientation (which constrains roll/pitch — foot flatness).
    Together they enforce: flat foot AND turned sideways, in the correct direction = correct
    block posture.
    """
    if not hasattr(env, "_ifos_ss_prev"):
        env._ifos_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ifos_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._ifos_ss_prev[just_reset] = False
    env._ifos_flag[just_reset]    = False

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    just_fired = softstop_fired & ~env._ifos_ss_prev
    env._ifos_ss_prev[:] = softstop_fired

    if not just_fired.any():
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]

    # FIX 2026-07-10: was the geometrically-closest foot to the ball's live
    # position (dist[:,0] <= dist[:,1]) -- see docstring. Now the fixed,
    # task-assigned foot, matching correct_foot's own gate below.
    foot_idx = _get_correct_foot_idx(env, ball_name)                       # (N,) 0=left, 1=right
    is_left = foot_idx == 0
    expected_sign = torch.where(is_left, 1.0, -1.0)                        # (N,)

    # FIX 2026-08-16 (user found live via the assigned_foot_angle_deg
    # P-panel): first tried the lateral (local Y) body axis (see
    # foot_inner_face_continuous's matching fix for the full derivation),
    # but ANY body-orientation vector read off the foot link entangles
    # Ankle_Pitch and/or Ankle_Roll -- user directly observed the plotted
    # angle rising from pure Ankle_Pitch tilt with no yaw change at all.
    # Switched to reading Hip_Yaw's JOINT VALUE directly (T1's ankle has no
    # yaw DOF) -- zero Ankle_Pitch/Ankle_Roll contamination by construction.
    #
    # FIX 2026-08-16 (SAME DAY, user found via live viewer comparison
    # against the P-panel): Hip_Yaw alone still undercounted -- user
    # visually saw ~90deg on a left_near save while the plot showed ~40deg.
    # First tried summing Waist + Hip_Yaw (t1_headless.xml: the legs are
    # children of a "Waist" body with its own yaw joint, range +/-90deg,
    # between Trunk and the legs -- completely missed by Hip_Yaw alone).
    # That got closer (~65-67deg live-checked against model_13000.pt) but
    # still undercounted.
    #
    # FIX 2026-08-16 (THIRD revision, same day): root-caused the remaining
    # gap by comparing the joint-sum against the foot's true toe-axis WORLD
    # azimuth (atan2 of the toe/local-X axis's horizontal projection) on
    # the same live save moments -- ~89deg, matching the user's visual read
    # almost exactly, vs. the joint-sum's ~67deg. The extra ~22deg is
    # real and NOT roll (roll is exactly, provably invariant here: rotating
    # a vector about its own axis leaves it unchanged, confirmed live too --
    # Ankle_Roll sits at ~-26deg, its own range limit, on these same saves,
    # yet contributes ~0 to the toe's true direction). It's Ankle_Pitch
    # GIMBAL COUPLING: Ankle_Pitch's own rotation axis is itself downstream
    # of Hip_Yaw/Waist in the kinematic chain, so once the leg is already
    # yawed ~65deg, Ankle_Pitch's "vertical tip" rotation happens about an
    # axis that's ALSO been yawed -- at large existing yaw, pitching is no
    # longer purely vertical, some of it leaks into horizontal-plane
    # direction. This is a real geometric effect from composing two
    # non-parallel rotation axes, not summable from individual joint
    # values. Fix: stopped enumerating joints -- now reads the toe axis's
    # true azimuth directly. Originally set this to ROOT-LOCAL frame
    # (subtracting Trunk's own world yaw via quat_apply_inverse) to
    # preserve the 2026-08-01 "a dive yaw shouldn't earn free credit"
    # design intent for foot_inner_face_continuous.
    #
    # FIX 2026-08-16 (SAME DAY, user request, "i want 80 degrees in global
    # frame... i want to have more control how it saves the ball"):
    # switched to WORLD frame (dropped the quat_apply_inverse step) -- the
    # ball always approaches from a fixed WORLD direction (this project's
    # own Frame Convention, world -X), so "did the foot block at the right
    # angle" is fundamentally a world-frame question, not a robot-local
    # one. This also restores this function's ORIGINAL pre-2026-08-16
    # design (its own docstring always said "world-frame target
    # direction" -- the joint-space/root-local revisions earlier today had
    # silently drifted away from that without it being a deliberate
    # decision for THIS function). Known trade-off, flagged to the user
    # before applying: this project has an extensive history of fighting
    # Hip_Yaw-reaction-torque whole-body yaw spin (postheadingorientation,
    # ang_vel_z, several target-angle revisions) specifically BECAUSE
    # world-frame credit is exploitable that way -- postheadingorientation/
    # ang_vel_z/stayonline still push back against it, just not this term
    # anymore. Provably exact: local-X is invariant to Ankle_Roll
    # regardless of any other joint's value, so this measurement is still
    # immune to roll -- only the world-vs-root-local choice changed. Same
    # target angle, same expected_sign convention, same
    # alignment_threshold. See docs/BugFixes.md, 2026-08-16.
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]     # (N, 2, 4)
    assigned_quat_w = torch.where(is_left[:, None], foot_quat_w[:, 0, :], foot_quat_w[:, 1, :])  # (N, 4)
    x_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    foot_x_w = quat_apply(assigned_quat_w, x_local)                        # (N, 3), toe direction, world
    yaw_deg = torch.rad2deg(torch.atan2(foot_x_w[:, 1], foot_x_w[:, 0]))   # (N,) signed, world frame

    signed_progress_deg = expected_sign * yaw_deg  # (N,) positive = rotated toward the block-posture target
    oriented_correctly = (signed_progress_deg - _FOOT_TARGET_ANGLE_DEG).abs() < tolerance_deg

    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = just_fired & oriented_correctly & correct_foot & ~env._ifos_flag
    env._ifos_flag |= fired
    return fired.float()


def foot_inner_face_continuous(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Continuous reward for rotating the assigned foot's inner face toward the ball,
    in the correct (side-matched) direction.

    Active until `softstop` fires (env._softstop_flag) -- the same event
    `inner_face_orientation_save`'s one-shot bonus itself fires on -- not the
    compound `~behind` gate. Uses the same assigned foot as footreach
    (_get_correct_foot_idx: left=0 for +Y balls, right=1 for -Y balls).

    Metric: foot_long_axis_w · target_w — negative when foot points opposite the
    target, up to 1 when exactly aligned with the target direction (60° off
    robot-forward, 30° off robot-local Y, toward +Y for the left foot / -Y for
    the right foot). Uses robot-local axes (not world) so a dive yaw doesn't
    degrade the signal.

    FIX 2026-07-10: was `.abs()` — same bug as inner_face_orientation_save (see
    that function's docstring for the full analysis and live evidence). This is
    the DENSE, per-step version of the same reward, so it was actually the
    dominant source of the wrong-direction-tolerant gradient (far more total
    signal than the one-shot bonus) -- fixing this one specifically should matter
    most. Deliberately left unclamped (can go negative for a wrong-direction
    rotation, not just zero) -- a mild, informative penalty gives clearer gradient
    toward the correct mirror-image direction than merely withholding reward,
    which would look identical to "never rotated at all" from the policy's
    perspective.

    FIX 2026-08-01 (user request): retargeted from full-sideways (robot-local Y)
    to 60° off robot-forward (30° off Y), matching inner_face_orientation_save's
    new target exactly (_FOOT_TARGET_COS/_FOOT_TARGET_SIN) so the dense per-step
    pull and the one-shot save bonus no longer disagree on where "correct" is.

    FIX 2026-08-05 (user request): the metric above (raw cos(angle-target)) is
    kept for the UNDERSHOOT/wrong-side region, but the OVERSHOOT region (foot
    rotated past the target, on the correct side -- i.e. drifting back toward
    the old 90° value) now uses a steeper foot_clearance-style Gaussian instead
    of cosine's own near-flat tail there. See _FOOT_OVERSHOOT_SIGMA's docstring
    for the live-checkpoint evidence and calibration. The two pieces agree
    exactly at the target (both equal 1.0), so this is a continuous, just not
    smooth, splice.

    FIX 2026-08-06 (user request -- a same-day simplification attempt to a
    binary threshold check was reverted; user wanted ONLY the timing
    changed, not the shaping mechanism): gate changed from `(~behind)` to
    `~softstop_fired` (env._softstop_flag). First pass used `~sb_flag`
    (stopball's own, EARLIER/easier threshold, delta_vx > 1.0 m/s) per a
    literal "on until stopball" reading -- corrected per follow-up user
    request to key off `_softstop_flag` instead, since that's the exact
    event `inner_face_orientation_save`'s one-shot bonus fires on ("i want
    the foot continuous to stop when that one launches"). This creates a
    clean handoff with no gap and no overlap: the dense per-step term shapes
    the angle through the whole approach and up to the precise instant the
    one-shot bonus evaluates and fires, then goes silent, handing judgment
    of that exact moment entirely to the one-shot term. `_ball_is_behind`
    (the original gate) is a compound condition (ball_x<0 OR softstop_flag
    OR sb_flag) that can flip True purely from ball position before any
    real contact, which could have cut this term off before the actual
    save event -- keying off `_softstop_flag` directly avoids that. Reward
    math/shaping itself is unchanged from the 2026-08-05 state.
    """
    robot: Entity = env.scene[asset_cfg.name]

    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) — 0=left, 1=right
    expected_sign = torch.where(foot_idx == 0, 1.0, -1.0)                   # (N,) left=+Y, right=-Y

    # FIX 2026-08-16 (user found live via the assigned_foot_angle_deg
    # P-panel): the 2026-08-16 local-Y fix above (see its own comment)
    # closed the Ankle_Roll blind spot but is STILL a body-orientation
    # read, and the user directly observed the plotted angle rising just
    # from Ankle_Pitch tilt (foot pointing "downwards") with no yaw change
    # at all -- confirmed by that fix's own derivation: local Y is provably
    # NOT invariant to Ankle_Pitch except in the degenerate zero-roll case
    # (Y_world = A @ R_pitch(phi) @ R_roll(theta) @ e_y, not independent of
    # phi once theta != 0). Any body-orientation vector read off the foot
    # link necessarily entangles Ankle_Pitch and/or Ankle_Roll, because
    # both are real rotations between the hip and the foot.
    #
    # Real fix: don't infer yaw from the foot's world orientation at all --
    # read the Hip_Yaw JOINT VALUE directly. T1's ankle has no yaw DOF
    # (t1_headless.xml, confirmed repeatedly elsewhere in this file) --
    # Hip_Yaw (axis "0 0 1" for BOTH sides, not mirrored; range +/-1 rad)
    # is the ONLY LEG joint that changes the foot's heading. A joint-space
    # scalar is, by construction, completely independent of every other
    # joint's value (Ankle_Pitch, Ankle_Roll, Knee, Hip_Pitch, Hip_Roll all
    # downstream/upstream but orthogonal in joint space) -- zero
    # contamination, not just reduced. `default_joint_pos["Hip_Yaw"]` is
    # 0.0 (confirmed via _POST_SAVE_STANCE_MAP's own comment), so the raw
    # joint value IS the delta-from-neutral in radians already, no
    # subtraction needed.
    #
    # FIX 2026-08-16 (SAME DAY, user found via live viewer comparison
    # against the P-panel): Hip_Yaw alone still undercounted -- first tried
    # summing Waist + Hip_Yaw (t1_headless.xml: legs are children of a
    # "Waist" body with its own yaw joint, range +/-90deg, between Trunk
    # and the legs), which got closer (~65-67deg live-checked) but still
    # undercounted vs. the ~90deg visually observed.
    #
    # FIX 2026-08-16 (THIRD revision, same day, user confirmed frame choice
    # via AskUserQuestion): root-caused the remaining gap -- see
    # inner_face_orientation_save's matching fix (above) for the full
    # derivation. The foot's true toe-axis WORLD azimuth read ~89deg on the
    # same live save moments, matching the user's visual read almost
    # exactly; the ~22deg the joint-sum missed is Ankle_Pitch GIMBAL
    # COUPLING (Ankle_Pitch's own rotation axis is downstream of Hip_Yaw/
    # Waist, so once the leg is already yawed, pitching is no longer purely
    # vertical -- confirmed NOT roll: Ankle_Roll sits at its own range
    # limit on these saves yet contributes ~0 to the toe's true direction,
    # exactly as its roll-invariance identity predicts). Stopped enumerating
    # joints -- now reads the toe axis's true azimuth directly. Originally
    # set to ROOT-LOCAL frame (via quat_apply_inverse against
    # root_link_quat_w) to preserve this reward's "robot-local... so a dive
    # yaw doesn't degrade the signal" design intent.
    #
    # FIX 2026-08-16 (SAME DAY, user request, "i want 80 degrees in global
    # frame... i want to have more control how it saves the ball"):
    # switched to WORLD frame (dropped the quat_apply_inverse step),
    # matching inner_face_orientation_save's own same-day fix -- the ball
    # always approaches from a fixed WORLD direction, so the actual
    # blocking angle is fundamentally a world-frame question. Known
    # trade-off flagged to the user before applying: this reopens some
    # exposure to the well-documented Hip_Yaw-reaction-torque whole-body
    # yaw-spin exploit this project has repeatedly fought
    # (postheadingorientation/ang_vel_z/stayonline still push back against
    # it, just not via this term anymore). Still exactly immune to roll
    # (local-X's own invariance to Ankle_Roll is unaffected by the
    # world-vs-root-local choice) -- only which frame the yaw itself is
    # measured in changed. Same target angle, same expected_sign
    # convention, same overshoot shape/constants. See docs/BugFixes.md,
    # 2026-08-16.
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]     # (N, 2, 4)
    assigned_quat_w = torch.where((foot_idx == 0)[:, None], foot_quat_w[:, 0, :], foot_quat_w[:, 1, :])  # (N, 4)
    x_local = torch.zeros(env.num_envs, 3, device=env.device)
    x_local[:, 0] = 1.0
    foot_x_w = quat_apply(assigned_quat_w, x_local)                        # (N, 3), toe direction, world
    yaw_deg = torch.rad2deg(torch.atan2(foot_x_w[:, 1], foot_x_w[:, 0]))   # (N,) signed, world frame

    target_signed_deg = expected_sign * _FOOT_TARGET_ANGLE_DEG              # (N,)

    # FIX 2026-08-16 (user request, then reconsidered same day, "have below
    # 80 degrees better think unbiasedly how to improve it"): a fully
    # symmetric sharp Gaussian (tried first, same day) makes this DENSE,
    # per-step term behave like a near-binary spike -- it stops providing
    # gradient until the foot is already close to target, which defeats
    # the point of a dense shaping signal (the one-shot
    # inner_face_orientation_save already handles "was it correct at the
    # exact save moment" as a threshold check; this term's job is guiding
    # the approach from wherever the policy currently is). Restored the
    # ORIGINAL two-branch asymmetry: undershoot/wrong-side vs. steep
    # Gaussian (_FOOT_OVERSHOOT_SIGMA=0.03, unchanged) ONLY for overshoot
    # past target -- overshoot (drifting back toward the old ~90deg value)
    # has an extensive multi-fix history of being a real, exploited
    # failure mode in this project, undershoot doesn't and should keep
    # climbable gradient rather than a cliff.
    #
    # FIX 2026-08-16 (SAME DAY, follow-up, user request: "the difference
    # between 45 deg and the wanted 80 deg is only 1.3 of reward is this
    # good enough?"): plain cos(err) on the undershoot branch was
    # confirmed too flat -- at this term's max curriculum weight (6.94),
    # 45deg off (35deg short of 80deg target) scored 82% of peak, only a
    # 1.3-point gap. First tried a plain Gaussian (_FOOT_UNDERSHOOT_SIGMA,
    # chosen via AskUserQuestion from gentle/moderate/steep) -- steeper
    # than cosine, but a plain Gaussian is floored at 0, which silently
    # undid the deliberate 2026-07-10 fix above (cos(err) going NEGATIVE
    # for a wrong-direction rotation, distinguishing it from "never
    # rotated at all" -- a Gaussian collapses both cases to ~0, exactly
    # the ambiguity that fix existed to prevent). Caught before shipping,
    # not by the user.
    #
    # Real fix (confirmed via AskUserQuestion): sign-preserving power of
    # cosine, `sign(cos(err)) * |cos(err)|^_FOOT_UNDERSHOOT_POWER` --
    # keeps cos's natural sign (still negative past +/-90deg from target,
    # preserving the 2026-07-10 distinguishing behavior) while raising the
    # magnitude to a power steepens the falloff near the peak. Power=9
    # chosen by matching the approved Gaussian's own curve (45deg-off
    # ~16% of peak) via cos(35deg)^n=0.16 -> n=9.18, rounded to 9 (odd, to
    # keep the sign flip) -- verified numerically within ~0.05 of the
    # Gaussian's own values across the full undershoot range, so this
    # keeps the steepness the user already approved while restoring the
    # wrong-direction penalty. See docs/BugFixes.md, 2026-08-16.
    undershoot_err = target_signed_deg - yaw_deg                            # (N,)
    cos_err = torch.cos(torch.deg2rad(undershoot_err))
    alignment = torch.sign(cos_err) * cos_err.abs() ** _FOOT_UNDERSHOOT_POWER  # (N,) in [-1, 1]

    side_sign = yaw_deg * expected_sign                  # (N,) >0 = correct side
    angle_from_forward_deg = yaw_deg.abs()                 # (N,) magnitude of yaw from neutral

    overshoot_mask = (side_sign > 0.0) & (angle_from_forward_deg > _FOOT_TARGET_ANGLE_DEG)
    overshoot_err = angle_from_forward_deg - _FOOT_TARGET_ANGLE_DEG
    overshoot_reward = torch.exp(-_FOOT_OVERSHOOT_SIGMA * overshoot_err ** 2)

    reward = torch.where(overshoot_mask, overshoot_reward, alignment)      # (N,)

    softstop_fired = getattr(env, "_softstop_flag", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return reward * (~softstop_fired).float()


def trailing_foot_forward_continuous(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Continuous reward for the TRAILING (non-assigned) foot pointing forward.

    NEW 2026-07-27 (user request). foot_inner_face_continuous/
    inner_face_orientation_save both only ever shape the LEADING/assigned
    foot's orientation (they pick one foot via _get_correct_foot_idx and
    never touch the other) -- the trailing foot had no orientation shaping
    anywhere in the reward table at all, which is the likely cause of the
    visually odd trailing-foot orientation reported while watching sgk_play.
    No G1 equivalent exists for this or its leading-foot sibling (checked --
    no hand-orientation-during-catch reward anywhere in legged_robot.py), so
    this is a pure SGK design addition, not a G1-parity change.

    Unlike the leading foot's sideways (robot-local Y) target, the trailing
    foot should stay forward-facing (robot-local +X) -- it isn't presenting
    a blocking face, it's just standing. No left/right sign flip is needed
    here (unlike the Y-axis version): "forward" is the same direction for
    both feet, no mirroring required.

    Metric: trailing_foot_long_axis_w . robot_forward_axis_w, in [-1, 1] --
    1 when the trailing foot's toe points the same way the robot's own body
    faces, negative if pointing backward. Deliberately unclamped (can go
    negative), same rationale as foot_inner_face_continuous: a plain
    zero-floor reward looks identical to "never engaged" as "pointing
    backward" from the policy's perspective, so a real (if mild) penalty for
    the wrong direction gives a clearer gradient.

    Active the ENTIRE episode, no ~behind gate (user's explicit choice) --
    the trailing foot's job (stay planted, forward-facing) doesn't end at
    the save moment the way the leading foot's blocking-face job does; the
    reported symptom was specifically about POST-save orientation.
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)

    foot_idx = _get_correct_foot_idx(env, ball_name)      # (N,) — leading foot, 0=left, 1=right
    trailing_idx = 1 - foot_idx                             # (N,) — the OTHER foot

    foot_long_local = torch.zeros(env.num_envs, 3, device=env.device)
    foot_long_local[:, 0] = 1.0                                              # local X = toe dir
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)        # (N, 3)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)        # (N, 3)
    trailing_long_w = torch.where((trailing_idx == 0)[:, None], left_long_w, right_long_w)  # (N, 3)

    robot_forward_local = torch.zeros(env.num_envs, 3, device=env.device)
    robot_forward_local[:, 0] = 1.0
    robot_forward_w = quat_apply(robot.data.root_link_quat_w, robot_forward_local)  # (N, 3)

    return (trailing_long_w * robot_forward_w).sum(dim=-1)  # (N,) in [-1, 1]


def _leading_foot_airborne_latched(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Sound, LATCHED "is the leading (assigned) foot still airborne since the
    save" flag -- (N,) bool. True from the exact instant `behind` first flips
    (the save moment), stays True while the foot is genuinely airborne, and
    the instant it registers its FIRST real ground contact, latches to False
    PERMANENTLY for the rest of that post-save period -- a later bounce or a
    brief contact-sensor dropout right at touchdown can never flip it back to
    True.

    NEW 2026-08-06 (user request, "spikes twice" bug). Before this fix,
    `postlegdofpos`, `postleadfootorientation`, and `postsave_foot_airtime`
    each independently recomputed `airborne = ~leading_in_contact` FRESH,
    EVERY STEP, straight from the raw `feet_contact` sensor (ball-contact
    excluded). That's unlatched by construction: MuJoCo contact sensors can
    miss a step or two right at the moment of touchdown (documented in the
    `debugging-mujoco-contact-sensors` skill -- "can miss brief/glancing
    touches"), so right after the foot's genuine first landing, `found` can
    briefly read 0 again, flipping `airborne` back to True for a step or two
    -- which the user observed as `postleadfootorientation` visibly firing a
    SECOND burst within the same post-save window, well after the real
    landing. Latching on first contact (rather than re-reading raw contact
    every step) makes a bounce/sensor-dropout physically unable to reopen an
    already-closed airborne period.

    Shared across all three consumers (mirrors `_postsave_airtime_window`'s
    own shared-helper pattern) so there's exactly one state machine computing
    "has the leading foot landed yet," not three independently-duplicated,
    independently-driftable copies of the same sensor logic.

    Re-arms on every `behind` RISING EDGE (a fresh save gets its own fresh
    airborne-until-landed tracking), not just once per episode -- this is a
    different concept from `_postsave_airtime_window`'s intentional
    one-shot-per-episode TIME window (that one exists to bound how long a
    reward can pay out at all; this one exists to correctly detect a single
    physical event -- the first landing -- regardless of how many times it's
    checked). Calling this from multiple reward functions in the same tick is
    safe: the state-mutating parts only react to a `behind` rising edge or a
    genuine contact reading, both idempotent to re-evaluate.
    """
    if not hasattr(env, "_leading_landed"):
        env._leading_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._leading_landed_prev_behind = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    behind = _ball_is_behind(env, ball_name)
    just_became_behind = behind & ~env._leading_landed_prev_behind
    env._leading_landed[just_reset | just_became_behind] = False
    env._leading_landed_prev_behind[:] = behind

    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) leading foot, 0=left, 1=right

    contact_sensor: ContactSensor = env.scene["feet_contact"]
    found = contact_sensor.data.found  # (N, 8)
    left_in_contact  = (found[:, :4] > 0).any(dim=-1)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)

    ball_sensor: ContactSensor = env.scene["ball_contact"]
    ball_found = ball_sensor.data.found  # (N, 8), same primary geom layout as feet_contact
    left_touching_ball  = (ball_found[:, :4] > 0).any(dim=-1)
    right_touching_ball = (ball_found[:, 4:] > 0).any(dim=-1)
    left_in_contact  = left_in_contact & ~left_touching_ball
    right_in_contact = right_in_contact & ~right_touching_ball

    leading_in_contact = torch.where(foot_idx == 0, left_in_contact, right_in_contact)  # (N,)

    # Latch: once genuine ground contact is seen while behind, lock landed
    # permanently until the next fresh save (behind rising edge) or episode
    # reset -- a later bounce/dropout can only OR further True bits in, never
    # clear the latch back to False.
    env._leading_landed |= behind & leading_in_contact

    return behind & ~env._leading_landed


def _postsave_airtime_window(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    window_steps: int = 20,
) -> torch.Tensor:
    """Shared one-shot-per-episode post-save airborne window (bool mask, (N,)).

    FIX 2026-08-03 (user request): factored out of postsave_foot_airtime so
    postleadfootorientation can share the EXACT same window instead of
    running unbounded for the rest of the post-save episode. Root-caused
    this session (docs/BugFixes.md): postleadfootorientation rotates the
    assigned foot back to forward using that leg's Hip_Yaw (T1's ankle has
    no yaw DOF at all -- confirmed in t1_headless.xml -- so Hip_Yaw is the
    ONLY joint that can change foot heading), and that Hip_Yaw rotation
    reaction-torques the trunk into a whole-body yaw drift (confirmed via
    live checkpoint replay: mirror-symmetric by which foot is assigned,
    +11.3deg for left-foot saves vs -5.1deg for right-foot saves at 40
    steps post-save, tracking each leg's own Hip_Yaw unwinding). Letting
    postleadfootorientation keep paying out for the ENTIRE post-save
    episode (its old gate: just `behind & airborne`, no time limit) gives
    the policy no reason to stop applying that reaction torque once the
    rotation is done -- capping it to the same short window
    postsave_foot_airtime already uses bounds how long the disturbance can
    be incentivized, without touching the rotation itself (foot orientation
    quality at the moment of landing is unaffected -- both terms already
    only reward WHILE airborne).

    Calling this from more than one reward function in the same tick is
    safe/idempotent: the state-mutating part (`_psa_window_start`/
    `_psa_used` latch) only reacts to a `behind` RISING EDGE, and
    `_psa_prev_behind` is updated to the current `behind` value on every
    call -- so a second call in the same tick sees `just_became_behind` as
    always False (`_psa_prev_behind` already equals `behind` from the
    first call) and only recomputes the (unchanged) `in_window` result. No
    double-consumption risk regardless of which reward term happens to be
    registered first in goalkeeper_env_cfg.py's RewardManager this tick.
    See postsave_foot_airtime's own docstring for the one-shot-latch
    rationale itself (2026-08-03, wrong-foot-deflection re-trigger
    concern, a separate earlier fix this same day).
    """
    if not hasattr(env, "_psa_prev_behind"):
        env._psa_prev_behind = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._psa_window_start = torch.full((env.num_envs,), -1, dtype=torch.int64, device=env.device)
        env._psa_used = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._psa_prev_behind[just_reset] = False
    env._psa_window_start[just_reset] = -1
    env._psa_used[just_reset] = False

    behind = _ball_is_behind(env, ball_name)
    just_became_behind = behind & ~env._psa_prev_behind & ~env._psa_used
    env._psa_window_start[just_became_behind] = env.episode_length_buf[just_became_behind]
    env._psa_used |= just_became_behind
    env._psa_prev_behind[:] = behind

    has_window = env._psa_window_start >= 0
    return has_window & ((env.episode_length_buf - env._psa_window_start) < window_steps)


def postleadfootorientation(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    window_steps: int = 20,
) -> torch.Tensor:
    """Continuous reward for the LEADING (assigned) foot rotating back toward
    forward (robot-local +X) once the save has happened.

    NEW 2026-07-27 (user request). foot_inner_face_continuous actively wants
    the leading foot turned SIDEWAYS while the save is in progress (gated
    `~behind`) -- this term wants the same foot to rotate back to FORWARD
    once the ball is behind, even while the foot is still airborne
    recovering from an in-the-air save (no ground-contact requirement, no
    settle wait -- continuous per-step, starts the instant `behind` flips).

    These two terms cannot contradict each other: foot_inner_face_continuous
    is gated `(~behind)` and this one is gated `behind` -- exactly
    complementary, so at most one is ever active on a given step for a given
    env. No code change to foot_inner_face_continuous was needed.

    No G1 equivalent (see trailing_foot_forward_continuous's docstring --
    no hand-orientation-during-catch reward exists anywhere in G1). Mirrors
    the same "active only when ball is behind" convention already used by
    postorientation/postangvel/postlinvel/postupperdofpos/postwaistdofpos/
    postlegdofpos, but targets foot HEADING directly rather than a generic
    joint-space default-pose pull -- postlegdofpos's Hip_Yaw term already
    does this indirectly (see the foot-yaw P-panel plots added earlier this
    session) but is a weak, indirect proxy; this targets the actual world-
    frame quantity that matters.

    FIX 2026-08-01 (user request): zero while the assigned foot is grounded.
    Was unconditional through the whole post-save window (airborne AND
    grounded) -- user found it was pulling a PLANTED foot to keep rotating,
    which slips against the ground instead of actually turning (same
    "active rotation while in ground contact = slippage" issue feet_slippage
    already has to guard against). Now only pays out while the foot is
    genuinely airborne, using the same feet_contact/ball_contact pattern
    feet_slippage uses to distinguish ground contact from a lingering
    foot-ball touch (a raw feet_contact reading can't tell the two apart).
    Net effect: rotation is only rewarded during the hangtime the policy
    already has post-save, with no pressure to rotate once planted.

    FIX 2026-08-03 (user request): additionally gated to the same fixed
    `window_steps`-long window `postsave_foot_airtime` uses (via the shared
    `_postsave_airtime_window` helper), not the entire post-save episode.
    Root cause: rotating the assigned foot back to forward requires that
    leg's Hip_Yaw (T1's ankle has no yaw DOF), and that rotation reaction-
    torques the trunk into a whole-body yaw drift -- confirmed via live
    checkpoint replay to be mirror-symmetric by assigned foot side (+11.3deg
    for left-foot saves vs -5.1deg for right, matching each leg's own
    Hip_Yaw unwinding on the same timescale). Capping how long this term
    keeps paying out caps how long the policy is incentivized to keep
    applying that reaction torque, without changing the rotation's target
    or its airborne-only gate. See `_postsave_airtime_window`'s and
    `postsave_foot_airtime`'s docstrings, and docs/BugFixes.md.

    FIX 2026-08-06 (later same day, user request -- "spikes twice" bug):
    airborne was a raw per-step `~leading_in_contact` read, computed fresh
    every tick straight from the `feet_contact` sensor -- unlatched, so a
    contact-sensor dropout or a small bounce right after the foot's genuine
    first landing could flip `airborne` back to True for a step or two
    later in the same `window_steps` window, firing this term a SECOND time
    well after the real landing (confirmed by the user watching the P-panel
    plot). Now sourced from `_leading_foot_airborne_latched`, which latches
    to "landed" permanently on first genuine contact and cannot reopen.

    FIX 2026-08-12 (user request, root-caused this session): gate changed
    from `_ball_is_behind(env, ball_name)` to `env._softstop_flag` directly.
    `foot_inner_face_continuous`'s 2026-08-06 gate change (`~softstop_fired`,
    see that function's docstring) broke the "exactly complementary, at most
    one active" invariant this function's own docstring above still
    (incorrectly) asserts: `_ball_is_behind` is a compound OR of
    `ball_x<0 | softstop_flag | sb_flag`, and `sb_flag` (stopball's earlier,
    easier delta_vx>1.0 threshold) or `ball_x<0` can both flip True well
    before `softstop_flag` (full velocity reversal) does. In that gap --
    confirmed real, not hypothetical, since `_ball_is_behind` is explicitly
    documented to fire on partial deflections `softstop` itself wouldn't --
    this term was pulling the assigned foot back toward forward (0 deg)
    while `foot_inner_face_continuous` was still actively pulling the SAME
    foot toward the save-angle target (was 60/75 deg off forward) on the
    same step: two reward terms fighting over the same foot's heading,
    netting out near forward regardless of the target angle -- the reported
    symptom ("foot just slightly off the x axis") that motivated this fix.
    Keying off `_softstop_flag` directly instead of `_ball_is_behind`
    restores the clean handoff `foot_inner_face_continuous`'s own docstring
    already assumed existed. Side effect: `_postsave_airtime_window`'s
    20-step window still starts counting from `_ball_is_behind`'s (earlier)
    rising edge, not softstop's, so the effective window this term can now
    be active in is shorter by however many steps elapse between the two --
    acceptable (bounds how long the reaction-torque risk this window exists
    to limit can fire, same direction as the window's own purpose), not
    re-widened as part of this fix. Not yet validated against a live
    training run.

    FIX 2026-08-13 (user request): reverted `env._softstop_flag` back to
    `_ball_is_behind(env, ball_name)` -- back to the exact gate active when
    model_17250.pt (airbornelatchfix) trained (this fix postdates that run).
    Reopens the `foot_inner_face_continuous` contradiction the 2026-08-12
    fix above closed -- a deliberate, informed tradeoff: done together with
    foot_ang_vel_xy's weight revert and foot_ang_vel_z's removal (same
    commit) to isolate whether these reward-config differences, not
    training maturity, explain the leading-foot rotation gap vs baseline.
    See docs/BugFixes.md.
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)

    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) — leading foot, 0=left, 1=right

    foot_long_local = torch.zeros(env.num_envs, 3, device=env.device)
    foot_long_local[:, 0] = 1.0
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)        # (N, 3)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)        # (N, 3)
    leading_long_w = torch.where((foot_idx == 0)[:, None], left_long_w, right_long_w)  # (N, 3)

    robot_forward_local = torch.zeros(env.num_envs, 3, device=env.device)
    robot_forward_local[:, 0] = 1.0
    robot_forward_w = quat_apply(robot.data.root_link_quat_w, robot_forward_local)  # (N, 3)

    alignment = (leading_long_w * robot_forward_w).sum(dim=-1)  # (N,) in [-1, 1]

    behind = _ball_is_behind(env, ball_name)
    airborne = _leading_foot_airborne_latched(env, ball_name)
    in_window = _postsave_airtime_window(env, ball_name, window_steps)

    return alignment * behind.float() * airborne.float() * in_window.float()


def postsave_foot_airtime(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    window_steps: int = 20,
) -> torch.Tensor:
    """Ramped bonus for the assigned/leading foot staying airborne, growing
    the longer it stays up, within a short fixed window right after the
    save (starting the instant `behind` first flips true).

    NEW 2026-08-01 (user request). Complements postleadfootorientation's
    airborne-only gate (FIX 2026-08-01, same day): that gate only pays out
    WHILE the foot happens to be airborne, it creates no pressure to delay
    landing in the first place. This term gives that direct, bounded push --
    "stay up a little longer" -- so postleadfootorientation actually has
    hangtime to work with instead of relying on whatever the dive physics
    happens to leave it.

    Time-boxed (window_steps=20, ~0.4s at dt=0.02s) rather than unconditional
    for the rest of the post-save window: foot_clearance's own docstring
    explains why an unbounded post-save airborne reward causes "post-save
    hopping" (it's deactivated via `~behind` for exactly that reason). A
    short fixed window buys real extra rotation time without opening the
    door to indefinite hopping -- once the window elapses this term is zero
    regardless of contact state, even if the foot is still airborne.

    FIX 2026-08-04 (user request): window_steps 10 -> 20 (~0.2s -> ~0.4s) --
    user judged the original window too short.

    FIX 2026-08-04 (2nd same-day change, user request): reshaped from a FLAT
    bonus (1.0 whenever airborne and in-window, regardless of how long
    already airborne) to a LINEAR RAMP that grows with elapsed time since
    the window opened: `(elapsed_steps + 1) / window_steps`, so a foot that
    lands almost immediately scores close to `1/window_steps` (barely
    anything) while a foot that stays up for the whole window scores close
    to 1.0 at the final in-window step. User's stated reasoning: a flat
    bonus gives no gradient rewarding LONGER hangtime specifically (landing
    at step 1 vs step 19 of a 20-step window scored identically before this
    fix) -- the ramp gives the policy a real incentive to keep extending
    airtime within the bounded window, not just to be airborne for a single
    qualifying instant. Peak magnitude unchanged (still tops out at 1.0 x
    weight, same ceiling as the old flat version) -- this reshapes WHERE the
    reward is concentrated within the window, not how much total reward is
    available.

    Ground-contact detection reuses postleadfootorientation's exact
    feet_contact/ball_contact pattern (distinguishes genuine ground contact
    from a lingering foot-ball touch). No G1 equivalent (see
    trailing_foot_forward_continuous's docstring -- no post-catch airborne
    shaping of any kind exists in G1).

    FIX 2026-08-03 (user request): window-open is now a one-shot-per-episode
    latch (env._psa_used), not just a `behind`-rising-edge restart. `behind`
    (_ball_is_behind) is NOT itself latched on its raw `ball_x_local < 0`
    component -- only its _sb_flag/_softstop_flag components are sticky, and
    those require correct_foot_contact. A WRONG-foot touch can knock the
    ball back to x>=0 without setting either flag, flipping `behind` back to
    False; if the ball then crosses behind again, the un-gated version would
    re-open a fresh airborne window with no genuine save behind it -- user
    was concerned this could let the policy learn to hop the foot
    repeatedly to farm the bonus. Now the window can only ever open once per
    episode: the first `behind` rising edge sets _psa_used, and every
    subsequent rising edge is ignored for the rest of that episode.

    FIX 2026-08-03 (2nd same-day change): window computation factored out
    into `_postsave_airtime_window` (now shared with `postleadfootorientation`,
    gated to this same window -- see that function's docstring for why).
    Pure extraction, no behavior change to this function.

    FIX 2026-08-06 (later same day, user request -- found while fixing the
    same bug in postleadfootorientation): airborne was a raw per-step
    `~leading_in_contact` read, unlatched -- a bounce/sensor dropout right
    after the genuine first landing could flip it back to True later in the
    same window, and since `ramp` here depends only on time-since-window-
    opened (not time-since-this-particular-liftoff), that second "airborne"
    reading could score a LARGER ramp value than it deserves -- effectively
    paying out for exactly the "land once, then hop again" pattern this
    term's own time-boxing was designed to discourage (see the foot_clearance
    "post-save hopping" reference above). Now sourced from
    `_leading_foot_airborne_latched`, which cannot reopen once landed.
    """
    in_window = _postsave_airtime_window(env, ball_name, window_steps)
    airborne = _leading_foot_airborne_latched(env, ball_name)

    # FIX 2026-08-04: linear ramp instead of a flat 1.0 -- see docstring.
    # env._psa_window_start is guaranteed set by _postsave_airtime_window
    # above (its hasattr-init block runs unconditionally on first call).
    elapsed = (env.episode_length_buf - env._psa_window_start).clamp(min=0).float()
    ramp = (elapsed + 1.0) / window_steps

    return (in_window & airborne).float() * ramp


def postleadfootplantspeed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    target_speed: float = 0.1,
    sigma: float = 300.0,
) -> torch.Tensor:
    """One-shot reward for the leading (assigned) foot's ground-impact speed
    landing near target_speed (default 0.1 m/s), the instant it first
    touches down during the post-save window.

    NEW 2026-08-07 (user request): user watched training and saw the
    assigned foot slamming down hard at landing rather than settling
    softly. Peaked Gaussian `exp(-sigma*(speed-target_speed)^2)`, same
    shape and same sigma=300 as `foot_clearance`'s established convention
    (that function's own docstring: "error = target_magnitude scores
    ~0.05, half that error scores ~0.47") -- here target_speed IS the
    literal error-normalizing magnitude (0.1 m/s), so the same sigma=300
    reused as-is gives the identical, already-validated calibration:
    landing at 0.0 or 0.2 m/s (error=0.1) scores ~0.05, landing at 0.05 or
    0.15 m/s (error=0.05) scores ~0.47, landing at exactly 0.1 m/s scores
    1.0. Deliberately symmetric/peaked (not a monotonic "slower is always
    better" shape like `cleanstop`'s) -- an unnaturally hovering
    near-zero-speed touch is also not the desired behavior, only a
    genuinely controlled ~0.1 m/s plant is.

    Speed is the full 3D linear velocity magnitude of the leading foot at
    the instant of touchdown (not vertical-only) -- a foot with high
    lateral/forward speed at contact is equally a "hard landing" concern
    for stability, and this mirrors the existing codebase's own
    full-magnitude convention (e.g. blue_ball_landed's assigned_foot_vel).
    Revisit to isolate the vertical (Z) component specifically if live
    evidence later shows horizontal speed dominates the measured impact.

    Fires once per save (re-arms on a fresh `behind` rising edge, mirroring
    `_leading_foot_airborne_latched`'s own re-arm semantics) -- uses its
    OWN independent contact-edge detection (not sourced from that shared
    helper) so this term's firing does not depend on call order relative
    to postlegdofpos/postleadfootorientation/postsave_foot_airtime, which
    may or may not have already run earlier in the same tick.

    No G1 equivalent (no post-catch landing-speed shaping of any kind
    exists in legged_robot.py -- G1 catches with hands, never plants a
    foot as the save effector).
    """
    n = env.num_envs
    if not hasattr(env, "_plant_speed_fired"):
        env._plant_speed_fired = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._plant_prev_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._plant_prev_behind = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._plant_last_tick = torch.full((n,), -1, dtype=torch.int64, device=env.device)

    just_reset = env.episode_length_buf <= 1
    behind = _ball_is_behind(env, ball_name)
    just_became_behind = behind & ~env._plant_prev_behind
    reset_mask = just_reset | just_became_behind
    env._plant_speed_fired[reset_mask] = False
    env._plant_prev_contact[reset_mask] = False
    env._plant_prev_behind[:] = behind

    robot: Entity = env.scene[asset_cfg.name]
    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) 0=left, 1=right
    arange_n = torch.arange(n, device=env.device)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    leading_vel = foot_vel_w[arange_n, foot_idx]                          # (N, 3)
    speed = torch.norm(leading_vel, dim=-1)

    contact_sensor: ContactSensor = env.scene["feet_contact"]
    found = contact_sensor.data.found                                    # (N, 8)
    left_in_contact = (found[:, :4] > 0).any(dim=-1)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)
    ball_sensor: ContactSensor = env.scene["ball_contact"]
    ball_found = ball_sensor.data.found                                  # (N, 8)
    left_touching_ball = (ball_found[:, :4] > 0).any(dim=-1)
    right_touching_ball = (ball_found[:, 4:] > 0).any(dim=-1)
    left_in_contact = left_in_contact & ~left_touching_ball
    right_in_contact = right_in_contact & ~right_touching_ball
    leading_in_contact = torch.where(foot_idx == 0, left_in_contact, right_in_contact)

    is_first_call_this_tick = env.episode_length_buf != env._plant_last_tick
    just_touched = leading_in_contact & ~env._plant_prev_contact & is_first_call_this_tick & behind
    env._plant_prev_contact = torch.where(is_first_call_this_tick, leading_in_contact, env._plant_prev_contact)
    env._plant_last_tick = torch.where(is_first_call_this_tick, env.episode_length_buf.clone(), env._plant_last_tick)

    fire = just_touched & ~env._plant_speed_fired
    env._plant_speed_fired |= fire

    reward = torch.exp(-sigma * (speed - target_speed) ** 2)
    return reward * fire.float()


def postheadingorientation(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    window_steps: int = 20,
) -> torch.Tensor:
    """Reward the robot's root YAW HEADING returning to whatever direction it
    was facing AT THE SAVE MOMENT, for a short window after the save.

    NEW 2026-07-28 (user request): user observed the whole body (not just a
    foot) drifting into a left/right yaw after a save. No existing term
    covers this. `postorientation` (this module) only tracks
    `projected_gravity_b[:, :2]` -- roll/pitch tilt -- which is YAW-INVARIANT
    (a robot standing upright but rotated 90 deg scores identically to one
    facing forward), so it cannot supply this signal. `ang_vel_z`
    (goalkeeper_env_cfg.py) only penalizes yaw ANGULAR VELOCITY, not final
    heading -- a robot that yaws once then holds still pays nothing there.
    `postlegdofpos`'s Hip_Yaw joint-space pull (rewards.py) is a weak,
    indirect proxy (joint angle, not the actual world-frame heading), noted
    as such in its own docstring and CLAUDE.md's divergence table.

    No G1 equivalent (checked -- G1 is a static catch task with hands, it
    never needs to reface after a catch). Reuses the same world-forward-axis
    `quat_apply` construction as `trailing_foot_forward_continuous`/
    `postleadfootorientation` above, but applied to the ROOT quaternion
    instead of a foot quaternion, and shaped as `exp(-k*err)` to match
    postorientation/postangvel/postlinvel's bounded [0,1] style rather than
    those two foot terms' raw [-1,1] dot product.

    FIX 2026-08-06 (user request): the target was a hardcoded world +X --
    i.e. a fixed compass direction the robot was forced to face for the
    ENTIRE rest of the episode after a save, with no time limit (unlike
    every other post-save foot/leg term, which are all either time-windowed
    or airborne-gated). User's complaint: forcing a specific facing
    direction on an otherwise-stationary, already-settled robot (no
    locomotion, no stepping) is physically arbitrary -- there's no reason a
    goalkeeper MUST face exactly world +X once it's done saving and holding
    still, only that it shouldn't have drifted mid-recovery. Fix: instead of
    a fixed world-frame target, capture the root's own forward-facing
    direction AT THE INSTANT `behind` first flips true (the save moment,
    same rising edge `_postsave_airtime_window`/`_leading_foot_airborne_
    latched` key off) and use THAT as the per-episode target -- rewards
    holding the heading it already had at the moment of the save, not
    snapping to an arbitrary global compass direction. Also gated to the
    same shared post-save `window_steps` (default 20,
    `_postsave_airtime_window`) instead of running unbounded for the rest of
    the episode -- once the window elapses (the robot should be settled by
    then), nothing further constrains heading, matching the "unnatural when
    stationary" complaint directly: this term now only fights genuine
    save-induced yaw drift during the recovery window, not indefinitely.
    `postorientation` (roll/pitch, upright) is intentionally UNCHANGED --
    it has no directional/heading component at all (see its own docstring),
    so the "forced to point somewhere" complaint doesn't apply to it.

    exp(-1.5 * 2*(1-alignment)) * behind * in_window -- bounded (0, 1],
    peaks at exactly 1.0 when the root's local +X axis matches its own
    save-moment heading. Not yet validated against a live training run.
    """
    robot: Entity = env.scene[asset_cfg.name]
    forward_local = torch.zeros(env.num_envs, 3, device=env.device)
    forward_local[:, 0] = 1.0
    forward_w = quat_apply(robot.data.root_link_quat_w, forward_local)  # (N, 3)

    if not hasattr(env, "_heading_target_w"):
        env._heading_target_w = forward_w.clone()
        env._heading_prev_behind = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._heading_prev_behind[just_reset] = False

    behind = _ball_is_behind(env, ball_name)
    just_became_behind = behind & ~env._heading_prev_behind
    # Snapshot the trunk's own forward direction at the exact save instant --
    # only meaningfully read once `behind` is true (return is gated on it
    # below), so no reset-on-episode-boundary is needed for the target
    # itself, only for the rising-edge tracker above.
    env._heading_target_w[just_became_behind] = forward_w[just_became_behind]
    env._heading_prev_behind[:] = behind

    alignment = (forward_w * env._heading_target_w).sum(dim=-1)  # (N,) in [-1, 1]

    err = 2.0 * (1.0 - alignment)  # squared-chord-distance analog, >= 0
    in_window = _postsave_airtime_window(env, ball_name, window_steps)
    return torch.exp(-1.5 * err) * behind.float() * in_window.float()


def cleanstop(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    speed_threshold: float = 1.0,
    steepness: float = 2.0,
    window_start: int = 8,   # ticks after softstop fires, ~0.16s @ dt=0.02s
    window_end: int = 12,    # ticks after softstop fires, ~0.24s @ dt=0.02s
) -> torch.Tensor:
    """One-time reward, fires a FIXED delay after softstop, scored by ball speed
    averaged over that window -- can go NEGATIVE for a hard/violent deflection.

    REWORKED 2026-08-29 (user request), replacing the settle-window design
    below (kept in this docstring for history). Root cause: investigating a
    separate "the robot learns to kick the ball" report, a live-checkpoint
    probe (`model_39750`, 1181 tracked contacts) found the ball is often
    still well above speed_threshold (1.0 m/s) for a long time after a
    genuine deflection -- mean total speed 1.84 m/s at the contact instant,
    still 1.73 m/s a full 0.4s later. The OLD design below only fired once
    the ball's speed happened to drop under 1.0 m/s and stay there for 10
    consecutive ticks -- for a hard kick, that could take far longer than
    any reasonable window, or never trigger within an episode at all. When
    it eventually did fire (via ordinary rolling friction, seconds later,
    fully decoupled from how violent the original contact was), it could
    only ever score in [0, 1] -- a violent kick that never settles just
    silently pays 0 and vanishes, giving ZERO gradient against kicking.
    Merged (per user, "why don't we integrate this in cleanstop... i think
    that is the better solution") into a single term rather than adding a
    second sibling reward alongside stopball/softstop/cleanstop/
    contact_yield_velocity: one FIXED-delay measurement now serves both
    "reward a genuinely clean stop" and "penalize a violent one" instead of
    two overlapping mechanisms.

    New design: once `softstop` fires (genuine correct-foot deflection,
    same eligibility gates as before -- correct foot, not a sole-contact
    save), average the ball's total speed over ticks [window_start,
    window_end] after that instant (8-12 ticks @ dt=0.02s = 0.16-0.24s,
    approximating the requested 0.15-0.25s average window under this sim's
    discrete tick size), then fire EXACTLY ONCE using that average --
    unconditional on whether the ball ever actually got slow. Averaging
    (not a single instant sample) absorbs a transient mid-bounce spike/dip,
    serving the same role the old settle-counter served, without the old
    design's open-ended "wait however long it takes" failure mode.

    Payout: `scale = tanh(steepness * (speed_threshold - avg_speed))` -- a
    single continuous function, symmetric around `speed_threshold`: exactly
    0.0 there, approaching +1 for slow/clean stops and -1 for fast/violent
    ones. `steepness=2.0` chosen for a gentle-ish slope near typical clean
    speeds (0.35 m/s -> +0.86, 0.5 m/s -> +0.76) while still reaching
    strongly negative for clearly bad speeds (1.5 m/s -> -0.76, 1.8 m/s ->
    -0.92).

    FIX 2026-08-29 (same day, user correction): the very first version of
    this payout kept the OLD `exp(-decay_rate*(speed-best_speed))` kernel
    and linearly rescaled it to try to reach `[-1,1]` -- but a plain
    exponential is always positive and only decays toward 0 as speed grows,
    so after rescaling it asymptoted around -0.05, NEVER actually reaching
    anywhere near -1 in practice despite the `clamp(-1,1)` being present in
    the code (the clamp just never engaged). Caught by plotting the shape
    and having the user look at it. A first fix tried a two-piece
    (separate positive/negative branches, different steepness each) design,
    which did reach -1 properly but the user asked for "just one function"
    instead -- replaced with the single symmetric `tanh` above, which
    reaches a genuine floor near -1 by construction (`tanh` is bounded)
    without any branching or explicit clamp needed at all.

    Not yet validated against a live training run -- first-guess window
    bounds/floor, same "document now, tune from real training data next"
    convention this project already uses throughout. Watch `cleanstop`'s
    own Episode_Reward for whether it goes meaningfully negative early in
    training (expected -- current checkpoints kick) and trends back toward
    positive as `contact_yield_velocity` (raised 25->50 same day) and this
    term's own new negative gradient both push against it.

    --- HISTORY (previous design, replaced above) ---
    FIX 2026-07-28 (user request): was a hard binary bonus gated on
    `speed < 0.10` -- an all-or-nothing cliff that gave zero gradient
    anywhere in the 0.10-1.0 m/s range a real (if imperfect) deflection
    lands in. Switched to a continuous `clamp(exp(-decay_rate*(speed-
    best_speed)), 0, 1)` payout, `decay_rate=3.75` chosen so
    `exp(-3.75*(1.0-0.2))~=0.05` at the old threshold edge.
    FIX 2026-07-28 (same day): added a settle window (`settle_steps=10`,
    consecutive ticks under `speed_threshold`, leaky-decremented per the
    2026-08-23 fix) so a transient mid-bounce dip wouldn't fire this on a
    ball that was still genuinely rolling fast.
    """
    ball: Entity = env.scene[ball_name]
    ball_speed = ball.data.root_link_lin_vel_w.norm(dim=-1)  # (N,)

    if not hasattr(env, "_cleanstop_flag"):
        env._cleanstop_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cleanstop_prev_softstop = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cleanstop_since = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._cleanstop_speed_sum = torch.zeros(env.num_envs, device=env.device)
        env._cleanstop_speed_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._cleanstop_flag[just_reset] = False
    env._cleanstop_prev_softstop[just_reset] = False
    env._cleanstop_since[just_reset] = -1
    env._cleanstop_speed_sum[just_reset] = 0.0
    env._cleanstop_speed_count[just_reset] = 0

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    # Same "only save with the side of the foot" gate as before (2026-08-15).
    sole_contact = getattr(env, "_episode_sole_contact", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    eligible = correct_foot & ~sole_contact & ~env._cleanstop_flag

    # Arm the fixed-delay window on the RISING EDGE of softstop's flag (this
    # function has no direct access to softstop's own one-shot pulse, so the
    # edge is detected locally against last call's snapshot -- cleanstop is
    # registered after softstop, so softstop_fired is already fresh this tick).
    was_armed = env._cleanstop_since >= 0
    env._cleanstop_since[was_armed] += 1
    just_softstopped = softstop_fired & ~env._cleanstop_prev_softstop & eligible & ~was_armed
    env._cleanstop_since[just_softstopped] = 0
    env._cleanstop_prev_softstop = softstop_fired.clone()

    armed = env._cleanstop_since >= 0
    in_window = armed & (env._cleanstop_since >= window_start) & (env._cleanstop_since <= window_end)
    env._cleanstop_speed_sum[in_window] += ball_speed[in_window]
    env._cleanstop_speed_count[in_window] += 1

    fired = armed & (env._cleanstop_since > window_end) & eligible
    avg_speed = env._cleanstop_speed_sum / env._cleanstop_speed_count.clamp(min=1)

    scale = torch.tanh(steepness * (speed_threshold - avg_speed))

    env._cleanstop_flag |= fired
    env._cleanstop_since[fired] = -1
    return fired.float() * scale


def contact_yield_velocity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    max_credit_speed: float = 0.5,
) -> torch.Tensor:
    """One-shot reward, fires exactly once at the genuine foot-ball contact
    instant, for the leading foot's velocity 'giving' backward along the
    direction NORMAL to (90 deg rotated from) its own orientation target
    (`_FOOT_TARGET_COS`/`_FOOT_TARGET_SIN`, mirrored per assigned side --
    the vector `foot_inner_face_continuous`/`inner_face_orientation_save`
    target for the foot's TOE axis), rather than presenting a rigid,
    stationary block face.

    NEW 2026-08-23 (user request), successor to an earlier CONTINUOUS
    version of this idea that the user explicitly corrected: "the other
    mechanism is wrong it should be a one off reward since you only have
    one contact moment for the booster to save the ball." Physical
    reasoning (user's own): via conservation of momentum, a foot moving
    backward (yielding) at contact absorbs some of the ball's incoming
    momentum instead of rigidly rebounding it, lowering the post-contact
    ball speed -- directly targets `cleanstop`'s own best_speed=0.2 goal,
    which a live checkpoint replay (2026-08-23) found the policy currently
    satisfies only marginally (mean ball speed at `cleanstop`-firing time
    0.79 m/s, mean payout scale 0.13) -- consistent with there being no
    reward anywhere that shapes the CONTACT itself, only the outcome after
    the fact.

    FIX 2026-08-23 (user correction, same day): first version used the
    SAME direction as the foot-orientation target (the toe axis) --
    corrected per explicit user request to use the NORMAL to it instead.
    The toe-axis target `(cos, sign*sin)` points mostly sideways (70 deg
    off forward); its normal points mostly along the robot's forward axis
    -- i.e. roughly toward where the ball is coming from, a physically
    sensible "block-face normal" (the direction the flat blocking surface
    actually faces the incoming ball, as opposed to the toe-axis direction
    the foot is rotated to). Yielding is then "backward" relative to THAT
    normal, which points back toward -X -- the same direction the ball
    itself is already travelling (`Frame Convention`: ball always
    approaches in world -X).

    FIX 2026-08-23 (user report, "the left green yield velocity direction
    is correct but the right one is not"): a FIXED -90 deg rotation
    (`(x,y) -> (y,-x)`) applied to each side's own already-mirrored
    target vector does NOT produce a mirrored pair -- rotation and mirror
    reflection don't commute. Applying `(x,y)->(y,-x)` to `(cos,sign*sin)`
    gives `(sign*sin,-cos)`: X now carries the sign flip instead of Y,
    which flips which LONGITUDINAL direction (forward vs backward) each
    foot's normal points in -- concretely, left ended up pointing mostly
    +X (forward) while right pointed mostly -X (backward), opposite
    senses, not a mirror pair. Fixed by making the rotation ITSELF mirror
    with `sign` (`+90 deg` for one side, `-90 deg` for the other, i.e.
    `-sign*90 deg` uniformly): `target_dir_w = (sin, -sign*cos)`. Verified
    algebraically: `dot(original, rotated) = cos*sin - sign^2*sin*cos = 0`
    for both signs (still perpendicular), `|rotated|^2 = sin^2+sign^2*cos^2
    = 1` (still unit length), and X (`sin`) is now sign-independent while
    Y (`-sign*cos`) carries the flip -- the same mirror structure the
    original toe-axis target itself has, this time correctly preserved
    through the rotation.

    Gated on `env._sb_deflection_now` -- `stopball`'s own raw per-tick
    "is a genuine single-contact deflection happening right now" flag
    (already correct-foot- and landing-gated, same event `softstop` keys
    off) -- the correct, already-proven instant to sample foot velocity at,
    reused rather than inventing a second contact-detection mechanism.

    `yield_component = -dot(foot_vel_w, target_dir_w)`: positive when the
    foot moves opposite to the block-face normal (retreating/yielding),
    negative when it moves INTO that normal instead (bracing/ramming
    rather than yielding). Scaled linearly by `max_credit_speed` (a plain
    ratio, not a peaked/exp kernel -- this is a "more yield is better, up
    to a point" quantity, not a target value to hit exactly, so a simple
    linear scale gives gradient across the whole useful range rather than
    concentrating it around one point) and clamped to `[-1, 1]`.

    FIX 2026-08-23 (user request, "make the reward negative if it is
    opposite the velocity because i only see a constant 0 reward"): was
    clamped to `[0, max_credit_speed]` -- a non-yielding contact scored
    exactly 0, indistinguishable from "no contact happened at all" from
    the policy's perspective, giving no gradient AWAY from the wrong
    behavior. Same reasoning already applied in this file to
    `foot_inner_face_continuous` (see that function's own docstring: "a
    mild, informative penalty gives clearer gradient... than merely
    withholding reward, which would look identical to 'never rotated at
    all'"). Now symmetric: `[-max_credit_speed, max_credit_speed]`,
    normalized to `[-1, 1]`.

    `max_credit_speed=0.5` m/s (user-set, was 1.0) is still a first guess
    for the ceiling itself, not yet calibrated against a measured
    foot-velocity-at-contact distribution; not yet validated against a
    live training run.
    """
    if not hasattr(env, "_contact_yield_flag"):
        env._contact_yield_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._contact_yield_flag[just_reset] = False

    deflection_now = getattr(
        env, "_sb_deflection_now", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    fired = deflection_now & ~env._contact_yield_flag
    env._contact_yield_flag |= fired

    robot: Entity = env.scene[asset_cfg.name]
    foot_idx = _get_correct_foot_idx(env, ball_name)                        # (N,) 0=left, 1=right
    expected_sign = torch.where(foot_idx == 0, 1.0, -1.0)                   # (N,) left=+Y, right=-Y

    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
    arange = torch.arange(env.num_envs, device=env.device)
    assigned_vel_w = foot_vel_w[arange, foot_idx]                           # (N, 3)

    # Normal to the foot-orientation target vector, rotation itself mirrored
    # by `sign` so left/right stay a proper mirror pair -- see FIX comment
    # above (a fixed-direction rotation does NOT commute with the mirror).
    target_dir_w = torch.zeros_like(assigned_vel_w)
    target_dir_w[:, 0] = _FOOT_TARGET_SIN
    target_dir_w[:, 1] = -expected_sign * _FOOT_TARGET_COS

    yield_component = -torch.sum(assigned_vel_w * target_dir_w, dim=-1)     # (N,)
    reward = torch.clamp(yield_component, min=-max_credit_speed, max=max_credit_speed) / max_credit_speed

    return fired.float() * reward


def foot_clearance(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    target_height: float = 0.10,
    clearance_sigma: float = 300.0,
) -> torch.Tensor:
    """Reward for lifting a foot to (not past) a target height during ball approach.

    Deactivates once the ball is behind to avoid rewarding post-save hopping.

    The robot sometimes shuffles without lifting feet (keeps both grounded), producing
    a slide rather than a committed step or dive. This reward creates a gradient for any
    foot-lift, making stepping/diving strictly better than shuffling.
    Weight: +2.0.

    FIX 2026-07-27 (user request): was a linear ramp clamped at target_height
    (10 cm) -- 10cm, 20cm, and 1m all scored the identical 1.0 ceiling, so
    nothing discouraged lifting the foot far higher than needed. Now a smooth
    bump `exp(-clearance_sigma * (height - target_height)^2)`, peaking at
    exactly 1.0 at target_height and decaying symmetrically on both sides --
    matches this reward table's existing exp(-k*err) kernel convention
    (feetorientation, footreach's phase1, etc.) instead of a hard clamp.
    clearance_sigma=300 chosen so the shape tracks the old linear ramp
    closely below the peak (h=0 -> ~0.05, h=0.05 -> ~0.47, close to the old
    ramp's exact 0.0/0.5) while now also decaying above 10cm instead of
    plateauing (h=0.15 -> ~0.47, h=0.20 -> ~0.05, mirroring the low side).
    No G1 equivalent exists for this term at all (checked -- no
    `_reward_feet_clearance`-style function anywhere in legged_robot.py), so
    this remains a pure SGK design choice, not a G1-parity change. Not yet
    validated against a live training run.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                     # (N,)
    foot_z_above_floor = (foot_pos_w[:, :, 2] - floor_z[:, None]).clamp(0.0, None)  # (N, 2)
    max_foot_height = foot_z_above_floor.max(dim=-1).values                   # (N,)
    reward = torch.exp(-clearance_sigma * (max_foot_height - target_height) ** 2)
    return reward * (~behind).float()


def trailing_foot_lift(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    target_height: float = 0.10,
    clearance_sigma: float = 300.0,
) -> torch.Tensor:
    """Reward for lifting the TRAILING (non-assigned) foot during the
    blue->orange and orange->red waypoint journey.

    NEW 2026-08-17 (user request): "an incentive that raises the foot" while
    the trailing foot travels start->orange and orange->red. `foot_clearance`
    already rewards lifting SOME foot to target_height, but takes the max
    across both feet -- fully satisfiable by the LEADING foot alone, leaving
    the trailing foot with no lift incentive of its own during this specific
    journey (mirrors the gap `orange_foot_proximity` closed for trailing-foot
    *position* on 2026-08-08, this time for height).

    Same Gaussian-bump shape/constants as foot_clearance (target_height=0.10,
    clearance_sigma=300) -- consistent kernel convention across the reward
    table, scoped to one foot instead of max(both).

    Active window: `env._orange_wide & ~env._red_landed_genuine` -- a single
    gate spanning BOTH requested spans (start->orange, i.e. before orange
    lands, AND orange->red, i.e. after orange lands but before red does),
    since `env._red_active` (gating red's own terms) only ever turns true
    once orange has already landed genuinely -- there is no gap between the
    two spans to leave uncovered. `env._orange_wide` (== `env._blue_wide`)
    keeps this zero on narrow crossings, where orange/red don't exist.
    Deliberately NOT gated on `~behind` like `foot_clearance` -- the
    orange->red leg of this journey routinely continues past the save.

    No G1 equivalent (same justification class as the orange/red waypoint
    terms -- G1 has no intermediate-waypoint concept for either foot).
    `_get_orange_reach_target_y`/`_get_red_reach_target_y` called explicitly
    (values discarded) purely to guarantee `env._orange_wide`/
    `env._red_landed_genuine` are fresh this tick regardless of registration
    order -- same freshness pattern `trailing_foot_reach` uses above. Not
    yet validated against a live training run.
    """
    _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    _get_red_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    robot: Entity = env.scene[asset_cfg.name]
    foot_idx = _get_correct_foot_idx(env, ball_name)      # (N,) — leading foot, 0=left, 1=right
    trailing_idx = 1 - foot_idx                             # (N,) — the OTHER foot

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]       # (N, 2, 3)
    trailing_z_w = torch.where(trailing_idx == 0, foot_pos_w[:, 0, 2], foot_pos_w[:, 1, 2])  # (N,)
    floor_z = env.scene.env_origins[:, 2]                                    # (N,)
    trailing_z = (trailing_z_w - floor_z).clamp(0.0, None)                   # (N,)

    reward = torch.exp(-clearance_sigma * (trailing_z - target_height) ** 2)

    active = env._orange_wide & ~env._red_landed_genuine
    return reward * active.float()


def clearance_at_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    target_height: float = 0.05,
    clearance_sigma: float = 300.0,
) -> torch.Tensor:
    """Continuous reward for the LEADING foot hovering near target_height
    during the blue-landed -> real-save window.

    NEW 2026-08-23 (user request), successor to the removed (2026-06-29)
    `airborne_at_save` -- that term was a one-shot BINARY bonus (airborne
    y/n) fired only on the exact softstop tick. This is deliberately
    different on both axes: continuous (graded toward a real height target,
    same Gaussian-bump kernel as foot_clearance/trailing_foot_lift) and
    windowed rather than instantaneous -- active for every step from the
    leading foot's genuine blue landing (`env._blue_landed_genuine`, already
    excludes cheap/free landings -- see `_get_reach_target_y`) through the
    real save (`_ball_is_behind`), not just the single save tick. User
    request: "in between blue ball and green ball save" -- "green ball" is
    this codebase's own term for the true/final target, as opposed to the
    blue/orange/red intermediate waypoint markers (see e.g.
    `_get_reach_target_y`'s docstring).

    Unlike foot_clearance (max over both feet) or trailing_foot_lift (the
    trailing foot), this scopes to the LEADING/assigned foot specifically --
    the one that will make contact -- via `_get_correct_foot_idx`, mirroring
    trailing_foot_lift's single-foot selection pattern.

    `_get_reach_target_y` is called explicitly here (return value discarded)
    purely to guarantee `env._blue_landed_genuine` is fresh this tick
    regardless of registration order -- same freshness pattern
    trailing_foot_lift already uses for `env._orange_wide`/
    `env._red_landed_genuine`.

    Not yet validated against a live training run.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    robot: Entity = env.scene[asset_cfg.name]
    foot_idx = _get_correct_foot_idx(env, ball_name)      # (N,) — leading foot, 0=left, 1=right

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    leading_z_w = torch.where(foot_idx == 0, foot_pos_w[:, 0, 2], foot_pos_w[:, 1, 2])  # (N,)
    floor_z = env.scene.env_origins[:, 2]                                     # (N,)
    leading_z = (leading_z_w - floor_z).clamp(0.0, None)                      # (N,)

    reward = torch.exp(-clearance_sigma * (leading_z - target_height) ** 2)

    behind = _ball_is_behind(env, ball_name)
    active = env._blue_landed_genuine & ~behind
    return reward * active.float()


def feet_slippage(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward feet not slipping while in ground contact.

    Mirrors upstream _reward_feet_slippage: exp(-10 * sum(foot_speed * in_contact)).
    Returns 1.0 when airborne or no slip; approaches 0 with high slip velocity.
    Geom layout in feet_contact sensor (sorted by name):
        0: left_foot_1, 1: left_foot_2 → left foot
        2: right_foot_1, 3: right_foot_2 → right foot

    Suppressed only for the correct foot when the ball is within 0.5 m of it: the
    contact sensor cannot distinguish ground from ball, so without this gate
    the policy learns to plant feet passively rather than sweep into the ball,
    preventing the force needed for stopball (delta_vx > 1 m/s).
    The trailing foot is NOT gated — it must remain non-sliding throughout.

    FIX 2026-07-27: "feet_contact" has secondary=None (goalkeeper_env_cfg.py:107),
    so it fires on contact with ANYTHING, not just the ground -- G1's identical
    "any contact" assumption (legged_robot.py:1469-1473) is harmless there
    because G1 catches with its hands, so feet never touch the ball. Here feet
    ARE the ball-contact effector, so a genuine foot-ball impact was being
    counted as ground slippage -- confirmed live (docs/BugFixes.md 2026-07-23
    entry: feet_slippage crashing 2.65->0.13 in sync with a hard-impact
    foot_ang_vel_xy spike, i.e. real contact, not sliding) and via sgk_play's
    P-panel plots. The suppress[] block below only ever covered the
    assigned/correct foot within 0.5m; the trailing foot got no such
    protection, so it could be wrongly penalized just for touching the ball.
    Now excludes genuine foot-ball contact for BOTH feet, using the
    "ball_contact" sensor already built for exactly this foot-vs-ball
    distinction (see penalize_wrong_foot_ball_contact). Known limitation
    (2026-07-15 finding on stopball/softstop, same sensor): MuJoCo's contact
    detection window for a small rolling ball can miss some frames of a
    brief/glancing touch, so this reduces rather than guarantees elimination
    of ball-contact false positives -- acceptable here since, unlike
    stopball/softstop, this reward doesn't need to catch one precise step,
    only reduce false penalization across a contact's duration.

    Sliding is measured as the worst-case horizontal speed across two contact
    points per foot: the center-bottom and the toe tip.  Rigid-body kinematics:
        v_contact = v_body + ω × r_contact
    Center-bottom: r = (0, 0, -0.03) in foot-local (capsule z=-0.01 + radius 0.02).
    Toe-tip: r = (0.105, 0.01, -0.03) in foot-local (foot4 forward tip + radius).
    When the trailing foot tilts toe-down, the center-bottom sample underestimates
    the actual toe sliding velocity; the toe-tip sample catches it.
    Weight: +3.0.
    """
    _CONTACT_Z_LOCAL = 0.03  # capsule z (-0.01) + radius (0.02)
    # foot4 extends to x=+0.105, y=+0.01 in foot-local frame
    _TOE_X_LOCAL = 0.105
    _TOE_Y_LOCAL = 0.01

    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]
    left_in_contact  = (found[:, :4] > 0).any(dim=-1)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)

    # Exclude genuine foot-ball contact (both feet) -- see FIX 2026-07-27 above.
    ball_sensor: ContactSensor = env.scene["ball_contact"]
    ball_found = ball_sensor.data.found  # [B, 8], same primary geom layout as feet_contact
    left_touching_ball  = (ball_found[:, :4] > 0).any(dim=-1)
    right_touching_ball = (ball_found[:, 4:] > 0).any(dim=-1)
    left_in_contact  = left_in_contact & ~left_touching_ball
    right_in_contact = right_in_contact & ~right_touching_ball

    in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1).float()  # [B, 2]

    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w   = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]   # [B, 2, 3]
    foot_omega_w = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]   # [B, 2, 3]
    foot_quat_w  = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]      # [B, 2, 4]

    def _contact_speed(r_local_offset: torch.Tensor) -> torch.Tensor:
        """Horizontal speed at a contact point given foot-local offset [B,2,3]."""
        r_world = quat_apply(
            foot_quat_w.reshape(-1, 4),
            r_local_offset.reshape(-1, 3),
        ).reshape(env.num_envs, 2, 3)
        v = foot_vel_w + torch.cross(foot_omega_w, r_world, dim=-1)
        return torch.norm(v[:, :, :2], dim=-1)  # [B, 2]

    # Center-bottom contact point
    r_center = torch.zeros_like(foot_vel_w)
    r_center[:, :, 2] = -_CONTACT_Z_LOCAL
    speed_center = _contact_speed(r_center)

    # Toe-tip contact point (left foot4 / right foot3 end, worst case for tilted foot).
    # Left foot (idx 0): foot4 tip at y=+0.01; right foot (idx 1): foot3 tip at y=-0.01.
    r_toe = torch.zeros_like(foot_vel_w)
    r_toe[:, :, 0] = _TOE_X_LOCAL
    r_toe[:, 0, 1] = _TOE_Y_LOCAL    # left foot
    r_toe[:, 1, 1] = -_TOE_Y_LOCAL   # right foot (mirrored)
    r_toe[:, :, 2] = -_CONTACT_Z_LOCAL
    speed_toe = _contact_speed(r_toe)

    # Take the worst-case (highest) sliding speed across the two sample points
    foot_speed = torch.maximum(speed_center, speed_toe)                      # [B, 2]

    ball: Entity = env.scene[ball_name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # [B, 2, 3]
    ball_pos_w = ball.data.root_link_pos_w                                   # [B, 3]

    # Suppress slippage penalty for the correct foot only when ball is near it.
    # The trailing foot keeps its full penalty so it cannot drag freely.
    foot_dist = torch.norm(foot_pos_w - ball_pos_w[:, None, :], dim=-1)    # [B, 2]
    foot_idx = _get_correct_foot_idx(env, ball_name)                        # (N,)
    arange = torch.arange(env.num_envs, device=env.device)
    correct_foot_near_ball = foot_dist[arange, foot_idx] < 0.5             # (N,)
    suppress = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    suppress[arange, foot_idx] = correct_foot_near_ball

    contactvel_per_foot = foot_speed * in_contact                           # [B, 2]
    contactvel_per_foot = contactvel_per_foot.masked_fill(suppress, 0.0)
    contactvel = contactvel_per_foot.sum(dim=-1)
    # FIX 2026-07-20 (reward audit item 6): was -50.0, 5x steeper than G1's
    # exp(-10*contactvel) (legged_robot.py:1473) -- the docstring above already
    # claimed to mirror G1's -10 kernel, so this was a straightforward
    # coefficient bug (doc/code mismatch), not a documented divergence. No
    # comment or commit anywhere justified -50 -- it has been -50.0 since this
    # function's first commit (c2b8b69), i.e. a porting error, not later drift.
    # Reverted to G1's literal -10.
    return torch.exp(-10.0 * contactvel)
