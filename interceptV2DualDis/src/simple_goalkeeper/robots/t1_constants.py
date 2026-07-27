"""Booster T1 robot constants for SimpleGoalKeeper."""
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

T1_XML = Path(__file__).parent / "xmls" / "t1.xml"
T1_HEADLESS_XML = Path(__file__).parent / "xmls" / "t1_headless.xml"
assert T1_XML.exists(), f"Missing {T1_XML}"
assert T1_HEADLESS_XML.exists(), f"Missing {T1_HEADLESS_XML}"

_rpm = lambda r: r * 2 * 3.14159265 / 60  # noqa: E731

ARM_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(21.8e-6, 36),
    velocity_limit=_rpm(89), effort_limit=36.0,
)
WAIST_HIP_ROLL_YAW_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(76.5e-6, 25),
    velocity_limit=_rpm(70), effort_limit=40.0,
)
HIP_PITCH_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(161.7e-6, 18),
    velocity_limit=_rpm(157), effort_limit=55.0,
)
KNEE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(196.3e-6, 18),
    velocity_limit=_rpm(140), effort_limit=65.0,
)
ANKLE_ACTUATOR = ElectricActuator(
    reflected_inertia=reflected_inertia(26.2e-6, 36),
    velocity_limit=_rpm(117), effort_limit=50.0,
)

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265   # 5 Hz natural frequency
DAMPING_RATIO = 2.0

def _kp(act: ElectricActuator) -> float:
    return act.reflected_inertia * NATURAL_FREQ ** 2

def _kv(act: ElectricActuator) -> float:
    return 2.0 * DAMPING_RATIO * act.reflected_inertia * NATURAL_FREQ

DELAY_MIN, DELAY_MAX = 1, 3

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"),
    stiffness=_kp(ARM_ACTUATOR), damping=_kv(ARM_ACTUATOR),
    effort_limit=ARM_ACTUATOR.effort_limit, armature=ARM_ACTUATOR.reflected_inertia,
    delay_min_lag=DELAY_MIN, delay_max_lag=DELAY_MAX,
)
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"),
    stiffness=_kp(WAIST_HIP_ROLL_YAW_ACTUATOR), damping=_kv(WAIST_HIP_ROLL_YAW_ACTUATOR),
    effort_limit=WAIST_HIP_ROLL_YAW_ACTUATOR.effort_limit,
    armature=WAIST_HIP_ROLL_YAW_ACTUATOR.reflected_inertia,
    delay_min_lag=DELAY_MIN, delay_max_lag=DELAY_MAX,
)
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Hip_Pitch",),
    stiffness=_kp(HIP_PITCH_ACTUATOR), damping=_kv(HIP_PITCH_ACTUATOR),
    effort_limit=HIP_PITCH_ACTUATOR.effort_limit, armature=HIP_PITCH_ACTUATOR.reflected_inertia,
    delay_min_lag=DELAY_MIN, delay_max_lag=DELAY_MAX,
)
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Knee_Pitch",),
    stiffness=_kp(KNEE_ACTUATOR), damping=_kv(KNEE_ACTUATOR),
    effort_limit=KNEE_ACTUATOR.effort_limit, armature=KNEE_ACTUATOR.reflected_inertia,
    delay_min_lag=DELAY_MIN, delay_max_lag=DELAY_MAX,
)
T1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
    stiffness=_kp(ANKLE_ACTUATOR), damping=_kv(ANKLE_ACTUATOR),
    effort_limit=ANKLE_ACTUATOR.effort_limit, armature=ANKLE_ACTUATOR.reflected_inertia,
    delay_min_lag=DELAY_MIN, delay_max_lag=DELAY_MAX,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.665),  # ~1 cm clearance above foot-geom bottom for headless XML (0.700 floats feet 4.5 cm)
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        r"(Left_Hip_Pitch|Right_Hip_Pitch)": -0.3,
        r"(Left_Knee_Pitch|Right_Knee_Pitch)": 0.6,
        r"(Left_Ankle_Pitch|Right_Ankle_Pitch)": -0.3,
        # FIX 2026-07-24: mirror-symmetric standing pose (was a one-sided
        # "right arm counterbalance" stance, Right_Shoulder_Roll=1.07 vs
        # Left=-0.41 -- a leftover from G1's hand-catching reach pose, not
        # a fit for this feet-only track). Shoulder_Pitch/Elbow_Pitch share
        # the same axis sign convention L/R (t1_headless.xml: both
        # axis="0 1 0", same range) so equal values are already symmetric;
        # Shoulder_Roll/Elbow_Yaw have mirrored axis conventions (axis
        # "1 0 0"/"0 0 1" but opposite-signed joint ranges L/R) so a
        # symmetric pose needs equal magnitude, opposite sign. See
        # docs/BugFixes.md.
        r"(Left_Shoulder_Pitch|Right_Shoulder_Pitch)": -0.21,
        # FIX 2026-07-27 (CORRECTED, user request, "look more soccer-like"):
        # three earlier same-day edits (0.41 -> 0.1745 -> 0.05 rad, i.e.
        # toward 0.0) had the axis direction backwards -- confirmed via an
        # offscreen mujoco.Renderer image (not just robot.data.joint_pos
        # numbers, which read "correct" the whole time but didn't reveal
        # the actual rendered pose) that Shoulder_Roll near 0.0 is a full
        # T-POSE (arms horizontal), not "hanging down" as assumed. Rendered
        # comparison images confirmed 1.5 rad (~86deg) -- close to this
        # joint's own range limit (Left: -1.74..1.57, Right: -1.57..1.74,
        # t1_headless.xml) rather than near its center -- is what actually
        # produces arms hanging naturally at the sides. Verified within
        # both the hard range and the 0.9 soft_joint_pos_limit_factor band
        # used elsewhere in training. Only Shoulder_Roll changed;
        # Shoulder_Pitch/Elbow_Pitch/Elbow_Yaw are untouched and still look
        # natural at 1.5 rad per the same renders. _POST_SAVE_STANCE_MAP
        # (rewards.py) mirrors this exact value -- keep them in sync if
        # either changes again. See docs/BugFixes.md for the full
        # misdiagnosis writeup and the actual reference images.
        r"Left_Shoulder_Roll": -1.5,
        r"Right_Shoulder_Roll": 1.5,
        r"(Left_Elbow_Pitch|Right_Elbow_Pitch)": -0.13,
        r"Left_Elbow_Yaw": -0.21,
        r"Right_Elbow_Yaw": 0.21,
    },
    joint_vel={".*": 0.0},
)

_foot_regex = r"^(left|right)_foot\d+_collision$"
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    solref=(0.01, 1),
    condim={_foot_regex: 6, ".*_collision": 3},
    friction={_foot_regex: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
    priority=1,
)

T1_ARTICULATION_HEADLESS = EntityArticulationInfoCfg(
    actuators=(
        T1_ACTUATOR_ARM, T1_ACTUATOR_WAIST, T1_ACTUATOR_HIP_PITCH,
        T1_ACTUATOR_KNEE, T1_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)

def get_t1_headless_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(T1_HEADLESS_XML)),
        articulation=T1_ARTICULATION_HEADLESS,
    )

T1_ACTION_SCALE_HEADLESS: dict[str, float] = {}
for _a in T1_ARTICULATION_HEADLESS.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    for _n in _a.target_names_expr:
        T1_ACTION_SCALE_HEADLESS[_n] = 0.25 * _a.effort_limit / _a.stiffness

# Explicit KP/KD values for each joint group (useful for deployment verification).
# Computed from NATURAL_FREQ=5 Hz, DAMPING_RATIO=2.0, motor specs above.
#   ARM:   kp=27.88  kd=3.55   effort=36 N·m   action_scale≈0.323
#   WAIST: kp=47.19  kd=6.01   effort=40 N·m   action_scale≈0.212
#   HIP_P: kp=51.71  kd=6.58   effort=55 N·m   action_scale≈0.266
#   KNEE:  kp=62.77  kd=7.99   effort=65 N·m   action_scale≈0.259
#   ANKLE: kp=33.51  kd=4.27   effort=50 N·m   action_scale≈0.373
T1_KP: dict[str, float] = {n: a.stiffness for a in T1_ARTICULATION_HEADLESS.actuators
                             if isinstance(a, BuiltinPositionActuatorCfg)
                             for n in a.target_names_expr}
T1_KD: dict[str, float] = {n: a.damping for a in T1_ARTICULATION_HEADLESS.actuators
                             if isinstance(a, BuiltinPositionActuatorCfg)
                             for n in a.target_names_expr}
