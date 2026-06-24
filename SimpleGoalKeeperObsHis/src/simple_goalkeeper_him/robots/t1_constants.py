"""Booster T1 robot constants for SimpleGoalKeeperHim."""
import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

T1_XML = Path(__file__).parent / "xmls" / "t1.xml"
T1_HEADLESS_XML = Path(__file__).parent / "xmls" / "t1_headless.xml"
assert T1_XML.exists(), f"Missing {T1_XML}"
assert T1_HEADLESS_XML.exists(), f"Missing {T1_HEADLESS_XML}"

# ─── Actuator design parameters ──────────────────────────────────────────────
# Natural frequency 10 Hz — mirrors Imitationlearningbooster exactly.
# The prior value (5 Hz) squared to 25 instead of 100 → all leg kp values were
# 4× too low (knee: 62.8 vs 200, hip_pitch: 51.7 vs 120).  Weak gains let the
# robot collapse under gravity from a standing pose with a random policy,
# producing 3-step episodes and a broken surrogate from iteration 1.
_NATURAL_FREQ = 2 * math.pi * 10   # 62.83 rad/s

# Actuator command delay (timesteps). ILB uses 2–8 at 200 Hz physics rate.
DELAY_MIN, DELAY_MAX = 2, 8


def _make_actuator(
    target_names_expr: tuple,
    effort: float,
    stiffness: float,
) -> BuiltinPositionActuatorCfg:
    """Create a PD position actuator with ILB-matched gain formula.

    armature = stiffness / ω²   (same as ILB _make_actuator)
    damping   = 2 × 2 × √(stiffness × armature)   (critically-damped × 2)
    """
    armature = stiffness / (_NATURAL_FREQ ** 2)
    damping  = 2 * 2.0 * math.sqrt(stiffness * armature)
    return BuiltinPositionActuatorCfg(
        target_names_expr=target_names_expr,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
        armature=armature,
        delay_min_lag=DELAY_MIN,
        delay_max_lag=DELAY_MAX,
    )


# Hardware-validated stiffness values — match ILB (Imitationlearningbooster/robots/t1_constants.py).
# kp values come from KaydenKnapik/BoosterT1mjlab hardware testing.
T1_ACTUATOR_ARM = _make_actuator(
    (".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"),
    effort=36.0, stiffness=15.0,
)
T1_ACTUATOR_WAIST = _make_actuator(
    ("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"),
    effort=40.0, stiffness=80.0,
)
T1_ACTUATOR_HIP_PITCH = _make_actuator(
    (".*_Hip_Pitch",),
    effort=55.0, stiffness=120.0,
)
T1_ACTUATOR_KNEE = _make_actuator(
    (".*_Knee_Pitch",),
    effort=65.0, stiffness=200.0,
)
T1_ACTUATOR_ANKLE_PITCH = _make_actuator(
    (".*_Ankle_Pitch",),
    effort=50.0, stiffness=50.0,
)
T1_ACTUATOR_ANKLE_ROLL = _make_actuator(
    (".*_Ankle_Roll",),
    effort=50.0, stiffness=40.0,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.700),  # 0.700 m = feet 4.5 cm above floor — matches ILB exactly.
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        r"(Left_Hip_Pitch|Right_Hip_Pitch)": -0.3,
        r"(Left_Knee_Pitch|Right_Knee_Pitch)": 0.6,
        r"(Left_Ankle_Pitch|Right_Ankle_Pitch)": -0.3,
        r"Left_Shoulder_Pitch": -0.21,
        r"Left_Shoulder_Roll": -0.41,
        r"Right_Shoulder_Pitch": -0.11,
        r"Right_Shoulder_Roll": 1.07,
        r"Right_Elbow_Pitch": -0.13,
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
        T1_ACTUATOR_ARM,
        T1_ACTUATOR_WAIST,
        T1_ACTUATOR_HIP_PITCH,
        T1_ACTUATOR_KNEE,
        T1_ACTUATOR_ANKLE_PITCH,
        T1_ACTUATOR_ANKLE_ROLL,
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
