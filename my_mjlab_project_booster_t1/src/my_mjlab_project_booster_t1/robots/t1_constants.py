"""Booster T1 robot constants for mjlab."""
from __future__ import annotations
import math
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg

T1_XML = Path(__file__).parent.parent / "assets" / "booster_t1" / "T1_serial_clean.xml"

_NATURAL_FREQ = 2 * math.pi * 10  # 10 Hz


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(T1_XML))


def _make_actuator(target: str, effort: float, stiffness: float) -> BuiltinPositionActuatorCfg:
    armature = stiffness / (_NATURAL_FREQ ** 2)
    damping = 2 * 2.0 * math.sqrt(stiffness * armature)
    return BuiltinPositionActuatorCfg(
        target_names_expr=(target,),
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
        armature=armature,
    )


T1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _make_actuator(r"(AAHead_yaw|Head_pitch)", 7.0, 20.0),
        _make_actuator(
            r"(Left_Shoulder_Pitch|Left_Shoulder_Roll|Left_Elbow_Pitch|Left_Elbow_Yaw"
            r"|Right_Shoulder_Pitch|Right_Shoulder_Roll|Right_Elbow_Pitch|Right_Elbow_Yaw)",
            18.0, 15.0,
        ),
        _make_actuator(r"Waist", 30.0, 80.0),
        _make_actuator(r"(Left_Hip_Pitch|Right_Hip_Pitch)", 45.0, 120.0),
        _make_actuator(r"(Left_Hip_Roll|Left_Hip_Yaw|Right_Hip_Roll|Right_Hip_Yaw)", 30.0, 80.0),
        _make_actuator(r"(Left_Knee_Pitch|Right_Knee_Pitch)", 60.0, 200.0),
        _make_actuator(r"(Left_Ankle_Pitch|Right_Ankle_Pitch)", 20.0, 50.0),
        _make_actuator(r"(Left_Ankle_Roll|Right_Ankle_Roll)", 15.0, 40.0),
    ),
    soft_joint_pos_limit_factor=0.9,
)

T1_ACTION_SCALE = {
    r"(AAHead_yaw|Head_pitch)": 7.0 / 20.0,
    # Arms: use 0.25 factor (same as G1) so motor saturates at 4x action_scale, not 1x.
    # Without this, 32% of N(0,1) policy outputs saturate the arm → arms appear locked.
    r"(Left_Shoulder_Pitch|Left_Shoulder_Roll|Left_Elbow_Pitch|Left_Elbow_Yaw"
    r"|Right_Shoulder_Pitch|Right_Shoulder_Roll|Right_Elbow_Pitch|Right_Elbow_Yaw)": 0.25 * 18.0 / 15.0,
    r"Waist": 30.0 / 80.0,
    r"(Left_Hip_Pitch|Right_Hip_Pitch)": 45.0 / 120.0,
    r"(Left_Hip_Roll|Left_Hip_Yaw|Right_Hip_Roll|Right_Hip_Yaw)": 30.0 / 80.0,
    r"(Left_Knee_Pitch|Right_Knee_Pitch)": 60.0 / 200.0,
    r"(Left_Ankle_Pitch|Right_Ankle_Pitch)": 20.0 / 50.0,
    r"(Left_Ankle_Roll|Right_Ankle_Roll)": 15.0 / 40.0,
}

T1_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.700),  # raised 3cm above ground so robot drops cleanly onto floor, avoiding underground spawn
    rot=(0.7071068, 0.0, 0.0, 0.7071068),  # +90° yaw around Z to match G1 orientation
    joint_pos={
        # Legs: kept bent for stability during early training
        r"(Left_Hip_Pitch|Right_Hip_Pitch)": -0.3,
        r"(Left_Knee_Pitch|Right_Knee_Pitch)": 0.6,
        r"(Left_Ankle_Pitch|Right_Ankle_Pitch)": -0.3,
        # Arms: set to motion average so action=0 sits near the reference, not T-pose.
        # Right_Shoulder_Roll is +1.07 avg (counterbalance arm) — the biggest driver of the lock.
        r"Left_Shoulder_Pitch": -0.21,
        r"Left_Shoulder_Roll": -0.41,
        r"Right_Shoulder_Pitch": -0.11,
        r"Right_Shoulder_Roll": 1.07,
        r"Right_Elbow_Pitch": -0.13,
        r"Right_Elbow_Yaw": 0.21,
    },
    joint_vel={".*": 0.0},
)

T1_COLLISION = CollisionCfg(
    geom_names_expr=(
        r"(torso_left|torso_right|head|"
        r"left_thigh|left_calf|left_forearm|left_hand|"
        r"right_thigh|right_calf|right_forearm|right_hand)",
    ),
    condim=1,
)
T1_FEET_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_[12]$",),
    condim=3,
    priority=1,
    friction=(0.6, 0.005, 0.001),
    disable_other_geoms=False,  # don't undo body geoms enabled by T1_COLLISION
)


def get_t1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=T1_STANDING_KEYFRAME,
        collisions=(T1_COLLISION, T1_FEET_COLLISION),
        spec_fn=get_spec,
        articulation=T1_ARTICULATION,
    )
