# SimpleGoalKeeper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone foot-based goalkeeper training environment for Booster T1 in `SimpleGoalKeeper/`, using beyondAMP for motion priors, with ball spawn in the robot's local +X frame.

**Architecture:** Start from `mjlab`'s base velocity env, strip terrain/velocity-command complexity, add ball entity + foot-based rewards. Ball always spawns in the robot's local +X frame and flies toward the robot. beyondAMP wraps the env for motion-prior training with a single discriminator.

**Tech Stack:** mjlab 1.3.0, beyondAMP (cloned from GitHub), rsl-rl-amp, mujoco ≥3.8, uv, Python 3.12.

---

## File Map

| File | Purpose |
|------|---------|
| `SimpleGoalKeeper/pyproject.toml` | uv project, beyondAMP editable deps |
| `SimpleGoalKeeper/CLAUDE.md` | Phase 1 scope notes |
| `SimpleGoalKeeper/src/simple_goalkeeper/robots/t1_constants.py` | T1 actuators, headless cfg, action scale |
| `SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/` | T1 XMLs + STL assets (copied) |
| `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py` | ball_pos_b, ball_vel_b (full visibility), foot pos |
| `SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py` | reset_ball_local_frame, visibility state init |
| `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` | 7 foot-based reward terms (local frame) |
| `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` | Full env config factory |
| `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | AMPRunnerCfg |
| `SimpleGoalKeeper/src/simple_goalkeeper/tasks/__init__.py` | Task registration |
| `SimpleGoalKeeper/src/simple_goalkeeper/scripts/pkl_to_npz.py` | Convert PKL motions → NPZ |
| `SimpleGoalKeeper/src/simple_goalkeeper/scripts/train.py` | Training entry point |
| `SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py` | Play/eval entry point |
| `SimpleGoalKeeper/src/simple_goalkeeper/motions/data/` | Converted NPZ motion files |

---

## Task 1: Scaffold Project Structure

**Files:**
- Create: `SimpleGoalKeeper/pyproject.toml`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/__init__.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/robots/__init__.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/__init__.py` (placeholder)
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/scripts/__init__.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/motions/data/.gitkeep`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls
mkdir -p /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/mdp
mkdir -p /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/tasks
mkdir -p /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/scripts
mkdir -p /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/motions/data
touch /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/__init__.py
touch /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/robots/__init__.py
touch /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py
touch /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/scripts/__init__.py
touch /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/motions/data/.gitkeep
```

- [ ] **Step 2: Write pyproject.toml**

`SimpleGoalKeeper/pyproject.toml`:
```toml
[project]
name = "simple-goalkeeper"
version = "0.1.0"
description = "Phase 1 foot-based goalkeeper for Booster T1"
requires-python = ">=3.10,<3.14"
dependencies = [
    "mjlab",
    "beyondAMP",
    "rsl-rl-amp",
    "amp-tasks",
    "amp-tasks-mjlab",
    "mujoco>=3.8.0,<3.9.0",
    "mujoco-warp>=3.8.0.1,<3.9.0",
    "numpy",
    "torch",
    "tyro",
    "tqdm",
]

[tool.uv.sources]
beyondAMP       = { path = "beyondAMP/source/beyondAMP",       editable = true }
rsl-rl-amp      = { path = "beyondAMP/source/rsl_rl_amp",      editable = true }
amp-tasks       = { path = "beyondAMP/source/amp_tasks",       editable = true }
amp-tasks-mjlab = { path = "beyondAMP/source/amp_tasks_mjlab", editable = true }

[project.scripts]
sgk_train    = "simple_goalkeeper.scripts.train:main"
sgk_play     = "simple_goalkeeper.scripts.play:main"
sgk_convert  = "simple_goalkeeper.scripts.pkl_to_npz:main"

[build-system]
requires = ["uv_build>=0.8.0,<0.9.0"]
```

- [ ] **Step 3: Commit scaffold**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git init
git add .
git commit -m "chore: scaffold SimpleGoalKeeper project"
```

---

## Task 2: Clone beyondAMP and Install

**Files:**
- Create: `SimpleGoalKeeper/beyondAMP/` (cloned)

- [ ] **Step 1: Clone beyondAMP**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git clone https://github.com/Renforce-Dynamics/beyondAMP.git
```

Expected: `beyondAMP/source/beyondAMP/`, `beyondAMP/source/rsl_rl_amp/`, etc. appear.

- [ ] **Step 2: Verify source structure**

```bash
ls /home/isaak/BEPImitationlearning/SimpleGoalKeeper/beyondAMP/source/
```

Expected output contains: `beyondAMP  rsl_rl_amp  amp_tasks  amp_tasks_mjlab`

- [ ] **Step 3: Create and sync uv venv**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
pip install uv --quiet 2>/dev/null; uv sync
```

Expected: `.venv` created, `beyondAMP`, `rsl_rl_amp`, `mjlab` all resolved.

- [ ] **Step 4: Verify beyondAMP import**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "from beyondAMP.mjlab.rsl_rl import AMPRunnerCfg; print('beyondAMP OK')"
```

Expected: `beyondAMP OK`

- [ ] **Step 5: Commit lock file**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add beyondAMP uv.lock pyproject.toml
git commit -m "chore: add beyondAMP subdir and lock file"
```

---

## Task 3: Copy Robot Assets

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/t1.xml`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/t1_headless.xml`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/ball.xml`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/assets/` (all STL files)

- [ ] **Step 1: Copy XML files and assets**

```bash
SRC=/home/isaak/HandWavingMotion/src/booster_t1_mjlab/robots/boostert1/xmls
DST=/home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls

cp $SRC/t1.xml $DST/
cp $SRC/t1_headless.xml $DST/
cp -r $SRC/assets $DST/

# Copy ball.xml from HandWavingMotion kick task
cp /home/isaak/HandWavingMotion/src/booster_t1_mjlab/robots/boostert1/xmls/ball.xml $DST/
```

- [ ] **Step 2: Verify all STLs are present**

```bash
ls /home/isaak/BEPImitationlearning/SimpleGoalKeeper/src/simple_goalkeeper/robots/xmls/assets/*.stl | wc -l
```

Expected: same count as in HandWavingMotion (≥20 STL files).

- [ ] **Step 3: Write t1_constants.py**

`SimpleGoalKeeper/src/simple_goalkeeper/robots/t1_constants.py`:
```python
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

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265
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
    pos=(0.0, 0.0, 0.665),
    joint_pos={
        "Left_Shoulder_Roll": -1.4, "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4, "Right_Elbow_Yaw": 0.4,
        ".*_Hip_Pitch": -0.2, ".*_Knee_Pitch": 0.4, ".*_Ankle_Pitch": -0.2,
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
```

- [ ] **Step 4: Verify t1_constants imports**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
from simple_goalkeeper.robots.t1_constants import get_t1_headless_robot_cfg, T1_ACTION_SCALE_HEADLESS
cfg = get_t1_headless_robot_cfg()
print('Robot cfg OK, action dims:', len(T1_ACTION_SCALE_HEADLESS))
"
```

Expected: `Robot cfg OK, action dims: 21`

- [ ] **Step 5: Commit assets**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/
git commit -m "feat: add T1 robot assets and constants"
```

---

## Task 4: Convert PKL Motions to NPZ

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/scripts/pkl_to_npz.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/motions/data/*.npz` (8 files)

- [ ] **Step 1: Write pkl_to_npz.py**

`SimpleGoalKeeper/src/simple_goalkeeper/scripts/pkl_to_npz.py`:
```python
"""Convert Booster T1 goalkeeper PKL motions → NPZ for beyondAMP.

PKL format: {fps: int, root_pos: (T,3), root_rot: (T,4) xyzw, dof_pos: (T,23)}
NPZ output: fps, joint_pos (T,21), joint_vel (T,21), body_pos_w, body_quat_w,
            body_lin_vel_w, body_ang_vel_w  — all (T, num_bodies, 3/4)

Usage:
    uv run sgk_convert --input-dir /path/to/Motions --output-dir src/simple_goalkeeper/motions/data
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

_HERE = Path(__file__).parent
_XML = _HERE.parent / "robots" / "xmls" / "t1_headless.xml"

# T1 joint order in MuJoCo qpos (after 7-DOF freejoint):
# index 0-1 = head (AAHead_yaw, Head_pitch) — not in headless actuators
# index 2-9 = arms, 10 = waist, 11-16 = left leg, 17-22 = right leg
_HEAD_JOINT_COUNT = 2  # skip in headless output


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert xyzw → wxyz (MuJoCo convention)."""
    return q[..., [3, 0, 1, 2]]


def _finite_diff_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central differences for velocity, forward/backward at endpoints."""
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def _quat_ang_vel(quats: np.ndarray, dt: float) -> np.ndarray:
    """Approximate angular velocity from quaternion sequence via log map."""
    # quats: (T, 4) wxyz
    ang_vel = np.zeros((len(quats), 3))
    for i in range(1, len(quats) - 1):
        q0 = quats[i - 1]
        q1 = quats[i + 1]
        # relative quat q_rel = q1 * conj(q0)
        q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        # quat multiply: q_rel = q1 * q0_conj
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q0_conj
        qr = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        # axis-angle from quat: omega = 2 * axis * angle / (2*dt)
        angle = 2.0 * np.arctan2(np.linalg.norm(qr[1:]), qr[0])
        axis_norm = np.linalg.norm(qr[1:])
        if axis_norm > 1e-8:
            axis = qr[1:] / axis_norm
        else:
            axis = np.zeros(3)
        ang_vel[i] = axis * angle / (2.0 * dt)
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def convert_one(pkl_path: Path, output_path: Path, output_fps: int = 50) -> None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    input_fps: int = data["fps"]
    root_pos = np.array(data["root_pos"], dtype=np.float32)   # (T, 3)
    root_rot_xyzw = np.array(data["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    dof_pos = np.array(data["dof_pos"], dtype=np.float32)     # (T, 23)
    T_in = root_pos.shape[0]

    # Resample to output_fps
    duration = (T_in - 1) / input_fps
    t_out = np.arange(0, duration, 1.0 / output_fps)
    T_out = len(t_out)
    t_in = np.linspace(0, duration, T_in)

    def resample(arr):
        out = np.zeros((T_out, arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(t_out, t_in, arr[:, j])
        return out

    root_pos_r = resample(root_pos)
    root_rot_r = resample(root_rot_xyzw)
    # Renormalise quaternions after lerp
    norms = np.linalg.norm(root_rot_r, axis=-1, keepdims=True).clip(min=1e-8)
    root_rot_r /= norms
    dof_pos_r = resample(dof_pos)

    # Convert root rot xyzw → wxyz for MuJoCo
    root_rot_wxyz = _quat_xyzw_to_wxyz(root_rot_r)

    # Floor-correct: shift Z so minimum foot height = 0
    model = mujoco.MjModel.from_xml_path(str(_XML))
    mdata = mujoco.MjData(model)

    # Find foot body indices
    foot_names = ["left_foot_link", "right_foot_link"]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]

    min_foot_z = float("inf")
    for i in range(T_out):
        mdata.qpos[:3] = root_pos_r[i]
        mdata.qpos[3:7] = root_rot_wxyz[i]
        mdata.qpos[7:] = dof_pos_r[i]
        mujoco.mj_kinematics(model, mdata)
        for bid in foot_ids:
            min_foot_z = min(min_foot_z, float(mdata.xpos[bid, 2]))

    root_pos_r[:, 2] -= min_foot_z
    print(f"  z-correction: {-min_foot_z:+.4f} m")

    # FK pass: extract body kinematics
    num_bodies = model.nbody
    body_pos_w = np.zeros((T_out, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T_out, num_bodies, 4), dtype=np.float32)

    for i in tqdm(range(T_out), desc=pkl_path.stem, ncols=80):
        mdata.qpos[:3] = root_pos_r[i]
        mdata.qpos[3:7] = root_rot_wxyz[i]
        mdata.qpos[7:] = dof_pos_r[i]
        mujoco.mj_kinematics(model, mdata)
        body_pos_w[i] = mdata.xpos.copy()
        body_quat_w[i] = mdata.xquat.copy()

    body_lin_vel_w = _finite_diff_vel(body_pos_w, 1.0 / output_fps)

    # Angular velocity per body from quaternions
    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for b in range(num_bodies):
        body_ang_vel_w[:, b, :] = _quat_ang_vel(body_quat_w[:, b, :], 1.0 / output_fps)

    # Joint data: skip head joints (first 2) → 21-DOF
    joint_pos = dof_pos_r[:, _HEAD_JOINT_COUNT:]  # (T, 21)
    joint_vel = _finite_diff_vel(joint_pos, 1.0 / output_fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        fps=np.array([output_fps]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos_absolute=np.array([1], dtype=np.int8),
    )
    print(f"  Saved → {output_path.name}  shape={joint_pos.shape}")


def main(
    input_dir: str = str(Path("/home/isaak/BEPImitationlearning/Motions")),
    output_dir: str = str(_HERE.parent / "motions" / "data"),
    output_fps: int = 50,
) -> None:
    """Convert all *.pkl in input_dir to NPZ files in output_dir."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    pkl_files = sorted(in_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {in_dir}")
    print(f"Converting {len(pkl_files)} files @ {output_fps} fps → {out_dir}")
    for pkl in pkl_files:
        convert_one(pkl, out_dir / (pkl.stem + ".npz"), output_fps)
    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
```

- [ ] **Step 2: Run conversion**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_convert \
    --input-dir /home/isaak/BEPImitationlearning/Motions \
    --output-dir src/simple_goalkeeper/motions/data \
    --output-fps 50
```

Expected: 8 NPZ files appear in `src/simple_goalkeeper/motions/data/`.

- [ ] **Step 3: Verify NPZ structure**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
import numpy as np
from pathlib import Path
f = next(Path('src/simple_goalkeeper/motions/data').glob('*.npz'))
d = np.load(f)
print('keys:', list(d.keys()))
print('joint_pos shape:', d['joint_pos'].shape)
print('body_pos_w shape:', d['body_pos_w'].shape)
print('fps:', d['fps'])
"
```

Expected: `joint_pos shape: (N, 21)`, `body_pos_w shape: (N, num_bodies, 3)`, `fps: [50]`

- [ ] **Step 4: Commit converted motions**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/scripts/pkl_to_npz.py src/simple_goalkeeper/motions/
git commit -m "feat: add PKL→NPZ converter and convert 8 goalkeeper motions"
```

---

## Task 5: Write Observations

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py`

- [ ] **Step 1: Write observations.py**

`SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py`:
```python
"""Goalkeeper observation terms — ball (with full visibility) + foot positions."""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_b = torch.zeros(env.num_envs, 3, device=env.device)
    x_b[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_b)


def _ball_visibility(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Compute per-env visibility mask (N,) bool with full 3-gate system.

    Gates:
      initial_vanish: _catchstep < _startstep (warmup countdown ~1 s)
      flying:  ball in view window (x_b ∈ (0.05, 3.4), |y_b| < 2, z < 1.8, approaching)
      random_vanish: _ball_visible_step > _vanish_step (per-episode random disappear)
    """
    if getattr(env, "_ball_vis_step", -1) == env.common_step_counter:
        return env._ball_vis_cache

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w

    ball_b = quat_apply(quat_inv(base_quat_w), ball_pos_w - base_pos_w)
    x_b = ball_b[:, 0]
    y_b = ball_b[:, 1]
    z_b = ball_b[:, 2]

    # initial_vanish gate
    catchstep = getattr(env, "_catchstep", None)
    startstep = getattr(env, "_startstep", None)
    if catchstep is None:
        initial_vanish = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif startstep is None:
        initial_vanish = catchstep < 43
    else:
        initial_vanish = catchstep < startstep

    catchstep_positive = (catchstep > 0) if catchstep is not None else torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(env.num_envs, device=env.device)
    approaching = (x_b < env._ball_obs_last_x) | (env._ball_obs_last_x == 0.0)
    env._ball_obs_last_x = x_b.clone()

    flying = (x_b > 0.05) & (x_b < 3.4) & (y_b.abs() < 2.0) & (z_b < 1.8) & catchstep_positive & approaching

    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.randint(0, 30, (env.num_envs,), device=env.device)
        env._ball_visible_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_ball_visible_step"):
        env._ball_visible_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._ball_visible_step = torch.where(
        flying, env._ball_visible_step + 1, torch.zeros_like(env._ball_visible_step)
    )
    random_vanish = env._ball_visible_step > env._vanish_step

    visible = initial_vanish & flying & ~random_vanish
    env._ball_vis_cache = visible
    env._ball_vis_step = env.common_step_counter
    return visible


def ball_pos_b(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ball position in robot body frame, zeroed when not visible. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    base_quat_w = robot.data.root_link_quat_w
    ball_b = quat_apply(quat_inv(base_quat_w), ball.data.root_link_pos_w - robot.data.root_link_pos_w)
    visible = _ball_visibility(env, ball_name)
    return ball_b * visible.float().unsqueeze(-1)


def ball_vel_b(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ball velocity in robot body frame, zeroed when not visible. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    base_quat_w = robot.data.root_link_quat_w
    vel_b = quat_apply(quat_inv(base_quat_w), ball.data.root_link_lin_vel_w)
    visible = _ball_visibility(env, ball_name)
    return vel_b * visible.float().unsqueeze(-1)


def left_foot_pos_b(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _FEET_CFG,
) -> torch.Tensor:
    """Left foot position in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[0]
    foot_pos_w = robot.data.body_link_pos_w[:, idx, :]
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), foot_pos_w - robot.data.root_link_pos_w)


def right_foot_pos_b(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _FEET_CFG,
) -> torch.Tensor:
    """Right foot position in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[1]
    foot_pos_w = robot.data.body_link_pos_w[:, idx, :]
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), foot_pos_w - robot.data.root_link_pos_w)
```

- [ ] **Step 2: Commit observations**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/mdp/observations.py
git commit -m "feat: add ball visibility + foot position observations"
```

---

## Task 6: Write Events

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py`

- [ ] **Step 1: Write events.py**

`SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py`:
```python
"""Event terms for SimpleGoalKeeper — ball reset in robot local frame."""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_ball_local_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    ball_name: str = "ball",
    dist_range: tuple[float, float] = (2.0, 4.0),
    lateral_range: tuple[float, float] = (-0.5, 0.5),
    height_range: tuple[float, float] = (0.1, 0.8),
    speed_range: tuple[float, float] = (2.0, 6.0),
) -> None:
    """Spawn the ball in the robot's local +X frame and fire it toward the robot.

    All ranges are in the robot's body frame:
      dist    = distance along local +X (forward direction)
      lateral = offset along local +Y (left/right)
      height  = absolute Z above the floor
    Ball initial velocity is -speed along local +X (toward robot).
    """
    n = len(env_ids)
    device = env.device

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]

    # Sample local-frame offsets
    dist = torch.empty(n, device=device).uniform_(*dist_range)
    lateral = torch.empty(n, device=device).uniform_(*lateral_range)
    height = torch.empty(n, device=device).uniform_(*height_range)
    speed = torch.empty(n, device=device).uniform_(*speed_range)

    local_offset = torch.stack([dist, lateral, torch.zeros(n, device=device)], dim=-1)  # (n, 3)

    robot_pos_w = robot.data.root_link_pos_w[env_ids]         # (n, 3)
    robot_quat_w = robot.data.root_link_quat_w[env_ids]       # (n, 4)

    # Rotate local offset into world frame
    world_offset = quat_apply(robot_quat_w, local_offset)     # (n, 3)
    ball_pos_w = robot_pos_w + world_offset
    ball_pos_w[:, 2] = height + env.scene.env_origins[env_ids, 2]  # absolute Z

    # Ball velocity: -speed along local +X in world frame
    local_vel = torch.stack([-speed, torch.zeros(n, device=device), torch.zeros(n, device=device)], dim=-1)
    ball_vel_w = quat_apply(robot_quat_w, local_vel)          # (n, 3)

    # Set ball state — position + linear velocity, zero angular velocity
    ball_state = torch.zeros(n, 13, device=device)
    ball_state[:, :3] = ball_pos_w
    ball_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, -1)
    ball_state[:, 7:10] = ball_vel_w
    ball.write_root_state_to_sim(ball_state, env_ids=env_ids)

    # Initialise visibility state for reset envs
    _init_visibility_state(env, env_ids)


def _init_visibility_state(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> None:
    """Initialise per-env visibility counters on episode reset."""
    n_total = env.num_envs
    device = env.device

    if not hasattr(env, "_catchstep"):
        env._catchstep = torch.zeros(n_total, dtype=torch.long, device=device)
    if not hasattr(env, "_startstep"):
        env._startstep = torch.zeros(n_total, dtype=torch.long, device=device)
    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.zeros(n_total, dtype=torch.long, device=device)
    if not hasattr(env, "_ball_visible_step"):
        env._ball_visible_step = torch.zeros(n_total, dtype=torch.long, device=device)
    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(n_total, device=device)

    env._catchstep[env_ids] = 50
    env._startstep[env_ids] = 50 - torch.randint(3, 11, (len(env_ids),), device=device)
    env._vanish_step[env_ids] = torch.randint(0, 30, (len(env_ids),), device=device)
    env._ball_visible_step[env_ids] = 0
    env._ball_obs_last_x[env_ids] = 0.0
```

- [ ] **Step 2: Commit events**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/mdp/events.py
git commit -m "feat: add local-frame ball reset event with visibility init"
```

---

## Task 7: Write Rewards

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py`

- [ ] **Step 1: Write rewards.py**

`SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py`:
```python
"""Foot-based goalkeeper reward terms — all direction rewards use robot local frame."""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_ALL_CFG = SceneEntityCfg("robot")


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_b = torch.zeros(env.num_envs, 3, device=env.device)
    x_b[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_b)


def foot_to_ball(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = _FEET_CFG,
) -> torch.Tensor:
    """Gaussian reward on XY distance from foot midpoint to ball.

    Dense signal from episode start — pulls feet toward incoming ball.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :2]  # (N, 2, 2)
    feet_mid_xy = foot_pos_w.mean(dim=1)                                  # (N, 2)
    ball_xy = ball.data.root_link_pos_w[:, :2]
    dist = torch.norm(ball_xy - feet_mid_xy, dim=-1)
    return torch.exp(-(dist ** 2) / (std ** 2))


def ball_vx_reduction(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    max_speed: float = 8.0,
) -> torch.Tensor:
    """Reward for neutralising ball's incoming local-X velocity.

    Ball starts with large negative vx_local (toward robot). Reward peaks
    when vx_local reaches 0 (ball fully stopped). Uses robot-frame projection.
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)                             # (N, 3)
    ball_vel_w = ball.data.root_link_lin_vel_w                   # (N, 3)
    vx_local = (ball_vel_w * x_axis_w).sum(dim=-1)              # (N,)
    incoming = torch.clamp(-vx_local, 0.0, max_speed)           # 0 when stopped/deflected
    return torch.exp(-(incoming ** 2) / ((max_speed / 2.0) ** 2))


def ball_positive_vx(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    target_speed: float = 5.0,
) -> torch.Tensor:
    """Primary success metric: ball deflected back toward the field (+local X).

    Normalised to [0, 1]: saturates at target_speed m/s deflection.
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)
    ball_vel_w = ball.data.root_link_lin_vel_w
    vx_local = (ball_vel_w * x_axis_w).sum(dim=-1)
    return torch.clamp(vx_local / target_speed, 0.0, 1.0)


def posture(
    env: "ManagerBasedRlEnv",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _ALL_CFG,
) -> torch.Tensor:
    """Gaussian reward for staying near the default joint pose."""
    robot: Entity = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_sq = torch.square(joint_pos - default_pos)
    return torch.exp(-torch.mean(error_sq / (std ** 2), dim=1))


def base_ang_vel_xy_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalise XY angular velocity (rolling/pitching) — keep upright."""
    robot: Entity = env.scene["robot"]
    return torch.sum(robot.data.root_link_ang_vel_b[:, :2] ** 2, dim=-1)

# Note: action_rate_l2 and dof_vel_l2 (joint_vel_l2) are provided by
# mjlab.envs.mdp — use those directly in the env config.
```

- [ ] **Step 2: Commit rewards**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/mdp/rewards.py
git commit -m "feat: add 7 foot-based goalkeeper reward terms (local frame)"
```

---

## Task 8: Write Env Config

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

- [ ] **Step 1: Write goalkeeper_env_cfg.py**

`SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`:
```python
"""Goalkeeper environment configuration for Booster T1 — Phase 1 (feet only)."""
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
import mjlab.envs.mdp.observations as mjlab_obs
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

from simple_goalkeeper.robots.t1_constants import get_t1_headless_robot_cfg, T1_ACTION_SCALE_HEADLESS
import simple_goalkeeper.mdp.observations as gk_obs
import simple_goalkeeper.mdp.rewards as gk_rew
from simple_goalkeeper.mdp.events import reset_ball_local_frame

_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_ALL_CFG = SceneEntityCfg("robot")


def _make_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.11, 0.0, 0.0),
        mass=0.42,
        rgba=(1.0, 1.0, 0.0, 1.0),
    )
    geom.friction = (0.4, 0.005, 0.0001)
    geom.solref = [0.002, 0.0001]
    geom.solimp = [0.0001, 0.001, 0.0001, 0.5, 2.0]
    geom.margin = 0.001
    geom.gap = 0.0001
    return spec


def goalkeeper_env_cfg(play: bool = False, num_envs: int = 4096) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # --- Scene ---
    cfg.scene.num_envs = num_envs
    cfg.scene.entities = {"robot": get_t1_headless_robot_cfg()}
    cfg.scene.entities["ball"] = EntityCfg(spec_fn=_make_ball_spec)

    # Flat terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors: only self-collision
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
    )

    cfg.sim.njmax = 500
    cfg.sim.nconmax = 100
    cfg.viewer.body_name = "Trunk"

    # --- Actions ---
    cfg.actions["joint_pos"].scale = T1_ACTION_SCALE_HEADLESS

    # --- Observations ---
    # Strip terrain/command/foot-contact obs from base env
    _remove_keys = ["command", "height_scan", "foot_height", "foot_air_time",
                    "foot_contact", "foot_contact_forces"]
    for k in _remove_keys:
        cfg.observations["actor"].terms.pop(k, None)
        cfg.observations["critic"].terms.pop(k, None)

    _robot_cfg = SceneEntityCfg("robot")
    _gk_obs = {
        "ball_pos_b": ObservationTermCfg(
            func=gk_obs.ball_pos_b,
            noise=Unoise(-0.05, 0.05) if not play else None,
            params={"ball_name": "ball"},
        ),
        "ball_vel_b": ObservationTermCfg(
            func=gk_obs.ball_vel_b,
            noise=Unoise(-0.1, 0.1) if not play else None,
            params={"ball_name": "ball"},
        ),
        "left_foot_pos_b": ObservationTermCfg(
            func=gk_obs.left_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
        "right_foot_pos_b": ObservationTermCfg(
            func=gk_obs.right_foot_pos_b,
            params={"asset_cfg": _FEET_CFG},
        ),
    }
    cfg.observations["actor"].terms.update(_gk_obs)
    cfg.observations["critic"].terms.update(_gk_obs)
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 1

    if not play:
        for term_cfg in cfg.observations["actor"].terms.values():
            term_cfg.delay_min_lag = 0
            term_cfg.delay_max_lag = 2
            term_cfg.delay_per_env = True

    # AMP observation group (beyondAMP handles the discriminator)
    cfg.observations["amp"] = amp_obs_basic_group()

    if play:
        cfg.observations["actor"].enable_corruption = False

    # --- Rewards ---
    # Remove all velocity-tracking rewards from base env
    cfg.rewards.clear()
    cfg.rewards.update({
        "foot_to_ball": RewardTermCfg(
            func=gk_rew.foot_to_ball, weight=3.0,
            params={"ball_name": "ball", "std": 0.15, "asset_cfg": _FEET_CFG},
        ),
        "ball_vx_reduction": RewardTermCfg(
            func=gk_rew.ball_vx_reduction, weight=5.0,
            params={"ball_name": "ball", "max_speed": 8.0},
        ),
        "ball_positive_vx": RewardTermCfg(
            func=gk_rew.ball_positive_vx, weight=10.0,
            params={"ball_name": "ball", "target_speed": 5.0},
        ),
        "posture": RewardTermCfg(
            func=gk_rew.posture, weight=1.0,
            params={"std": 0.25, "asset_cfg": _ALL_CFG},
        ),
        "ang_vel_xy": RewardTermCfg(
            func=gk_rew.base_ang_vel_xy_l2, weight=-0.1,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2, weight=-0.3,
        ),
        "dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2, weight=-0.001,
            params={"asset_cfg": _ALL_CFG},
        ),
    })

    # --- Terminations ---
    cfg.terminations.clear()
    cfg.terminations.update({
        "bad_orientation": TerminationTermCfg(
            func=mjlab_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": _robot_cfg},
            time_out=False,
        ),
        "base_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.4},
            time_out=False,
        ),
        "time_out": TerminationTermCfg(
            func=mjlab_mdp.time_out, time_out=True,
        ),
    })

    # --- Events ---
    # Keep push_robot, remove terrain/DR events
    for ev in ["foot_friction", "encoder_bias", "base_com", "out_of_terrain_bounds"]:
        cfg.events.pop(ev, None)

    cfg.events["reset_ball"] = EventTermCfg(
        func=reset_ball_local_frame,
        mode="reset",
        params={
            "ball_name": "ball",
            "dist_range": (2.0, 4.0),
            "lateral_range": (-0.5, 0.5),
            "height_range": (0.1, 0.8),
            "speed_range": (2.0, 6.0),
        },
    )

    # --- Curriculum ---
    cfg.curriculum.clear()

    # --- Episode length ---
    cfg.episode_length_s = 1e9 if play else 3.0

    if play:
        cfg.events.pop("push_robot", None)

    return cfg


def goalkeeper_play_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True, num_envs=1)
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    return cfg
```

- [ ] **Step 2: Commit env config**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat: add goalkeeper env config (flat, headless T1, ball local frame)"
```

---

## Task 9: Write AMP Config and Register Task

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py`
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/__init__.py`

- [ ] **Step 1: Write goalkeeper_amp_cfg.py**

`SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py`:
```python
"""AMPRunnerCfg for SimpleGoalKeeper training."""
from __future__ import annotations
from pathlib import Path

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import AMPPPOAlgorithmCfg, AMPRunnerCfg, RslRlPpoActorCriticCfg
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

_MOTIONS_DIR = Path(__file__).parent.parent / "motions" / "data"

# Lower-body bodies the AMP discriminator observes — feet, shanks, waist.
# This rewards natural leg movement without constraining arms.
_AMP_BODY_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "Shank_Left",
    "Shank_Right",
    "Waist",
    "Trunk",
]


def goalkeeper_amp_runner_cfg() -> AMPRunnerCfg:
    motion_files = sorted(str(p) for p in _MOTIONS_DIR.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No NPZ motion files found in {_MOTIONS_DIR}")

    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        experiment_name="simple_goalkeeper",
        run_name="phase1_feet",
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=AMPPPOAlgorithmCfg(
            class_name="AMPPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        amp_data=MotionDatasetCfg(
            motion_files=motion_files,
            body_names=_AMP_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name="Trunk",
        ),
        amp_discr_hidden_dims=[512, 256, 128],
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.7,
        amp_min_normalized_std=0.05,
    )
```

- [ ] **Step 2: Write tasks/__init__.py to register task**

`SimpleGoalKeeper/src/simple_goalkeeper/tasks/__init__.py`:
```python
"""Register SimpleGoalKeeper tasks in the mjlab task registry."""
from mjlab.tasks.registry import register_mjlab_task

from simple_goalkeeper.tasks.goalkeeper_env_cfg import (
    goalkeeper_env_cfg,
    goalkeeper_play_env_cfg,
)
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg

register_mjlab_task(
    task_id="Mjlab-BeyondAMP-Goalkeeper-T1",
    env_cfg=goalkeeper_env_cfg(),
    play_env_cfg=goalkeeper_play_env_cfg(),
    rl_cfg=goalkeeper_amp_runner_cfg(),
)
```

- [ ] **Step 3: Commit**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/tasks/
git commit -m "feat: add AMPRunnerCfg and register Mjlab-BeyondAMP-Goalkeeper-T1 task"
```

---

## Task 10: Write Train and Play Scripts

**Files:**
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/scripts/train.py`
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py`

- [ ] **Step 1: Write train.py**

`SimpleGoalKeeper/src/simple_goalkeeper/scripts/train.py`:
```python
"""Train SimpleGoalKeeper with beyondAMP."""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner

import simple_goalkeeper.tasks  # noqa: F401 — registers task


TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1"


def main(
    num_envs: int = 4096,
    log_root: str = "logs/rsl_rl",
    device: str = "cuda:0",
    resume: str | None = None,
) -> None:
    """Train SimpleGoalKeeper goalkeeper policy.

    Args:
        num_envs:  Number of parallel environments.
        log_root:  Directory for logs and checkpoints.
        device:    Torch device.
        resume:    Path to checkpoint to resume from (optional).
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    env_cfg = load_env_cfg(TASK_ID)
    agent_cfg: AMPRunnerCfg = load_rl_cfg(TASK_ID)  # type: ignore
    env_cfg.scene.num_envs = num_envs

    log_dir = (
        Path(log_root) / agent_cfg.experiment_name /
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

    runner = AMPOnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)

    if resume:
        runner.load(resume)
        print(f"[INFO] Resumed from {resume}")

    print(f"[INFO] Logging to: {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
```

- [ ] **Step 2: Write play.py**

`SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py`:
```python
"""Play / evaluate a trained SimpleGoalKeeper policy."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner

import simple_goalkeeper.tasks  # noqa: F401


TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1"


def main(
    checkpoint: str,
    num_envs: int = 1,
    device: str = "cuda:0",
) -> None:
    """Run a trained goalkeeper policy.

    Args:
        checkpoint: Path to .pt checkpoint file.
        num_envs:   Number of parallel environments (1 for visual playback).
        device:     Torch device.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg: AMPRunnerCfg = load_rl_cfg(TASK_ID)  # type: ignore
    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="human")
    env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    runner = AMPOnPolicyRunner(env, {}, log_dir=None, device=device)
    runner.load(checkpoint)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")

    policy = runner.get_inference_policy(device=device)
    obs, _ = env.reset()
    while True:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, done, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
```

- [ ] **Step 3: Commit scripts**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add src/simple_goalkeeper/scripts/train.py src/simple_goalkeeper/scripts/play.py
git commit -m "feat: add train and play scripts"
```

---

## Task 11: Write CLAUDE.md

**Files:**
- Create: `SimpleGoalKeeper/CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

`SimpleGoalKeeper/CLAUDE.md`:
```markdown
# SimpleGoalKeeper

Standalone foot-based goalkeeper training for Booster T1 using beyondAMP.

## Phase 1 Scope

**Feet only.** The robot uses its feet to intercept and deflect the ball.
No hand rewards, no arm observation terms. Arms are actuated (for balance)
but not rewarded for ball interaction.

## Frame Convention

All direction-dependent rewards use the **robot's local +X frame**:
- +X = robot forward direction (direction robot faces)
- Ball always spawns at positive local X (in front of the robot)
- Ball flies toward robot (negative local X velocity)
- Success = ball has positive local X velocity (deflected back to field)

This makes the goalkeeper task robot-orientation-independent — the policy
works regardless of where in the world the robot is facing.

## beyondAMP Location

`./beyondAMP/source/` — cloned from https://github.com/Renforce-Dynamics/beyondAMP.git

## Motion Files

Expected at `src/simple_goalkeeper/motions/data/*.npz`
- Format: `fps`, `joint_pos (T,21)`, `joint_vel (T,21)`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- Joint order: 21-DOF headless T1 (excludes AAHead_yaw and Head_pitch)
- Convert PKL files: `uv run sgk_convert --input-dir /path/to/Motions`

## Running

```bash
# Train
uv run sgk_train --num-envs 4096

# Play (with checkpoint)
uv run sgk_play --checkpoint logs/rsl_rl/simple_goalkeeper/DATE/model_ITER.pt

# Convert motions
uv run sgk_convert --input-dir /path/to/pkl/files
```

## Reward Intent (Phase 1)

1. `foot_to_ball` (3.0) — pull feet toward ball, dense from episode start
2. `ball_vx_reduction` (5.0) — reward neutralising incoming velocity
3. `ball_positive_vx` (10.0) — primary success: ball deflected back (+X)
4. `posture` (1.0) — stay near default pose
5. `ang_vel_xy` (-0.1) — keep upright
6. `action_rate_l2` (-0.3) — smooth actions
7. `dof_vel` (-0.001) — penalise excessive joint velocity

## Do NOT

- Add hand-based rewards — this is Phase 1 (feet only)
- Import from `Imitationlearningbooster`, `BoosterT1mjlab`, or `HandWavingMotion`
- Modify files inside `beyondAMP/` — treat as frozen upstream
```

- [ ] **Step 2: Commit CLAUDE.md**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with Phase 1 scope and frame convention notes"
```

---

## Task 12: Integration Test — Import and Env Instantiation

**Files:** No new files — smoke-test that everything works together.

- [ ] **Step 1: Test task registration**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
import simple_goalkeeper.tasks
from mjlab.tasks.registry import list_tasks
tasks = list_tasks()
assert 'Mjlab-BeyondAMP-Goalkeeper-T1' in tasks, f'Task not registered, got: {tasks}'
print('Task registered OK:', tasks)
"
```

Expected: `Task registered OK: ['Mjlab-BeyondAMP-Goalkeeper-T1']`

- [ ] **Step 2: Test env config builds without error**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg
cfg = goalkeeper_env_cfg(num_envs=4)
print('Env config OK')
print('  obs actor terms:', list(cfg.observations['actor'].terms.keys()))
print('  rewards:', list(cfg.rewards.keys()))
print('  terminations:', list(cfg.terminations.keys()))
print('  events:', list(cfg.events.keys()))
print('  has amp obs:', 'amp' in cfg.observations)
"
```

Expected output contains:
```
Env config OK
  obs actor terms: ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'ball_pos_b', 'ball_vel_b', 'left_foot_pos_b', 'right_foot_pos_b']
  rewards: ['foot_to_ball', 'ball_vx_reduction', 'ball_positive_vx', 'posture', 'ang_vel_xy', 'action_rate_l2', 'dof_vel']
  has amp obs: True
```

- [ ] **Step 3: Test AMP runner cfg**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg
cfg = goalkeeper_amp_runner_cfg()
print('AMP runner cfg OK')
print('  experiment:', cfg.experiment_name)
print('  motion files:', len(cfg.amp_data.motion_files), 'files')
print('  amp_reward_coef:', cfg.amp_reward_coef)
"
```

Expected: `motion files: 8 files`

- [ ] **Step 4: Final commit**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
git add .
git commit -m "test: verified imports, env config, and AMP runner cfg"
```

---

## Self-Review Checklist

- [x] Ball spawns in robot local +X frame (`events.py::reset_ball_local_frame`)
- [x] Full visibility system ported (`observations.py::_ball_visibility`)
- [x] All rewards use robot local frame for direction (`rewards.py::_robot_x_axis_w`)
- [x] beyondAMP AMP obs group added (`amp_obs_basic_group()` in env config)
- [x] Motion conversion script handles xyzw→wxyz, floor correction, 21-DOF strip
- [x] No imports from `Imitationlearningbooster`/`BoosterT1mjlab`/`HandWavingMotion`
- [x] play config sets `auto_reset=True`, removes push_robot
- [x] `action_rate_l2` uses `mjlab_mdp.action_rate_l2` (confirmed exists, uses `env.action_manager.prev_action`)
- [x] `dof_vel` uses `mjlab_mdp.joint_vel_l2` (confirmed exists)
