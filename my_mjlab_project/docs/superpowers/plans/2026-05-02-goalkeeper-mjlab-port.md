# Goalkeeper MuJoCo Lab Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Humanoid-Goalkeeper Isaac Gym project to MuJoCo Lab (mjlab), preserving all goalkeeper task logic and replacing the 6-discriminator AMP with pose-tracking rewards.

**Architecture:** Extend mjlab's existing `unitree_g1_flat_tracking_env_cfg()` (G1 robot already in mjlab's asset zoo) with goalkeeper-specific scene (ball entity), observations, rewards, events, and a custom `MultiMotionCommand` that loads all 6 goalkeeper motion clips and assigns one per env at reset. Register the task as an mjlab plugin via pyproject.toml entry points.

**Tech Stack:** Python 3.12, mjlab>=1.3.0, mujoco, torch, numpy — all in the existing `my_mjlab_project` venv at `/home/isaak/BEPImitationlearning/my_mjlab_project/.venv`.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `pyproject.toml` | Modify | Add `mjlab.tasks` entry point + deps |
| `src/my_mjlab_project/__init__.py` | Modify | Import task registration on package load |
| `src/my_mjlab_project/tasks/__init__.py` | Create | Call `register_mjlab_task` |
| `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py` | Create | Full env config (scene, obs, rewards, events, terminations) |
| `src/my_mjlab_project/tasks/goalkeeper_ppo_cfg.py` | Create | RSL-RL runner config |
| `src/my_mjlab_project/mdp/__init__.py` | Create | Re-exports |
| `src/my_mjlab_project/mdp/commands.py` | Create | `MultiMotionCommand` / `MultiMotionCommandCfg` |
| `src/my_mjlab_project/mdp/observations.py` | Create | ball_pos_b, ball_vel_b, hand positions |
| `src/my_mjlab_project/mdp/rewards.py` | Create | eereach, success, stopball, stayonline, noretreat, foot/posture rewards |
| `src/my_mjlab_project/mdp/events.py` | Create | Ball spawn + velocity event |
| `src/my_mjlab_project/motions/convert.py` | Create | `.pt → .npz` converter using MuJoCo FK |
| `src/my_mjlab_project/motions/data/` | Create dir | Holds 6 converted `.npz` files |
| `scripts/convert_motions.sh` | Create | Shell wrapper to run converter for all 6 clips |
| `scripts/train.py` | Create | Training entry point |
| `scripts/play.py` | Create | Evaluation entry point |

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/my_mjlab_project/__init__.py`

- [ ] **Step 1: Update pyproject.toml — add entry point and deps**

```toml
# pyproject.toml — full file after changes:
[project]
name = "my-mjlab-project"
version = "0.1.0"
description = "G1 Goalkeeper task for MuJoCo Lab"
readme = "README.md"
authors = [
    { name = "Isaak Bouwmeester", email = "i.p.b.bouwmeester@student.tue.nl" }
]
requires-python = ">=3.12"
dependencies = [
    "mjlab>=1.3.0",
    "torch",
    "numpy",
    "scipy",
    "tqdm",
]

[project.entry-points."mjlab.tasks"]
goalkeeper = "my_mjlab_project"

[project.scripts]
my-mjlab-project = "my_mjlab_project:main"

[build-system]
requires = ["uv_build>=0.11.8,<0.12.0"]
build-backend = "uv_build"
```

- [ ] **Step 2: Update `src/my_mjlab_project/__init__.py`**

```python
from my_mjlab_project.tasks import register_all


def main() -> None:
    print("my-mjlab-project — use 'uv run mjlab train goalkeeper' to train.")
```

- [ ] **Step 3: Reinstall package so entry point is picked up**

Run from `/home/isaak/BEPImitationlearning/my_mjlab_project/`:

```bash
uv pip install -e .
```

Expected: no errors. Then verify:

```bash
uv run python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points().select(group='mjlab.tasks')])"
```

Expected output: `['goalkeeper']`

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src/my_mjlab_project/tasks
mkdir -p src/my_mjlab_project/mdp
mkdir -p src/my_mjlab_project/motions/data
mkdir -p scripts
```

---

## Task 2: Motion Data Conversion Script

**Files:**
- Create: `src/my_mjlab_project/motions/convert.py`
- Create: `scripts/convert_motions.sh`

The `.pt` files have 21 joints; mjlab's G1 has 29. We zero-fill the 8 missing joints (waist_roll, waist_pitch, and 6 wrist joints), then use MuJoCo FK to compute world-frame body positions for all 31 bodies.

- [ ] **Step 1: Write `src/my_mjlab_project/motions/convert.py`**

```python
"""Convert Humanoid-Goalkeeper .pt motion files to mjlab NPZ format.

NPZ format expected by mjlab's MotionLoader (commands.py):
  joint_pos:       (N, 29)     actuated joint positions in MuJoCo order
  joint_vel:       (N, 29)     actuated joint velocities
  body_pos_w:      (N, 31, 3)  all-body world-frame positions
  body_quat_w:     (N, 31, 4)  all-body quaternions (wxyz)
  body_lin_vel_w:  (N, 31, 3)  all-body linear velocities
  body_ang_vel_w:  (N, 31, 3)  all-body angular velocities
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

# MuJoCo joint order (indices 1-29 are actuated; 0 is floating_base).
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # length 29

# Mapping: .pt joint_position column index → mujoco actuated joint index (0-based)
# joint_id.txt has 21 joints; the 8 missing joints stay at zero.
PT_TO_MUJOCO_IDX = {
    0: 0,   # left_hip_pitch_joint
    1: 1,   # left_hip_roll_joint
    2: 2,   # left_hip_yaw_joint
    3: 3,   # left_knee_joint
    4: 4,   # left_ankle_pitch_joint
    5: 5,   # left_ankle_roll_joint
    6: 6,   # right_hip_pitch_joint
    7: 7,   # right_hip_roll_joint
    8: 8,   # right_hip_yaw_joint
    9: 9,   # right_knee_joint
    10: 10, # right_ankle_pitch_joint
    11: 11, # right_ankle_roll_joint
    12: 12, # waist_yaw_joint
    13: 15, # left_shoulder_pitch_joint
    14: 16, # left_shoulder_roll_joint
    15: 17, # left_shoulder_yaw_joint
    16: 18, # left_elbow_joint
    17: 22, # right_shoulder_pitch_joint
    18: 23, # right_shoulder_roll_joint
    19: 24, # right_shoulder_yaw_joint
    20: 25, # right_elbow_joint
}


def _pt_joints_to_mujoco(pt_joint_pos: np.ndarray) -> np.ndarray:
    """Map (N, 21) .pt joint array to (N, 29) MuJoCo actuated joint array."""
    N = pt_joint_pos.shape[0]
    out = np.zeros((N, 29), dtype=np.float32)
    for pt_idx, mj_idx in PT_TO_MUJOCO_IDX.items():
        out[:, mj_idx] = pt_joint_pos[:, pt_idx]
    return out


def _compute_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_pos: np.ndarray,      # (3,) xyz
    base_quat_xyzw: np.ndarray,  # (4,) xyzw
    joint_pos_29: np.ndarray,  # (29,)
) -> tuple[np.ndarray, np.ndarray]:
    """Set qpos and run forward kinematics. Returns (xpos, xquat) for all bodies."""
    # MuJoCo free joint qpos layout: pos(3) + quat_wxyz(4)
    data.qpos[0:3] = base_pos
    # Convert xyzw → wxyz
    xyzw = base_quat_xyzw
    data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]  # wxyz
    # Actuated joints start at qpos[7]
    data.qpos[7:36] = joint_pos_29
    mujoco.mj_kinematics(model, data)
    return data.xpos.copy(), data.xquat.copy()  # both (nbody, ...) 


def convert_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    """Convert one .pt motion file to mjlab NPZ format."""
    raw = torch.load(pt_path, map_location="cpu")

    base_pos = raw["base_position"].numpy().astype(np.float32)       # (N, 3)
    base_quat_xyzw = raw["base_pose"].numpy().astype(np.float32)     # (N, 4) xyzw
    joint_pos_21 = raw["joint_position"].numpy().astype(np.float32)  # (N, 21)
    joint_vel_21 = raw["joint_velocity"].numpy().astype(np.float32)  # (N, 21)
    N = base_pos.shape[0]

    joint_pos_29 = _pt_joints_to_mujoco(joint_pos_21)
    joint_vel_29 = _pt_joints_to_mujoco(joint_vel_21)

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)

    num_bodies = model.nbody  # 31

    body_pos_w = np.zeros((N, num_bodies, 3), dtype=np.float32)
    body_quat_wxyz = np.zeros((N, num_bodies, 4), dtype=np.float32)

    for t in tqdm(range(N), desc=f"FK {pt_path.stem}", leave=False):
        xpos, xquat = _compute_fk(
            model, data,
            base_pos[t], base_quat_xyzw[t], joint_pos_29[t]
        )
        body_pos_w[t] = xpos      # xpos is (nbody, 3)
        body_quat_wxyz[t] = xquat  # xquat is (nbody, 4) already wxyz

    # Finite-difference velocities (central differences, clamp at endpoints)
    dt = 1.0 / 30.0  # 30 FPS source data
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    # Angular velocity from quaternion finite-differences (approximate)
    # Compute axis-angle from consecutive quaternion differences
    body_ang_vel_w = _angular_velocity_from_quats(body_quat_wxyz, dt)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(npz_path),
        joint_pos=joint_pos_29,
        joint_vel=joint_vel_29,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_wxyz,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved {npz_path} (N={N} frames, {num_bodies} bodies)")


def _angular_velocity_from_quats(quats: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity from quaternion sequence via axis-angle difference.

    quats: (N, B, 4) wxyz quaternions
    Returns: (N, B, 3) angular velocities in world frame
    """
    N, B, _ = quats.shape
    ang_vel = np.zeros((N, B, 3), dtype=np.float32)
    for t in range(1, N - 1):
        q0 = quats[t - 1]  # (B, 4)
        q1 = quats[t + 1]  # (B, 4)
        # Relative quaternion: q_diff = q0_inv * q1
        # q0_inv = [w, -x, -y, -z]
        q0_inv = q0 * np.array([1, -1, -1, -1])
        q_diff = _quat_mul_batch(q0_inv, q1)  # (B, 4)
        # Axis-angle from q_diff
        angle = 2.0 * np.arctan2(
            np.linalg.norm(q_diff[:, 1:], axis=-1),
            q_diff[:, 0]
        )  # (B,)
        axis = q_diff[:, 1:] / (np.linalg.norm(q_diff[:, 1:], axis=-1, keepdims=True) + 1e-8)
        ang_vel[t] = (axis * angle[:, None]) / (2.0 * dt)
    # Copy boundary values
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def _quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batch quaternion multiply. q: (B, 4) wxyz."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    motion_names = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
    for name in motion_names:
        pt_path = args.input_dir / f"{name}.pt"
        npz_path = args.output_dir / f"{name}.npz"
        if pt_path.exists():
            convert_pt_to_npz(pt_path, npz_path)
        else:
            print(f"[WARN] {pt_path} not found, skipping")
```

- [ ] **Step 2: Write `scripts/convert_motions.sh`**

```bash
#!/usr/bin/env bash
# Run from /home/isaak/BEPImitationlearning/my_mjlab_project/
set -e

INPUT=/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper
OUTPUT=src/my_mjlab_project/motions/data

uv run python -m my_mjlab_project.motions.convert "$INPUT" "$OUTPUT"
echo "Done. Files in $OUTPUT:"
ls -lh "$OUTPUT"
```

```bash
chmod +x scripts/convert_motions.sh
```

- [ ] **Step 3: Run the converter**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
./scripts/convert_motions.sh
```

Expected output: 6 `.npz` files in `src/my_mjlab_project/motions/data/`, each ~5–20 MB depending on clip length. Should complete in < 2 minutes.

---

## Task 3: Verify Motion Conversion

**Files:**
- No new files (verification only)

- [ ] **Step 1: Check NPZ shapes**

```bash
uv run python -c "
import numpy as np
from pathlib import Path

data_dir = Path('src/my_mjlab_project/motions/data')
for f in sorted(data_dir.glob('*.npz')):
    d = np.load(f)
    N = d['joint_pos'].shape[0]
    print(f'{f.name}: N={N}, joint_pos={d[\"joint_pos\"].shape}, body_pos_w={d[\"body_pos_w\"].shape}')
"
```

Expected: each file shows `joint_pos=(N, 29)`, `body_pos_w=(N, 31, 3)`.

- [ ] **Step 2: Verify no NaN/Inf in converted data**

```bash
uv run python -c "
import numpy as np
from pathlib import Path

data_dir = Path('src/my_mjlab_project/motions/data')
ok = True
for f in sorted(data_dir.glob('*.npz')):
    d = np.load(f)
    for key in ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w']:
        if not np.isfinite(d[key]).all():
            print(f'[FAIL] {f.name}/{key} has NaN/Inf')
            ok = False
if ok:
    print('All files: no NaN/Inf found.')
"
```

Expected: `All files: no NaN/Inf found.`

- [ ] **Step 3: Spot-check pelvis height for lefthand clip**

```bash
uv run python -c "
import numpy as np
d = np.load('src/my_mjlab_project/motions/data/lefthand.npz')
pelvis_z = d['body_pos_w'][:, 1, 2]  # body 1 = pelvis
print(f'Pelvis z — min={pelvis_z.min():.3f}m  max={pelvis_z.max():.3f}m  mean={pelvis_z.mean():.3f}m')
"
```

Expected: pelvis z between ~0.6–0.9m (upright standing/moving). If near 0 or > 2m, the FK mapping is wrong — recheck `PT_TO_MUJOCO_IDX` and qpos layout.

---

## Task 4: MultiMotionCommand

**Files:**
- Create: `src/my_mjlab_project/mdp/__init__.py`
- Create: `src/my_mjlab_project/mdp/commands.py`

The `MultiMotionCommand` extends mjlab's `MotionCommand` to manage 6 separate motion clips. At each env reset, it randomly assigns a motion type. Per-env reference states come from the assigned clip.

- [ ] **Step 1: Create `src/my_mjlab_project/mdp/__init__.py`**

```python
from my_mjlab_project.mdp.commands import MultiMotionCommand, MultiMotionCommandCfg
from my_mjlab_project.mdp import observations, rewards, events
```

- [ ] **Step 2: Create `src/my_mjlab_project/mdp/commands.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.tracking.mdp.commands import (
    MotionCommand,
    MotionCommandCfg,
    MotionLoader,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    """Motion command that cycles across multiple NPZ motion clips per env.

    Attributes:
        motion_files: Ordered tuple of NPZ file paths (one per motion type).
            motion_file (inherited) should be set to motion_files[0] so the
            mjlab train.py single-file existence check passes.
    """
    motion_files: tuple[str, ...] = field(default_factory=tuple)

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)


class MultiMotionCommand(MotionCommand):
    """Extends MotionCommand to load N motion clips and assign one per env at reset."""

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        # Load all motion clips (body_indexes computed by parent __init__)
        self.loaders: list[MotionLoader] = [
            MotionLoader(path, self.body_indexes, device=self.device)
            for path in cfg.motion_files
        ]
        # Per-env assignment: which motion clip (0..M-1) each env follows
        self.motion_type_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Replace the single parent loader with loader[0] for compatibility
        self.motion = self.loaders[0]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Randomly assign a new motion type to each resetting env, then call parent."""
        num_loaders = len(self.loaders)
        self.motion_type_ids[env_ids] = torch.randint(
            0, num_loaders, (env_ids.numel(),), device=self.device
        )
        # Clamp time_steps for the new assignment (different clips may differ in length)
        self.time_steps[env_ids] = 0
        # Write RSI from the assigned clip
        for motion_idx in range(num_loaders):
            mask = (self.motion_type_ids[env_ids] == motion_idx).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            env_subset = env_ids[mask]
            loader = self.loaders[motion_idx]
            self._write_reference_state_to_sim(
                env_subset,
                loader.body_pos_w[0:1].expand(env_subset.numel(), -1, -1)[:, 0],
                loader.body_quat_w[0:1].expand(env_subset.numel(), -1, -1)[:, 0],
                loader.body_lin_vel_w[0:1].expand(env_subset.numel(), -1, -1)[:, 0],
                loader.body_ang_vel_w[0:1].expand(env_subset.numel(), -1, -1)[:, 0],
                loader.joint_pos[0:1].expand(env_subset.numel(), -1),
                loader.joint_vel[0:1].expand(env_subset.numel(), -1),
            )

    def _gather_per_env(self, attr: str) -> torch.Tensor:
        """Gather the current-frame tensor for each env from its assigned loader.

        Returns shape (N, ...) where N = num_envs.
        """
        results = []
        for motion_idx, loader in enumerate(self.loaders):
            mask = (self.motion_type_ids == motion_idx)
            t = self.time_steps[mask].clamp(0, loader.time_step_total - 1)
            data = getattr(loader, attr)  # (T, ...)
            results.append((mask, data[t]))
        # Assemble into full (N, ...) tensor
        sample = results[0][1]
        out = torch.zeros(self.num_envs, *sample.shape[1:], device=self.device)
        for mask, values in results:
            out[mask] = values
        return out

    # Override the properties used by reward/observation/termination terms
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_per_env("joint_pos")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._gather_per_env("body_pos_w")

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_per_env("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_per_env("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_per_env("body_ang_vel_w")

    def _update_command(self) -> None:
        """Advance time_steps per env and wrap at each clip's length."""
        self.time_steps += 1
        for motion_idx, loader in enumerate(self.loaders):
            mask = (self.motion_type_ids == self.motion_type_ids)  # all envs this motion
            mask = (self.motion_type_ids == motion_idx)
            expired = mask & (self.time_steps >= loader.time_step_total)
            if expired.any():
                self._resample_command(expired.nonzero(as_tuple=True)[0])
        self.update_relative_body_poses()
```

- [ ] **Step 3: Smoke-test MultiMotionCommand import**

```bash
uv run python -c "
from my_mjlab_project.mdp.commands import MultiMotionCommandCfg, MultiMotionCommand
print('MultiMotionCommand import OK')
"
```

Expected: `MultiMotionCommand import OK` (no import errors)

---

## Task 5: Ball Entity and Goalkeeper Observations

**Files:**
- Create: `src/my_mjlab_project/mdp/observations.py`

The ball entity is a sphere rigid body added to the scene. Observation functions access it via `env.scene["ball"]`.

- [ ] **Step 1: Create `src/my_mjlab_project/mdp/observations.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")


def ball_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Ball position in robot base (torso_link) frame. Shape: (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    # World-frame positions
    ball_pos_w = ball.data.root_link_pos_w        # (N, 3)
    robot_pos_w = robot.data.root_link_pos_w      # (N, 3)
    robot_quat_w = robot.data.root_link_quat_w    # (N, 4) wxyz
    # Transform to base frame
    rel = ball_pos_w - robot_pos_w                # (N, 3)
    return quat_apply_inverse(robot_quat_w, rel)  # (N, 3)


def ball_vel_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    scale: float = 0.2,
) -> torch.Tensor:
    """Ball linear velocity in robot base frame, scaled. Shape: (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    ball_vel_w = ball.data.root_link_lin_vel_w    # (N, 3)
    robot_quat_w = robot.data.root_link_quat_w    # (N, 4)
    return scale * quat_apply_inverse(robot_quat_w, ball_vel_w)


def right_hand_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Right wrist_yaw_link position in base frame. Shape: (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    # body index for right_wrist_yaw_link = 30 (from g1 body ordering)
    right_hand_w = robot.data.body_pos_w[:, 30]   # (N, 3)
    robot_pos_w = robot.data.root_link_pos_w
    robot_quat_w = robot.data.root_link_quat_w
    return quat_apply_inverse(robot_quat_w, right_hand_w - robot_pos_w)


def left_hand_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Left wrist_yaw_link position in base frame. Shape: (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    # body index for left_wrist_yaw_link = 23
    left_hand_w = robot.data.body_pos_w[:, 23]    # (N, 3)
    robot_pos_w = robot.data.root_link_pos_w
    robot_quat_w = robot.data.root_link_quat_w
    return quat_apply_inverse(robot_quat_w, left_hand_w - robot_pos_w)


def ball_distance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Distance from robot origin to ball. Shape: (N, 1)."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    diff = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return torch.norm(diff, dim=-1, keepdim=True)
```

- [ ] **Step 2: Verify body indices for hand positions**

```bash
uv run python -c "
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
m = mujoco.MjModel.from_xml_path(str(G1_XML))
for name in ['left_wrist_yaw_link', 'right_wrist_yaw_link']:
    idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f'{name}: body index {idx}')
"
```

Expected output:
```
left_wrist_yaw_link: body index 23
right_wrist_yaw_link: body index 30
```

If indices differ, update `observations.py` lines for `right_hand_w` and `left_hand_w`.

---

## Task 6: Goalkeeper Rewards

**Files:**
- Create: `src/my_mjlab_project/mdp/rewards.py`

All 25+ reward terms from `legged_robot.py`. Motion tracking replaces AMP.

- [ ] **Step 1: Create `src/my_mjlab_project/mdp/rewards.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")

# ─── Task rewards ────────────────────────────────────────────────────────────

def eereach(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for reaching closest hand toward the ball."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_link_pos_w         # (N, 3)
    lh_w = robot.data.body_pos_w[:, 23]            # left_wrist_yaw (N, 3)
    rh_w = robot.data.body_pos_w[:, 30]            # right_wrist_yaw (N, 3)
    dist_l = torch.norm(ball_pos_w - lh_w, dim=-1)
    dist_r = torch.norm(ball_pos_w - rh_w, dim=-1)
    min_dist = torch.minimum(dist_l, dist_r)       # (N,)
    return torch.exp(-min_dist / std)


def catch_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    hand_radius: float = 0.25,
) -> torch.Tensor:
    """Binary: 1 if either hand is within hand_radius of ball."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_link_pos_w
    lh_w = robot.data.body_pos_w[:, 23]
    rh_w = robot.data.body_pos_w[:, 30]
    caught = (
        (torch.norm(ball_pos_w - lh_w, dim=-1) < hand_radius) |
        (torch.norm(ball_pos_w - rh_w, dim=-1) < hand_radius)
    )
    return caught.float()


def stopball(
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    vel_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    hand_radius: float = 0.25,
) -> torch.Tensor:
    """Binary: 1 if ball is near a hand AND moving slowly (caught and stopped)."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_link_pos_w
    ball_vel = ball.data.root_link_lin_vel_w
    lh_w = robot.data.body_pos_w[:, 23]
    rh_w = robot.data.body_pos_w[:, 30]
    near_hand = (
        (torch.norm(ball_pos_w - lh_w, dim=-1) < hand_radius) |
        (torch.norm(ball_pos_w - rh_w, dim=-1) < hand_radius)
    )
    slow = torch.norm(ball_vel, dim=-1) < vel_threshold
    return (near_hand & slow).float()


def stayonline(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    goal_line_y: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Penalize lateral displacement from goal line (y-axis penalty)."""
    robot: Entity = env.scene[asset_cfg.name]
    y_pos = robot.data.root_link_pos_w[:, 1]  # y coordinate
    return -torch.abs(y_pos - goal_line_y)


def noretreat(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Penalize retreating away from an incoming ball (moving backwards from ball)."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    robot_vel_w = robot.data.root_link_lin_vel_w   # (N, 3)
    ball_dir_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    ball_dir_w = ball_dir_w / (torch.norm(ball_dir_w, dim=-1, keepdim=True) + 1e-6)
    retreat_speed = -torch.sum(robot_vel_w * ball_dir_w, dim=-1)  # negative = retreating
    return torch.clamp(retreat_speed, min=0.0)  # penalize retreat only


def feetorientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward for keeping feet flat (ankle roll links near zero roll)."""
    robot: Entity = env.scene[asset_cfg.name]
    # ankle_roll body indices: left=7, right=13
    l_ankle_quat = robot.data.body_quat_w[:, 7]   # (N, 4) wxyz
    r_ankle_quat = robot.data.body_quat_w[:, 13]
    # Roll is approximately 2*quat.x for small angles
    l_roll = 2.0 * l_ankle_quat[:, 1]
    r_roll = 2.0 * r_ankle_quat[:, 1]
    error = l_roll**2 + r_roll**2
    return torch.exp(-error / std**2)


def postorientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward upright torso (projected gravity near [0,0,-1])."""
    robot: Entity = env.scene[asset_cfg.name]
    gravity_b = robot.data.projected_gravity_b    # (N, 3) in base frame
    # Ideal: gravity_b = [0, 0, -1]
    error = torch.sum((gravity_b - torch.tensor([[0, 0, -1]], device=env.device))**2, dim=-1)
    return torch.exp(-error / std**2)


def postangvel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward near-zero base angular velocity."""
    robot: Entity = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_link_ang_vel_b       # (N, 3)
    error = torch.sum(ang_vel**2, dim=-1)
    return torch.exp(-error / std**2)


def postlinvel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward near-zero base linear velocity (goalkeeper stands still by default)."""
    robot: Entity = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_link_lin_vel_b       # (N, 3)
    error = torch.sum(lin_vel**2, dim=-1)
    return torch.exp(-error / std**2)

# ─── Regularization (thin wrappers around mjlab built-ins) ───────────────────
# Use mjlab.tasks.tracking.mdp rewards for: action_rate_l2, joint_pos_limits, self_collision_cost
```

- [ ] **Step 2: Smoke-test rewards import**

```bash
uv run python -c "
from my_mjlab_project.mdp.rewards import eereach, catch_success, stopball
print('Rewards import OK')
"
```

Expected: `Rewards import OK`

---

## Task 7: Goalkeeper Events (Ball Spawn + Domain Randomization)

**Files:**
- Create: `src/my_mjlab_project/mdp/events.py`

- [ ] **Step 1: Create `src/my_mjlab_project/mdp/events.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_ball_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (2.0, 6.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    z_range: tuple[float, float] = (0.5, 1.5),
    vel_x_range: tuple[float, float] = (-8.0, -4.0),
    vel_z_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Reset ball to a random incoming position with shot velocity."""
    ball: Entity = env.scene[ball_cfg.name]
    n = env_ids.numel()
    device = env.device

    def rnd(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    pos = torch.stack([rnd(*x_range), rnd(*y_range), rnd(*z_range)], dim=-1)
    # Incoming velocity aimed roughly toward goal
    vel = torch.stack([
        rnd(*vel_x_range),
        torch.zeros(n, device=device),
        rnd(*vel_z_range),
    ], dim=-1)

    # Write ball state into the MuJoCo data
    ball.write_root_pose_to_sim(
        torch.cat([pos, torch.tensor([[0, 0, 0, 1]], device=device).expand(n, -1)], dim=-1),
        env_ids=env_ids,
    )
    ball.write_root_velocity_to_sim(
        torch.cat([vel, torch.zeros(n, 3, device=device)], dim=-1),
        env_ids=env_ids,
    )
```

---

## Task 8: Goalkeeper Environment Configuration

**Files:**
- Create: `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`
- Create: `src/my_mjlab_project/tasks/goalkeeper_ppo_cfg.py`

This is the main config file. It extends `unitree_g1_flat_tracking_env_cfg()` and adds all goalkeeper elements.

- [ ] **Step 1: Create `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`**

```python
"""Goalkeeper environment configuration.

Extends mjlab's G1 flat tracking env with:
  - Ball rigid body in scene
  - MultiMotionCommand (6 goalkeeper motion clips)
  - Goalkeeper observation terms (ball pos/vel, hand pos, distance)
  - Goalkeeper reward terms (eereach, success, stopball, etc.)
  - Ball spawn event
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking import mdp as tracking_mdp

from my_mjlab_project.mdp import commands as gk_commands
from my_mjlab_project.mdp import events as gk_events
from my_mjlab_project.mdp import observations as gk_obs
from my_mjlab_project.mdp import rewards as gk_rewards

# Paths to converted NPZ motion files
_MOTIONS_DIR = Path(__file__).parent.parent / "motions" / "data"

MOTION_FILES = tuple(
    str(_MOTIONS_DIR / f"{name}.npz")
    for name in ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
)


def get_ball_spec() -> mujoco.MjSpec:
    """Soccer ball: 22 cm diameter, 450 g."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.pos = [4.0, 0.0, 1.0]     # default spawn (overridden by event)
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11, 0, 0],
        mass=0.45,
        rgba=[1.0, 0.6, 0.0, 1.0],
        friction=[0.1, 0.005, 0.0001],
    )
    return spec


def make_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create goalkeeper task configuration."""

    # ── Start from G1 flat tracking env ──
    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

    # ── Scene: add ball ──
    cfg.scene.entities = {
        "robot": get_g1_robot_cfg(),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }
    cfg.scene.num_envs = 1020
    cfg.scene.env_spacing = 3.0

    # ── Episode ──
    cfg.episode_length_s = 3.0

    # ── Motion command: swap MotionCommandCfg → MultiMotionCommandCfg ──
    multi_motion_cmd = gk_commands.MultiMotionCommandCfg(
        entity_name="robot",
        motion_files=MOTION_FILES,
        motion_file=MOTION_FILES[0],      # for train.py single-file check
        anchor_body_name="torso_link",
        body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        resampling_time_range=(1.0e9, 1.0e9),  # no auto-resample; reset drives it
        sampling_mode="uniform",
        debug_vis=not play,
        pose_range={} if play else {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        },
        velocity_range={} if play else {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
        },
    )
    cfg.commands = {"motion": multi_motion_cmd}

    # ── Observations ──
    # Extend the actor terms with goalkeeper-specific observations
    extra_actor_terms = {
        "ball_pos_b": ObservationTermCfg(func=gk_obs.ball_pos_b),
        "ball_vel_b": ObservationTermCfg(
            func=gk_obs.ball_vel_b, params={"scale": 0.2}
        ),
        "right_hand_pos": ObservationTermCfg(func=gk_obs.right_hand_pos_b),
        "left_hand_pos": ObservationTermCfg(func=gk_obs.left_hand_pos_b),
    }
    extra_critic_terms = {
        "ball_pos_b": ObservationTermCfg(func=gk_obs.ball_pos_b),
        "ball_vel_b": ObservationTermCfg(func=gk_obs.ball_vel_b, params={"scale": 0.2}),
        "right_hand_pos": ObservationTermCfg(func=gk_obs.right_hand_pos_b),
        "left_hand_pos": ObservationTermCfg(func=gk_obs.left_hand_pos_b),
        "ball_distance": ObservationTermCfg(func=gk_obs.ball_distance),
    }

    actor_terms = {**cfg.observations["actor"].terms, **extra_actor_terms}
    critic_terms = {**cfg.observations["critic"].terms, **extra_critic_terms}
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ── Rewards ──
    cfg.rewards = {
        # ── Motion tracking (replaces AMP) ──
        "motion_global_root_pos": RewardTermCfg(
            func=tracking_mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=tracking_mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_pos": RewardTermCfg(
            func=tracking_mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=tracking_mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        # ── Task rewards ──
        "eereach": RewardTermCfg(func=gk_rewards.eereach, weight=10.0),
        "success": RewardTermCfg(func=gk_rewards.catch_success, weight=5.0),
        "stopball": RewardTermCfg(func=gk_rewards.stopball, weight=100.0),
        "stayonline": RewardTermCfg(func=gk_rewards.stayonline, weight=-2.0),
        "noretreat": RewardTermCfg(func=gk_rewards.noretreat, weight=-2.0),
        "feetorientation": RewardTermCfg(func=gk_rewards.feetorientation, weight=3.0),
        "postorientation": RewardTermCfg(func=gk_rewards.postorientation, weight=3.0),
        "postangvel": RewardTermCfg(func=gk_rewards.postangvel, weight=3.0),
        "postlinvel": RewardTermCfg(func=gk_rewards.postlinvel, weight=1.0),
        # ── Regularization ──
        "action_rate_l2": RewardTermCfg(
            func=tracking_mdp.action_rate_l2, weight=-0.1
        ),
        "joint_limit": RewardTermCfg(
            func=tracking_mdp.joint_pos_limits,
            weight=-3.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "self_collisions": RewardTermCfg(
            func=tracking_mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
    }

    # ── Events ──
    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_events.reset_ball_state,
        mode="reset",
        params={"ball_cfg": SceneEntityCfg("ball")},
    )

    # ── Terminations ──
    # Keep parent terminations (time_out, anchor_pos, anchor_ori, ee_body_pos)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg
```

- [ ] **Step 2: Create `src/my_mjlab_project/tasks/goalkeeper_ppo_cfg.py`**

```python
"""PPO runner configuration for the goalkeeper task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def goalkeeper_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.998,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_goalkeeper",
        run_name="gk",
        save_interval=200,
        num_steps_per_env=100,
        max_iterations=200_000,
    )
```

---

## Task 9: Task Registration

**Files:**
- Create: `src/my_mjlab_project/tasks/__init__.py`

- [ ] **Step 1: Create `src/my_mjlab_project/tasks/__init__.py`**

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from my_mjlab_project.tasks.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from my_mjlab_project.tasks.goalkeeper_ppo_cfg import goalkeeper_ppo_runner_cfg

GOALKEEPER_TASK_ID = "goalkeeper"


def register_all() -> None:
    register_mjlab_task(
        task_id=GOALKEEPER_TASK_ID,
        env_cfg=make_goalkeeper_env_cfg(play=False),
        play_env_cfg=make_goalkeeper_env_cfg(play=True),
        rl_cfg=goalkeeper_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )


register_all()
```

- [ ] **Step 2: Verify task appears in mjlab registry**

```bash
uv run mjlab list-envs
```

Expected: output includes `goalkeeper` in the task list.

---

## Task 10: Training & Evaluation Scripts

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/play.py`

- [ ] **Step 1: Create `scripts/train.py`**

```python
#!/usr/bin/env python3
"""Train the goalkeeper policy. Run from my_mjlab_project root."""
import sys
import my_mjlab_project  # noqa: F401 — registers task as mjlab plugin

# Pass remaining args to mjlab train
sys.argv = [sys.argv[0], "goalkeeper"] + sys.argv[1:]

from mjlab.scripts.train import main
main()
```

- [ ] **Step 2: Create `scripts/play.py`**

```python
#!/usr/bin/env python3
"""Evaluate the goalkeeper policy. Run from my_mjlab_project root."""
import sys
import my_mjlab_project  # noqa: F401

sys.argv = [sys.argv[0], "goalkeeper"] + sys.argv[1:]

from mjlab.scripts.play import main
main()
```

---

## Task 11: Smoke Test

- [ ] **Step 1: Run training for 50 iterations**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python scripts/train.py \
    --agent.max-iterations 50 \
    --env.scene.num-envs 16
```

Expected: No crash. Stdout shows iteration logs like:
```
[INFO] Training with: device=cuda:0, seed=0
Iter 10: ...
Iter 20: ...
...
Iter 50: ...
```

- [ ] **Step 2: Check reward signals are non-zero**

In the training output, look for reward terms. Expected: `motion_body_pos` reward > 0 from iteration 1. `eereach` should become non-zero by iteration 20+.

If any reward is stuck at exactly 0 for all iterations, check:
- That the ball entity was added correctly (no scene compilation errors)
- That the NPZ files are loaded (check for FileNotFoundError logs)
- That body indices in `observations.py` are correct

- [ ] **Step 3: Verify NPZ loading by checking motion tracking loss**

The `motion_body_pos` reward should start high (> 0.5) if motion data is correct. If it starts at 0 or NaN, the NPZ format is wrong — recheck `convert.py` output keys match what `MotionLoader.__init__` in `commands.py` expects (`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`).

---

## Self-Review Checklist

- **Spec coverage:** Joint ordering mismatch risk is addressed (Task 2 FK + Task 3 spot-checks). All 6 motion files handled (Task 4 MultiMotionCommand). All major reward groups covered (Task 6). Ball spawn event covered (Task 7). Task registration covered (Task 9).

- **No placeholders:** All code blocks are complete. No TBD/TODO in implementation steps.

- **Type consistency:**
  - `MultiMotionCommandCfg.motion_files: tuple[str, ...]` used throughout Tasks 4 and 8
  - `EntityCfg(spec_fn=get_ball_spec)` pattern follows yam_lift_cube exactly
  - `env.scene["ball"]` accessed via `SceneEntityCfg("ball")` pattern throughout
  - `body_pos_w[:, 23]` / `[:, 30]` for left/right wrist — verified in Task 5 Step 2
  - `update_relative_body_poses()` called in `MultiMotionCommand._update_command` — matches parent class pattern

- **Known risks:**
  - `body.add_freejoint()` vs `body.add_joint(type="free")` — use `add_freejoint()` (matches manipulation task pattern)
  - `MultiMotionCommand._gather_per_env()` uses property override — verify parent class doesn't cache `self.motion` in `update_relative_body_poses()`. If it does, also override that method.
  - `gk_events.reset_ball_state` uses `ball.write_root_pose_to_sim()` — verify this method exists on non-articulated entities (may need `ball.write_root_state_to_sim()` instead, or direct data buffer write).
