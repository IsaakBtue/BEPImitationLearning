# Goalkeeper Deployment — Complete Session Documentation

**Last updated:** 2026-05-26  
**Author:** Isaak Bouwmeester  
**Project:** BEP Imitation Learning — Booster T1 Goalkeeper  

This document is written for continuity. The next session should read every section. Nothing is summarised; everything that matters is written out in full.

---

## 1. What This Repository Is

The repository at `/home/isaak/BEPImitationlearning` is a BEP (bachelor end project) that adapts the **InternRobotics Humanoid-Goalkeeper** pipeline (originally for Unitree G1) to the **Booster Robotics T1** humanoid robot.

The upstream G1 pipeline lived in `Humanoid-Goalkeeper/` and used Isaac Gym + HIM-PPO. The T1 port uses MuJoCo Lab (mjlab) + standard PPO and is in `my_mjlab_project_booster_t1/`.

The deployment framework used is `booster_deploy/` which is a Booster Robotics open-source package for running trained policies on the T1 hardware (sim2sim via MuJoCo, or real robot via ROS 2 + Booster SDK).

---

## 2. The Immovable Rule About `booster_deploy/`

**NEVER modify anything inside `/home/isaak/BEPImitationlearning/booster_deploy/`.**

This is constraint #7 in `CLAUDE.md`. The folder is a frozen upstream deployment framework. Read it for patterns; never change it. All goalkeeper-specific deployment code lives in `goalkeeper_deploy/` instead.

---

## 3. Trained Model Being Deployed

```
/home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1/logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt
```

This is an rsl_rl checkpoint dictionary with keys:
- `actor_state_dict` — the actor network weights + running normalizer stats
- `critic_state_dict`
- `optimizer_state_dict`
- `iter` — 2000 (training iteration at save time)
- `infos`

The actor network architecture:
- Input: 870 = 87 obs/step × 10 history steps
- Hidden layers: [512, 256, 128] with ELU activations
- Output: 23 (one per T1 joint)
- Observation normalizer: `EmpiricalNormalization` (running mean + std stored in `obs_normalizer._mean`, `obs_normalizer._std`, both shape `[1, 870]`)

The forward pass: `normalized = (obs - mean) / (std + 1e-2)` then MLP.

Other checkpoints available in the same run directory:
- `model_1000.pt` — iteration 1000
- `model_19000.pt` — iteration 19000 (longer training, may behave differently)

---

## 4. Training Environment (What the Model Learned)

Training lives in `my_mjlab_project_booster_t1/` and uses:
- Framework: **mjlab** (MuJoCo Lab, Booster's internal RL library built on MuJoCo + rsl_rl)
- Python: 3.12 via uv venv at `my_mjlab_project_booster_t1/.venv`
- Task: `goalkeeper_booster_t1`
- URDF: `src/my_mjlab_project_booster_t1/assets/booster_t1/T1_serial_clean.xml`

### 4.1 Observation Space (87 dims per step, history=10 → 870 input)

All observations are computed in the robot's **body frame**. Ordering matters exactly.

| Slice | Name | Dims | Notes |
|---|---|---|---|
| [0:3] | `base_lin_vel_b` | 3 | Robot base linear velocity in body frame. MuJoCo freejoint qvel[:3] is already body-frame. |
| [3:6] | `base_ang_vel_b` | 3 | Robot base angular velocity in body frame. MuJoCo freejoint qvel[3:6]. |
| [6:29] | `joint_pos_rel` | 23 | `joint_pos - TRAINING_DEFAULT_POS`. Uses training keyframe default, NOT deployment default. |
| [29:52] | `joint_vel` | 23 | Raw joint velocities. |
| [52:75] | `last_action` | 23 | Raw actor output from the previous inference step (before action_scale multiply). |
| [75:78] | `ball_pos_b` | 3 | Ball position in robot body frame. **Zeroed when ball is not visible** (catchstep warmup or random vanish). |
| [78:81] | `ball_vel_b` | 3 | Ball linear velocity in robot body frame. Also zeroed when ball hidden. |
| [81:84] | `left_hand_pos_b` | 3 | Left hand (`left_hand_link`) world position transformed to body frame. Always visible. |
| [84:87] | `right_hand_pos_b` | 3 | Right hand (`right_hand_link`) world position transformed to body frame. Always visible. |

**History format:** 10 consecutive observation vectors, flattened: `[obs_t-9, obs_t-8, ..., obs_t]`. Oldest first, newest last. Initialized as all zeros.

### 4.2 Joint Order

The training uses the MuJoCo XML joint order from `T1_serial_clean.xml`, which happens to be **identical** to `T1_23DOF_CFG.joint_names` in `booster_deploy`. This means no joint remapping is needed between training and deployment.

```
Index  Joint Name
  0    AAHead_yaw
  1    Head_pitch
  2    Left_Shoulder_Pitch
  3    Left_Shoulder_Roll
  4    Left_Elbow_Pitch
  5    Left_Elbow_Yaw
  6    Right_Shoulder_Pitch
  7    Right_Shoulder_Roll
  8    Right_Elbow_Pitch
  9    Right_Elbow_Yaw
  10   Waist
  11   Left_Hip_Pitch
  12   Left_Hip_Roll
  13   Left_Hip_Yaw
  14   Left_Knee_Pitch
  15   Left_Ankle_Pitch
  16   Left_Ankle_Roll
  17   Right_Hip_Pitch
  18   Right_Hip_Roll
  19   Right_Hip_Yaw
  20   Right_Knee_Pitch
  21   Right_Ankle_Pitch
  22   Right_Ankle_Roll
```

### 4.3 Training Default Joint Positions (TRAINING_DEFAULT_POS)

These come from `T1_STANDING_KEYFRAME` in `t1_constants.py`. They are the reference pose that defines `joint_pos_rel = joint_pos - default`. They are NOT the same as `T1_23DOF_CFG.default_joint_pos` in `booster_deploy` (which has zeros for legs).

```python
TRAINING_DEFAULT_POS = [
    0.0,   # AAHead_yaw
    0.0,   # Head_pitch
    -0.21, # Left_Shoulder_Pitch
    -0.41, # Left_Shoulder_Roll
    0.0,   # Left_Elbow_Pitch
    0.0,   # Left_Elbow_Yaw
    -0.11, # Right_Shoulder_Pitch
    1.07,  # Right_Shoulder_Roll   ← large positive: counterbalance arm
    -0.13, # Right_Elbow_Pitch
    0.21,  # Right_Elbow_Yaw
    0.0,   # Waist
    -0.3,  # Left_Hip_Pitch
    0.0,   # Left_Hip_Roll
    0.0,   # Left_Hip_Yaw
    0.6,   # Left_Knee_Pitch
    -0.3,  # Left_Ankle_Pitch
    0.0,   # Left_Ankle_Roll
    -0.3,  # Right_Hip_Pitch
    0.0,   # Right_Hip_Roll
    0.0,   # Right_Hip_Yaw
    0.6,   # Right_Knee_Pitch
    -0.3,  # Right_Ankle_Pitch
    0.0,   # Right_Ankle_Roll
]
```

### 4.4 Action Scale (per joint)

Actions from the actor are NOT joint targets directly. Final joint target = `TRAINING_DEFAULT_POS + action * ACTION_SCALE`.

Formula: `scale = 0.25 * effort_limit / stiffness` (for arm/leg joints).  
Head joints: `scale = effort_limit / stiffness` (no 0.25 factor).

```python
ACTION_SCALE = [
    7.0/20.0,           # AAHead_yaw    = 0.350
    7.0/20.0,           # Head_pitch    = 0.350
    0.25*36.0/15.0,     # L_Shoulder_Pitch  = 0.600
    0.25*36.0/15.0,     # L_Shoulder_Roll   = 0.600
    0.25*36.0/15.0,     # L_Elbow_Pitch     = 0.600
    0.25*36.0/15.0,     # L_Elbow_Yaw       = 0.600
    0.25*36.0/15.0,     # R_Shoulder_Pitch  = 0.600
    0.25*36.0/15.0,     # R_Shoulder_Roll   = 0.600
    0.25*36.0/15.0,     # R_Elbow_Pitch     = 0.600
    0.25*36.0/15.0,     # R_Elbow_Yaw       = 0.600
    0.25*40.0/80.0,     # Waist             = 0.125
    0.25*55.0/120.0,    # L_Hip_Pitch       ≈ 0.1146
    0.25*40.0/80.0,     # L_Hip_Roll        = 0.125
    0.25*40.0/80.0,     # L_Hip_Yaw         = 0.125
    0.25*65.0/200.0,    # L_Knee_Pitch      = 0.08125
    0.25*50.0/50.0,     # L_Ankle_Pitch     = 0.250
    0.25*50.0/40.0,     # L_Ankle_Roll      = 0.3125
    0.25*55.0/120.0,    # R_Hip_Pitch       ≈ 0.1146
    0.25*40.0/80.0,     # R_Hip_Roll        = 0.125
    0.25*40.0/80.0,     # R_Hip_Yaw         = 0.125
    0.25*65.0/200.0,    # R_Knee_Pitch      = 0.08125
    0.25*50.0/50.0,     # R_Ankle_Pitch     = 0.250
    0.25*50.0/40.0,     # R_Ankle_Roll      = 0.3125
]
```

### 4.5 PD Gains (Actuator Dynamics)

Training used `BuiltinPositionActuatorCfg` from mjlab with these stiffness values:

| Joint group | Effort limit | Stiffness (kp) | Damping (kd) |
|---|---|---|---|
| Head (AAHead_yaw, Head_pitch) | 7 | 20 | 1.273 |
| Arms (8 joints) | 36 | 15 | 0.955 |
| Waist | 40 | 80 | 5.093 |
| Hip Pitch (L/R) | 55 | 120 | 7.632 |
| Hip Roll, Yaw (L/R) | 40 | 80 | 5.093 |
| Knee Pitch (L/R) | 65 | 200 | 12.732 |
| Ankle Pitch (L/R) | 50 | 50 | 3.183 |
| Ankle Roll (L/R) | 50 | 40 | 2.546 |

Damping formula: `kd = 2 * 2.0 * stiffness / (2π * 10) ≈ 0.06366 * stiffness` (critical damping at 10 Hz natural frequency).

The deployment uses `ctrl = clip(kp * (target - pos) - kd * vel, -effort, effort)` PD control. The `kp` and `kd` match training to replicate the actuator dynamics as closely as possible.

### 4.6 Initial Orientation

The robot is initialised at `pos=(0,0,0.75)` with a **+90° yaw** rotation `q=[0.7071068, 0, 0, 0.7071068]` (w,x,y,z). This means the robot faces world +Y direction. Balls come from world +Y (y=5m) toward the robot at (0,0,0.75). In the robot's body frame, the ball approaches from +x_body, lateral is y_body, up is z_body.

### 4.7 Training Commands

```bash
# Training (from repo root)
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1
uv run python -m mjlab.scripts.train goalkeeper_booster_t1 --gpu-ids '[0]'

# Play (evaluation, requires checkpoint path)
uv run python -m mjlab.scripts.play goalkeeper_booster_t1 \
  --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt

# Play with motion overlay (shows reference ghost robot)
uv run python -m mjlab.scripts.play goalkeeper_booster_t1_withoverlay \
  --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt

# Smoke test (fast: 2 envs, 3 iterations)
uv run python -m mjlab.scripts.train goalkeeper_booster_t1 \
  --env.scene.num-envs 2 --agent.max-iterations 3 --gpu-ids '[0]'
```

---

## 5. The `booster_deploy` Framework

Located at `/home/isaak/BEPImitationlearning/booster_deploy/`. **Read-only.**

### 5.1 What It Does

Provides a unified abstraction for running trained control policies on:
- **Real robot** via ROS 2 (`/low_state` + `/joint_ctrl` topics)
- **MuJoCo simulation** (sim2sim)
- **Webots simulation** (internal only)

### 5.2 Directory Layout

```
booster_deploy/
├── booster_deploy/           # Core Python package
│   ├── controllers/
│   │   ├── base_controller.py       # BaseController, RobotData, Policy (ABCs)
│   │   ├── controller_cfg.py        # Config dataclasses
│   │   ├── mujoco_controller.py     # MujocoController (MuJoCo sim)
│   │   └── booster_robot_controller.py  # Real robot via Booster SDK + ROS 2
│   ├── robots/
│   │   └── booster.py           # K1_CFG and T1_23DOF_CFG (RobotCfg instances)
│   └── utils/
│       ├── registry.py          # _TASK_REGISTRY dict + register_task()
│       └── isaaclab/            # Config helpers (configclass decorator)
├── scripts/
│   └── deploy.py               # Entry point — discovers tasks/, dispatches controller
└── tasks/
    ├── locomotion/locomotion.py     # T1WalkControllerCfg, K1WalkControllerCfg
    └── beyond_mimic/beyond_mimic.py # BeyondMimicPolicy for dance/fight tasks
```

### 5.3 How deploy.py Works

```python
# From booster_deploy/scripts/deploy.py:
sys.path.append(".")   # adds CWD to sys.path (important!)
import tasks as tasks_pkg   # imports CWD/tasks/
for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
    __import__(mod_info.name)  # each __init__.py calls register_task()
task_cfg = get_task(args.task)
if args.mujoco:
    MujocoController(task_cfg).run()
```

**Critical:** `sys.path.append(".")` means the `tasks` package that gets imported is from wherever you run the script. This is why our `goalkeeper_deploy/deploy.py` wrapper works — it runs from `goalkeeper_deploy/` so Python finds `goalkeeper_deploy/tasks/`.

### 5.4 MujocoController Details

From `booster_deploy/booster_deploy/controllers/mujoco_controller.py`:

- Loads MJCF from `cfg.robot.mjcf_path` (resolved via `{BOOSTER_ASSETS_DIR}`)
- Sets initial qpos as `[init_pos(3), init_quat(4), default_joint_pos(N)]`
- `update_state()` reads: `qpos[7:]` = joint pos, `qvel[6:]` = joint vel, `qpos[:7]` = base pose, `qvel[:6]` = base vel
- **PROBLEM:** This hardcoding breaks when extra joints exist (like our ball freejoint adds 7 more qpos). That's why we subclass with `GoalkeeperMujocoController`.
- PD control: `ctrl = clip(kp * (targets - pos) - kd * vel, -limit, limit)`

### 5.5 Standard T1 Config (T1_23DOF_CFG)

From `booster_deploy/booster_deploy/robots/booster.py`:

- 23 DOF (includes head, arms, waist, legs)
- `joint_names` order = identical to training MuJoCo XML order (verified empirically)
- `mjcf_path = "{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml"` — requires `booster_assets` package
- **`booster_assets` is NOT installed on this machine** — so the standard walk task won't run without installing it first
- `sim_joint_names` order is different from `joint_names` (IsaacLab ordering vs SDK ordering) — but for goalkeeper this doesn't matter because we don't use the remapping

### 5.6 Existing Tasks in booster_deploy

| Task name | File | Notes |
|---|---|---|
| `t1_walk` | `tasks/locomotion/__init__.py` | T1 locomotion, needs booster_assets + `models/t1_walk.pt` |
| `k1_mj2` | `tasks/beyond_mimic/__init__.py` | K1 dance (MJ2 motion), has its model |
| `k1_fight` | `tasks/beyond_mimic/__init__.py` | K1 fight motion, has its model |

---

## 6. The `goalkeeper_deploy/` Package

Located at `/home/isaak/BEPImitationlearning/goalkeeper_deploy/`. **This is where all our code lives.**

### 6.1 Why It's Separate

We cannot modify `booster_deploy/`. But we need a new task with a custom controller (because of the ball). The solution: create a separate directory with its own `tasks/` package and a wrapper `deploy.py` that:
1. Adds `goalkeeper_deploy/` first to `sys.path` so Python finds our `tasks/`
2. Also adds `booster_deploy/` to `sys.path` so `booster_deploy` package is importable
3. Replicates the deploy dispatch logic, with one extension: honours `_mujoco_controller_cls` on task configs

### 6.2 File Map

```
goalkeeper_deploy/
├── deploy.py                          # Wrapper entry point (run from here)
├── export_model.py                    # Exports rsl_rl checkpoint → TorchScript
├── docs/
│   └── GOALKEEPER_DEPLOY_COMPLETE.md  # This file
└── tasks/
    ├── __init__.py                    # imports goalkeeper subpackage
    └── goalkeeper/
        ├── __init__.py                # registers "goalkeeper_t1" task
        ├── task.py                    # GoalkeeperPolicy + GoalkeeperT1ControllerCfg
        ├── controller.py              # GoalkeeperMujocoController (custom scene with ball)
        └── models/
            └── goalkeeper_t1_2000.pt  # TorchScript model (exported from checkpoint)
```

### 6.3 deploy.py — The Entry Point

```python
# Sets up sys.path correctly, then dispatches:
# - auto-discovers goalkeeper_deploy/tasks/ (NOT booster_deploy/tasks/)
# - if task config has _mujoco_controller_cls, uses that instead of MujocoController
```

**Always run from inside `goalkeeper_deploy/`:**
```bash
cd /home/isaak/BEPImitationlearning/goalkeeper_deploy
python deploy.py --task goalkeeper_t1 --mujoco
python deploy.py --list
```

**Do NOT run from repo root** — Python won't find our `tasks/` package correctly.

### 6.4 export_model.py — Model Export

Converts the rsl_rl checkpoint to a TorchScript `GoalkeeperActor` module. Run once; output at `tasks/goalkeeper/models/goalkeeper_t1_2000.pt`.

The exported model takes a flat 870-dim tensor (10 × 87) and returns 23 raw actions. It includes the frozen normalizer buffers from training.

```bash
# Run from repo root
python goalkeeper_deploy/export_model.py
```

**Only re-run if you change the checkpoint** (e.g., switching from model_2000.pt to model_19000.pt). If you re-export, update `CHECKPOINT_PATH` and `OUTPUT_PATH` in `export_model.py` and update `checkpoint_path` in `GoalkeeperPolicyCfg` in `task.py`.

### 6.5 task.py — Policy and Config

**`GoalkeeperPolicy`:**
- Loads the TorchScript model from `cfg.checkpoint_path`
- Maintains 87×10 observation history (zeros-initialized, rolls on each step)
- `_compute_obs()`: assembles the 87-dim observation using `controller.robot.data` + controller ball/hand fields
- `inference()`: rolls history, runs model, stores `last_action`, returns `default_pos + action * action_scale`
- Safety fallback: if `projected_gravity[2] > -0.5` (robot falling), calls `controller.stop()`

**`GoalkeeperT1RobotCfg`:**
- Inherits from `RobotCfg`
- `default_joint_pos = TRAINING_DEFAULT_POS` — bent-legged standing pose
- `joint_stiffness = KP` — matches training actuator stiffness
- `joint_damping = KD` — critical-damping formula
- `mjcf_path = ""` — ignored; controller builds scene internally
- `prepare_state`: stiff upright pose for entering custom mode on real robot

**`GoalkeeperT1ControllerCfg`:**
- `policy_dt = 0.02` (50 Hz)
- `mujoco.decimation = 10` → `physics_dt = 0.002` (500 Hz physics)

**`GoalkeeperT1Cfg`** (in `__init__.py`):
- Adds `_mujoco_controller_cls = GoalkeeperMujocoController` attribute
- Registered as `"goalkeeper_t1"` in registry

### 6.6 controller.py — Custom MuJoCo Controller

**Why we need a custom controller:** The standard `MujocoController` reads `qpos[7:]` as all joint positions and `qvel[6:]` as all joint velocities. When we add a ball (which has a freejoint = 7 more qpos elements), those indices break. We need to explicitly read `qpos[7:30]` (robot joints) and `qpos[30:37]` (ball).

**`_build_scene(physics_dt)`:**
- Loads `T1_serial_clean.xml` via `mujoco.MjSpec.from_file()`
- Adds a floor plane geom to worldbody
- Adds a ball body with freejoint, sphere geom (radius=0.11m, mass=0.42kg), yellow rgba
- Adds 23 motor actuators (one per joint, `set_to_motor()` type, forcerange = ±effort_limit)
- Returns compiled `mujoco.MjModel` and `mujoco.MjData`

**qpos/qvel layout after build:**
```
qpos[0:3]   robot base position (world)
qpos[3:7]   robot base quaternion [w,x,y,z] (world)
qpos[7:30]  robot 23 joint positions
qpos[30:33] ball position (world)
qpos[33:37] ball quaternion [w,x,y,z] (world)

qvel[0:3]   robot base linear velocity  (BODY FRAME — MuJoCo freejoint convention)
qvel[3:6]   robot base angular velocity (BODY FRAME)
qvel[6:29]  robot 23 joint velocities
qvel[29:32] ball linear velocity (ball BODY FRAME — must rotate to world frame for obs)
qvel[32:35] ball angular velocity (ball body frame)
```

**Body IDs (determined at compile time):**
- `ball` = 25
- `left_hand_link` = 7
- `right_hand_link` = 11
- `Trunk` = 1

**`update_state()`:** Reads the correct qpos/qvel slices. Ball world velocity is obtained from `mj_data.cvel[ball_id, 3:]` which gives world-frame linear velocity (avoids needing to rotate ball body frame vel to world frame manually). Hand world positions come from `mj_data.xpos[hand_id]`.

**`ctrl_step()`:** Standard PD control on robot joints only. Also handles ball auto-relaunch: if `current_time - last_launch > 3.5s` OR `ball_y < -1.5m` OR `ball_z < 0.05m`, calls `_launch_ball()`.

**`_launch_ball()`:**
- Resets ball to `(x_rand ∈ [-1.5, 1.5], y=5.0, z_rand ∈ [0.5, 1.5])` world frame
- Computes direction from ball → robot base + 0.2m height
- Sets speed `∈ [4.0, 7.0]` m/s in that direction
- Calls `mj_forward` to update derived quantities

**`run()`:** Standard MuJoCo viewer loop. Camera follows robot base. Press `q` or close window to stop.

### 6.7 The Exported TorchScript Model

`goalkeeper_deploy/tasks/goalkeeper/models/goalkeeper_t1_2000.pt` is a `torch.jit.ScriptModule`. Call signature:
```python
action = model(obs_flat: torch.Tensor) -> torch.Tensor
# obs_flat: shape (870,)   float32
# action:   shape (23,)    float32  — raw actions before scaling
```

Internals: normalizes input using frozen `obs_mean` and `obs_std` buffers, passes through MLP.

---

## 7. How to Run the Deployment

### 7.1 Prerequisites

The deployment runs on the local machine (not the robot) for sim2sim.

Required Python packages (all should already be installed in system Python or a venv):
```
torch       (any version with TorchScript support)
mujoco      (3.x with MjSpec API)
numpy
```

`booster_assets` is NOT required for goalkeeper deployment because we build the scene from `T1_serial_clean.xml` directly.

### 7.2 Sim2Sim (MuJoCo)

```bash
cd /home/isaak/BEPImitationlearning/goalkeeper_deploy
python deploy.py --task goalkeeper_t1 --mujoco
```

A MuJoCo viewer opens showing:
- Grey T1 robot in bent-leg standing pose
- Yellow ball that launches itself toward the robot every ~3.5 seconds from y=5m
- Robot attempts to intercept the ball using the trained goalkeeper policy

**To stop:** Close the MuJoCo viewer window, or Ctrl+C.

**If ball is not launching:** The ball auto-launches when `ctrl_step` detects >3.5s elapsed or ball has left the field. First launch happens within 3.5s of start.

### 7.3 Real Robot (TODO — Not Yet Tested)

Requirements:
- Booster T1 with firmware ≥ v1.4
- `booster_robotics_sdk_python` Python package installed on the robot
- ROS 2 Humble active (`source /opt/booster/BoosterRos2Interface/install/setup.bash`)
- Deploy directory copied to the robot

**WARNING:** Real-robot deployment has not been tested yet. The `GoalkeeperPolicy` is designed to support it, but `GoalkeeperMujocoController` inherits from `BaseController` (not `MujocoController`) and does NOT implement the real-robot path. To add real-robot support, you would need to:
1. Create a `GoalkeeperBoosterRobotPortal` that reads ball position from cameras or a separate perception pipeline
2. Or use the ball position as zero (robot just stands and reacts to learned instinct) which may not work well

For now, use `--mujoco` only.

### 7.4 List Available Tasks

```bash
cd /home/isaak/BEPImitationlearning/goalkeeper_deploy
python deploy.py --list
```

Expected output:
```
Available tasks:
  goalkeeper_t1  :  tasks.goalkeeper.GoalkeeperT1Cfg
```

### 7.5 Re-export Model (If Checkpoint Changes)

```bash
# From repo root
python goalkeeper_deploy/export_model.py
```

If you want to use a different checkpoint (e.g., `model_19000.pt`), edit `export_model.py`:
```python
CHECKPOINT_PATH = ".../2026-05-23_18-35-15/model_19000.pt"
OUTPUT_PATH = ".../models/goalkeeper_t1_19000.pt"
```

Then update `GoalkeeperPolicyCfg.checkpoint_path` in `task.py`:
```python
checkpoint_path: str = "models/goalkeeper_t1_19000.pt"
```

---

## 8. Training System Background (What the Model Was Trained On)

### 8.1 Architecture: mjlab on MuJoCo

Unlike the upstream Humanoid-Goalkeeper (which uses Isaac Gym's GPU-parallel physics), the T1 port uses **mjlab** which wraps MuJoCo with an rsl_rl-compatible RL interface. Key differences:

| Aspect | Upstream (G1, Isaac Gym) | T1 Port (mjlab) |
|---|---|---|
| Physics | Isaac Gym (GPU) | MuJoCo |
| Motion priors | AMP (adversarial) | Motion tracking rewards |
| PPO variant | HIM-PPO | Standard PPO |
| Num envs | 4096+ | 6144 (training) |
| Policy frequency | 50 Hz | 50 Hz |
| Physics timestep | ~0.005s | 0.002s (500 Hz) |

### 8.2 Motion Data

Training uses **motion tracking rewards** to shape the robot's pose during interception. The reference motion is `lefthand_t1.npz` — a captured dive/reach motion for the left hand interception. The motion is used for:
- RSI (Reference State Initialization): robot spawns at diverse frames of the reference motion
- Tracking rewards: `motion_global_root_pos`, `motion_body_pos`, etc. penalize deviation

The motion tracking rewards are **not in observations** — they are training-only signals. The policy learns to intercept balls without needing the reference trajectory at test time.

### 8.3 Reward Structure

The reward has two parts:

**Task rewards (the actual goal):**
| Reward | Weight | Description |
|---|---|---|
| `stopball` | 100–200 (curriculum) | Ball velocity zeroed by hand contact |
| `eereach` | 10–20 (curriculum) | End-effector reaches predicted ball intercept |
| `hand_proximity_strict` | 5–10 (curriculum) | Hand within 15cm of ball |
| `stayonline` | -2 | Robot stays near x=0 line |
| `noretreat` | -2 | Robot doesn't retreat from ball approach |
| `feetorientation` | +3 | Feet flat on ground |
| `postorientation` | +3 | Upright after ball passes |
| `postangvel` | +3 | Low angular velocity after ball passes |
| `postlinvel` | +1 | Low linear velocity after ball passes |
| `successland` | +4 | Safe landing after dive |
| `postupperdofpos` | +1 | Arms return to neutral after save |
| `postwaistdofpos` | +1 | Waist returns to neutral after save |

**Regularization:**
| Reward | Weight | Description |
|---|---|---|
| `action_rate_l2` | -0.1 | Smoothness (2nd-order jerk) |
| `dof_acc` | -2.5e-7 | Joint acceleration |
| `torques` | -1e-5 | Torque magnitude |
| `dof_vel` | -5e-4 | Joint velocity |
| `ang_vel_xy` | -0.1 | Base tilt velocity |
| `dof_pos_limits` | -3 to -9 (curriculum) | Soft joint limits |
| `dof_vel_limits` | -2 | Soft velocity limits |
| `torque_limits` | -3 to -9 (curriculum) | Soft torque limits |
| `penalize_sharpcontact` | -100 | Hard foot contact forces |
| `penalize_kneeheight` | -100 | Knees hitting ground |
| `penalize_self_collision` | -50 | Self-collision |
| `deviation_waist_joint` | -0.001 | Waist joint deviation |
| `feet_slippage` | +3 | Feet not slipping |

**Motion tracking (training only, not in deployment obs):**
| Reward | Weight |
|---|---|
| `motion_global_root_pos` | 4.0 |
| `motion_global_root_ori` | 3.0 |
| `motion_body_pos` | 4.0 |
| `motion_body_ori` | 2.0 |
| `motion_body_lin_vel` | 2.0 |
| `motion_body_ang_vel` | 2.0 |

### 8.4 Episode Structure

- Duration: 3.0 seconds (150 steps at 50 Hz)
- Ball arrives within the episode window (catchstep warmup ~7 steps, then ball launches)
- Termination: bad_orientation (>57° tilt), base_height < 0.4m, sharpforce > 1500N, timeout

### 8.5 Ball Observation Masking

The policy was trained with sophisticated ball visibility masking (features 7 & 8 from upstream):
- **Catchstep warmup** (feature 7): ball is hidden for first ~7 steps while it's being launched (not yet on a real trajectory)
- **Flying gate** (feature 8): ball must be in camera field-of-view range (0.05 < y_body < 3.4, |x_body| < 2.0, z < 1.8) to be visible
- **Random vanish** (feature 8): ball randomly disappears at a random step to prevent policy from locking on a fixed trajectory

In deployment, we currently set ball_pos_b and ball_vel_b from MuJoCo data without any of this masking. The policy should still work because it was trained to handle zero ball obs (the hidden periods).

---

## 9. Upstream Features Implemented in This Port

Eight features from the upstream Humanoid-Goalkeeper were ported as part of commit `cab43bd`. These are documented in detail in `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md`.

Short summary:
1. **P1B: successland** — proper `_has_in_air` tracking, landing bonus (+5×), one-foot penalty
2. **P1C: force averaging** — per-foot max then mean (not flat mean across all geoms)
3. **P2A: ball difficulty curriculum** — easy→hard shot range over training
4. **P2B: dof/torque limits curriculum** — weights scale -3→-6→-9
5. **P2C: hand_proximity_strict curriculum** — weight scales 5→7.5→10
6. **P3A: eereach intercept point** — Phase 1 pre-positioning reward + predicted intercept point vs current ball pos
7. **Feature 7: catchstep warmup** — ball obs zeroed for first ~7 steps
8. **Feature 8: ball visibility masking** — flying gate + random vanish

---

## 10. Known Issues and Next Steps

### 10.1 Real-Robot Deployment Not Implemented

`GoalkeeperMujocoController` only handles sim2sim. For real-robot deployment, ball position must come from an external perception system (cameras). The booster_deploy `BoosterRobotPortal` handles robot state, but there is no ball perception integration.

**To add real-robot support:**
- Create `GoalkeeperBoosterRobotPortal` subclassing `BoosterRobotPortal`
- Override `update_state()` to additionally read ball pose from a ROS 2 topic or camera pipeline
- Register a separate "goalkeeper_t1_real" task that uses this portal

### 10.2 Ball Visibility Masking Not Applied in Deployment

Training applied three layers of ball masking (catchstep, flying gate, random vanish). Deployment currently gives the policy the raw MuJoCo ball position every step. This is a distribution shift. The policy may handle it gracefully (it saw many zero-ball steps during training) but could behave differently than in training.

**To fix:** Implement the `_compute_ball_visibility()` function from `mdp/observations.py` inside `GoalkeeperPolicy._compute_obs()` using the controller's physics state.

### 10.3 model_2000 Is Early Training

`model_2000.pt` is at iteration 2000. The run continued to iteration 19000+ (`model_19000.pt` exists in the same directory). Consider testing `model_19000.pt` which may have better ball-catching behavior. To switch:
1. Re-run `export_model.py` with updated paths
2. Update `checkpoint_path` in `task.py`

### 10.4 booster_assets Not Installed

The standard `t1_walk` task in `booster_deploy/tasks/locomotion/` requires the `booster_assets` Python package (Booster's robot model repository). It is not installed. To install:
```bash
git clone https://github.com/BoosterRobotics/booster_assets
pip install -e booster_assets
```

The goalkeeper task does NOT need `booster_assets` because it uses `T1_serial_clean.xml` from the training assets.

---

## 11. File Locations Quick Reference

| What | Path |
|---|---|
| Training project | `/home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1/` |
| Model checkpoint (deployed) | `my_mjlab_project_booster_t1/logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt` |
| Model checkpoint (longer run) | `my_mjlab_project_booster_t1/logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_19000.pt` |
| T1 MJCF (training) | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/assets/booster_t1/T1_serial_clean.xml` |
| T1 robot constants | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py` |
| Training env config | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py` |
| Training PPO config | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py` |
| Observation functions | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/observations.py` |
| Reward functions | `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py` |
| Booster deploy framework | `/home/isaak/BEPImitationlearning/booster_deploy/` (**read-only**) |
| Goalkeeper deploy entry | `goalkeeper_deploy/deploy.py` |
| Goalkeeper policy | `goalkeeper_deploy/tasks/goalkeeper/task.py` |
| Goalkeeper MuJoCo controller | `goalkeeper_deploy/tasks/goalkeeper/controller.py` |
| TorchScript model | `goalkeeper_deploy/tasks/goalkeeper/models/goalkeeper_t1_2000.pt` |
| Model export script | `goalkeeper_deploy/export_model.py` |
| Commands | `commands.txt` at repo root |
| This document | `goalkeeper_deploy/docs/GOALKEEPER_DEPLOY_COMPLETE.md` |
| Upstream G1 (frozen) | `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/` |
| Divergence log | `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` |
| Bug fixes history | `my_mjlab_project_booster_t1/docs/03_BUG_FIXES.md` |
| Reward comparison doc | `my_mjlab_project_booster_t1/REWARD_COMPARISON_AMP_vs_TRACKING.md` |
| Observation consistency doc | `my_mjlab_project_booster_t1/OBSERVATION_REWARD_CONSISTENCY.md` |

---

## 12. Technical Details That Bit Us (Don't Repeat)

### 12.1 MuJoCo freejoint qvel is body-frame, NOT world-frame

`mj_data.qvel[:3]` for a free joint is the body's linear velocity expressed in the **body frame** (local frame), not world frame. Same for `qvel[3:6]` (angular). This is correct for the observation — training also uses body-frame velocities. Do not transform these.

For ball velocity, we use `mj_data.cvel[ball_id, 3:]` which gives world-frame linear velocity, then transform to robot body frame via quaternion inverse rotation.

### 12.2 Training default pose ≠ deployment default pose

`T1_23DOF_CFG.default_joint_pos` in booster_deploy has zeros for all leg joints. The training `T1_STANDING_KEYFRAME` has bent legs (`Hip_Pitch=-0.3, Knee=0.6, Ankle=-0.3`). Using the wrong default causes wrong `joint_pos_rel` observations and wrong joint targets.

**Always use `TRAINING_DEFAULT_POS` from `task.py` — never use `T1_23DOF_CFG.default_joint_pos`.**

### 12.3 sys.path ordering matters for task discovery

`booster_deploy/scripts/deploy.py` does `sys.path.append(".")` (appends, not inserts). If `booster_deploy/` is already in sys.path before `"."`, Python finds `booster_deploy/tasks/` before our `goalkeeper_deploy/tasks/` and imports the wrong one.

Our `goalkeeper_deploy/deploy.py` uses `sys.path.insert(0, _HERE)` to put `goalkeeper_deploy/` first. This is why you must use our wrapper `deploy.py` and not `booster_deploy/scripts/deploy.py`.

### 12.4 TorchScript requires instance attributes for float constants

When scripting with `torch.jit.script`, global float constants used in `forward()` cause:
`RuntimeError: python value of type 'float' cannot be used as a value`

The fix: store the constant as an instance attribute in `__init__`:
```python
self.eps: float = EPS  # annotation required
```

### 12.5 Ball adds 7 qpos elements — indices shift

Standard MuJoCo T1 model: `qpos = [7_freejoint + 23_joints] = 30` elements.  
With ball: `qpos = [7_freejoint + 23_joints + 7_ball_freejoint] = 37` elements.  
The standard `MujocoController.update_state()` reads `qpos[7:]` as joint positions — with a ball this gives 30 values, causing shape mismatch with `robot.data.joint_pos` tensor of size 23.

Our controller explicitly reads `qpos[7:30]` for robot joints and `qpos[30:37]` for ball.

---

## 13. Commands Reference

### Training
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1
uv run python -m mjlab.scripts.train goalkeeper_booster_t1 --gpu-ids '[0]'
```

### Play (MuJoCo training viewer)
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1
uv run python -m mjlab.scripts.play goalkeeper_booster_t1 \
  --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt
```

### Goalkeeper sim2sim deployment
```bash
cd /home/isaak/BEPImitationlearning/goalkeeper_deploy
python deploy.py --task goalkeeper_t1 --mujoco
```

### List all registered tasks
```bash
cd /home/isaak/BEPImitationlearning/goalkeeper_deploy
python deploy.py --list
```

### Re-export TorchScript model
```bash
python goalkeeper_deploy/export_model.py
```

### Check booster_deploy built-in tasks (from its own deploy.py)
```bash
cd /home/isaak/BEPImitationlearning/booster_deploy
python scripts/deploy.py --list
```
