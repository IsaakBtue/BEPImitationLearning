# Comparison: Our Goalkeeper Task vs Unitree RL MjLab Reference Implementation

## Overview
We're comparing our custom goalkeeper task implementation against the proven Unitree RL MjLab implementation (`https://github.com/unitreerobotics/unitree_rl_mjlab`), which is a working baseline for motion tracking with G1 on mjlab.

---

## Directory Structure Comparison

### Reference (Unitree)
```
src/tasks/
├── tracking/
│   ├── __init__.py                 (empty)
│   ├── tracking_env_cfg.py         (base config factory)
│   ├── config/
│   │   └── g1/
│   │       ├── __init__.py         (registers task)
│   │       ├── env_cfgs.py         (G1-specific env config)
│   │       └── rl_cfg.py           (G1-specific RL config)
│   ├── mdp/
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   └── ...
│   └── rl/
│       └── runner.py               (custom MotionTrackingOnPolicyRunner)
├── velocity/
│   └── ...
└── __init__.py                     (uses import_packages for auto-discovery)

scripts/
└── train.py                        (generic trainer, task-agnostic)
```

### Our Implementation (Goalkeeper)
```
src/my_mjlab_project/
├── tasks/
│   ├── __init__.py                 (manually calls register_all())
│   ├── goalkeeper_env_cfg.py       (monolithic config file)
│   └── goalkeeper_ppo_cfg.py       (RL config)
├── mdp/
│   ├── commands.py                 (MultiMotionCommandCfg - custom)
│   ├── observations.py             (goalkeeper-specific obs)
│   └── rewards.py                  (goalkeeper-specific rewards)
└── motions/
    ├── convert.py
    └── data/
        └── *.npz

scripts/
├── train.py                        (task-specific wrapper)
└── play.py                         (task-specific wrapper)
```

### Assessment
**✅ OK:** Both register tasks properly via `register_mjlab_task()`
**⚠️ Potential issue:** Our monolithic `goalkeeper_env_cfg.py` (1 file) vs their modular base config + robot-specific config separation

---

## Configuration Structure Differences

### 1. Base Environment Config Factory

**Reference (Unitree):** `make_tracking_env_cfg()` is a reusable base that returns a complete `ManagerBasedRlEnvCfg`
- Single source of truth for all tracking tasks
- Robot-specific configs just inherit and override specific fields
- Easy to maintain and extend to new robots

**Ours:** Direct factory in `goalkeeper_env_cfg.py` starting from `unitree_g1_flat_tracking_env_cfg()`
- ✅ Correct: We call the base tracking config as a starting point
- ⚠️ **Issue:** We modify it in-place within one function. If we later add multiple goalkeeper variants (e.g., different ball sizes, reward weights), code duplication would increase.

### 2. Robot Assets

**Reference:**
```python
cfg.scene.entities = {"robot": get_g1_robot_cfg()}
```

**Ours:**
```python
cfg.scene.entities = {
    "robot": get_g1_robot_cfg(),
    "ball": EntityCfg(spec_fn=get_ball_spec),
}
cfg.scene.num_envs = 1020
```

**Assessment:**
✅ **Correct:** We properly add the ball entity and increase num_envs for the goalkeeper task
- Reference doesn't have a ball (velocity tracking, not goalkeeper)
- Our addition is appropriate for the task

### 3. MuJoCo Model Parameters (CRITICAL)

**Reference:** 
No explicit nconmax/njmax settings found in the reference config

**Ours (Fixed in recent commit):**
```python
def get_ball_spec(...):
    spec = mujoco.MjSpec()
    spec.nconmax = 100
    spec.njmax = 500
```

**Assessment:**
✅ **Fixed:** We now set contact and joint buffers in the ball spec
- **Why this matters:** With 1020 parallel environments + ball collisions, default buffers overflow
- Reference doesn't need this because they don't have ball collisions

---

## Motion Command Implementation

### Reference (Unitree)
```python
from mjlab.tasks.tracking.mdp import MotionCommandCfg
motion_cmd = cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.anchor_body_name = "torso_link"
motion_cmd.body_names = (...)
```

**Single motion file per training run.** The motion file is passed at runtime:
```bash
uv run python scripts/train.py Unitree-G1-Tracking --motion-file motion.npz
```

### Ours
```python
from my_mjlab_project.mdp import MultiMotionCommandCfg
cfg.commands["motion"] = MultiMotionCommandCfg(
    entity_name="robot",
    motion_files=motion_files,  # 6 files hardcoded
    ball_name="ball",
    ...
)
```

**Multiple motion files per training run.** All 6 motions embedded in config, randomly sampled at reset.

**Assessment:**
⚠️ **Design difference (not necessarily wrong):**
- **Reference approach:** One motion per run, switch runs to try different motions
- **Our approach:** All 6 motions in parallel, agent learns all simultaneously
- **Pros of ours:** Faster convergence, generalization across motions
- **Cons of ours:** Higher complexity, more dependencies on custom MultiMotionCommandCfg

**Potential issue:** If MultiMotionCommandCfg has bugs, training will fail. Reference approach is simpler and less likely to break.

---

## Training Script Comparison

### Reference (scripts/train.py)
```python
# Generic trainer that works for ANY registered task
all_tasks = list_tasks()
chosen_task, remaining_args = tyro.cli(...)  # User picks task at CLI
args = TrainConfig.from_task(chosen_task)
runner_cls = load_runner_cls(task_id)
runner.learn(num_learning_iterations=...)
```

**Strengths:**
✅ Task-agnostic: works with velocity, tracking, custom tasks, etc.
✅ Flexible: motion file passed at CLI, not hardcoded
✅ Standard: uses mjlab's native motion file handling

### Ours (scripts/train.py and scripts/play.py)
```python
# Task-specific wrappers
cmd = ["uv", "run", "mjlab", "train", "goalkeeper"] + args
subprocess.run(cmd, cwd=PROJECT_DIR)
```

**Issues:**
⚠️ **Wrapper overhead:** Just calls mjlab's built-in trainer with a specific task
⚠️ **Not flexible:** Motion files baked into config, can't change at runtime
⚠️ **Redundant:** These scripts add no value over calling mjlab directly

---

## Observations & Rewards Comparison

### Observations (MotionTrackingOnPolicyRunner expectations)

**Reference tracking_env_cfg.py actor terms:**
- `command` (motion command, 58-dim for 14 bodies)
- `motion_anchor_pos_b` (3-dim)
- `motion_anchor_ori_b` (6-dim)
- `base_lin_vel` (3-dim, IMU)
- `base_ang_vel` (3-dim, IMU)
- `joint_pos` (29-dim, relative to default)
- `joint_vel` (29-dim)
- `actions` (29-dim, last action)
**Total: ~130-180 dim** (varies by body count)

**Our goalkeeper config actor terms:**
- All of the above, plus:
- `ball_pos_b` (3-dim)
- `ball_vel_b` (3-dim)
- `left_hand_pos_b` (3-dim)
- `right_hand_pos_b` (3-dim)

**Assessment:**
✅ **Correct:** We add goalkeeper-specific observations (ball, hand positions)
✅ **Reasonable:** Actor obs count is ~172 dim (matches expected range)

---

## Rewards Comparison

### Reference G1 Tracking Rewards
- **Motion tracking:** reward = exp(-||Δq||² / σ²) for joints
- **Minimal task rewards:** Just tracking the motion clips

### Our Goalkeeper Rewards (18 terms total)

**Motion tracking (6 terms):** 
- root_pos, root_ori, body_pos, body_ori, body_lin_vel, body_ang_vel
- ✅ Matches reference approach

**Goalkeeper task (9 terms):**
- eereach, catch_success, stopball, stayonline, noretreat, feetorientation, postorientation, postangvel, postlinvel
- ✅ Task-specific, appropriate

**Regularization (3 terms):**
- action_rate_l2, joint_limit, self_collisions
- ✅ Standard for legged robot tasks

**Assessment:**
✅ **Comprehensive:** 18 terms is reasonable and balanced

---

## Critical Issues & Recommendations

### 🔴 Issue 1: Training Startup Hangs (CURRENT)
**Symptom:** `uv run python -m mjlab.scripts.train goalkeeper --gpu-ids 0` times out after 10+ seconds

**Possible causes:**
1. Model compilation (MuJoCo → CUDA)
2. Ball spec nconmax/njmax affecting compilation
3. Large motion loader initialization (6 × ~1000 frame clips)
4. GPU memory exhaustion on large environment creation

**Testing steps:**
1. Reduce num_envs to 100 and retry
2. Check GPU memory usage with `nvidia-smi`
3. Add timing logs to goalkeeper_env_cfg.py `__init__`
4. Verify motion files load correctly

### ⚠️ Issue 2: MultiMotionCommandCfg Complexity
**Current state:** Works in smoke tests but full-scale failures not yet observed

**Risk:** If motion loading fails partway through training, recovery is difficult

**Mitigation:**
- Add comprehensive error checking in MultiMotionCommandCfg
- Log motion loader stats at initialization
- Consider fallback to single motion (like reference) if needed

### ⚠️ Issue 3: Scripts are Redundant
**Current:** `scripts/train.py` and `scripts/play.py` are thin wrappers

**Better approach:**
- Use mjlab's built-in CLI directly OR
- Make scripts configure environment variables (e.g., `--motion-file`)

**Example (current best practice):**
```bash
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids 0
```

### ✅ Strength: Config Structure is Sound
- ✅ Properly registers with mjlab
- ✅ Observations/rewards are well-formed
- ✅ Uses native mjlab managers (event, reward, termination)
- ✅ PPO config follows reference patterns

---

## Comparison Summary Table

| Aspect | Reference (Unitree) | Ours (Goalkeeper) | Assessment |
|--------|---------------------|-------------------|------------|
| **Task Registration** | register_mjlab_task ✅ | register_mjlab_task ✅ | Identical, correct |
| **Base Config** | make_tracking_env_cfg() | unitree_g1_flat_tracking_env_cfg() | Both correct |
| **Motion Handling** | Single file @ CLI | Multi-file embedded | Different but valid |
| **Ball/Collisions** | N/A | nconmax=100, njmax=500 | Correct addition |
| **Observations** | ~130-180 dim | ~172 dim | Match expected |
| **Rewards** | ~8-10 terms | 18 terms | Ours is richer |
| **RL Config** | Standard PPO | Standard PPO | Identical patterns |
| **Training Script** | Generic mjlab trainer | mjlab trainer wrapper | Ours: redundant |
| **Status** | ✅ Proven | ⚠️ Startup hangs | Need debugging |

---

## Immediate Next Steps

1. **Debug startup hang:**
   ```bash
   cd /home/isaak/BEPImitationlearning/my_mjlab_project
   uv run python -c "
   import sys; import time
   start = time.time()
   from src.my_mjlab_project.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg
   print(f'Config load: {time.time() - start:.2f}s')
   cfg = goalkeeper_env_cfg()
   print(f'Config create: {time.time() - start:.2f}s')
   print(f'Num envs: {cfg.scene.num_envs}')
   "
   ```

2. **Test with reduced scale:**
   ```bash
   uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 10 --gpu-ids 0
   ```

3. **Compare with reference:**
   ```bash
   cd /tmp/unitree_rl_mjlab
   uv run python scripts/train.py Unitree-G1-Tracking --motion-file /path/to/motion.npz
   ```

4. **Profile MuJoCo model creation:**
   - Add timing in `get_ball_spec()` and `goalkeeper_env_cfg()`
   - Check if ball spec or motion loader is the bottleneck

---

## Files to Monitor

- `/home/isaak/BEPImitationlearning/my_mjlab_project/src/my_mjlab_project/mdp/commands.py` — MultiMotionCommandCfg stability
- `/home/isaak/BEPImitationlearning/my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py` — Config initialization
- `/home/isaak/BEPImitationlearning/my_mjlab_project/src/my_mjlab_project/motions/motion_loader.py` — Motion loading performance

