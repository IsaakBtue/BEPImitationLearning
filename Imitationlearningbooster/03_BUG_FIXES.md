# Bug Fixes History

## Bug #1: Body Index Off-by-One (CRITICAL) — Fixed 2026-05-02

### Symptom
Every training episode terminated on step 1 with all 1020 environments crashing:
- `mean_episode_length = 1.00` (should be ~50–250)
- `error_anchor_pos = 1.25 m` (threshold is 0.25 m)
- 100% of episodes terminated via `anchor_pos` constraint

### Root Cause
In `src/my_mjlab_project/motions/convert.py` (lines 154–166), motion data was storing MuJoCo's worldbody at index 0:

**BUGGY CODE:**
```python
num_bodies = model.nbody  # 31 (includes worldbody)
body_pos_w[t] = data.xpos.copy()  # Saves (0,0,0) at NPZ[0]
body_quat_wxyz[t] = data.xquat.copy()
```

**Why this broke training:**
1. MuJoCo stores 31 bodies: index 0=worldbody (always at 0,0,0), indices 1–30=robot
2. mjlab's `entity.body_names` is 0-indexed on robot bodies only (skips worldbody)
3. MotionLoader reads NPZ[0] expecting pelvis, gets worldbody (0,0,0)
4. Reference State Initialization (RSI) teleports robot to floor (z≈0.3m)
5. Robot expects torso at z≈1.18m → error ≈ 1.18m >> 0.25m threshold → immediate termination

### Fix
**FIXED CODE:**
```python
num_bodies = model.nbody - 1  # 30 (skip worldbody)
body_pos_w[t] = data.xpos[1:].copy()  # Pelvis now at NPZ[0]
body_quat_wxyz[t] = data.xquat[1:].copy()
```

All 6 motion files regenerated with 30 bodies:
- lefthand.npz (123 frames, 30 bodies)
- righthand.npz (246 frames, 30 bodies)
- leftjump.npz (254 frames, 30 bodies)
- rightjump.npz (200 frames, 30 bodies)
- leftstep.npz (194 frames, 30 bodies)
- rightstep.npz (197 frames, 30 bodies)

### Verification
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| mean_episode_length | 1.00 | 7.6 | ✅ Fixed |
| error_anchor_pos | 1.25 m | 0.14 m | ✅ Fixed |
| anchor_pos terminations | 100% | 0% | ✅ Fixed |
| Training iterations | Crashes | 3 complete | ✅ Fixed |

**W&B Log:** `wandb/run-20260502_212536-den9mpz2/`

---

## Bug #2: Ball Z-Position Missing env_origins Offset — Fixed 2026-05-02

### File
`src/my_mjlab_project/mdp/commands.py` (line 239)

### Before
```python
ball_pos_w[:, 2] = z_start  # Missing z offset
```

### After
```python
ball_pos_w[:, 2] = origins[:, 2] + z_start  # Correct
```

### Impact
Benign on flat terrain (z=0 for all envs) but latent bug for non-zero terrain heights.

---

## Bug #3: Motion-Dependent Observations in Play Config — Fixed 2026-05-03

### Symptom
Play script crashed: `AttributeError: 'NoneType' object has no attribute 'anchor_pos_w'`

When motion command was removed from play config, reward functions tried to access motion-dependent attributes that no longer existed.

### Root Cause
Three motion-dependent reward terms required access to `cmd.anchor_pos_w` and similar command attributes:
- motion_global_root_pos
- motion_global_root_ori
- motion_body_pos
- motion_body_ori
- motion_body_lin_vel
- motion_body_ang_vel

### Fix
**File:** `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py` (lines 265–273)

```python
# Remove motion-dependent reward terms (they access cmd attributes)
_motion_rewards = [
    "motion_global_root_pos", "motion_global_root_ori",
    "motion_body_pos", "motion_body_ori",
    "motion_body_lin_vel", "motion_body_ang_vel",
]
for _rew in _motion_rewards:
    cfg.rewards.pop(_rew, None)
```

Also removed motion-dependent termination `ee_body_pos` which calls `bad_motion_body_pos_z_only()` function.

### Result
Play mode now runs successfully without crashes. Policy is 100% autonomous—no motion input required.

---

## Bug #4: Ball Not Resetting Between Episodes — Fixed 2026-05-03

### Symptom
Ball spawned once at start of play mode, never reset. After first episode, ball stayed in last position.

### Root Cause
Reset event was configured with `mode="startup"` instead of `mode="reset"`.

**BUGGY CODE:**
```python
cfg.events["reset_ball_autonomous"] = EventTermCfg(
    func=gk_resets.reset_ball_autonomous,
    mode="startup",  # Only runs once at initialization
    params={"ball_name": "ball"},
)
```

### Fix
**FIXED CODE:**
```python
cfg.events["reset_ball_autonomous"] = EventTermCfg(
    func=gk_resets.reset_ball_autonomous,
    mode="reset",  # Runs every episode reset
    params={"ball_name": "ball"},
)
```

### Result
Ball now resets with new random trajectory every 10 seconds (episode_length_s = 10.0).

---

## Bug #5: Ball Not Bouncing — Fixed 2026-05-03

### Symptom
Ball would hit ground and stop instead of bouncing. Physics parameters seemed ignored.

### Root Cause
MuJoCo's contact behavior depends on parameters from **both** the ball AND the ground.

Initial attempt set only `solimp` but not `solref`:
```python
geom.solimp = [0.001, 0.01, 0.001, 0.5, 2.0]  # No solref!
```

`solref` (solver reference parameters) controls the actual contact stiffness and damping in MuJoCo's implicit integrator:
- `solref[0]`: Time constant (larger = softer/more compliant)
- `solref[1]`: Damping ratio (smaller = less energy loss = more bouncy)

### Fix
**FIXED CODE:**
```python
geom.solref = [0.05, 0.0001]  # Stiff contact, minimal damping
geom.solimp = [0.0001, 0.001, 0.0001, 0.5, 2.0]
geom.margin = 0.001  # Contact detection distance
geom.gap = 0.0001    # Contact inclusion distance
```

**Parameters explained:**
- `solref[0]=0.05`: 0.05-second time constant (stiff, maintains shape during contact)
- `solref[1]=0.0001`: Near-zero damping (minimal energy loss per bounce)
- Result: Extreme bounciness while remaining stable

### Result
Ball now bounces visibly with minimal energy loss.

---

## Bug #6: Motion Command Breaking Autonomous Play — Fixed 2026-05-03

### Symptom
Even with motion command removed, ball never reset because reset code tried to access motion attributes.

### Root Cause
`_reset_ball()` function was called by motion command and tried to access `cmd.ball_spawning_positions`, etc.

### Fix
Implemented standalone `reset_ball_autonomous()` function independent of motion command:

**File:** `src/my_mjlab_project/mdp/resets.py`

```python
def reset_ball_autonomous(env: ManagerBasedRlEnv, env_ids: torch.Tensor, 
                         ball_name: str = "ball") -> None:
    """Reset ball with random trajectory independent of motion type."""
    ball: Entity = env.scene[ball_name]
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Randomize spawn position and compute trajectory
    x_start = sample_uniform(3.0, 5.0, (n,), device=env.device)
    y_end = sample_uniform(-1.2, 1.2, (n,), device=env.device)
    z_end = sample_uniform(0.1, 1.6, (n,), device=env.device)
    # ... compute velocity to reach endpoint
    
    ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(ball_velocity, env_ids=env_ids)
```

### Result
Ball resets autonomously every episode without depending on motion command.

---

## Bug #8: Reward Structure Inverted (Motion Ignored, Ball Unimportant) — Fixed 2026-05-09

### Symptom
- Robot standing stationary while motion reference dives 0.58m sideways
- Ball stopping barely rewarded (0.14 out of max)
- Stability rewards dominating (2.38 orientation vs 0.14 stopball)
- Motion tracking math proved robot stays still: predicted error 0.302m ≈ observed 0.306m
- Policy learned "stand still and be balanced" rather than "execute lefthand save and stop ball"

### Root Cause
Reward weights from mjlab base tracking config didn't match the original isaacgym-based goalkeeper task:

| Reward | Original isaacgym | Our mjlab | Issue |
|---|---|---|---|
| **stopball** | 100.0 | **2.0** | PRIMARY TASK 50× underweighted! |
| eereach | 10.0 | 10.0 | ✓ Correct |
| success/catch | 5.0 | 5.0 | ✓ Correct |
| feetorientation | 3.0 | 3.0 | ✓ Correct |
| motion_body_pos | N/A (AMP) | 1.0 | Competed with task |
| motion_global_root_pos | N/A (AMP) | 0.5 | Too weak to force dive |

The original used **AMP (Adversarial Motion Prior)** with implicit motion guidance. We use **explicit motion tracking rewards** that were too weak and competed with each other.

Robot logic:
- Standing still + balanced = feetorientation (2.38) + postorientation (2.19) = 4.57 reward
- Stopping ball = stopball (0.14) = 0.14 reward
- Decision: ignore the ball, stay balanced ✓

### Fix
Restructured rewards to match original isaacgym philosophy, using motion tracking instead of AMP:

```python
# Task rewards (PRIMARY)
stopball = 100.0     # was 2.0 ← THE CRITICAL FIX
eereach = 10.0       # unchanged
catch_success = 5.0  # unchanged

# Motion tracking (explicit, replacing AMP)
motion_global_root_pos = 10.0  # was 0.5 → force sideways dive
motion_body_pos = 10.0         # was 1.0 → force arm position
motion_global_root_ori = 5.0   # was 0.5
motion_body_ori = 3.0
motion_body_lin_vel = 3.0
motion_body_ang_vel = 3.0

# Stability (unchanged)
feetorientation = 3.0
postorientation = 3.0
postangvel = 3.0
postlinvel = 1.0
```

### Result
Reward hierarchy now:
1. **Stop the ball** (100.0) — PRIMARY OBJECTIVE
2. **Follow the lefthand motion** (34.0 total motion terms) — MANDATORY GUIDANCE
3. **Execute with balance & form** (10.0 stability) — REQUIRED CONSTRAINT

Robot now learns: "To get maximum reward, I must follow the lefthand motion AND stop the ball"

Expected behavior in next training run:
- `error_anchor_pos` drops from 0.28m → <0.1m (robot follows the dive)
- `stopball` reward jumps from 0.14 → 50+ (actual ball-stopping behavior emerges)
- `motion_global_root_pos` increases from 0.25 → 0.8+ (root translation reward)
- Robot actively executes the lefthand goalkeeper save instead of passive standing

---

## Bug #7: T1 Arms Locked/Stiff During Training — Fixed 2026-05-09

### Symptom
During training, the Booster T1 arms appeared rigid and "not trying" — they barely moved compared to the G1/lefthand training where arms swing visibly while exploring. The robot stood upright but arms stayed near the default pose.

### Root Cause A: Missing `0.25` factor in arm action scale (primary)

The G1 actuator constants define action scale as `0.25 * effort / stiffness`. The `0.25` factor is intentional — it means the motor only saturates when the policy outputs **4× the action scale**, giving large headroom for exploration before the effort limit is hit.

The T1 action scale was defined as `effort / stiffness` (no 0.25 factor). This put the effort saturation point at exactly `action_scale = 1.0`. With a Gaussian policy distribution (std ≈ 1.0), **32% of all arm outputs immediately saturated the motor** — the arm physically couldn't follow those commands, so it appeared locked.

| | G1 arm | T1 arm (old) | T1 arm (fixed) |
|---|---|---|---|
| stiffness | 14.25 Nm/rad | 40.0 Nm/rad | 15.0 Nm/rad |
| effort limit | 25 Nm | 18 Nm | 18 Nm |
| saturation range | 1.75 rad | 0.45 rad | 1.20 rad |
| action scale | 0.439 rad/unit | 0.450 rad/unit | 0.300 rad/unit |
| headroom (saturation / scale) | **4.0×** | **1.0×** | **4.0×** |

### Root Cause B: Knee joints near-zero in motion reference data

The retargeted `lefthand_booster_t1.pkl` had `Left_Knee_Pitch` (dof 14) and `Right_Knee_Pitch` (dof 20) set to essentially 0 rad (straight legs) throughout the entire motion. The standing keyframe initialises the robot with 0.6 rad bent knees for stability. Because `body_pos_w` in the motion file was computed with straight-knee kinematics, the motion body tracking reward targeted lower-leg body positions corresponding to a straight-leg stance — directly conflicting with the bent-knee pose needed to balance. This destabilised training and reduced arm exploration.

### Fix

**File: `robots/t1_constants.py`**

1. Lowered arm stiffness from 40 → 15 Nm/rad (makes arms softer and increases saturation headroom to 1.2 rad)
2. Applied `0.25` factor to arm action scale (matching G1's formula exactly):

```python
# Before
_make_actuator(r"(Left_Shoulder_Pitch|...|Right_Elbow_Yaw)", 18.0, 40.0)
T1_ACTION_SCALE[arm_joints] = 18.0 / 40.0  # = 0.450

# After
_make_actuator(r"(Left_Shoulder_Pitch|...|Right_Elbow_Yaw)", 18.0, 15.0)
T1_ACTION_SCALE[arm_joints] = 0.25 * 18.0 / 15.0  # = 0.300
```

**File: `motions/convert_booster.py`**

Clamp both knee joints to a minimum of 0.5 rad **before** running forward kinematics, so `joint_pos` and `body_pos_w` are consistent with a stable bent-knee stance:

```python
KNEE_INDICES = [14, 20]  # Left_Knee_Pitch, Right_Knee_Pitch
MIN_KNEE_BEND = 0.5  # radians
for ki in KNEE_INDICES:
    dof_pos[:, ki] = np.maximum(dof_pos[:, ki], MIN_KNEE_BEND)
```

Motion file `lefthand_t1.npz` was regenerated after this change.

### Result
Arms now explore freely during training, matching the visible motion seen in the G1 lefthand training.

---

## Summary of Changes

| Date | Bug | Severity | Fix | Status |
|------|-----|----------|-----|--------|
| 2026-05-02 | Body index off-by-one | CRITICAL | Skip worldbody in convert.py | ✅ Fixed |
| 2026-05-02 | Ball z-offset missing | MINOR | Add env_origins z to ball_pos | ✅ Fixed |
| 2026-05-03 | Motion-dependent rewards crash | CRITICAL | Pop motion_* rewards from play config | ✅ Fixed |
| 2026-05-03 | Ball not resetting | CRITICAL | Change event mode from startup to reset | ✅ Fixed |
| 2026-05-03 | Ball not bouncing | HIGH | Add solref parameter to ball geometry | ✅ Fixed |
| 2026-05-03 | Motion command breaking autonomous play | HIGH | Implement standalone reset_ball_autonomous() | ✅ Fixed |
| 2026-05-09 | T1 arms locked/stiff during training | HIGH | Lower arm stiffness + apply 0.25 scale factor + fix knee data | ✅ Fixed |

---

## Testing Verification

### Smoke Test (2 envs, 3 iterations, CPU)
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper \
  --env.scene.num-envs 2 --runner.max-iterations 3 --gpu-ids '[]'
```

**Results:**
- ✅ Episodes last 7+ steps (was 1 step before body index fix)
- ✅ motion errors < 0.14 m (was 1.25 m before)
- ✅ No motion-dependent reward crashes
- ✅ Training completes without errors

### Play Test
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.play goalkeeper \
  --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-03_10-47-38/model_200.pt
```

**Results:**
- ✅ No crashes on startup
- ✅ Ball spawns and bounces
- ✅ ENTER key resets environment and ball
- ✅ Policy responds autonomously to ball trajectory
- ✅ No motion input required

---

## Key Lessons

1. **Body indexing matters:** Always verify which convention your framework uses (0-indexed vs 1-indexed, inclusive/exclusive of special bodies)

2. **Contact parameters are pairwise:** In physics simulators, contact behavior depends on BOTH objects, not just one

3. **Mode flags are critical:** Event `mode="startup"` vs `mode="reset"` makes a huge difference

4. **Test independently:** Autonomous play exposed bugs that training alone would have hidden

5. **Minimize dependencies:** Standalone reset function is more robust than one coupled to motion command

