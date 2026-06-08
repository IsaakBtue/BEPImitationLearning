# RL Port Wiki: Problems Encountered & Lessons Learned

This document is the technical companion to the BEP report section on reinforcement learning. It records every significant bug, misalignment, and design decision encountered while porting the InternRobotics Humanoid-Goalkeeper pipeline (IsaacGym / Unitree G1) to the Booster T1 in MuJoCo / mjlab. Each entry includes the symptom, the root cause, and the fix with code samples.

---

## Table of Contents

1. [Simulator Differences (PhysX vs MuJoCo)](#1-simulator-differences)
   - 1.1 Contact force API migration
   - 1.2 Soft vs hard contact model — threshold recalibration
   - 1.3 Memory model and num_envs
2. [Robot Migration (G1 → T1)](#2-robot-migration)
   - 2.1 Action scale missing 0.25 factor — arms locked
   - 2.2 Arm stiffness too high
   - 2.3 Motion data worldbody off-by-one
   - 2.4 Coordinate system rotation — all task rewards broken
3. [Training Bugs (Single Motion Project)](#3-training-bugs-single-motion)
   - 3.1 Ball never moved — event mode bug
   - 3.2 Stopball continuous vs event-based (50× overweighted)
   - 3.3 Reward weight imbalance (stopball 2 → 100)
   - 3.4 Motion playback speed wrong (30 fps vs 50 Hz)
   - 3.5 Entropy coefficient without AMP — std runaway
   - 3.6 Redundant ball reset competing with motion command
4. [AMP-Specific Bugs (Full Goalkeeper Agent)](#4-amp-specific-bugs)
   - 4.1 AMP normalizer variance collapse (Chan's algorithm)
   - 4.2 Discriminator: 120 backward passes collapsed to 1
   - 4.3 AMP reward scale 10× too weak
   - 4.4 Discriminator update frequency 5× too low
   - 4.5 Spectral normalization missing
5. [Deployment Bugs](#5-deployment-bugs)
   - 5.1 base_lin_vel in world frame instead of body frame
   - 5.2 Ball visibility masking absent at deployment
6. [Ball Visibility Masking System](#6-ball-visibility-masking-system)
7. [Lessons Learned](#7-lessons-learned)

---

## 1. Simulator Differences

### 1.1 Contact Force API Migration

**Symptom:** Force-based reward terms and termination conditions could not be ported directly — the code referring to `cfrc_ext` tensors simply had no equivalent in mjlab.

**Root cause:** Isaac Gym exposes `net_contact_force_tensor` as a global tensor of shape `[num_envs × num_bodies, 3]`, updated every physics step automatically. MuJoCo via mjlab has no such global tensor. Every sensor must be declared explicitly as a `ContactSensorCfg`.

**Fix:** Declare a `feet_contact` sensor and a `self_collision` sensor in the environment config, then rewrite each reward/termination function to read from `env.scene["feet_contact"].data.force`.

```python
# OLD (Isaac Gym): global tensor, direct indexing
contact_forces = self.contact_forces[:, self.contact_feet_indices, :]  # [N, 2, 3]
mean_force = torch.mean(torch.norm(contact_forces, dim=-1), dim=-1)

# NEW (MuJoCo mjlab): must declare sensor first in scene config
ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(mode="geom", pattern=r"^(left|right)_foot_[12]$", entity="robot"),
    secondary=None,       # any contact partner
    fields=("found", "force"),
    reduce="netforce",    # sum all contact points per geom
    history_length=0,
)

# Then read in reward function:
def penalize_sharpcontact(env, force_threshold=1000.0):
    sensor = env.scene["feet_contact"]
    force = sensor.data.force          # [B, 4, 3]
    mean_force = torch.norm(force, dim=-1).mean(-1)  # [B]
    return (mean_force > force_threshold).float()
```

**Why `reduce="netforce"`:** A single foot geom touching the ground generates 2–4 contact points. `maxforce` would discard all but the strongest; `netforce` sums them into the true total force vector — matching Isaac Gym's `net_contact_force_tensor`.

**Why `secondary=None`:** Captures contacts with any partner (ground, ball, other bodies), equivalent to Isaac Gym's unconditional contact tensor.

---

### 1.2 Soft vs Hard Contact Model — Threshold Recalibration

**Symptom:** The `stopball` reward never fired at all during early training runs, even when visual inspection showed the robot touching the ball.

**Root cause:** The original `stopball` threshold of `delta_vy > 2.0 m/s` was calibrated for PhysX (hard LCP contacts), which produce instantaneous large velocity impulses. MuJoCo uses a soft penalty solver (`solref`/`solimp`) that produces smaller, smoother velocity changes for the same physical deflection. A ball deflected from -1.0 m/s to +0.5 m/s (Δv = 1.5 m/s) — a valid save — never fired the reward.

**Fix:** Lower the threshold from 2.0 to 1.0 m/s:

```python
# In mdp/rewards.py
def stopball(env, ball_name="ball", delta_vel_threshold=1.0):  # was 2.0
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    ...
```

**Also:** The `_ball_is_behind` helper used the same 2.0 threshold and had to be updated consistently.

**Key insight:** Physics simulator constants are not portable. Any threshold that depends on contact impulse magnitude (velocity changes, force values) must be recalibrated after a simulator change.

---

### 1.3 Memory Model and num_envs

**Symptom:** The original paper trains with 6144 parallel environments. Initial mjlab runs used 1020 to fit in 8 GB VRAM — but this never needed to be reduced.

**Root cause:** Isaac Gym stores per-environment physics state in separate GPU tensors (joint positions, velocities, contact forces, rigid body transforms, Jacobians). For 6144 envs × 29 DOF, this accumulates to ~10–15 GB just for simulation state. MuJoCo Warp batches all environments into a single contiguous MJX data structure. For 23 DOF, the per-environment footprint is approximately:

```
23 DOF × (pos + vel + acc) ≈ 69 floats × 4 bytes = 276 bytes
6144 envs × 276 bytes ≈ 1.7 MB of simulation state
```

The neural network (~500K params), rollout buffer (6144 × 24 steps × ~200 values), and optimizer states add up to well under 1 GB. Total consumption on an RTX 3070 Laptop (8 GB): stable at 47,693 steps/s.

**Impact:** Restoring 6144 envs gave 6× more samples per gradient update (147,456 vs 24,576), resulting in 6× less noisy gradient estimates and faster convergence with no wall-clock overhead.

---

## 2. Robot Migration

### 2.1 Action Scale Missing 0.25 Factor — Arms Locked

**Symptom:** During early training the T1 robot's arms barely moved. They stayed near the default pose despite the reference motion diving with a wide arm swing. W&B showed `eereach` (hand-to-ball reward) stuck near 0 while `motion_body_pos` (tracking reward) was active — the robot was tracking the motion but not actually moving its arms.

**Root cause:** The G1 actuator formula is `action_scale = 0.25 × effort / stiffness`. The `0.25` factor means the actuator only saturates when the policy outputs ±4.0, not ±1.0. A Gaussian policy with std ≈ 1.0 generates outputs in roughly [-2, +2]; with the 0.25 factor most outputs are in the linear PD regime. Without the 0.25 factor, saturation occurs at ±1.0 — 32% of all Gaussian samples saturate the motor, producing bang-bang torque switching. The arm physically cannot follow commanded positions, appearing locked.

```python
# WRONG — arm saturates at policy output ±1.0
_make_actuator(r"Left_Shoulder_Pitch|...|Right_Elbow_Yaw", effort=18.0, stiffness=40.0)
T1_ACTION_SCALE[arm_joints] = 18.0 / 40.0  # = 0.450 rad/unit

# CORRECT — arm saturates at policy output ±4.0 (matching G1 headroom)
_make_actuator(r"Left_Shoulder_Pitch|...|Right_Elbow_Yaw", effort=18.0, stiffness=15.0)
T1_ACTION_SCALE[arm_joints] = 0.25 * 18.0 / 15.0  # = 0.300 rad/unit
```

**The fix had two parts:**
1. Apply the 0.25 factor to all joints (arms, legs, waist) — not just arms.
2. Lower arm stiffness from 40 → 15 Nm/rad to increase the saturation range to a physically sensible 1.2 rad.

| Parameter | G1 arm | T1 (broken) | T1 (fixed) |
|---|---|---|---|
| Saturation range | ±1.75 rad | ±0.45 rad | ±1.20 rad |
| Action scale | 0.439 | 0.450 | 0.300 |
| Headroom (sat/scale) | **4.0×** | **1.0×** | **4.0×** |

---

### 2.2 Motion Data Worldbody Off-by-One

**Symptom:** Every training episode terminated immediately (step 1) with 100% of environments crashing via `anchor_pos` termination. `error_anchor_pos = 1.25 m` vs threshold of 0.25 m.

**Root cause:** The motion converter used MuJoCo's `data.xpos` directly, which includes the worldbody at index 0 with position (0, 0, 0). mjlab's `MotionLoader` is 0-indexed on robot bodies only (worldbody excluded). As a result, `NPZ[0]` stored the worldbody's fixed (0,0,0) position instead of the pelvis/trunk. Reference State Initialisation (RSI) teleported the robot to z ≈ 0 (the worldbody's position) instead of z ≈ 0.7 m (the trunk standing height).

```python
# WRONG — includes worldbody at index 0
num_bodies = model.nbody          # e.g. 25 (includes worldbody)
body_pos_w[t] = data.xpos.copy()  # NPZ[0] = (0,0,0) worldbody

# CORRECT — skip worldbody
num_bodies = model.nbody - 1      # 24 robot bodies only
body_pos_w[t] = data.xpos[1:].copy()  # NPZ[0] = Trunk position ✓
```

This bug appeared **twice** (in `convert_booster.py` and later in `convert_all.py` for the multi-motion converter), because each converter was written independently without the knowledge of the earlier fix.

---

### 2.3 Coordinate System Rotation — All Task Rewards Broken

**Symptom:** After what appeared to be successful training (the robot could stand and balance), the robot never attempted to stop the ball. The `stopball` reward was 0.0 throughout training. The robot drifted clockwise over time.

**Root cause:** The original G1 goalkeeper has the ball approaching from the +X direction; all reward functions use world-X velocity and position. The T1 port placed the ball on the +Y axis (90° rotation). The reward functions were never updated:

```python
# WRONG — uses world X (original G1 convention)
def stopball(env, ...):
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 0]   # world X
    ball_y_local = ball.data.root_link_pos_w[:, 0]     # world X
    ...

# Since the ball approaches from +Y and ends at negative Y,
# ball_y_local was always negative → stopball condition never satisfied → reward = 0

# CORRECT — use world Y for T1 port
def stopball(env, ...):
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]   # world Y ✓
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env_origins[:, 1]
```

**Affected functions:** `stopball`, `_ball_is_behind`, `stayonline`, `noretreat` — four of the primary task rewards were silently dead. The `noretreat` reward was penalising sideways motion (world -X) instead of retreating from the ball (world -Y), providing no meaningful signal.

**The clockwise drift** was caused by `noretreat` applying a penalty on the wrong axis with no counter-force — the policy rotated to minimise an axis it was being penalised on.

---

## 3. Training Bugs (Single Motion Project)

### 3.1 Ball Never Moved — Event Mode Bug

**Symptom:** The `stopball` reward was consistently 0.14 (near-zero) throughout training. Visual inspection showed the ball always at the robot's feet in the same position, never launching.

**Root cause:** The ball reset event was configured with `mode="startup"` instead of `mode="reset"`. In mjlab, `"startup"` runs exactly once when the simulation initialises; `"reset"` runs at every episode reset.

```python
# WRONG — ball is placed once at startup and never moves
cfg.events["reset_ball"] = EventTermCfg(
    func=gk_resets.reset_ball_training,
    mode="startup",   # only runs once at init!
    params={"ball_name": "ball"},
)

# CORRECT — ball resets with new random trajectory every episode
cfg.events["reset_ball"] = EventTermCfg(
    func=gk_resets.reset_ball_training,
    mode="reset",     # runs at every episode reset ✓
    params={"ball_name": "ball"},
)
```

**Impact:** Without a moving ball, `stopball` (weight 100) never fired. The entire primary task reward was dead. The policy learned to stand still and be balanced because that maximised the stability rewards, which were the only active signals.

---

### 3.2 Stopball Continuous vs Event-Based (50× Overweighted)

**Symptom:** After fixing the ball movement, the `stopball` reward logged 22 units/episode in W&B but the robot was visibly doing less — standing still and letting the ball roll into the body rather than diving.

**Root cause:** The initial port implemented `stopball` as a continuous per-step reward: "if ball is in front and moving slowly, reward = 1". With weight 100, a stationary ball in front of the robot earned `100 × ~200 steps = 20,000 points/episode`. The optimal strategy was passive blocking.

The original fires **exactly once** per episode using an explicit latch flag:

```python
# WRONG — continuous reward, gameable by passive blocking
def stopball_wrong(env):
    in_front = ball_y_local > 0.0
    deflected = ball_y_vel > -0.5       # "slow ball"
    return (in_front & deflected).float()   # fires every step → 20,000 pts/ep

# CORRECT — event-based, fires exactly once per episode
def stopball(env, ball_name="ball", delta_vel_threshold=1.0):
    ball_y_vel   = ball.data.root_link_lin_vel_w[:, 1]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env_origins[:, 1]

    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset]    = False
    env._sb_init_vy[just_reset] = ball_y_vel[just_reset].clone()

    delta_vy = ball_y_vel - env._sb_init_vy
    in_front = ball_y_local > 0.0
    fired    = (delta_vy > delta_vel_threshold) & in_front & ~env._sb_flag

    env._sb_flag |= fired      # latch prevents re-firing
    return fired.float()       # max 1.0 per episode
```

**Key insight:** A logged `stopball` value of 1.0 means the robot reliably deflects the ball once every episode. Values above 1.0 indicate a double-fire bug in the latch logic.

---

### 3.3 Reward Weight Imbalance (stopball 2 → 100)

**Symptom:** Robot standing stationary while the reference motion dived 0.58 m sideways. Motion tracking reward dominated; ball-stopping had no influence.

**Root cause:** The initial mjlab port copied reward weights from the base tracking config rather than the Humanoid-Goalkeeper config. The tracking config had `stopball = 2.0` (a debugging value). The original uses `stopball = 100.0 × dt` in effective per-step scale, which at the 50 Hz episode rate translates to the one-shot bonus being worth the equivalent of a full second of stability reward.

```python
# mjlab base config (wrong starting point)
rewards = {
    "stopball": 2.0,           # 50× too low
    "motion_global_root_pos": 0.5,  # too weak to force the dive
    "motion_body_pos": 1.0,
}

# Corrected weights (matching G1 intent)
rewards = {
    "stopball": 100.0,         # PRIMARY TASK — must dominate
    "motion_global_root_pos": 10.0,  # strong enough to force sideways dive
    "motion_body_pos": 10.0,
    "eereach": 10.0,
    "catch_success": 5.0,
}
```

**The policy logic before the fix:** staying balanced = 4.57 pts; stopping ball = 0.14 pts → "stand still". After the fix: stopping ball = potential 100 pts per episode → "must dive".

---

### 3.4 Motion Playback Speed Wrong (30 fps vs 50 Hz)

**Symptom:** The robot's reference dive animation looked compressed — the full sideways dive completed in ~2.4 s instead of the intended ~4 s, forcing the policy to move faster than physically plausible.

**Root cause:** The motion PKL file was recorded at 30 fps and contained 123 frames. The converter saved all 123 frames as-is. mjlab plays NPZ frames at the policy timestep (dt = 0.02 s = 50 Hz), so the motion ran at `123 × 0.02 = 2.46 s` — 1.37× too fast. The G1 version is also 123 frames at 30 fps but runs in a 4.1 s episode covering only the first 90 frames.

```python
# WRONG — frames saved at source rate (30 fps), played at 50 Hz
frames = pkl_data["dof_pos"]  # 123 frames at 30 fps
np.savez("lefthand_t1.npz", joint_pos=frames, fps=30)
# mjlab plays at 50 Hz → 2.46 s episode for 123 frames (1.37× too fast)

# CORRECT — resample to target rate, trim to target duration
TARGET_FPS = 50.0
TARGET_DURATION = 3.0  # seconds
source_fps = 30.0
source_frames = len(pkl_data["dof_pos"])

# Build new time axis at target fps
t_source = np.arange(source_frames) / source_fps
t_target = np.arange(0, TARGET_DURATION, 1.0 / TARGET_FPS)

# Interpolate joint positions
frames_resampled = np.stack([
    np.interp(t_target, t_source, pkl_data["dof_pos"][:, j])
    for j in range(n_joints)
], axis=-1)  # shape: (150, 23) — exactly 3.0 s at 50 Hz
np.savez("lefthand_t1.npz", joint_pos=frames_resampled, fps=TARGET_FPS)
```

**Also fixed in the same session:** The motion command was calling `_reset_ball` every time the motion clip looped (approximately every 2.4 s), launching a new ball mid-episode. Added a `reset_ball: bool = True` parameter so only true episode resets launch the ball.

---

### 3.5 Entropy Coefficient Without AMP — std Runaway

**Symptom:** After copying `entropy_coef = 0.01` from the G1 config, training collapsed: `mean_std` grew from 1.0 to 5.3+ over 1000 iterations. The robot fell in fewer than 15 steps; episode reward dropped from 42 to 1.

**Root cause:** The G1 training uses AMP discriminators, which provide stabilising counter-gradients: as the policy std grows, AMP rewards decrease (unnatural motion is penalised), creating a negative feedback loop. Without AMP, the entropy gradient is unopposed:

```
entropy gradient → std increases → noisier advantages → larger PPO gradient → std grows further
```

At std = 5.3, approximately 42% of actions exceed ±4.0 (the saturation point), collapsing training.

**Fix:** Reduce entropy coefficient based on empirical testing:

| `entropy_coef` | Observed `mean_std` | Outcome |
|---|---|---|
| 0.01 (G1 original) | 1.0 → 5.3+ | Runaway collapse |
| 0.004 | 1.0 → 2.5+ | Slow runaway |
| 0.002 | Stable ~0.5 | Over-conservative arm exploration |
| 0.005 | Stable ~1.0–1.5 | Good balance (final value) |

The mjlab G1 tracking reference uses `entropy_coef = 0.005`. The port converged on the same value through empirical testing.

---

### 3.6 Redundant Ball Reset Competing With Motion Command

**Symptom:** Ball trajectories during training appeared to ignore the per-motion target zone — some episodes had the ball flying to a random position rather than the intended left-arm intercept zone.

**Root cause:** Two ball-reset mechanisms were active simultaneously:
1. `MultiMotionCommand._reset_ball` — fires at every episode reset, uses motion-biased target ranges (ball aimed at the lefthand intercept zone).
2. `cfg.events["reset_ball"]` — an additional event-based reset using uniform target ranges (-1.2, 1.2) with no directional bias.

Both fired at episode reset; mjlab execution ordering was non-deterministic, so the event-based reset sometimes overwrote the motion-command's carefully computed trajectory.

**Fix:** Remove `cfg.events["reset_ball"]` from the training path entirely. Ball reset is handled solely by the motion command.

---

## 4. AMP-Specific Bugs (Full Goalkeeper Agent)

### 4.1 AMP Normalizer Variance Collapse

**Symptom:** AMP reward was near zero throughout training despite the discriminator appearing to update. Debug prints showed the online normalizer's running variance converging to ~0.01 instead of ~1.0.

**Root cause:** Three compounding errors in the implementation of Chan's parallel algorithm for online variance:

```python
# BROKEN — three compounding errors
def update(self, batch):
    n = batch.shape[0]
    batch_mean = batch.mean(0)
    batch_var  = batch.var(0)

    delta  = batch_mean - self.mean
    # Error 1: count incremented BEFORE cross-term calculation
    self.count += n                             # wrong: should be after
    # Error 2: m_a multiplied by batch size n instead of old_count
    m_a = self.var * n                          # wrong: should be self.var * old_count
    m_b = batch_var * n
    # Error 3: division by self.count + n (double-counts n)
    M2 = m_a + m_b + delta**2 * old_count * n / (self.count + n)  # wrong denominator
    self.var = M2 / (self.count + n)            # wrong: old_count used wrong

# CORRECT — Chan's algorithm
def update(self, batch):
    n = batch.shape[0]
    old_count = self.count                      # save BEFORE increment
    batch_mean = batch.mean(0)
    batch_var  = batch.var(0)
    delta  = batch_mean - self.mean

    self.mean  = self.mean + delta * n / (old_count + n)
    m_a = self.var * old_count                  # correct: use old_count
    m_b = batch_var * n
    M2  = m_a + m_b + delta**2 * old_count * n / (old_count + n)
    self.var   = M2 / (old_count + n)           # correct denominator
    self.count = old_count + n                  # increment LAST
```

**Effect of the bug:** `normalize_torch` divides by `sqrt(var + 1e-8)`. With `var ≈ 0.01` (std ≈ 0.1), the discriminator received inputs amplified by ~10,000×. Gradients exploded and the discriminator produced near-random outputs.

---

### 4.2 Discriminator: 120 Backward Passes Collapsed to 1

**Symptom:** Training crashed with CUDA OOM after a few iterations. Before the OOM, W&B showed disc_loss ≈ 0.0 (suspiciously perfect) from iteration 0.

**Root cause:** The discriminator training loop accumulated all forward passes before calling `backward()` once:

```python
# BROKEN — accumulates all 120 forward passes, one backward
disc_loss_total = 0.0
for _ in range(num_disc_updates):          # 20 iterations
    for motion_name in MOTION_NAMES:       # 6 motions
        expert_obs = get_expert_obs(motion_name)
        policy_obs = get_policy_obs(motion_name)
        expert_loss = disc.compute_loss(expert_obs)
        policy_loss = disc.compute_loss(policy_obs)
        # compute_grad_pen uses create_graph=True — retains computation graph!
        grad_pen = disc.compute_grad_pen(expert_obs, lambda_=5.0)
        disc_loss_total += expert_loss + policy_loss + grad_pen * 0.1

# One backward for 120 forward passes — 120× gradient magnitude
optimizer.zero_grad()
disc_loss_total.backward()
optimizer.step()
```

With `create_graph=True` in `compute_grad_pen`, all 120 computation graphs were kept alive simultaneously — O(120²) VRAM growth — and the single backward step applied a 120× larger gradient than intended.

```python
# CORRECT — one zero_grad/backward/step per disc update iteration
for _ in range(num_disc_updates):
    disc_loss_iter = 0.0
    for motion_name in MOTION_NAMES:
        expert_obs = get_expert_obs(motion_name)
        policy_obs = get_policy_obs(motion_name)
        disc_loss_iter += (
            disc.compute_loss(expert_obs)
            + disc.compute_loss(policy_obs)
            + disc.compute_grad_pen(expert_obs, lambda_=5.0) * 0.1
        )
    optimizer.zero_grad()
    disc_loss_iter.backward()
    optimizer.step()
```

---

### 4.3 AMP Reward Scale 10× Too Weak

**Symptom:** AMP reward contribution was negligible compared to task rewards; the discriminator had no practical influence on policy behaviour.

**Root cause:** The G1 runner uses `predict_reward(...) * 0.5` in the per-discriminator reward formula. The port used `* 0.1`. With `amp_coef = 0.4`, the max AMP contribution per env per step was:
- G1: `0.4 × 0.5 = 0.20`
- Port: `0.4 × 0.1 = 0.04` (5× weaker)

```python
# WRONG — 5× too weak
amp_reward = disc.predict_reward(policy_obs) * 0.1

# CORRECT — matches G1 upstream
amp_reward = disc.predict_reward(policy_obs) * 0.5
```

---

### 4.4 Discriminator Update Frequency 5× Too Low

**Symptom:** The discriminator appeared undertrained relative to the policy — policy learned fast, discriminator reward remained flat.

**Root cause:** G1 trains the discriminator for `5 epochs × 4 mini-batches = 20 gradient steps per iteration`, jointly with PPO. The port ran the discriminator for only `num_mini_batches = 4` steps — 5× fewer.

```python
# WRONG — only 4 disc gradient steps per iteration
for _ in range(num_mini_batches):
    disc_update(...)

# CORRECT — 20 steps (matching G1's 5 epochs × 4 mini-batches)
for _ in range(num_mini_batches * num_learning_epochs):
    disc_update(...)
```

---

### 4.5 Spectral Normalization Missing

**Background:** The G1 discriminator uses plain `nn.Linear` layers; the T1 port added `spectral_norm` as a stability measure.

**Reason:** G1 trains with ~17,000 diverse samples per discriminator per update (1020 envs × 100 steps / 6 motions). With smaller mini-batches, without Lipschitz constraints the discriminator can develop unbounded gradients. Spectral normalization bounds the spectral norm of each weight matrix to ≤1, preventing gradient explosion when batch diversity is lower.

```python
import torch.nn.utils as utils

# Standard nn.Linear
layer = nn.Linear(in_features, out_features)

# Spectrally normalised (T1 port)
layer = utils.spectral_norm(nn.Linear(in_features, out_features))
```

---

## 5. Deployment Bugs

### 5.1 base_lin_vel in World Frame Instead of Body Frame

**Symptom:** The deployed policy behaved erratically on the real robot, drifting unpredictably when the robot's heading was not aligned with the default +X direction.

**Root cause:** MuJoCo freejoint `qvel[0:3]` is the derivative of world-frame position — it is the **world-frame** linear velocity, not the body frame. The deployment code labelled it as body frame:

```python
# WRONG — world-frame velocity labelled as body-frame
base_lin_vel_b = dq[:3]   # qvel[0:3] is WORLD frame!
```

Training (via mjlab) explicitly rotates to body frame:
```python
# mjlab training code (entity/data.py)
root_link_lin_vel_b = quat_apply_inverse(root_link_quat_w, root_link_lin_vel_w)
```

At a 90° yaw, world X and body Y are completely swapped — two of the three velocity axes were wrong.

```python
# CORRECT — rotate world velocity into body frame at deployment
def _rot_world_to_body(q_wxyz: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])
    return R.T @ v_w

base_lin_vel_b = _rot_world_to_body(base_quat, dq[:3])
```

**Note:** `qvel[3:6]` (angular velocity) is already body-frame in MuJoCo — only linear velocity needed the fix.

---

### 5.2 Ball Visibility Masking Absent at Deployment

**Symptom:** The deployed policy reacted unexpectedly during the ball launch window — making premature arm movements when the ball was still at a physically inconsistent location.

**Root cause:** Training applies three masking gates to ball observations. The deployment controller provided raw ball position/velocity with no masking. During the catchstep warmup (~7 steps), the ball has not yet reached a consistent trajectory; exposing raw position gives out-of-distribution values.

```python
# Deployment controller — added visibility state
class GoalkeeperMujocoController:
    def __init__(self):
        self._ball_step = 50           # countdown timer
        self._ball_prev_y = None
        self._ball_vanish_step = random.randint(0, 29)
        self._ball_visible_step = 0
        self.ball_visible = False

    def _compute_ball_visibility(self, ball_pos_b):
        # Gate 1: initial warmup (catchstep)
        initial_vanish = self._ball_step >= 43

        # Gate 2: flying zone
        y_b = ball_pos_b[1]  # local Y (approach axis)
        approaching = (self._ball_prev_y is None) or (y_b < self._ball_prev_y)
        flying = (0.05 < y_b < 3.4) and (abs(ball_pos_b[0]) < 2.0) \
                 and (ball_pos_b[2] < 1.8) and approaching and (self._ball_step > 0)

        # Gate 3: random vanish
        if flying:
            self._ball_visible_step += 1
        else:
            self._ball_visible_step = 0
        random_vanish = self._ball_visible_step <= self._ball_vanish_step

        return (not initial_vanish) and flying and random_vanish
```

---

## 6. Ball Visibility Masking System

The upstream G1 goalkeeper applies three masking conditions to hide the ball observation from the policy, forcing it to learn to act on uncertainty:

| Gate | Condition | Purpose |
|---|---|---|
| `initial_vanish` | `catchstep < startstep` (~7 steps at episode start) | Ball hasn't reached a consistent trajectory yet; hide to prevent the policy from reacting to physically implausible states |
| `flying` | Ball in zone (y: 0.05–3.4 m, \|x\| < 2 m, z < 1.8 m) and approaching | Ball is visible only while in the valid catch volume and moving toward the robot |
| `random_vanish` | `ball_visible_step > vanish_step` (random 0–30 steps per episode) | Simulate occlusion; force the policy to commit to a trajectory without perfect observation |

The full mjlab implementation:

```python
def _compute_ball_visibility(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Returns boolean [N] mask — True when ball is visible to the policy."""
    ball = env.scene["ball"]
    robot = env.scene["robot"]

    ball_pos_w = ball.data.root_link_pos_w       # [N, 3] world frame
    env_origins = env.scene.env_origins           # [N, 3]
    root_pos_w  = robot.data.root_link_pos_w      # [N, 3]
    root_quat_w = robot.data.root_link_quat_w     # [N, 4]

    # Ball in robot body frame
    ball_pos_b = quat_rotate_inverse(root_quat_w, ball_pos_w - root_pos_w)

    y_b = ball_pos_b[:, 1]                        # approach axis
    x_b = ball_pos_b[:, 0]
    z_b = ball_pos_b[:, 2]

    # Gate 1: initial vanish (catchstep warmup)
    catchstep  = env._catchstep                   # [N] int, counts down from 50
    startstep  = env._startstep                   # [N] int, ~43
    initial_vanish = catchstep < startstep        # True = hidden

    # Gate 2: flying zone + approaching
    approaching = (ball_pos_b[:, 1] < env._ball_last[:, 1]) | (env._ball_last[:, 1] == 0)
    flying = ((y_b > 0.05) & (y_b < 3.4)
              & (x_b.abs() < 2.0)
              & (z_b < 1.8)
              & approaching
              & (catchstep > 0))

    # Gate 3: random vanish (per-env threshold sampled at reset)
    env._ball_visible_step = torch.where(flying, env._ball_visible_step + 1,
                                         torch.zeros_like(env._ball_visible_step))
    random_vanish = env._ball_visible_step > env._vanish_step   # True = visible

    env._ball_last = ball_pos_b.clone()

    visible = (~initial_vanish) & flying & random_vanish
    return visible.float()
```

Ball position and velocity observations multiply by `visible.float()`, zero-ing them when the ball is not in the valid state.

---

## 7. Lessons Learned

**1. Simulator constants are not portable.** Contact impulse magnitudes, force thresholds, and velocity deltas all depend on the contact model. Any threshold calibrated in one simulator must be re-measured in the new one — do not copy constants blindly.

**2. Check every event mode flag.** The `mode="startup"` vs `mode="reset"` bug completely broke training without a visible error. When a reward is consistently near zero, the first check should be whether the underlying event even fires.

**3. The 0.25 action scale factor is load-bearing.** Without it, 32% of Gaussian policy samples saturate the actuator, locking joints. All joint groups must use `0.25 × effort / stiffness`.

**4. AMP bugs fail silently.** The normalizer variance collapse, backward pass accumulation, and reward scale errors each produced training runs that looked plausible in W&B but were fundamentally broken. Validate AMP by printing discriminator reward statistics (should be non-zero and varying) and normalizer variance (should converge to ~1.0 for standardised inputs).

**5. Coordinate system must be decided once and enforced everywhere.** The 90° rotation bug touched four reward functions, two observation functions, and the ball reset. A single source-of-truth constant (`BALL_APPROACH_AXIS = 1` for +Y) and systematic grep of all axis indices would have caught this in one pass.

**6. Worldbody indexing appears in every MuJoCo converter independently.** When writing a new motion converter, always check: does `data.xpos[0]` refer to the worldbody (always at origin) or the first robot body? The answer is always: worldbody first, so always `data.xpos[1:]`.

**7. Increase num_envs before tuning.** Many training instabilities vanished when `num_envs` was restored from 1020 to 6144. The 6× larger batch gave gradient estimates stable enough to reveal whether reward signals were actually working.

**8. Cross-reference the upstream source verbatim.** The most effective debugging approach was reading every upstream function with exact line numbers before implementing. Multiple reactive debug sessions failed to find bugs that a single exhaustive source comparison caught in one pass.
