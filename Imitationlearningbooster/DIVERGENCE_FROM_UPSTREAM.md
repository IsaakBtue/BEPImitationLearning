# Divergence from Upstream (Humanoid-Goalkeeper)

## 2026-05-24 — 8 missing features implemented in mjlab port

**Scope:** Eight features from the upstream `Humanoid-Goalkeeper` Isaacgym training pipeline were implemented into the mjlab port (`my_mjlab_project_booster_t1`). All upstream code was read verbatim before implementation. Coordinate system note: upstream uses X as the ball approach axis; port uses Y.

---

### Feature P1B: `successland` — add `_has_in_air` tracking

**Original (`legged_robot.py` lines 1445–1466):**
```python
def _reward_successland(self):
    foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
    jump = self.root_states[:,2] > 1.0
    self.has_in_air = torch.logical_or(self.has_in_air, jump)
    has_contact = (foot_contact_forces_z[:, 0] > 1.) & (foot_contact_forces_z[:, 1] > 1.)
    one_feet_contact = (((foot_contact_forces_z[:, 0] >  1.) & (foot_contact_forces_z[:, 1] < 1.)) | ((foot_contact_forces_z[:, 0] <  1.) & (foot_contact_forces_z[:, 1] > 1.))) & (self.has_in_air)
    successful_landings = torch.logical_and(has_contact, self.has_in_air)
    air_reward = self.has_in_air.float()
    landing_reward = successful_landings.float() * 5.0
    one_feet_punish = one_feet_contact.float() * -1.0
    jump_ids = (self.end_regions == 2) | (self.end_regions == 3)
    return (air_reward + landing_reward + one_feet_punish) * jump_ids
```

**What was wrong:** Port version fired whenever `behind AND feet_down` — it never tracked whether the robot had actually left the ground, and had no landing bonus (+5×) or one-foot penalty (-1). The `jump_ids` gate (jump-region envs only) was also absent.

**Fix:** Added `env._has_in_air` bool tensor (reset on `episode_length_buf <= 1`), set via `root_z - env_z > 1.0`, gate reward to require `_has_in_air`. Landing bonus (+5×) and one-foot penalty (-1) added. Since port has no region partitioning, `jump_ids` gate replaced by the `_has_in_air` requirement (equivalent — only fires after actual jumps).

**File:** `mdp/rewards.py` → `successland()`

---

### Feature P1C: Fix force averaging in `penalize_sharpcontact` and `sharpforce_termination`

**Original (`legged_robot.py` lines 1475–1477, 258):**
```python
def _reward_penalize_sharpcontact(self):
    return (torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim=-1) > self.cfg.rewards.max_contact_force) * 1.0
# termination:
sharpforce_buf = torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim=-1) > 1.5 * self.cfg.rewards.max_contact_force
```
`contact_feet_indices` has 2 entries (two ankle-roll-link bodies).

**What was wrong:** Port averaged over all 4 foot geoms with a flat mean — this underestimates peak force by 50% on single-foot impacts, since the two geoms of the unloaded foot contribute zeros that halve the mean.

**Fix:** Per-foot max over geoms, then mean over two feet — matches upstream 2-body mean semantics:
```python
force_per_geom = sensor.data.force.norm(dim=-1)   # [B, 4]
left_max  = force_per_geom[:, :2].max(dim=-1).values
right_max = force_per_geom[:, 2:].max(dim=-1).values
mean_force = (left_max + right_max) / 2.0
```
Applied to both `penalize_sharpcontact()` (rewards.py) and `sharpforce_termination()` (resets.py).

---

### Feature P2A: Ball difficulty curriculum

**Original (`legged_robot.py` lines 333–336, `assign_ball_states` lines 784–788):**
```python
# reset_idx curriculum:
self.command_ranges[:, 0] = torch.clip(self.command_ranges[:, 0] - 0.3 * self.curriculumupdate, self.command_bound[:,0], self.command_bound[:,1])
self.command_ranges[:, 1] = torch.clip(self.command_ranges[:, 1] + 0.3 * self.curriculumupdate, self.command_bound[:,0], self.command_bound[:,1])
# ... similar for height ranges
# assign_ball_states:
ball_end_local = torch.stack([...
    torch.rand(len(ball_ids)) * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0]) + self.command_ranges[ball_ids, 0],
    torch.rand(len(ball_ids)) * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2]) + self.command_ranges[ball_ids, 2]
], dim=1)
```

**What was wrong:** Port's `_reset_ball` used fixed `_BALL_END_RANGES` with no curriculum expansion.

**Fix:**
- Added `_BALL_END_RANGES_EASY` (difficulty=0): x_end ∈ (−0.40, −0.20), z_end ∈ (0.55, 1.05)
- Added `env._ball_difficulty` float (0.0–1.0), linearly interpolated in `_reset_ball`
- Added `ball_difficulty_curriculum` class in `resets.py`, registered in `cfg.curriculum`
- Stages: step=0→0.0, step=stage1→0.5, step=stage2→1.0

**Files:** `mdp/commands.py` → `_reset_ball()`, `mdp/resets.py` → `ball_difficulty_curriculum`, `tasks/goalkeeper_env_cfg.py` → curriculum

---

### Feature P2B: `dof_pos_limits` and `torque_limits` curriculum

**Original (`legged_robot.py` lines 366–373):**
```python
if self.curriculumupdate > 1.0:
     self.reward_scales["dof_pos_limits"] = self.dof_pos_init * 2.0
     self.reward_scales["torque_limits"]  = self.torque_init  * 2.0
if self.curriculumupdate > 2.0:
     self.reward_scales["dof_pos_limits"] = self.dof_pos_init * 3.0
     self.reward_scales["torque_limits"]  = self.torque_init  * 3.0
```
`dof_pos_init = torque_init = -3.0 × dt`.

**What was wrong:** Port used fixed weights (-3.0) with no scaling.

**Fix:** Added `dof_pos_limits_curriculum` and `torque_limits_curriculum` `CurriculumTermCfg` entries in `cfg.curriculum`, scaling from -3.0 → -6.0 → -9.0 at stage1/stage2 steps.

**File:** `tasks/goalkeeper_env_cfg.py`

---

### Feature P2C: `hand_proximity_strict` curriculum

**Original (`legged_robot.py` lines 362–363):**
```python
if "success" in self.reward_scales:
    self.reward_scales["success"] = self.success_init * (1 + 0.5 * self.curriculumupdate)
```
`success_init = 5.0 × dt`. curriculumupdate 0→1→2 gives weight 5→7.5→10.

**What was wrong:** Port's `hand_proximity_strict` had fixed weight 5.0.

**Fix:** Added `hand_proximity_strict_curriculum` `CurriculumTermCfg` in `cfg.curriculum`, scaling 5.0 → 7.5 → 10.0 at stage1/stage2 steps.

**File:** `tasks/goalkeeper_env_cfg.py`

---

### Feature P3A: `eereach` target uses predicted intercept point + Phase 1

**Original (`legged_robot.py` lines 797–808, 1361–1400):**
```python
# assign_ball_states: compute intercept point
catch_prop = (0.1 - ball_start_local[:,0:1]) / (ball_end_local[:,0:1] - ball_start_local[:,0:1])
self.end_target[ball_ids,:] = self.ball_start[ball_ids,:] + delta_pos * catch_prop

# post_physics_step: update when ball is close (0.1–0.5 m from robot)
approachidx = ((balllocal < 0.5) & (balllocal > 0.1) & ...).nonzero(as_tuple=False).flatten()
self.end_target[approachidx, :] = self.ball_states[approachidx, :3].clone()

# _reward_eereach Phase 1 (ball far, x_local > 1.5):
asidegoal = clip(end_target_local[:, 1], -1, 1)
asidegoal[|asidegoal| < 0.3] = 0
verticalgoal = clip(torso_z - clip(end_target[:, 2], 0.3, 1.2), 0, 1)
phase1_rew = 1 - (verticalgoal + |asidegoal|) / 2
taskrew[phase1] = phase1_rew[phase1]
```

**What was wrong:** Port's `eereach()` always used current ball position for distance computation — no prediction of intercept point, no Phase 1 pre-positioning reward.

**Fix:**
- `_reset_ball` now computes `catch_prop = (y_start - 0.1) / (y_start + 0.3)` (Y-axis equivalent of upstream X formula) and stores `env._ball_end_target` in world frame.
- `eereach()` now uses Phase 1 (ball y_local > 1.5): lateral+vertical pre-positioning reward.
- Phase 2 (ball y_local ≤ 1.5): uses `end_target` when ball > 0.5 m out, snaps to ball when ≤ 0.5 m.

**Files:** `mdp/commands.py` → `_reset_ball()`, `mdp/rewards.py` → `eereach()`

---

### Feature 7: Catchstep warmup — mask ball observations during launch

**Original (`legged_robot.py` lines 643, 968, 178, 392–403):**
```python
# _compute_torques:
self.joint_pos_target[self.catchstep > self.startstep] = self.init_dof_pos[self.catchstep > self.startstep]
# _init_buffers:
self.catchstep = 50 * torch.ones(self.num_envs, dtype=torch.int, device=self.device)
self.startstep = 50 - random.randint(3, 10)
# post_physics_step:
self.catchstep -= 1
# compute_observations:
initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
end_target_local = ... * initial_vanish   # zero ball obs during warmup
```

**What was wrong:** Port had no catchstep warmup — ball observations were visible immediately at episode start, giving the policy an unrealistic view of a ball that hasn't reached a physically consistent trajectory yet.

**Fix:**
- `_reset_ball` stores `env._catchstep[env_ids] = 50` (int tensor).
- `_update_command` decrements it each step: `_catchstep = (_catchstep - 1).clamp(min=0)`.
- `ball_pos_b` and `ball_vel_b` in observations.py apply `initial_vanish = (catchstep < 43)` — ball is hidden for the first ~7 steps (when catchstep ≥ 43 = startstep equivalent).
- This also initialises `env._ball_end_target` storage for the intercept point (P3A).

**Files:** `mdp/commands.py` → `_reset_ball()`, `_update_command()`, `mdp/observations.py`

---

### Feature 8: Ball visibility masking curriculum

**Original (`legged_robot.py` lines 392–428):**
```python
initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
end_target_local = quat_rotate_inverse(...) * initial_vanish

flying = ((end_target_local[:,0] > 0.05) & (end_target_local[:,0] < 3.4) &
          (end_target_local[:,1] > -2.0) & (end_target_local[:,1] < 2.0) &
          (end_target_local[:,2] < 1.8) & (self.catchstep > 0.) &
          ((end_target_local[:,0] < self.ball_last[:,0]) | (self.ball_last[:,0] == 0.))).view(-1, 1)
random_vanish = (self.catchstep > self.vanish_step).view(-1, 1)
self.ball_last = end_target_local

# actor_obs[:, :num_ballobs] = actor_obs[:, :num_ballobs] * flying * random_vanish
# (with noise: also multiplied by random_vanish; without noise: only by flying)
```

**What was wrong:** Port exposed ball position/velocity to the policy at all times — no flying-zone check, no random disappearance, no warmup masking. This lets the policy see the ball when it's behind the robot or in physically implausible states.

**Fix:** Added `_compute_ball_visibility()` helper in `observations.py` that implements all three conditions:
1. `initial_vanish`: `_catchstep < 43` (startstep ≈ 43)
2. `flying`: y_local ∈ (0.05, 3.4), |x_local| < 2.0, z < 1.8, approaching, catchstep > 0
3. `random_vanish`: `_ball_visible_step > _vanish_step` (per-env random threshold 0–30 steps)

Both `ball_pos_b` and `ball_vel_b` multiply output by `visible.float()`, zeroing the observation when ball is not visible.

**File:** `mdp/observations.py`

---

## 2026-05-24 — Full conversion feasibility: AMP, HIM-PPO, and algorithmic gap analysis

**Scope:** Feasibility assessment for a complete port of the Humanoid-Goalkeeper Isaac Gym pipeline (HIM-PPO + 6× AMP + MotionLib) into mjlab/MuJoCo. Two independent research passes cross-referenced upstream source code, the mjlab port, and Isaac Lab's installed AMP infrastructure. Key finding: **the full conversion is feasible** — it is a research engineering task (~1500 lines of new code), not a configuration exercise. No component is a blocking limitation.

**AMP availability clarification:** AMP is **not** in rsl_rl 5.0.1 (the mjlab venv backend), but Isaac Lab ships a working AMP pipeline at `/home/isaak/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/` (SKRL backend). A Booster T1 AMP prototype already exists at `/home/isaak/Humanoid_Imitation-Learning/Isaaclab_Humanoid_Booster_AMP/`. The path for the mjlab port is to port the upstream custom rsl_rl fork directly rather than use the SKRL route.

---

### 1. AMP (Adversarial Motion Priors)

**Upstream architecture** (`Humanoid-Goalkeeper/rsl_rl/rsl_rl/`):

| Component | Location | Description |
|---|---|---|
| `AMP` discriminator class | `modules/amp.py` | GAIL + spectral norm + gradient penalty λ=5; architecture: 58→512→ReLU→256→1 |
| 6 motion-keyed discriminators | `algorithms/him_ppo.py` lines 31–37 | One per motion type; routed per-env via `critic_obs[:, num_one_step_obs+3]` |
| LSGAN loss + gradient penalty | `modules/amp.py compute_loss()` | `(expert_d−1)² + (policy_d+1)²` + one-sided expert penalty |
| MC-smoothed reward | `modules/amp.py predict_reward()` | 20 perturbations σ=0.3, `clamp(1 − 0.25·min_se, 0)` |
| AMP reward blending | `runners/him_on_policy_runner.py` line 185 | `rewards = amp_reward × 0.4 + task_reward × 0.6` |
| AMP obs: 58-D | `envs/base/legged_robot.py get_amp_observations()` | Two consecutive frames of 29 joint positions |
| Expert buffer with temporal jitter | `envs/g1/g1_utils.py MotionLib.get_expert_obs()` lines 158–189 | Bilinear interp + fps jitter U(0.25, 1.25) |
| Running AMP normalizer | `rsl_rl/utils/utils.py` lines 108–160 | `RunningMeanStd` over 58-D, updated from both expert and policy batches |

**Current port status:** Not implemented. The port uses explicit L2 penalty terms (`dof_acc`, `torques`, `dof_vel`, `ang_vel_xy`, `action_acc_l2`) as a substitute for the discriminator's implicit regularization. AMP's 40% reward contribution is absent, and `entropy_coef` required recalibration (0.01 → 0.01 with supporting penalties) to avoid std runaway.

**Feasibility:** Portable with medium effort. The `AMP` class and `SpectralNorm` (~100 lines) are pure PyTorch with no Isaac Gym dependency. Key porting steps:
1. Copy `AMP` class into `Imitationlearningbooster/` — zero dependency changes.
2. Add `amp_obs` field (shape `[T, N, 40]` for T1's 20 joints × 2 frames) to a `RolloutStorage` subclass.
3. Add `MotionLib.get_expert_obs()` equivalent with bilinear temporal jitter over `.npz` joint_pos tensors.
4. Subclass rsl_rl `PPO.update()` to add: discriminator optimizer groups, `AMP.compute_loss()`, `Normalizer.update()`.
5. Add `get_amp_observations()` env method → `robot.data.joint_pos` (shape `[N, 20]`).
6. Down-weight explicit `motion_body_*` tracking rewards (they conflict with discriminator gradients).
- **Estimated scope:** ~400 lines. No mjlab framework changes required.

**Residual limitation (scientific, not blocking):** The upstream uses 6 motion-keyed discriminators. The mjlab port currently has only 1 motion type (`lefthand_t1.npz`). A single-discriminator AMP is fully valid and matches what Isaac Lab's canonical AMP task implements. Expanding to 6 motion types requires converting 5 additional motion files to `.npz`, which is a data preparation task.

---

### 2. HIM-PPO Internal Model

**Upstream architecture** (`Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/actor_critic.py`):

Three auxiliary sub-networks operate on the full 960-D history:

| Head | Architecture | Loss | Purpose |
|---|---|---|---|
| **History encoder** | 960→128→64→16 | PPO gradient (implicit) | 16-D latent compressing hidden system state |
| **Ball estimator** | 960→128→32→6 | MSE vs `critic_obs[:, −13:−7]` | Supervises ball 3D pos+vel inference |
| **Region estimator** | 960→128→32→6 | CrossEntropy vs `critic_obs[:, −14]` | 6-class interception region |

Actor MLP input = **119-D** (96 one-step + 16 encoder latent + 6 ball est + 1 region argmax), not the raw 960-D history.

Additional loss terms in `him_ppo.py` `update()`:
- `est_loss`: MSE(ball_estimate, privileged_ball_state)
- `region_loss`: CrossEntropy(region_logits, region_id)
- `smooth_loss`: `‖π(obs)−π(interp_obs)‖² + ‖V(obs)−V(interp_obs)‖²` — temporal Lipschitz regularization via mixup interpolation of consecutive obs pairs

**Current port status:** Not implemented. Port uses flat 900-D input to standard MLP. `action_acc_l2` penalty substitutes for `smooth_loss` but is not mathematically equivalent.

**Feasibility:** Requires framework fork (portable, medium effort). The three sub-networks are ~150 lines of pure PyTorch. The `HIMPPO`, `HIMRolloutStorage`, `HIMOnPolicyRunner` classes total ~600 lines with zero Isaac Gym dependency — they depend only on `torch` and the rsl_rl `VecEnv` interface. Strategy: port the entire upstream rsl_rl fork into `Imitationlearningbooster/` as a standalone package and adapt `HIMOnPolicyRunner` to unpack mjlab's `TensorDict` observations into the flat `actor_obs`/`critic_obs` tensors `HIMPPO.act()` expects.
- **Estimated scope:** ~600 lines (mostly copy from upstream with interface adapter).

---

### 3. Asymmetric Actor-Critic (Privileged Observations)

**Upstream:** `HIMRolloutStorage` stores separate `observations` (actor) and `privileged_observations` (critic). Critic receives 113-D: actor obs + lin_vel + region_id + end_target + ball_vel + hand positions + hand-ball dist. Source: `storage/him_rollout_storage.py`.

**Current port status:** Both actor and critic receive identical 900-D observations. Ball velocity and hand positions were dissolved into the actor observation instead of being privileged.

**Feasibility:** **Native in rsl_rl v5 / mjlab** — no porting required. The `obs_groups` dict in `RslRlOnPolicyRunnerCfg` directly supports named actor/critic groups:
```python
obs_groups: dict = {"actor": ["policy"], "critic": ["critic"]}
```
Source: `/home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1/.venv/lib/python3.12/site-packages/mjlab/rl/config.py` lines 92–93. Restoring the asymmetric split requires only redefining the observation groups in `goalkeeper_env_cfg.py` — moving ball_vel, hand_pos, and hand-ball-dist out of the actor group and into a separate critic group.
- **Estimated scope:** ~20 lines in `goalkeeper_env_cfg.py`. Zero new code.

---

### 4. Ball Masking Curriculum

**Upstream** (`legged_robot.py` lines 397–428): Three masking conditions applied to actor ball_pos:
- `initial_vanish`: ball hidden until step `catchstep >= startstep` (~40–47 steps, decays with curriculum)
- `random_vanish`: ball hidden after `catchstep > vanish_step` (random 0–30 per reset)
- `flying`: ball visible only when inside valid catch volume (x 0.05–3.4 m, y ±2 m, z < 1.8 m) and moving toward robot

**Current port status:** Absent. Policy sees continuous unmasked ball position throughout every episode — a sim-to-real gap if vision is occluded.

**Feasibility:** Portable (low effort). mjlab `ObservationTermCfg` accepts custom `func` callables. The three masking conditions are pure tensor logic on environment state, wrappable as a custom observation term in `mdp/observations.py` (file already exists).
- **Estimated scope:** ~50 lines.

---

### 5. MotionLib Expert Buffer with Temporal Jitter

**Upstream** (`g1_utils.py` lines 158–189): `MotionLib.get_expert_obs()` samples random clip + random time with per-sample temporal ratio `ratio = (fps/env_fps) × U(0.25, 1.25)`, interpolates between adjacent frames, returns concatenated two-frame vector.

**Current port status:** `MotionLoader` in mjlab only exposes the current reference frame for the current timestep — no random-access sampling for discriminator training.

**Feasibility:** Portable (medium effort). `MotionLib` class is ~200 lines of pure PyTorch with no Isaac Gym dependency. Copy verbatim into `Imitationlearningbooster/`. Integration into the AMP training loop (Component 1 Step 3) is the main work.
- **Estimated scope:** ~150 lines (class copy + integration).

---

### 6. 6-Motion RSI Coverage

**Upstream:** `num_envs` partitioned into 6 equal groups at init, each assigned a motion type and catch region. Reset uses the partition to initialize joint state from the relevant motion trajectory.

**Current port status:** 1 motion type, `sampling_mode="start"` (always frame 0).

**Feasibility:** Portable (low effort). A custom `EventTermCfg` reset function in `mdp/resets.py` (file exists) can partition envs across motion types and sample init poses from `MotionLib`.
- **Estimated scope:** ~40 lines.

---

### 7. Physics: PhysX vs MuJoCo Contact Model

| Property | G1 / PhysX | T1 / MuJoCo | Portability |
|---|---|---|---|
| Contact model | Hard complementarity (LCP/TGS) | Soft penalty (solref/solimp) | Scientific difference — not portable, requires recalibration |
| Ball restitution DR [0.0, 1.0] | ✅ direct coefficient | No native DR primitive | Custom `EventTermCfg` modifying `model.pair_solref` (~30 lines) |
| Foot friction DR [0.1, 2.0] | ✅ | ✅ via `mjlab_dr.geom_friction` | Already partially implemented |
| Joint limits | Hard constraint | Soft solimp penalty | Behavioral difference at limits |
| Armature | Not applicable | Added: `stiffness/(2π×10)²` | Already implemented |

**Practical calibration already done:** `stopball delta_vy` threshold recalibrated 2.0→1.0 m/s (MuJoCo softer contacts produce smaller velocity deltas). Force thresholds (sharpcontact 1000 N, termination 1500 N) carried from upstream but not yet validated against MuJoCo profiles.

**Residual scientific limitation:** PhysX and MuJoCo use fundamentally different contact solvers. Ball-bounce trajectories at the same nominal friction/restitution values will differ quantitatively. This is a **scientific validity concern for the BEP** (policies trained in MuJoCo may behave differently than a PhysX-equivalent), not a blocking implementation concern.

---

### 8. Domain Randomization — Partially Disabled

| DR Type | G1 Original | mjlab Port | Status |
|---|---|---|---|
| Kp gain scale [0.8, 1.2] | ✅ per reset | Configured but **disabled** | Re-enable after stable locomotion |
| Kd gain scale [0.8, 1.2] | ✅ per reset | Configured but **disabled** | Same |
| Link mass scale [0.8, 1.2] | ✅ per reset | Configured but **disabled** | Same |
| Initial joint pos offset ±0.1 rad | ✅ per reset | Configured but **disabled** | Same |
| Ball mid-trajectory perturbation ±0.5 m/s every 0.5 s | ✅ | **Not implemented** | ~20 lines in `mdp/commands.py` |
| Ball restitution [0.0, 1.0] | ✅ per reset | **Not implemented** | Custom event, ~30 lines |
| Push robot ±1.5 m/s | ✅ every 15 s | ✅ reduced ±0.5 m/s | Intentional for early training |
| Foot friction [0.1, 2.0] | ✅ | ✅ [0.3, 1.2] | Narrower range |
| Actuator command delay (2–8 steps) | ✅ action-level | ✅ actuator-pipeline level | KaydenKnapik values |
| Obs noise: joint_vel ±1.5 | ✅ | ✅ | Verified vs KaydenKnapik |

The four disabled DR terms must be re-enabled before any sim-to-real transfer attempt.

---

### 9. Feasibility Summary

| Component | Verdict | Effort estimate |
|---|---|---|
| AMP 6× discriminators | Portable — custom rsl_rl subclass | ~400 lines |
| HIM-PPO internal model + aux losses | Portable — port upstream rsl_rl fork | ~600 lines |
| Asymmetric actor-critic | **Native** in rsl_rl v5 / mjlab | ~20 lines (config only) |
| Ball masking curriculum | Portable (low) | ~50 lines |
| MotionLib expert buffer + temporal jitter | Portable (medium) | ~150 lines |
| 6-motion RSI | Portable (low) | ~40 lines |
| Physics DR (friction ✅, restitution custom) | Portable (medium) | ~30 lines |
| PhysX vs MuJoCo contact semantics | Scientific gap — recalibrate thresholds | Ongoing |

**Overall:** Full conversion is feasible. No blocking limitations. Total estimated new code: ~1,300 lines in `Imitationlearningbooster/`, no mjlab framework modifications. Strategy is to port the upstream rsl_rl custom fork as a standalone package and wire it to mjlab's `VecEnv` interface. The MuJoCo vs PhysX contact model difference is a scientific consideration to characterize in the BEP, not an obstacle.

---

## 2026-05-20 — KaydenKnapik hardware-verified actuator config; joint_vel noise; actuator delay

**Files:** `robots/t1_constants.py`, `mdp/rewards.py`, `assets/booster_t1/T1_serial_clean.xml`, `tasks/goalkeeper_env_cfg.py`

**Reference:** `https://github.com/KaydenKnapik/BoosterT1mjlab` — successfully deployed RL on real T1 hardware. Cloned at `/home/isaak/BEPImitationlearning/BoosterT1mjlab/`. Two Haiku subagents independently verified all values before applying.

### 1. Effort limits updated to KaydenKnapik hardware-verified values

**What:** `robots/t1_constants.py` actuator `effort_limit` values and `mdp/rewards.py` `_T1_EFFORT_MAP` updated.

**Why it was wrong:** Our original values came from `T1_serial_clean.xml` `actuatorfrcrange`. KaydenKnapik's values are higher (especially ankles: 15→50 Nm, arms: 18→36 Nm) and represent what the real hardware can actually sustain. Using artificially low limits over-penalised torques that are physically achievable, biasing the policy toward weaker actions than necessary.

| Joint group | Old value | New value (KaydenKnapik) |
|---|---|---|
| Arms (Shoulder/Elbow 4×2) | 18 Nm | **36 Nm** |
| Waist | 30 Nm | **40 Nm** |
| Hip Pitch | 45 Nm | **55 Nm** |
| Hip Roll / Hip Yaw | 30 Nm | **40 Nm** |
| Knee Pitch | 60 Nm | **65 Nm** |
| Ankle Pitch | 20 Nm | **50 Nm** |
| Ankle Roll | 15 Nm | **50 Nm** |

**Impact on T1_ACTION_SCALE:** `T1_ACTION_SCALE[joint] = 0.25 × effort / stiffness`. Stiffness values unchanged. Arm scale doubles (0.30 → 0.60), ankle scales increase significantly (0.10 → 0.31 for ankle_pitch).

---

### 2. actuatorfrcrange removed from T1_serial_clean.xml

**What:** All `actuatorfrcrange` attributes removed from `T1_serial_clean.xml`.

**Why it was wrong:** MuJoCo applies `actuatorfrcrange` as a hard joint-level force clamp independent of Python's `effort_limit`. With both active, the tighter (XML) value wins — making the Python effort_limit irrelevant for arms and ankles. KaydenKnapik's XML has **no actuatorfrcrange at all**; Python `effort_limit` is their only hard clamp. After updating Python effort_limits to KaydenKnapik values, the old XML clamps (e.g., ankle ±15 Nm) would override the new Python limits (50 Nm), defeating the upgrade entirely. Solution: remove XML clamps and let Python be the sole constraint, matching KaydenKnapik exactly.

**Evidence:** KaydenKnapik XML `grep actuatorfrcrange` → 0 results. Our XML had 22 instances.

---

### 3. Actuator command delay added (2–8 timesteps)

**What:** `delay_min_lag=2, delay_max_lag=8` added to all actuators in `robots/t1_constants.py`.

**Why it was missing:** KaydenKnapik applies delay at the actuator command level (not obs level) to simulate network/motor controller latency on real hardware. Without this, the policy sees immediate actuator response which is unrealistic. At 200 Hz, 2–8 timesteps = 10–40 ms latency, consistent with real T1 motor controller response time.

**Implementation:** Added `_DELAY_MIN = 2, _DELAY_MAX = 8` constants and passed to `_make_actuator()`.

---

### 4. Joint velocity observation noise increased (±0.5 → ±1.5)

**What:** `joint_vel` obs noise in actor observations raised from ±0.5 (base `tracking_env_cfg.py` default) to ±1.5 (KaydenKnapik value).

**Why it was wrong:** The base tracking config uses ±0.5, but KaydenKnapik's hardware-tuned setup uses ±1.5. Real joint velocity encoders have higher noise than position encoders; ±0.5 rad/s underestimates the noise seen on real hardware, creating a sim2real gap.

**Implementation:** `goalkeeper_env_cfg.py` overrides `joint_vel` in all obs groups after calling `make_tracking_env_cfg()`.

---

## 2026-05-20 — IMU velocity noise added; torque_limits upgraded to per-joint map

**Files:** `tasks/goalkeeper_env_cfg.py`, `mdp/rewards.py`

### 1. IMU velocity observation noise

**What:** Added `noise=Unoise(n_min=-0.1, n_max=0.1)` to `base_lin_vel` and `noise=Unoise(n_min=-0.2, n_max=0.2)` to `base_ang_vel`.

**Why it was wrong:** Both obs were replaced with direct state reads (no IMU sensor on T1) but noise was not added. G1 explicitly applies `lin_vel=0.1` and `ang_vel=0.2` noise in `g1_29_config.py`. The absence of any noise created a training/real gap.

**Evidence:** Cross-reference with G1 original at `legged_gym/legged_gym/envs/g1/g1_29_config.py`. Confirmed by sub-agent code verification.

---

### 2. torque_limits: universal 50 Nm cap → per-joint _T1_EFFORT_MAP

**What:** `gk_rew.torque_limits` now uses `_T1_EFFORT_MAP` for per-joint soft limit enforcement (updated to KaydenKnapik values above).

**Why it was wrong:** Arms (now 36 Nm) were penalised only above 47.5 Nm with the old universal 50 Nm cap. After KaydenKnapik update, arms are penalised above 34.2 Nm — more than 2× stricter. Universal caps mask per-joint constraint violations.

**Implementation:** `_T1_EFFORT_MAP` dict in `rewards.py`, cached as `env._t1_effort_limits` tensor.

---

## 2026-05-15 — num_envs restored from 1020 → 6144 (MuJoCo Warp vs Isaac Gym memory model)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** `cfg.scene.num_envs` raised back to 6144 (original upstream value). Previously set to 1020 to fit in 8 GB VRAM.

**Why this is now possible — Isaac Gym vs MuJoCo Warp memory model:**

RL training is GPU-compute bound, not VRAM bound, once you understand how each simulator stores state.

**Isaac Gym (original)** stored every environment's full physics state — joint positions, velocities, contact forces, rigid body transforms, jacobians — as separate GPU tensors, one slice per environment. For 6144 envs × a 29-DOF humanoid, this accumulated to ~10–15 GB just for simulation state, before counting the neural network, rollout buffers, and optimizer. Hence the reduction to 1020 envs.

**MuJoCo Warp (mjlab)** uses a fundamentally different GPU memory layout. It batches all environments into a single contiguous MJX data structure using Warp kernels. The physics state per env is extremely compact: 23 DOF × (pos + vel + acc) ≈ ~70 floats = 280 bytes. For 6144 envs that is ~1.7 MB of simulation state. The neural network (87→512→256→128→23) is ~500K parameters = ~2 MB. The rollout buffer (6144 envs × 24 steps × ~200 values) is ~120 MB. Total: well under 1 GB for the core training data. The remaining VRAM headroom (the GPU has 8 GB) is used by CUDA kernels, warp compilation cache, and the MJX contact solver.

**Training is GPU-compute bound, not VRAM bound:** The bottleneck is how fast the GPU can run physics steps and backpropagation. Adding more environments doesn't cost more compute-per-step — it just runs more environments in the same parallel kernel launch. This is why `steps_per_second` stays at ~48,000 whether running 1024 or 6144 envs.

**Impact of restoring 6144 envs:**
- Samples per gradient update: 1024 × 24 = **24,576** → 6144 × 24 = **147,456** (6× more)
- Gradient estimates are 6× less noisy — fewer iterations needed for convergence
- Curriculum fires at the same iteration numbers (600/1200) but after 6× more environment experience, matching the original's data scale more closely
- Same wall-clock time per iteration (~3 s)

**Evidence:** Empirically confirmed stable training at 6144 envs on RTX 3070 Laptop (8 GB VRAM) at 47,693 steps/sec, GPU memory usage within limits.

---

## 2026-05-15 — Fix feet_slippage to match original formula + sign

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** Rewrote `feet_slippage` to match `_reward_feet_slippage` exactly.

**Old (wrong):**
```python
foot_xy_vel_sq = sum(foot_vel[:, :2]**2)   # XY only, quadratic
in_contact = foot_z < threshold             # height proxy
return sum(foot_xy_vel_sq * in_contact)     # positive slippage value
# weight: -3.0
```

**New (matches original):**
```python
foot_speed = norm(foot_vel_3d)              # 3D velocity, upstream uses 7:10
contactvel = sum(foot_speed * in_contact)
return exp(-10 * contactvel)                # 1.0 when no slip, ~0 when slipping
# weight: +3.0
```

**Why it was wrong:**
1. **Sign + formula**: Old returned raw slippage with weight -3.0 (penalty). Original returns `exp(-10*vel)` with weight +3.0 (reward for not slipping). Both push gradient in the same direction, but the exponential formulation provides non-zero gradient everywhere — even when slippage is small the policy is encouraged to reduce it further. The quadratic gives near-zero gradient when slip is already low.
2. **XY-only vs 3D**: Original uses full 3D foot velocity (`rigid_body_states[:, :, 7:10]`). Port only used XY, missing vertical impact velocity at landing.

**Contact detection:** Original uses `contact_forces > 1N`; port keeps height proxy (< 0.05 m) since no per-foot contact sensor is configured. Semantically equivalent at normal walking speeds.

**Evidence:** Original config `g1_29_config.py`: `feet_slippage: 3.0`. Original function `_reward_feet_slippage` returns `torch.exp(contactvel * -10)`.

---

## 2026-05-15 — Restore sampling_mode="uniform" (RSI) + add deviation_waist_joint penalty

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:**
1. **`sampling_mode` reverted `"start"` → `"uniform"` (training only).** Play mode keeps `"start"` so the robot always begins from the standing pose at inference time. See [[2026-05-15 RSI desync entry]] for the previous reversal that is now itself being reversed.
2. **`deviation_waist_joint` penalty added** (weight -0.001). New function `gk_rew.deviation_waist_joint` returns `|waist_joint_pos - default|` summed across the single T1 Waist joint. Active continuously (not gated on ball position). Registered as `cfg.rewards["deviation_waist_joint"]`.

**Why — sampling_mode:**
Direct comparison of runs `2026-05-14_15-18-21` (RSI, hands moved) vs `2026-05-15_16-02-28` (start, hands stayed still) showed RSI was the dominant factor. With `"start"`, the policy must rediscover the full stand→lunge→reach sequence every episode; `eereach` gradient only arrives late in the sequence after the lunge is already working. With `"uniform"`, a fraction of episodes spawn the robot mid-dive with arms near the ball — direct, immediate gradient on hand positioning. The G1 original uses RSI and successfully learns arm movement.

The 2026-05-15 "RSI desync" reason (ball arrives while robot is recovering) is now mitigated: episodes are 3 s (not 5 s), the ball is reset exactly once at episode start, and the ball travel time (1–2 s) means a robot spawning at frame 75+ of the 150-frame clip (mid-recovery) is the minority case. The majority of random spawn frames are before or at the interception window.

**Why — deviation_waist_joint:**
The original Humanoid-Goalkeeper has `_reward_deviation_waist_pitch_joint` (weight -0.001) penalising the waist pitch joint deviation from 0 at all times — an always-on trunk stability term. G1 has a dedicated waist pitch DOF; T1 has a single `Waist` joint which serves the same role. This term was missing from the port entirely. The `postwaistdofpos` term already existed but is gated on ball-behind, leaving the waist unpenalised during the approach and dive phases.

**Evidence:** Codebase comparison against `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py` confirmed `deviation_waist_pitch_joint: -0.001` present in original, absent in port. Run comparison confirmed RSI loss as primary cause of hands-not-moving symptom.

**Impact:** Retrain required. Expect `hand_proximity_strict` (currently ~0.003, near zero) to rise as the policy gets direct gradient on hand-to-ball distance from RSI spawn frames.

---

## 2026-05-15 — Curriculum timing made proportional to num_steps_per_env

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/__init__.py`

**What:** Curriculum step thresholds changed from hardcoded values (60,000 / 120,000) to `600 × num_steps_per_env` and `1200 × num_steps_per_env`. The `num_steps_per_env` value is now read from the PPO runner config at task registration time and passed through to `goalkeeper_env_cfg()` as a parameter.

**Why it was wrong:** The thresholds 60K / 120K were calibrated for `num_steps_per_env=100` — they fire at iterations 600 and 1200 respectively (since `common_step_counter` counts individual `env.step()` calls, not rollout iterations). When `num_steps_per_env` was changed to 24, the same thresholds would only fire at iterations 2,500 and 5,000 — 4× too late. The policy trained through most of the budget at curriculum stage 0 weights, losing the staged reward scaling entirely.

**Correct formula:** `stage1_step = 600 × num_steps_per_env`, `stage2_step = 1200 × num_steps_per_env`. At `num_steps_per_env=24`: 14,400 / 28,800 steps. At `num_steps_per_env=100`: 60,000 / 120,000 steps (original values exactly). Changing `num_steps_per_env` in `goalkeeper_ppo_cfg.py` is now the only file that needs editing; the curriculum and all other configs derive from it automatically.

**Evidence:** With `num_steps_per_env=24` and hardcoded 60K threshold, training output at iter 6,495 showed `Curriculum/stopball_curriculum/weight: 200.0` — already maxed out, confirming the threshold had fired. With the formula fix and a fresh run, stages will fire at the intended training iterations.

---

## 2026-05-15 — Raise entropy_coef 0.002 → 0.005 to encourage arm exploration

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:** `entropy_coef` raised from 0.002 to 0.005.

**Why it was wrong:** Original G1 uses `entropy_coef=0.01`. Port was set to 0.002 (5× lower) to prevent std runaway without AMP. By iter 1581, `mean_std` had stabilized at 0.50 — no runaway risk — but arm motion (`eereach`) was growing very slowly. The lower entropy was causing the policy to over-exploit its current (leg-dominant) strategy without sufficiently exploring arm trajectories toward the ball.

**Correct value:** 0.005 — halfway between original (0.01) and previous (0.002). Provides more arm exploration gradient while staying well below the AMP-free runaway threshold. Resume training from an existing checkpoint is valid; entropy_coef only affects the PPO update step, not the collected rollouts.

**Evidence:** At iter 1581, `stopball=0.317` and `eereach=0.600` were both still low despite stable locomotion (episode_length=146/150, bad_orientation=0.25 envs). Increasing entropy encourages the policy to explore different arm positions during the ball approach phase.

---

## 2026-05-15 — Match episode structure to original (3 s, one ball per episode)

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`

**Bug 1 — Motion played at wrong speed (30 fps source, 50 Hz policy)**

`convert_booster.py` wrote the npz with 123 frames as-is from the 30 fps pkl. mjlab plays npz frames at policy dt (0.02 s = 50 Hz), so the motion ran in 123 × 0.02 = 2.46 s — 1.37× too fast. The original G1 `lefthand.pt` is also 123 frames at 30 fps = 4.1 s; its 3 s episode covered the first 90 source frames. Fix: resample from 30 fps → 50 Hz and trim to 3.0 s → 150 frames. `convert_booster.py` now uses `TARGET_FPS=50`, `TARGET_DURATION=3.0`; the npz has shape (150, ...). `episode_length_s = 3.0` matches exactly.

**Bug 2 — Ball relaunched every motion loop (multiple balls per episode)**

`MultiMotionCommand._update_command` called `_resample_command` when the motion looped, which called `_reset_ball` — relaunching the ball mid-episode. The original launches the ball exactly once per episode at reset. Fix: added `reset_ball: bool = True` parameter to `_resample_command`; `_update_command` now passes `reset_ball=False` so only true episode resets launch the ball.

---

## 2026-05-15 — Fix training ball-reset and RSI desync

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**Bug 1 — Redundant event-based ball reset conflicted with MultiMotionCommand**

`cfg.events["reset_ball"]` (training only) called `reset_ball_training` → `_shoot_ball` with `x_end = uniform(-1.2, 1.2)` (no bias). But `MultiMotionCommand._reset_ball` already fires at every episode reset AND every motion loop, using `_BALL_END_RANGES = [(-1.2, -0.2, ...)]` which correctly biases the ball toward the lefthand intercept zone. The two ball resets competed with unknown mjlab ordering; the event-based one injected uniform x_end that partially or fully clobbered the biased reset.

Fix: Removed `cfg.events["reset_ball"]` from the training path. Ball reset is now solely handled by `MultiMotionCommand._reset_ball`.

**Bug 2 — `sampling_mode="uniform"` (RSI) desynchronized motion phase from ball arrival**

Training used RSI: the robot started at a random frame of the lefthand motion. The ball was launched 0.5–1.0 s later. If the robot started mid-dive (frame 50 of 100), the ball arrived as the robot was recovering — `stopball`/`eereach` could not fire. Play used `sampling_mode="start"` (always frame 0), so the robot always executed the full dive before the ball arrived. This created a systematic training/play gap.

Fix: Changed `sampling_mode` to `"start"` for training as well. The robot now always starts at the standing pose (frame 0) both in training and play. The motion loops every ~2 s during the 5 s episode (robot resets to standing, ball relaunches), giving 2–3 clean dive attempts per episode.

## 2026-05-15 — Fix 4 bugs from critical G1↔T1 comparison

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/resets.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**Bug 1 — Ball never moved during training (CRITICAL)**

`reset_ball_autonomous` was only registered in play mode. Training used `reset_scene_to_default` which placed the ball at `env_origin + (0,0,0)` with zero velocity every episode. `stopball` (weight 100) never fired. `eereach` gradient was only on a stationary ball at the robot's feet.

Fix: Added `reset_ball_training()` to `resets.py` and registered it as `cfg.events["reset_ball"]` (mode="reset") in the training path. Ball now spawns at Y∈[3,5] m with a random trajectory aimed at the goal line, matching the original's ball setup.

**Bug 2 — `postupperdofpos` sigma 20× too sharp**

Original `_reward_postupperdofpos` uses `exp(sum_sq_err × -1)` (sigma=1 on SUM). Booster used `exp(-20 × mean_sq_err)`. For 8 arm joints this is 2.5× too sharp per joint, giving near-zero reward unless arm is within ~0.1 rad of default — almost no gradient signal for arm recovery.

Fix: Changed to `err = sum(square(delta)); exp(-1.0 * err)`. Matches original code exactly. The config value `target_dof_pos_sigma=-20` exists in the G1 config but is NOT used in the original's reward code; the code hardcodes -1.

**Bug 3 — `postwaistdofpos` sigma 6.7× too sharp**

Original `_reward_postwaistdofpos` uses `exp(-3 × sum_sq_err)` (sigma=3). Booster used `exp(-20 × mean_sq_err)` — same mistake as postupperdofpos.

Fix: Changed to `err = sum(square(delta)); exp(-3.0 * err)`. Matches original code.

**Bug 4 — `noretreat` used world-Y instead of body-frame forward**

Original uses `base_lin_vel[:, 0]` (body-frame X = forward direction). Booster used `root_link_lin_vel_w[:, 1]` (world Y). These diverge at 30-45° yaw during a dive, applying the retreat penalty on the wrong axis.

Fix: Changed to `root_link_lin_vel_b[:, 0]` (body-frame forward). Semantically: "don't move backward in the direction you're facing", which remains correct during dives regardless of yaw.

**Addition — `hand_proximity_strict` reward (mirrors `_reward_success`)**

Original has a `success` reward (weight 5) firing at `dist < 0.15m` (strict threshold). Booster only had `catch_success` at 0.5m — 3.3× too coarse, no precision gradient. Added `hand_proximity_strict` (weight 5, threshold 0.15m) to provide dense gradient signal for the final hand-to-ball approach.

**Evidence:** From code comparison against `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py`. Ball reset confirmed by tracing `make_tracking_env_cfg()` → `reset_scene_to_default` in mjlab venv source. Sigma values confirmed by reading original reward function code (not config).

---

## 2026-05-15 — Temporarily disable heavy DR to unblock early training

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:** Disabled `pd_gains`, `link_mass` (pseudo_inertia), and `reset_joints` DR terms. Reverted `foot_friction` range back to base config [0.3, 1.2]. Reduced push_robot to milder defaults.

**Why:** After 38 iterations the policy was getting worse (base_height terminations 36→44, zero timeouts, motion errors increasing). The three heavy DR terms make the optimization landscape too difficult before the policy has learned to stand and balance. Will re-enable once the policy shows stable upright behaviour.

---

## 2026-05-14 — Fix feet_slippage sign + replace tracking terminations with G1-equivalent fall detection

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. **`feet_slippage` weight corrected: +3.0 → -3.0.** The reward function returns positive squared foot velocities; the positive sign was rewarding slippage instead of penalizing it.
2. **All three mjlab tracking terminations removed** (`anchor_pos`, `anchor_ori`, `ee_body_pos`). These check deviation from the reference motion pose — a tracking-framework concept not present in the G1 upstream.
3. **Two G1-equivalent fall terminations added:**
   - `bad_orientation` with `limit_angle=1.0 rad` (57°): matches G1's `gravity_termination_buf` which fires when `projected_gravity XY norm > 0.8` (≈ 53°). Slightly more permissive to allow diving tilts.
   - `base_height` with `minimum_height=0.4 m`: catches kneeling/folding collapses where the trunk drops to half its standing height. Mirrors G1's `knee_height < 0.10 m` intent.

**Why it was wrong:**
- `feet_slippage` sign: oversight — upstream uses weight −3.0 on a positive-valued function. Confirmed by iter 0 showing +0.046 instead of −0.046.
- Tracking terminations: `ee_body_pos` (`bad_motion_body_pos_z_only`, threshold 0.25 m) fired when any of the 4 specified bodies deviated > 0.25 m in Z from the motion reference. With random actions at iter 0, this killed 74% of episodes in ~0.25 s. Removing all three without adding replacements caused the robot to fall through its own legs indefinitely (no fall detection at all).
- Correct approach from G1: terminate on tilt + trunk height, not on motion pose deviation.

**Evidence:** Training log iter 0–1: `Episode_Termination/ee_body_pos: 70.1 → 74.9`, `time_out: 0.66 → 0.00`, mean episode length 14 → 13 steps (0.25 s).

---

## 2026-05-14 — Domain randomization, feet_slippage, observation delay

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What — 3 items:**

1. **Domain randomization expanded to match upstream G1 suite:**
   - `push_robot`: interval 1–3 s → 15 s constant, max push velocity 1.5 m/s (upstream values).
   - `foot_friction`: geom friction range [0.3, 1.2] → [0.1, 2.0] (matches upstream `randomize_friction`).
   - `pd_gains` (new, `mode="reset"`): kp and kd scaled ∈ [0.8, 1.2] per episode. Matches upstream `randomize_kp/kd`. Uses `mjlab_dr.pd_gains`.
   - `link_mass` (new, `mode="reset"`): `mjlab_dr.pseudo_inertia` with `alpha_range=(0.8, 1.2)`. Scales body mass and inertia jointly (physically consistent). Matches upstream `randomize_link_mass [0.8, 1.2]`.
   - `reset_joints` (new, `mode="reset"`): initial joint position offset ±0.1 rad at episode start. Matches upstream `randomize_initial_joint_pos` with scale [0.5, 1.5] and offset [-0.1, 0.1].
   - Note: `encoder_bias` (joint injection ±0.01 rad) was already in the base tracking config. Not duplicated.
   - Not ported: `randomize_com_displacement`, `randomize_restitution`, `ball_interval_s` (ball perturbation) — these are secondary and can be added later.

2. **`feet_slippage` reward added** (weight 3.0): custom `gk_rew.feet_slippage`. Penalises foot XY velocity while foot height < 0.05 m (contact proxy). Uses `body_link_lin_vel_w` for foot linear velocity. Matches upstream `_reward_feet_slippage`. Without this, the policy can slide sideways during saves without penalty.

3. **Observation delay added** (training only): 0–2 step uniform random lag per env applied to all actor observation terms (`delay_min_lag=0`, `delay_max_lag=2`, `delay_per_env=True`). At 50 Hz, 2 steps = 40 ms — realistic sensor + communication latency for real hardware. Matches upstream `delay=True`. Not applied in play mode (real hardware already has its own latency).

**Why — domain randomization:** Without kp/kd and mass randomization, the policy overfits to exact sim physics. Knees and hips have stiffness 80–200 Nm/rad; ±20% randomization spans the manufacturing tolerance and calibration uncertainty for real hardware. Observation delay models the loop latency between sensor reading and torque command execution (typically 20–60 ms on real robots).

**Impact:** Full retrain required. Expect slower initial learning (harder optimization landscape) but more robust policy that transfers better to real hardware.

## 2026-05-14 — Port 5 missing upstream reward terms + fix stopball threshold

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What — 5 gap fixes:**

1. **Gap 1 — Soft limit penalties added** (`dof_pos_limits` -3.0, `dof_vel_limits` -2.0, `torque_limits` -3.0):
   - `dof_pos_limits`: uses `mjlab_mdp.joint_pos_limits` (soft_joint_pos_limits from URDF). Prevents the policy from exploiting joint limits without penalty.
   - `dof_vel_limits`: custom `gk_rew.dof_vel_limits` with `vel_limit=20 rad/s` universal cap (mjlab has no per-joint URDF velocity limits stored). Clips `max(0, |qd| - 18)` summed across joints.
   - `torque_limits`: custom `gk_rew.torque_limits` using `qfrc_actuator` with `torque_limit=50 Nm` (median T1 effort; knees are 60 Nm so slightly under-penalised, arms 18 Nm so over-penalised but their torques are naturally small). Clips `max(0, |tau| - 47.5)` summed.

2. **Gap 2 — Safety penalties added** (`penalize_sharpcontact` -100, `penalize_kneeheight` -100):
   - `penalize_sharpcontact`: custom. mjlab does not expose `cfrc_ext` per body, so trunk height < 0.35 m is used as a proxy for hard body–ground contact (normal standing is ~0.7 m). Binary flag: fires once when trunk collapses.
   - `penalize_kneeheight`: custom. Checks `Shank_Left` / `Shank_Right` body heights. If either shank drops below 0.12 m, the robot is kneeling — fires as a binary flag per step.

3. **Gap 3 — Successland added** (`successland` 4.0): custom `gk_rew.successland`. Rewards both feet within 0.05 m of ground simultaneously when ball is behind. Mirrors upstream `_reward_successland` but uses foot height instead of contact sensor (no foot-ground contact sensor configured in this env). Without this, the policy has no incentive to land safely after a dive — it just falls over.

4. **Gap 4 — `num_steps_per_env` 50 → 100**: Restores upstream value. `stopball` is a sparse, one-shot reward; with 50-step rollouts (1 second at 50 Hz) the save often falls outside the GAE credit-assignment window. 100 steps = 2 seconds, covering the full ball-approach and interception window. Curriculum thresholds restored to 60K / 120K (were 30K / 60K when at 50 steps) to keep stage transitions at ~600 and ~1200 training iterations.

5. **Gap 5 — Post-save recovery rewards added** (`postupperdofpos` 1.0, `postwaistdofpos` 1.0): custom `gk_rew.postupperdofpos` / `postwaistdofpos`. Both use `exp(-20 * mean_sq_err)` matching upstream `target_dof_pos_sigma=-20`. Arms and waist are penalised for deviating from the standing-keyframe default after ball passes. Without these, the policy locks the diving arm out indefinitely and never recovers to a stable post-save posture.

**Stopball threshold fix:** `delta_vel_threshold` lowered from 2.0 → 1.0 m/s. At 2.0, slow-ball saves (approaching at -1 m/s, deflected to +0.5 m/s → delta=1.5 m/s) never triggered the reward. 1.0 catches slow deflections while remaining above typical ball-bouncing noise (<0.5 m/s delta).

**Why missing:** The original G1 version has all of these. They were never ported during the initial mjlab migration because the focus was on getting task rewards working. Without safety penalties and post-save recovery rewards, the optimizer finds degenerate strategies: hard falls, kneeling, and frozen diving poses.

**Impact:** Full retrain required. Expect: (1) no kneeling/collapse postures during training, (2) active landing after saves, (3) arms return to neutral between saves, (4) slower std blowup from joint/torque limit gradients dampening aggressive motions.

## 2026-05-14 — Full regularization alignment with original Humanoid-Goalkeeper

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Ported 5 missing regularization terms from original and fixed 4 parameter mismatches:

1. **smoothness weight -0.01 → -0.1**: Original `_reward_smoothness` uses second-order jerk at -0.1. Our weight was 10× too weak, causing jerk to balloon unchecked (saw `action_rate_l2` grow to -14 per episode).
2. **Added `dof_acc` (-2.5e-7)**: `mjlab_mdp.joint_acc_l2` — penalises joint acceleration. Missing entirely from our port.
3. **Added `torques` (-1e-5)**: `mjlab_mdp.joint_torques_l2` — penalises torque magnitude. Missing entirely.
4. **Added `dof_vel` (-5e-4)**: `mjlab_mdp.joint_vel_l2` — penalises joint velocity. Missing entirely.
5. **Added `ang_vel_xy` (-0.1)**: Custom `gk_rew.base_ang_vel_xy_l2` — penalises base roll/pitch rate (XY axes only). Missing entirely. Suppresses wobbling/tipping.
6. **`eereach reach_th` 0.3 → 0.2**: Original uses 0.2 m sigmoid midpoint. Ours was 50% more generous.
7. **`catch_success catch_th` 0.3 → 0.5**: Original uses 0.5 m catch threshold for continuous reward. Ours was tighter than original for this continuous term.
8. **`postangvel` XY only**: Original penalises only roll/pitch rate after ball passes; we were penalising all 3 axes including yaw. Fixed to match.
9. **`entropy_coef` 0.004 → 0.002**: Cannot safely use original's 0.01 without AMP (empirically confirmed collapse). 0.002 is a calibrated middle ground: above the over-converging 0.001 (gave std=0.43), below the runaway 0.004 (gave std=2.5+). The 5 new regularization terms provide additional implicit pressure against high-variance actions, backing the lower entropy target.

**Why missing:** The original Humanoid-Goalkeeper runs in Isaac Gym which has AMP providing natural regularization via discriminator gradients. Without AMP, these explicit penalties are the primary mechanism for smooth, stable motion. Our initial port carried the task rewards but omitted the regularization layer.

**Impact:** Full retrain required. Expect smoother joint motion, less jitter, and `mean_std` stabilizing in 1.0–2.0 range rather than drifting to 2.5+.

## 2026-05-14 — stopball: continuous → event-based (port from original)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Rewrote `stopball` from a continuous per-step reward to a one-time event-based reward, matching the original Humanoid-Goalkeeper's `_reward_stopball()`.

**Old behaviour (broken):**
```python
in_front = ball_y_local > 0.0
deflected = ball_y_vel > -0.5
return (in_front & deflected).float()  # fires every step
```
With weight 100, a stationary ball in front of the robot earned 100 pts × ~200 steps = **20,000 pts/episode**. The optimal policy was to stand still and let the ball roll into the body.

**New behaviour (matches original):**
```python
delta_vy = ball_y_vel - prev_ball_y_vel
fired = (delta_vy > 2.0) & in_front & ~stop_flag
stop_flag |= fired  # fires exactly once per episode
return fired.float()
```
Fires exactly once when the ball first decelerates by >2 m/s while still in front. After that, `stop_flag` blocks further firings. Cannot be gamed by passive blocking.

**Why this was wrong:** The original `_reward_stopball` uses an explicit `stop_flag` per environment (reset in `reset_idx`) and only returns `1.0 * (stop_flag == 0) * changevel`. Our port mistakenly used a continuous velocity threshold, turning a one-time 100-point bonus into a massive continuous reward that dominated all other signals.

**Evidence:** Comparison between run `15-18-21` (active interception) and `20-28-27` (passive blocking) showed the new run scoring 22 stopball vs 14, yet visually doing less work. Passive blocking was the cheaper strategy under continuous reward. Event-based reward removes this exploit entirely.

**Impact:** Full retrain required. `eereach` and `catch_success` will now dominate — both require hand proximity to the ball, so the robot is incentivized to actively reach.

## 2026-05-14 — num_steps_per_env 100 → 50, entropy_coef 0.001 → 0.004 (encourage arm exploration)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`
**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. `num_steps_per_env`: 100 → 50. Curriculum thresholds halved accordingly (60K/120K → 30K/60K) since `common_step_counter` increments per `env.step()` call.
2. `entropy_coef`: 0.001 → 0.004.

**Why — num_steps_per_env=50:**
100 steps = 2-second rollouts on a 5-second episode. The ball arrives within 2–3 seconds, so a 2-second rollout already covers the full interception window — no benefit in going longer. 50 steps = 1 second, 2× faster wall-clock per iteration, ~5 rollouts per episode which still gives adequate credit assignment through GAE.

**Why — entropy_coef=0.004:**
After 2400 iterations at entropy_coef=0.001, `mean_std` converged to 0.43. At this point the policy is too narrow to explore arm-reaching behaviors: `eereach` reward reached only 1.02 out of a possible ~20 (hand ~0.9m from ball on average), despite strong `motion_body_pos=6.03` (robot IS tracking reference motion). The policy settled in a conservative local optimum — standing stable, partially tracking the reference pose — before discovering that extending the arm to the ball gives much larger reward. Raising entropy_coef to 0.004 pushes std back toward ~1.0–1.5, allowing exploration of arm extension. Chosen below 0.005 (which caused slow runaway) and well below 0.01 (catastrophic). Monitor `mean_std`: if it exceeds 2.5, reduce entropy_coef again.

## 2026-05-14 — entropy_coef 0.01 → 0.001 (IsaacGym value causes std runaway without AMP)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:** `entropy_coef` set to `0.001` instead of the IsaacGym Humanoid-Goalkeeper value of `0.01`.

**Why:** The IsaacGym reference uses `entropy_coef=0.01`, but this value only works there because: (1) AMP discriminator provides stabilising counter-gradients, (2) 6144 parallel environments give 3× larger batch size. Without AMP, `0.01` triggers a self-reinforcing positive feedback loop in mjlab:

- Entropy gradient continuously pushes `mean_std` upward
- Larger std → more saturated actions → noisier advantages → std grows further
- Observed in training run `2026-05-14_17-04-22`: std grew 1.0 → 5.3 over 1000 iters
- At std=5.3, ~42% of actions exceed ±4.0 (saturation point after the 0.25 action scale fix)
- Robot fell in <15 steps; reward collapsed from 42 → 1

The mjlab G1 tracking reference (`mjlab/tasks/tracking/config/g1/rl_cfg.py`) uses `entropy_coef=0.005`. With 100 steps/env and no AMP, `0.001` is the safe value.

**Evidence:** TensorBoard `Policy/mean_std` monotonically increasing every iteration with no stabilisation. `Episode_Termination/ee_body_pos` exploded from 7 → 70 falls/episode.

## 2026-05-14 — Action jerk penalty + reward curriculum (mirroring G1 training strategy)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_env_cfg.py`

**What:**
1. Replaced `action_rate_l2` (first-order, weight -0.1) with `action_acc_l2` (second-order jerk, weight -0.01).
2. Added `cfg.curriculum` with step-based reward scaling for `stopball`, `eereach`, and `catch_success`.

**Why — jerk penalty:**
After 1800 training iterations the first-order `action_rate_l2` term grew from -1.7 to -15.0 per episode, eventually dominating the total reward and plateauing it at ~59 even as `stopball` kept climbing. The G1 reference uses `sum((a_t - 2*a_{t-1} + a_{t-2})^2)` (second-order / jerk) at an effective weight of ~0.0002. The second-order formulation penalises oscillation harder than a smooth fast dive: an ankle alternating ±1 each step scores 4× larger per step on jerk vs rate, which is exactly the pathological behaviour we want to suppress. The smaller weight (-0.01 vs -0.1) reduces overall drag so task rewards can keep growing.

**Why — curriculum:**
G1 scales `stopball`, `eereach`, and `success` upward as training progresses: `weight × (1 + 0.5 × curriculum_level)`. Without this, fixed-weight smoothness penalties become proportionally larger relative to task rewards as the policy lengthens episodes and takes more actions. The mjlab `reward_curriculum` applies staged step-based multipliers: ×1.0 → ×1.5 → ×2.0 at ~600 and ~1200 training iterations respectively (thresholds: 15M and 30M env steps = 1020 envs × 24 steps/iter × target_iter). This keeps the reward landscape competitive throughout long runs.

**Impact:** Prior checkpoints remain incompatible (action space unchanged, but reward signal structure changed). Recommend starting a fresh run.

This document tracks substantive changes where the Booster T1 adaptation deviates from the G1 tracking task pipeline.

## 2026-05-14 — Fix reward axis mismatch after 90° rotation (ball from +Y not +X)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Changed `_ball_is_behind`, `stopball`, `stayonline`, and `noretreat` from using world X (index 0) to world Y (index 1).

**Why:** The original Humanoid-Goalkeeper had the ball coming from +X and the robot facing +X, so all rewards used world X. When the T1 port rotated the setup 90° (ball from +Y, robot faces +Y), the reward functions were never updated. As a result:
- `stopball` (weight=100) **never fired** — ball ends at negative X so `ball_x_local > 0` was always false. The entire main task reward was dead.
- `_ball_is_behind` was true from the start of every episode (ball starts at negative X), so `postorientation`/`postangvel`/`postlinvel` activated immediately and at the wrong time.
- `noretreat` penalised moving in world -X (sideways to the ball) rather than -Y (actually retreating from the ball), giving the policy no signal to prevent retreating.
- These misaligned rewards likely caused the observed clockwise rotation: no penalty for drifting/spinning, and the dominant 100-weight reward not firing at all.

**Impact:** Full retrain required. `stopball` will now actually fire, which is the primary learning signal.

---

## 2026-06-05 — Fix goalkeeper_amp_env_cfg: restore 870-dim obs, 16 missing rewards, sensors, and config params

**Files:**
- `Imitationlearningbooster/src/imitationlearningbooster/tasks/goalkeeper_amp_env_cfg.py`

**What — 8 bug fixes:**

### 1. Observation space restored to 870-dim (CRITICAL)

The AMP env cfg was missing `left_hand_pos_b` (3 dims) and `right_hand_pos_b` (3 dims) from both actor and critic observation groups. Without them the AMP task produced 81 dims/step × 10 history = **810-dim** input — incompatible with the deployed `model_2000.pt` which expects `87 × 10 = 870`. Added both to actor/critic groups identical to `goalkeeper_env_cfg.py`.

### 2. eereach reach_th corrected from 0.3 → 0.2

The AMP config called `RewardTermCfg(func=gk_rew.eereach, weight=10.0)` with no params, using the function default `reach_th=0.3`. The correct value (matching G1 upstream and `goalkeeper_env_cfg.py`) is 0.2 m. Added `params={"reach_th": 0.2}`.

### 3. 16 missing reward terms added

Rewards present in `goalkeeper_env_cfg.py` and G1 upstream but absent from AMP config:
`successland` (4.0), `penalize_sharpcontact` (-100.0), `penalize_kneeheight` (-100.0), `penalize_self_collision` (-50.0), `feet_slippage` (3.0), `postupperdofpos` (1.0), `postwaistdofpos` (1.0), `dof_acc` (-2.5e-7), `action_rate_l2` (-0.1), `torques` (-1e-5), `dof_vel` (-5e-4), `dof_pos_limits` (-3.0), `dof_vel_limits` (-2.0), `torque_limits` (-3.0), `deviation_waist_joint` (-0.001), `ang_vel_xy` (-0.1).

Without these, AMP training was entirely unguarded: no safety penalties, no joint-limit regularization, no smoothness constraints. Degenerate behaviours (kneeling, hard falls, jitter) would dominate.

### 4. sharpforce_termination added

The base `goalkeeper_env_cfg` has a hard episode termination when ground impact force exceeds 1500 N (`sharpforce_termination`). The AMP config was missing this. Without it, the AMP policy faces no hard termination for catastrophic falls.

### 5. Contact sensors added

`ContactSensorCfg` for `feet_contact` (geom pattern `^(left|right)_foot_[12]$`) and `self_collision` (Trunk self-collision) were missing. These are required by `penalize_sharpcontact`, `penalize_self_collision`, `feet_slippage`, and `sharpforce_termination`. Added with identical configuration to `goalkeeper_env_cfg.py`.

### 6. num_envs set to 6144

The AMP config never explicitly set `cfg.scene.num_envs`, inheriting whatever `make_tracking_env_cfg()` defaulted to. Set to 6144 matching the baseline.

### 7. sim contact capacity set

Added `cfg.sim.nconmax = 100` and `cfg.sim.njmax = 500` (same as baseline) to handle ball contact simulation without contact truncation.

### 8. Viewer tracking added

Added `cfg.viewer.body_name = "Trunk"` so the sim viewer follows the robot (not stuck at origin).

**Why they were wrong:** The AMP config was created as a skeleton and was never brought up to parity with the baseline `goalkeeper_env_cfg.py`. Because the AMP runner adds discriminator infrastructure on top, it is easy to focus only on the AMP-specific additions and miss the base config requirements.

**Impact:** AMP training (`goalkeeper_booster_t1_amp`) is now functionally equivalent to the baseline in terms of safety constraints and reward structure, with AMP rewards layered on top.

---

## 2026-06-05 — Fix AMP discriminator: gradient penalty scale and spectral normalization

**File:** `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/modules/discriminator.py`

### 1. Gradient penalty scale corrected (10× too weak → matches G1)

The discriminator's `compute_grad_pen()` was returning `lambda_ * grad_pen * 0.1` (with `lambda_=5`), giving an effective gradient penalty of **0.5**. G1 upstream uses `lambda_=5` with no extra multiplier. Removed the `* 0.1` factor; effective penalty is now 5.0, matching upstream.

**Why it was wrong:** The `* 0.1` multiplier appears to have been added as a conservative scaling but has no upstream justification. Weak gradient penalty allows the discriminator to violate the Lipschitz constraint, destabilising GAN training with sharp discriminator gradients.

### 2. Spectral normalization added to all discriminator linear layers

G1 upstream applies spectral normalization to discriminator weights. The port did not. Added `torch.nn.utils.spectral_norm` wrapping to all hidden `nn.Linear` layers in the MLP trunk and the output `amp_linear` layer.

**Why it was wrong:** Spectral normalization constrains the spectral norm of each layer's weight matrix to ≤1, bounding the Lipschitz constant of the discriminator. Without it, discriminator gradients can become unboundedly large, causing mode collapse or training instability.

**Impact:** Discriminator training stability significantly improved. Gradient flow to policy from AMP rewards will be more consistent.

---

## 2026-06-05 — Fix GoalkeeperMotionLoader: add temporal jitter to expert sampling

**File:** `Imitationlearningbooster/src/imitationlearningbooster/rsl_rl_amp/utils/motion_loader.py`

**What:** Added per-sample temporal jitter to `feed_forward_generator()`. Before: always sampled consecutive frame pairs `(idx, idx+1)`. After: second frame sampled at `idx + round(U(0.75, 1.25))`, clamped to valid range.

**Why it was wrong:** G1's `MotionLib.get_expert_obs()` applies `ratio = (fps / env_fps) × U(0.25, 1.25)` temporal jitter before sampling the "next" frame, then bilinearly interpolates between frames. The port sampled raw consecutive frames. This locked the expert transition distribution to exactly 1-frame-apart deltas, while G1's distribution spans 0.75–1.25 frame gaps. The narrower distribution makes the discriminator more brittle — small deviation from exactly-1-frame timing is penalised even if the motion is qualitatively identical.

**Impact:** Expert data diversity increased; discriminator generalises better to variations in motion speed.

---

## 2026-06-05 — Fix convert_all.py: worldbody included in body positions (all 6 npz files broken)

**File:** `Imitationlearningbooster/src/imitationlearningbooster/motions/convert_all.py`

**What:**
1. `n_bodies = model.nbody` → `n_bodies = model.nbody - 1` (exclude worldbody at index 0)
2. `body_pos_w[t] = data_mj.xpos.copy()` → `body_pos_w[t] = data_mj.xpos[1:].astype(np.float32)`
3. `body_quat_w[t] = data_mj.xquat.copy()` → `body_quat_w[t] = data_mj.xquat[1:].astype(np.float32)`
4. `foot_body_ids` index computation corrected: offset by -1 to account for worldbody exclusion.
5. `apply_foot_fix()` call removed.

**Why it was wrong:**
- MuJoCo's `data.xpos` and `data.xquat` include the worldbody at index 0. `mjlab`'s `MotionLoader` indexes bodies starting at 0 assuming worldbody is already excluded. With worldbody at npz[0], every body position was off by 1 — body 0 (Trunk) was reading worldbody's fixed position (0,0,0), making the Reference State Initialization place all robots at the world origin rather than their reference poses. All 6 npz files had shape `(150, 25, 3)` (worldbody included); correct shape is `(150, 24, 3)`.
- `apply_foot_fix()` enforces `ankle = -(hip + knee)` which is not applied in `convert_booster.py`. The deployed `model_2000.pt` was trained using npz files without this fix. Using foot-fixed npz files for AMP fine-tuning would create a mismatch between expert motion distribution and the baseline policy's learned dynamics.

**Impact:** All 6 npz files regenerated with corrected converter (body shape now `(150, 24, 3)`). AMP Reference State Initialization now places robots at correct reference poses. PKL source files were present; regeneration succeeded.

---

## 2026-06-05 — Fix head joint action scale (missing 0.25 factor)

**Files:**
- `Imitationlearningbooster/src/imitationlearningbooster/robots/t1_constants.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py`

**What:** `T1_ACTION_SCALE` for `(AAHead_yaw|Head_pitch)` changed from `7.0 / 20.0 = 0.35` to `0.25 * 7.0 / 20.0 = 0.0875`.

**Why it was wrong:** All other joints use the `0.25` factor (effort / stiffness × 0.25), which matches the G1 upstream's `action_scale = 0.25`. The head joints were accidentally omitted from this convention. Without the 0.25 factor, a policy output of ±1.0 moves the head ±0.35 rad — 4× the intended ±0.0875 rad. During random initialization the head bangs against its limits, injecting spurious contact forces and noise into early training.

**Both t1_constants.py files kept in sync** (Imitationlearningbooster and my_mjlab_project_booster_t1 are identical after this fix).

---

## 2026-06-05 — Add imitationlearningbooster as editable dependency of my_mjlab_project_booster_t1

**File:** `my_mjlab_project_booster_t1/pyproject.toml`

**What:** Added `"imitationlearningbooster"` to `[project] dependencies` and added to `[tool.uv.sources]`:
```toml
imitationlearningbooster = { path = "../Imitationlearningbooster", editable = true }
```

**Why it was wrong:** The `Imitationlearningbooster` package was not installed in the project venv. Its entry point (`goalkeeper_booster_t1_amp = "imitationlearningbooster"`) never fired, making the entire AMP training infrastructure (`GoalkeeperAmpRunner`, `GoalkeeperMotionLoader`, `MultiDiscriminator`, all 6 motion files) inaccessible. Running `uv sync` will install it.

**Next step required:** Run `uv sync` inside `my_mjlab_project_booster_t1/` to install the editable dependency.

## 2026-05-14 — Fix bang-bang ankle/leg control and PPO rollout length

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:**
1. Applied the `0.25` action-scale factor to all non-arm joints (waist, hips, knees, ankles). Previously only arms had this factor.
2. Set `num_steps_per_env=24` (original upstream used 100) and `gamma=0.99` (was 0.998, now matches upstream).

**Why — action scale:**
Without the 0.25 factor, every non-arm joint saturated at max torque when the policy output ±1.0. Because the initial policy is Gaussian(0,1), outputs above ±1 are common, causing bang-bang (on/off) torque switching — visible as fast ankle jitter and micro-bouncing. The G1 reference applies `0.25 × effort/stiffness` to ALL joints so saturation only occurs at action=4.0.

**Why — gamma:**
0.998 was an unexplained deviation from both the original Humanoid-Goalkeeper (0.99) and the G1 mjlab config (0.99). Reverted to 0.99.

**Why — num_steps_per_env=100:**
Matches upstream G1 exactly. 100 steps = 2 seconds of experience per rollout at 50 Hz, giving 204,800 samples per gradient update (2048 envs × 100 steps). Empirically, reducing to 24 steps caused the policy std to grow unchecked (1.26 → 3.55 over 2800 iters), because 4× fewer samples per update produced noisier advantage estimates and larger policy steps. Keeping 100 steps provides stable gradient estimates matching G1's training regime.

**Impact:**
- Ankle pitch: action=1 now produces 5 Nm (was 20 Nm = full saturation). Control stays in PD linear regime.
- Smooth foot contact expected; micro-bouncing should be eliminated.
- **All prior checkpoints are incompatible** (action space meaning changed). Full retrain required.

## 2026-05-03 — Fully autonomous goalkeeper (no motion input at play time)

**Files:** 
- `my_mjlab_project/src/my_mjlab_project/mdp/resets.py` (new)
- `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** 
1. Created `reset_ball_autonomous()` - standalone ball reset function with no motion tracking dependency
2. Removed motion command entirely from play config
3. Added autonomous ball reset as startup event
4. Removed 6 motion-dependent reward + 3 motion-dependent termination terms

**Why:** The end goal is a **100% autonomous goalkeeper**:
- **Training:** Learn from all 6 motion types (left/right hand, jump, step) via RSI (RL + imitation)
- **Observations:** Ball position, ball velocity, joint state **only** (no motion in obs)
- **Play:** Policy autonomously chooses best response to any incoming ball

By removing the motion command, the policy receives no motion input at inference time. The autonomous reset function ensures the ball still gets randomized trajectories.

**Impact:** 
- Policy trained on diverse motion examples but is **completely autonomous at play time**
- No `--motion-file` argument needed
- Ball resets with random trajectory **every 10 seconds** (3-5m away, random y/z, timed arc)
- Episode auto-resets every 10 seconds to generate new ball trajectory
- Play command: `uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/.../model_N.pt`
- Play mode runs stably, policy continuously faces new ball trajectories
- Robot returns to random pose on each reset (domain randomization)

## 2026-05-02 — Remove motion-reference observations from actor/critic

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** Removed `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b` from
the actor and critic observation groups in `goalkeeper_env_cfg()`.

**Why:** The upstream G1 tracking task treats reference motion as an explicit
input to the policy. This requires a motion file at inference time. For an
autonomous goalkeeper agent that decides its own response based on ball
position, the motion reference must not appear in the observation space.
The 6 motion files are retained for RSI (reference state initialisation) and
style-shaping rewards during training only.

**Impact:** All checkpoints trained before 2026-05-02 are incompatible with the
new observation space (actor dim changed). Full retrain required.

## 2026-05-22 — Physics-based collision detection replacing height proxies

**Files:**
- `my_mjlab_project_booster_t1/src/.../mdp/rewards.py`
- `my_mjlab_project_booster_t1/src/.../tasks/goalkeeper_env_cfg.py`
- `my_mjlab_project_booster_t1/src/.../mdp/resets.py`

### 1. New `feet_contact` ContactSensor

**What:** Added a `ContactSensorCfg(name="feet_contact")` monitoring the 4 foot geoms
(`left_foot_1`, `left_foot_2`, `right_foot_1`, `right_foot_2`) with `secondary=None`
(any contact partner, including ground) and `reduce="netforce"`.

**Why it was missing:** mjlab has no global `cfrc_ext`-equivalent tensor. Isaac Gym
exposed `net_contact_force_tensor` per-body; mjlab requires explicit `ContactSensorCfg`
declarations. The original port used height proxies everywhere because this API gap
was not addressed.

**Shape:** `data.found [B, 4]`, `data.force [B, 4, 3]`. Geom index order (lexicographic):
0=left_foot_1, 1=left_foot_2, 2=right_foot_1, 3=right_foot_2.

---

### 2. `penalize_sharpcontact` — trunk height proxy → foot force sensor

**What was wrong:** Used `trunk_z < 0.35 m` as a fall detector. Original uses
`mean(norm(foot_forces)) > 1000 N` — a direct impact measurement.

**Fix:** Now reads `env.scene["feet_contact"].data.force` and returns binary
`(mean_force_norm > 1000.0).float()`. Weight: -100.0 unchanged.

**Evidence:** Original `legged_robot.py:1477`. Proxy missed hard landings at standing
height; falsely fired during controlled crouches.

---

### 3. `feet_slippage` — foot height proxy → sensor-based contact detection

**What was wrong:** Used `foot_z < 0.05 m` for `in_contact`. When feet were airborne
(mid-dive), `in_contact=0` → `contactvel=0` → `exp(0)=1.0` — full reward even while
the robot was completely in the air. Inflated WandB `feet_slippage` curves throughout
training.

**Fix:** Now uses `env.scene["feet_contact"].data.found > 0` per foot geom. Left foot
in_contact = `(found[:,0]>0) | (found[:,1]>0)`, right similarly. Rest of formula
unchanged. Weight: +3.0 unchanged.

**Evidence:** Original `legged_robot.py:1472` uses `contact_forces > 1 N` as the
in_contact threshold.

---

### 4. `penalize_self_collision` — new reward wiring `self_collision` sensor

**What was missing:** The `self_collision` ContactSensor was already registered in
`goalkeeper_env_cfg.py` but its output was never consumed by any reward or observation.

**Fix:** Added `penalize_self_collision` reward that reads `self_collision.data.found`
and returns `(found > 0).any(dim=-1).float()`. Registered at weight -50.0.

**Reasoning:** Self-collisions (arm hitting torso during dive) are recoverable but
undesirable. Weight -50 (half of sharpcontact -100) reflects lower severity.

---

### 5. `sharpforce_termination` — force-based episode termination

**What was missing:** Original terminates at `mean_foot_force > 1500 N`
(`sharpforce_buf` at `legged_robot.py:258`). mjlab port had no equivalent.

**Fix:** Added `sharpforce_termination` in `resets.py`, registered as
`TerminationTermCfg(time_out=False)` with `max_contact_force=1500.0`. Uses the
same `feet_contact` sensor force computation as `penalize_sharpcontact`.

**Impact:** New termination fires only on catastrophic impacts. Checkpoints trained
before this date are fully compatible — this does not change observation or action space.

---

## 2026-05-26 — Deployment bug fixes: base_lin_vel frame and ball visibility masking

### Bug Fix 1: `base_lin_vel` was fed in world frame instead of body frame

**What was wrong (`goalkeeper_deploy/tasks/goalkeeper/controller.py`):**
```python
base_lin_vel_b = dq[:3]  # WRONG — labelled body frame but is world frame
```
MuJoCo freejoint `qvel[0:3]` is the derivative of world-frame position, i.e. the **world-frame** linear velocity. It is NOT the body-frame velocity. The comment claiming it was body frame was incorrect.

**What the training does (`mjlab/entity/data.py:597`):**
```python
root_link_lin_vel_b = quat_apply_inverse(root_link_quat_w, root_link_lin_vel_w)
```
Training explicitly rotates the world-frame velocity into body frame before adding it to the observation. At the robot's 90° yaw, world-X ↔ body-Y are completely swapped, making this a critical error — the policy received the wrong velocities in two out of three axes.

**Angular velocity (`qvel[3:6]`) was actually correct.** MuJoCo freejoint `qvel[3:6]` is already body-frame angular velocity. Training also produces body-frame angular velocity (via `quat_apply_inverse(quat_w, ang_vel_w)`). Both resolve to the same value.

**Fix:** Added `_rot_world_to_body(q_wxyz, v_w)` numpy helper (same rotation matrix as `_quat_rot_inv` in task.py) and applied it to `dq[:3]` in `update_state()`:
```python
base_lin_vel_b = _rot_world_to_body(base_quat, dq[:3])
```
Also fixed `qfrc_actuator[:23]` → `qfrc_actuator[6:29]` (robot joints are dofs 6-28 in the scene with ball).

**Evidence:** Trace through `mjlab/entity/data.py` lines 594-602 and `observations.py` lines 24-28 confirmed training always gives body-frame velocity. Rotation test: world `[0,1,0]` at 90° yaw → body `[1,0,0]` ✓.

---

### Bug Fix 2: Ball visibility masking was completely absent

**What was wrong:** Deployment always provided raw ball position and velocity. The policy was trained with three masking gates:
1. **initial_vanish** — ball hidden for the first ~7-8 policy steps after each episode reset (catchstep warmup, mirrors upstream `startstep ≈ 43`)
2. **flying** — ball hidden unless approaching, within 3.4 m in Y, |X| < 2 m, Z < 1.8 m
3. **random_vanish** — ball disappears for a random number of steps (sampled 0-29) per episode

Without masking, ball_pos_b and ball_vel_b carried out-of-distribution values during the warmup window and after the ball passed the robot.

**Fix:** Added ball visibility state to `GoalkeeperMujocoController`:
- `_ball_step` countdown (50→0 per launch, decremented in `update_state()`)
- `_ball_prev_y` for approach direction tracking
- `_ball_vanish_step` random threshold (re-sampled in `_launch_ball()`)
- `_ball_visible_step` flying-step counter
- `_compute_ball_visibility()` method mirroring `observations.py` logic exactly
- `ball_visible: bool` attribute read by the policy

In `task.py` `_compute_obs()`:
```python
vis = 1.0 if getattr(self.controller, 'ball_visible', True) else 0.0
ball_pos_b = _quat_rot_inv(q, ball_pos_w - base_pos) * vis
ball_vel_b = _quat_rot_inv(q, ball_vel_w) * vis
```

**Evidence:** Integration test confirmed `_ball_step` decrements correctly, visibility stays False during warmup (steps 0-7), and activates when ball enters flying zone.

**Files changed:** `goalkeeper_deploy/tasks/goalkeeper/controller.py`, `goalkeeper_deploy/tasks/goalkeeper/task.py`

---

## 2026-06-05 — Re-enable catchstep warmup initialization and decrement (Feature 7 fix)

**What was wrong:** Feature 7 (catchstep warmup) was implemented in `mdp/commands.py` but both initialization (`_reset_ball`) and decrement (`_update_command`) were commented out with note "Skipped for now due to indexing issues; ball starts with zero velocity instead." This disabled Feature 7 and Feature 8 (ball visibility masking), which depend on `env._catchstep` being set and tracked.

Without active catchstep:
- Ball observations are always visible during warmup window (fallback in `observations.py` line 57-60)
- Policy receives out-of-distribution ball pos/vel during the first ~7-8 steps after reset
- Feature 8 visibility masking was only partially active

**Root cause:** Original code used `scatter_()` operation which failed with certain `env_ids` tensor shapes/dtypes. However, new `env_ids` validation guard (lines 188-192) now ensures `env_ids` is 1D and of dtype `torch.long` or `torch.int64`, making direct indexing safe.

**Fix:** Re-enabled both blocks using direct indexing (matching upstream and `my_mjlab_project_booster_t1` project):
- `_reset_ball()`: `self._env._catchstep[env_ids] = 50`
- `_update_command()`: `self._env._catchstep = (self._env._catchstep - 1).clamp(min=0)`

The `env_ids` validation guard (lines 188-192) prevents the original indexing crashes.

**Files changed:** `src/imitationlearningbooster/mdp/commands.py` (lines 280-294, 306-309)
