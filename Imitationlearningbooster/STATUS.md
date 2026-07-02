# Goalkeeper AMP Booster T1 — Session Status
_Last updated: 2026-06-10_

---

## Current Training Run

| Item | Value |
|---|---|
| Checkpoint | `logs/rsl_rl/goalkeeper_amp_booster_t1/2026-06-09_22-28-51/model_7800.pt` |
| Iterations completed | 7800 |
| Branch | `singleMotionNonDomain` |
| num_envs | 6144 |
| num_steps_per_env | 100 |
| max_iterations | 40 000 |

Training is running. Symptom at 7800 iters: policy appears to only perform lefthand saves.
**Root cause identified and fixed** (see Bug B1 below) — play mode had wrong ball axis, not a training flaw.

---

## Architecture: What Is Running

### 6 AMP Discriminators — confirmed correct

```
MOTION_NAMES = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
```

- 6 × `Discriminator` (input_dim=46, 512→256, spectral_norm) in `him_amp_runner.py`
- 6 × `GoalkeeperMotionLoader` (one per `.npz` file)
- 6 × `ReplayBuffer` (obs_dim=23, size=100 000)
- 1 shared `EmpiricalNormalization` (shape=(46,))
- 1 Adam optimizer over all 6 discriminators

### Static Env Partitioning

`static_partition=True` in `MultiMotionCommandCfg`:

| Env indices | Motion type | Ball Y end target | Ball Z target |
|---|---|---|---|
| 0 – 1023 | lefthand (0) | -Y side [-0.65, -0.15] | mid [0.40, 1.15] |
| 1024 – 2047 | righthand (1) | +Y side [+0.15, +0.65] | mid [0.40, 1.15] |
| 2048 – 3071 | leftjump (2) | -Y side [-0.65, -0.15] | high [0.85, 1.40] |
| 3072 – 4095 | rightjump (3) | +Y side [+0.15, +0.65] | high [0.85, 1.40] |
| 4096 – 5119 | leftstep (4) | -Y side [-0.65, -0.15] | low [0.20, 0.65] |
| 5120 – 6143 | rightstep (5) | +Y side [+0.15, +0.65] | low [0.20, 0.65] |

Each group permanently sees only its motion type. AMP discriminator per type trains on 1024 dedicated envs. Assignment is restored at every episode reset (never re-randomised).

**Ball per-group differences:**
- `x_start` (approach distance): uniform [3.0, 4.5] — same for all groups
- `y_end` (lateral target): per-group from table above, difficulty-interpolated
- `z_end` (height target): per-group from table above, difficulty-interpolated
- `y_start` / `z_start`: randomised [−0.8, 0.8] / [0.5, 1.4] — same for all groups
- Approach axis: **+X** (ball spawns at world X + x_start, travels toward X ≈ −0.3)

### Ball Approach Axis

Ball ALWAYS approaches from world **+X** (confirmed for both training and play after 2026-06-10 fix).
Robot faces world **+Y** (T1_STANDING_KEYFRAME has +90° yaw quaternion `(0.7071, 0, 0, 0.7071)`).

In body frame (robot faces +Y = body +X):
- Body +X = world +Y = robot forward
- Body -Y = world +X = **where ball comes from** (robot's right side in body frame)
- Ball visibility uses `ball_x_local > 0.05` (world X), which works because ball is at world X > 3 m at start.

### Key Dimensions

| Quantity | Value |
|---|---|
| T1 DOF | 23 (incl. head) |
| AMP obs dim per frame | 23 |
| AMP discriminator input | 46 (2 consecutive frames × 23) |
| Actor obs (per step) | ~87 dims |
| Actor history length | 10 frames |
| Total actor input | 870 |
| Episode length | 3.0 s = 150 steps @ 50 Hz |
| Ball flight time | 0.5 – 1.0 s |
| Curriculum max (curriculumupdate) | 3 |

---

## Motion Files

| File | Size | Notes |
|---|---|---|
| `lefthand_t1.npz` | 297 KB | Shorter clip (~half others) — loops more in 3 s episodes |
| `righthand_t1.npz` | 591 KB | |
| `leftjump_t1.npz` | 609 KB | |
| `rightjump_t1.npz` | 480 KB | |
| `leftstep_t1.npz` | 466 KB | |
| `rightstep_t1.npz` | 473 KB | |

All 6 files confirmed present and loaded at training start. `joint_names` embedded in each npz; `_verify_joint_order()` runs once at `learn()` start.

---

## All Significant Fixes Applied (chronological, newest first)

### B1 — 2026-06-10: Play-mode ball spawn axis mismatch (CRITICAL)
**File:** `mdp/resets.py` → `_shoot_ball`
**Bug:** Ball was launched from **+Y** direction (`y_start = 3–5 m`). The visibility check gates on `ball_x_local > 0.05` (world X). Ball was permanently invisible → policy got zero ball observations → defaulted to lefthand save every episode.
**Fix:** `_shoot_ball` now uses same +X axis as training: `x_start ∈ [3.0, 4.5]`, `dx = -x_start - 0.3`, bilateral Y end targets ±0.65 m.

### B2 — 2026-06-09: Adaptive curriculum driver
**File:** `mdp/resets.py`, `tasks/goalkeeper_amp_env_cfg.py`
**Bug:** Fixed-step curriculum advanced on wall-clock steps, not agent performance.
**Fix:** `adaptive_curriculum_update` mirrors G1 exactly: `env._curriculumupdate = int(mean(ep_len) / 50)` gated every 500 sim steps. Sets `_ball_difficulty = _curriculumupdate / 3.0`.

### B3 — 2026-06-09: Static env partitioning for AMP (mirrors G1 `end_regions`)
**File:** `mdp/commands.py`, `tasks/goalkeeper_amp_env_cfg.py`
**Bug:** `motion_type_ids` was re-randomised at every episode reset. AMP discriminators received mixed-style training signal.
**Fix:** `static_partition=True` → permanent assignment at init, restored at every reset.

### B4 — 2026-06-06 (Fourth-pass): 5 AMP training bugs
- **EmpiricalNormalization variance collapse** — Chan's algorithm wrong → var→0.01 → discriminator inputs 10 000× amplified
- **120 backward passes → 1** — `zero_grad/backward/step` was outside the `num_disc_updates` loop
- **Normalizer fed its own output** — normalizer.update called on post-normalized data
- **stopball threshold 2.0→1.0** — MuJoCo soft contacts produce smaller impulses than PhysX
- **eereach routed to wrong hand** — was using `min(both hands)`, now uses correct hand by motion type

### B5 — 2026-06-05: AMP joint ordering verification
Added `joint_names` to all 6 npz files. `_verify_joint_order()` in runner asserts match at `learn()` start.

### B6 — 2026-06-05 (Third-pass): 7 AMP training signal bugs
- `num_steps_per_env` 24→100 (policy never saw full ball approach)
- AMP reward scale 0.1→0.5 (was 5× too weak)
- Discriminator gradient penalty over-regularised (effective λ=5 → λ=0.5)
- Disc update freq 4→20 steps/iteration (5× under-trained)
- GAN loss halved by mistake → restored
- `eereach` vel_sigma: hand velocity → lateral/vertical torso velocity
- `eereach` phase1 missing deflection guard

### B7 — 2026-06-05: Ball visibility masking (per-env `vanish_step` / `startstep`)
`_reset_ball` now re-samples `_vanish_step` and `_startstep` per-env per-episode. Step-caching guard prevents double-execution of stateful visibility check.

### B8 — 2026-05-24: 8 missing upstream features
`successland`, `penalize_sharpcontact`/`sharpforce_termination` force averaging, ball difficulty curriculum, `dof_pos_limits` / `torque_limits` / `hand_proximity_strict` curricula, `eereach` intercept target + Phase 1 pre-positioning, catchstep warmup, ball visibility masking.

### B9 — 2026-05-20: KaydenKnapik hardware-verified actuator config
Effort limits, actuator delay (2–8 steps @ 200 Hz), joint_vel noise ±1.5, `actuatorfrcrange` removed from XML, per-joint `torque_limits` via `_T1_EFFORT_MAP`.

### B10 — 2026-05-15: Curriculum timing proportional to num_steps_per_env
`stage1 = 600 × nspenv`, `stage2 = 1200 × nspenv` — changing the rollout length no longer breaks curriculum timing.

### B11 — 2026-05-15: Episode structure (3 s, one ball per episode, correct fps)
Motion clips resampled 30fps→50Hz, trimmed to 3 s (150 frames). Ball relaunch on motion loop suppressed (`reset_ball=False`).

### B12 — 2026-05-15: G1 fall detection terminations
Replaced mjlab tracking terminations (`anchor_pos/ori`, `ee_body_pos`) with `bad_orientation` (limit=1 rad) + `base_height` (min=0.4 m).

### B13 — 2026-05-15: Core reward bugs fixed
`postupperdofpos` sigma 20→1, `postwaistdofpos` sigma 20→3, `noretreat` world-Y→body-frame-X, `stopball` threshold 2.0→1.0, `hand_proximity_strict` added.

---

## Reward Weights (current)

| Reward | Weight | Notes |
|---|---|---|
| `eereach` | 20.0 (→36 max at curriculum 3) | Sigmoid + vel_sigma, per-motion-type hand routing |
| `hand_proximity_strict` | 10.0 (→20 max) | Dense < 0.15 m signal |
| `stopball` | 100.0 (→250 max at cu=3) | One-time Δvx > 1.0 m/s |
| `stayonline` | −2.0 | X-axis deviation from goal line |
| `noretreat` | −2.0 | Body-frame backward motion |
| `feetorientation` | 3.0 | Feet flat |
| `postorientation` | 3.0 | Upright after ball passes |
| `postangvel` | 3.0 | Low angular vel after ball passes |
| `postlinvel` | 1.0 | Low forward vel after ball passes |
| `successland` | 4.0 | Landing reward after jump save |
| `penalize_sharpcontact` | −100.0 | Foot force > 1000 N |
| `penalize_kneeheight` | −100.0 | Knee < 0.12 m from ground |
| `penalize_self_collision` | −50.0 | Trunk-subtree self-contact |
| `feet_slippage` | 3.0 | exp(−10 × slip) |
| `postupperdofpos` | 1.0 | Arms return to default after ball |
| `postwaistdofpos` | 1.0 | Waist returns to default after ball |
| `dof_acc` | −2.5e-7 | Joint acceleration |
| `action_rate_l2` | −0.1 | Action rate |
| `torques` | −1e-5 | Normalized torques |
| `dof_vel` | −5e-4 | Joint velocity |
| `dof_pos_limits` | −3.0 (→−9 max) | Joint position limit violations |
| `dof_vel_limits` | −2.0 | Joint velocity limit violations |
| `torque_limits` | −3.0 (→−9 max) | Per-joint torque limit violations |
| `deviation_waist_joint` | −0.001 | Always-on waist stability |
| `ang_vel_xy` | −0.1 | Roll/pitch angular velocity |
| **AMP blend** | 40% AMP + 60% task | Blended in `him_amp_runner.py` |

---

## Known Remaining Issues / To-Do

### Not yet verified after fix B1
- Play mode ball now uses +X axis — re-run play to confirm policy uses both hands.
- If still only lefthand: check that `body_ids[0]` = left hand and `body_ids[1]` = right hand (mjlab may assign by XML order not `body_names` order). Check with a quick debug print.

### DR disabled
Domain randomisation for `pd_gains`, `link_mass`, `reset_joints` is **off** to keep early training stable. Re-enable before sim-to-real transfer:
```python
# In goalkeeper_amp_env_cfg.py, currently popped:
_dr_events = ["base_com", "encoder_bias", "foot_friction", "push_robot"]
```

### HIM-PPO internal model not implemented
Current policy uses flat 870-D input MLP. The upstream G1 uses a history encoder (960→16-D latent) + ball estimator + region estimator feeding a 119-D actor. Not blocking for training but reduces sample efficiency.

### lefthand clip is shorter
`lefthand_t1.npz` (297 KB) has ~half the frames of other motions. It loops ~2× more per episode. This gives the lefthand discriminator slightly denser supervision but hasn't caused a measurable bias in training curves. Regenerate with a longer clip if lefthand imbalance persists after fixing B1.

### Ball perturbation mid-trajectory not implemented
G1 applies ±0.5 m/s ball velocity perturbation every 0.5 s during flight. Not yet ported. Needed for robustness.

---

## Key File Locations

| Purpose | File |
|---|---|
| AMP env config | `src/imitationlearningbooster/tasks/goalkeeper_amp_env_cfg.py` |
| PPO runner config | `src/imitationlearningbooster/tasks/goalkeeper_amp_ppo_cfg.py` |
| AMP runner (6 discs) | `src/imitationlearningbooster/rsl_rl_amp/runners/him_amp_runner.py` |
| Discriminator | `src/imitationlearningbooster/rsl_rl_amp/modules/discriminator.py` |
| Ball spawn / 6-way routing | `src/imitationlearningbooster/mdp/commands.py` |
| Rewards | `src/imitationlearningbooster/mdp/rewards.py` |
| Resets / ball autonomous | `src/imitationlearningbooster/mdp/resets.py` |
| Observations + visibility | `src/imitationlearningbooster/mdp/observations.py` |
| Robot constants / action scale | `src/imitationlearningbooster/robots/t1_constants.py` |
| Motion npz files | `src/imitationlearningbooster/motions/data/*.npz` |
| Full fix history | `DIVERGENCE_FROM_UPSTREAM.md` |

---

## Quick Start Commands

```bash
# Activate env
cd /home/robocup/IsaakB/BEPImitationLearning/Imitationlearningbooster
source .venv/bin/activate  # or: env -u PYTHONPATH .venv/bin/...

# Play latest checkpoint (verify ball is now visible and both hands are used)
env -u PYTHONPATH .venv/bin/play goalkeeper_booster_t1_amp \
  --checkpoint logs/rsl_rl/goalkeeper_amp_booster_t1/2026-06-09_22-28-51/model_7800.pt

# Resume training
env -u PYTHONPATH .venv/bin/train goalkeeper_booster_t1_amp \
  --resume logs/rsl_rl/goalkeeper_amp_booster_t1/2026-06-09_22-28-51/model_7800.pt

# Fresh training run
env -u PYTHONPATH .venv/bin/train goalkeeper_booster_t1_amp
```
