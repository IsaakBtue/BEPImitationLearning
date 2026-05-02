# Original Isaac Gym vs Isaac Lab Port — Full Comparison

**Date:** 2026-05-02  
**Source (frozen):** `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/`  
**Port:** `Humanoid-Goalkeeper-isaaclab/goalkeeper/`

Read this document **before** changing any reward scale, observation term, network architecture, or training parameter. Every claim here is verified against source code line numbers.

---

## 1. Algorithm: HIM-PPO vs Standard PPO

The biggest divergence. The original uses a **custom HIM-PPO** (`rsl_rl` fork), not standard PPO.

### 1.1 Network Architecture

| Component | Original (`actor_critic.py`) | Port (`rsl_rl_ppo_cfg.py`) |
|---|---|---|
| History encoder | 960-D → 128 → 64 → **16-D latent** | ❌ absent |
| Ball estimator | 960-D → 128 → 32 → **6-D** (auxiliary task) | ❌ absent |
| Region estimator | 960-D → 128 → 32 → **6-D logits** (CrossEntropy) | ❌ absent |
| Actor input | 96 + 16 + 6 + 1 = **119-D** | 960-D raw → 512 → 256 → 256 |
| Critic input | **113-D** privileged | **113-D** privileged ✅ |
| AMP discriminators | 6 × (58-D → 512 → 256 → 1) | ❌ absent |

### 1.2 Loss Function

| Loss term | Original | Port |
|---|---|---|
| PPO surrogate (clipped) | ✅ | ✅ |
| Value loss (clipped) | ✅ | ✅ |
| Ball estimation MSE | ✅ | ❌ absent |
| Region CrossEntropy | ✅ | ❌ absent |
| Smoothness (interpolation) | ✅ | ❌ absent |
| AMP discriminator loss (×6) | ✅ | ❌ absent |
| Gradient penalty (λ=5) | ✅ | ❌ absent |
| Entropy bonus (−0.01) | ✅ | ✅ |

### 1.3 Reward Blending

```
Original:  final_reward = 0.4 × AMP_reward + 0.6 × task_reward
Port:      final_reward = task_reward  (AMP = 0)
```

The AMP reward penalises unnatural motion by measuring how "discriminator-fooling" the joint trajectories are. Without it, the agent has no incentive to move naturally or limit joint velocities beyond the explicit penalty terms.

### 1.4 Training Hyperparameters (both match)

| Param | Original | Port |
|---|---|---|
| lr | 1e-3 (adaptive) | 1e-3 (adaptive) ✅ |
| clip_param | 0.2 | 0.2 ✅ |
| gamma | 0.99 | 0.99 ✅ |
| lambda (GAE) | 0.95 | 0.95 ✅ |
| num_mini_batches | 4 | 4 ✅ |
| num_learning_epochs | 5 | 5 ✅ |
| steps_per_env | 100 | 100 ✅ |
| max_grad_norm | 1.0 | 1.0 ✅ |
| entropy_coef | 0.01 | 0.01 ✅ |
| desired_kl | 0.01 | 0.01 ✅ |

---

## 2. Observations

### 2.1 Per-step policy observation (96-D) — matches exactly

```
ball_local   (3)  — ball pos in torso frame; zeroed when hidden
ang_vel      (3)  — base angular velocity
gravity      (3)  — projected gravity vector
dof_pos      (29) — joint positions
dof_vel      (29) — joint velocities × 0.05
actions      (29) — last policy action
──────────────────
total        96   × 10 history steps = 960-D actor input
```

Noise applied to policy obs (not critic):
```
ball:     ±0.08   ang_vel: ±0.05   gravity: ±0.05
dof_pos:  ±0.01   dof_vel: ±0.075  actions:  0.0
```

### 2.2 Critic / privileged obs (113-D) — matches exactly

```
policy obs (96) + lin_vel(3) + region(1) + end_target(3) + ball_vel(3)
               + hand_r(3) + hand_l(3) + dist(1) = 113-D
```

No noise. Ground-truth ball position, velocity, and hand distances available.

### 2.3 What changed

| Item | Original | Port |
|---|---|---|
| Dims (policy history) | 960 (= 96 × 10) | 960 ✅ |
| Dims (critic) | 113 | 113 ✅ |
| Actor sees | encoded 119-D (latent + estimates) | raw 960-D |
| Ball masking logic | ✅ (vanish_step, catchstep) | ✅ identical |
| Noise | ✅ | ✅ identical scales |

---

## 3. Reward Functions

All 24 reward terms are ported with identical formulas. Both versions multiply raw scales by `dt = 0.02 s` in `_build_reward_scales()`.

Reporting note: the logging divides episode sums by `ep_len × dt`, so reported values are **reward per second**, not per step. Multiply by `ep_len × dt = 150 × 0.02 = 3` to get per-episode totals.

| Term | Raw scale | Per-step | Per-second |
|---|---|---|---|
| eereach | 10.0 | 0.20 | 10.0 |
| success | 5.0 | 0.10 | 5.0 |
| stopball | 100.0 | 2.00 | 100.0 |
| stayonline | −2.0 | −0.040 | −2.0 |
| noretreat | −2.0 | −0.040 | −2.0 |
| successland | 4.0 | 0.080 | 4.0 |
| feetorientation | 3.0 | 0.060 | 3.0 |
| penalize_sharpcontact | −100.0 | −2.00 (discrete) | up to −100 |
| penalize_kneeheight | −100.0 | −2.00 (discrete) | up to −100 |
| feet_slippage | 3.0 | 0.060 | 3.0 |
| postorientation | 3.0 | 0.060 | 3.0 |
| postangvel | 3.0 | 0.060 | 3.0 |
| postlinvel | 1.0 | 0.020 | 1.0 |
| postupperdofpos | 1.0 | 0.020 | 1.0 |
| postwaistdofpos | 1.0 | 0.020 | 1.0 |
| ang_vel_xy | −0.1 | −0.002 | −0.1 |
| dof_acc | −2.5e-7 | −5e-9 | −2.5e-7 |
| smoothness | −0.1 | −0.002 | −0.1 |
| torques | −1e-5 | −2e-7 | −1e-5 |
| **dof_vel** | **−5e-4** | **−1e-5** | **−5e-4** |
| dof_pos_limits | −3.0 | −0.060 | −3.0 |
| **dof_vel_limits** | **−2.0** | **−0.040** | **−2.0** |
| **torque_limits** | **−3.0** | **−0.060** | **−3.0** |
| deviation_waist_pitch_joint | −0.001 | −2e-5 | −0.001 |

Curriculum scaling (both versions): eereach/success/stopball scale with `(1 + 0.5 × curriculumupdate)`. dof_pos_limits and torque_limits multiply by 2–3× at higher curriculum levels.

---

## 4. PD Control

Both versions implement the same formula manually:

```python
torques = p_gains × Kp_factors × (target - joint_pos) - d_gains × Kd_factors × joint_vel
torques += actuation_offset + joint_injection
torques = clip(torques, -torque_limits, torque_limits)
```

Gains per joint type:

| Joint | Kp (N·m/rad) | Kd (N·m·s/rad) |
|---|---|---|
| hip_yaw/roll/pitch | 150 | 2 |
| knee | 300 | 4 |
| ankle | 40 | 2 |
| shoulder/elbow/waist | 150 | 2 |
| wrist | 20 | 0.5 |

Original used Isaac Gym built-in drives (abstracted away); port applies torques via `set_joint_effort_target()`. Same math, different API.

---

## 5. Domain Randomization

| Type | Range | Original | Port |
|---|---|---|---|
| Kp scale | [0.8, 1.2] | ✅ every reset | ✅ every reset |
| Kd scale | [0.8, 1.2] | ✅ every reset | ✅ every reset |
| Actuation offset | [−0.01, 0.01] × limits | ✅ every reset | ✅ every reset |
| Joint injection | [−0.01, 0.01] × limits | ✅ per decimation step | ✅ per decimation step |
| Robot push | ±1.5 m/s xy every 15 s | ✅ | ✅ |
| Ball perturbation | ±0.5 m/s xyz every 0.5 s | ✅ | ✅ |
| Initial joint pos | scale [0.5,1.5] + offset | ✅ | ✅ |
| Friction | [0.1, 2.0] | ✅ applied to physics | ❌ sampled, **NOT applied** |
| Restitution | [0.0, 1.0] | ✅ applied to physics | ❌ sampled, **NOT applied** |
| Payload mass | [−5, 10] kg on base | ✅ applied | ❌ sampled, **NOT applied** |
| COM displacement | [−0.1, 0.1] m | ✅ applied | ❌ sampled, **NOT applied** |
| Link mass scale | [0.8, 1.2] | ✅ applied | ❌ sampled, **NOT applied** |

Missing DR reduces sim-to-real robustness. The robot learns to walk on fixed-friction ground only.

---

## 6. Ball Physics

Both versions use identical parabolic trajectory + drag:

```
drag = -0.5 × rho × Cd × A × |v| × v + uniform(-0.5, 0.5)
rho=1.225 kg/m³, Cd=0.47, A=π×(0.1)²=0.0314 m²
```

Ball trajectory: 6 catch regions, flight time [0.4, 1.0] s (training), [0.5, 1.0] s (play). Target interception point updated as ball approaches.

---

## 7. Reset & Termination

Both versions: identical termination conditions (knee < 0.10 m, gravity tilt > 0.8, sharp contact > 1500 N, timeout 3 s).

Reset procedure: 80% copy from random other env, 20% sample from standpos ± noise.

`init_pos` values match exactly between original and port (29-element list). The port reorders DFS → BFS at init time via name lookup (commit 705ef0a).

---

## 8. Root Cause Analysis: High rew_dof_vel_limits (−1145 /s)

The user observed after 6+ hours of training:
```
rew_dof_vel:       −149  /s
rew_dof_vel_limits: −1145 /s
rew_torque_limits:  −9.5  /s
```

**Back-calculation of what these numbers mean:**

rew_dof_vel_limits = −1145 /s:
- Per episode: −1145 × 150 steps × 0.02 s/step = −3435
- Per step: −3435 / 150 = −22.9
- dof_vel_limits scale per step = −2.0 × 0.02 = −0.04
- Sum of excess velocity per step = 22.9 / 0.04 = **573 rad/s** across 29 joints ≈ **~20 rad/s excess per joint average**

Normal joint soft velocity limits are ~5–15 rad/s. The agent is operating at 3–4× soft limits persistently.

**Why this happens without AMP:**

1. Standard PPO has no motion naturalness reward — large velocities are "free" unless explicitly penalised
2. `rew_dof_vel` (−5e-4) → per-step cost = −1e-5 per rad²/s² — extremely weak
3. Task rewards (eereach = +0.20/step) dominate; agent learns to swing joints aggressively to intercept ball
4. Without discriminator guiding toward reference mocap, chaotic explosive motion is the local optimum

**Comparison to original:**

In HIM-PPO, the AMP reward blends in at 40% and continuously penalises deviations from reference motions. A policy that swings joints at 20 rad/s would score ~0 on the discriminator regardless of task performance, making this strategy non-viable.

---

## 9. What to Fix (Priority Order)

### P1 — Strengthen velocity/torque penalties (immediate)

Without AMP, these penalty scales are too weak relative to task rewards:

| Config field | Current | Recommended |
|---|---|---|
| `rew_dof_vel` | −5e-4 | −5e-3 |
| `rew_dof_vel_limits` | −2.0 | −5.0 |
| `rew_torque_limits` | −3.0 | −5.0 |

Alternative: reduce `action_scale` from 0.25 → 0.15 to limit maximum joint velocity from policy.

### P2 — Apply friction/restitution DR to physics (moderate, sim-to-real)

Use `ArticulationView.set_material_properties()` in `_reset_idx()`.

### P3 — Apply mass/COM/link mass DR to physics (moderate, sim-to-real)

Use `ArticulationView.set_masses()` and `set_inertias()` in `_reset_idx()`.

### P4 — Implement AMP motion priors (long-term, motion quality)

**Option A (recommended):** Switch to skrl framework.
- Reference: `/home/isaak/BEPImitationlearning/humanoid_amp/`
- Add `extras["amp_obs"]` (2-frame dof_pos buffer, 58-D) to `_get_observations()`
- skrl AMP agent trains discriminator against `.npz` motion dataset
- Motion data: convert `.pt` → `.npz` (or write custom skrl MotionDataset adapter)

**Option B:** Compute AMP reward inside `_get_rewards()` with frozen or slowly-updated discriminator (no backprop, partial fidelity).

### P5 — Restore HIM-PPO auxiliary tasks (long-term, performance)

Ball estimator and region estimator auxiliary heads require either a custom rsl_rl fork or a completely custom training loop. Not possible within standard rsl_rl 5.0.1. Port to HIM-PPO requires forking rsl_rl or porting the HIM-PPO code to work with Isaac Lab environments.
