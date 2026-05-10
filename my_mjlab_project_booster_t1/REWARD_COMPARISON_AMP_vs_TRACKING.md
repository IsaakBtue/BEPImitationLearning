# Reward Architecture Comparison: AMP (isaacgym) vs Motion Tracking (mjlab)

## 1. Side-by-Side Reward Table

### Motion / Style Rewards

| Reward | isaacgym (AMP) | Weight | mjlab (Motion Tracking) | Weight |
|---|---|---|---|---|
| Motion naturalness | AMP discriminator reward | 0.4 × [0..0.5]/step | `motion_global_root_pos` (exp decay on global anchor pos error) | **10.0** |
| Motion orientation | (covered by AMP) | — | `motion_global_root_ori` (exp decay on anchor orientation error) | **5.0** |
| Whole-body pose | (covered by AMP) | — | `motion_body_pos` (exp decay on 14 body positions, relative frame) | **10.0** |
| Body orientation | (covered by AMP) | — | `motion_body_ori` | 3.0 |
| Body linear velocity | (covered by AMP) | — | `motion_body_lin_vel` | 3.0 |
| Body angular velocity | (covered by AMP) | — | `motion_body_ang_vel` | 3.0 |
| **Total motion weight** | | **~0.2/step max** | | **34.0** |

### Task Rewards

| Reward | isaacgym | Scale (×dt) | mjlab | Weight |
|---|---|---|---|---|
| Stop ball | `stopball` | +100 × 0.02 = **+2.0** | `stopball` | **+100.0** |
| Reach ball | `eereach` (sigmoid dist × upright factor) | +10 × 0.02 = **+0.2** | `eereach` | **+10.0** |
| Catch success | `success` (binary dist<0.15m) | +5 × 0.02 = **+0.1** | `catch_success` (binary dist<0.3m) | **+5.0** |
| Stay on line | `stayonline` | −2 × 0.02 = **−0.04** | `stayonline` | **−2.0** |
| No retreat | `noretreat` | −2 × 0.02 = **−0.04** | `noretreat` | **−2.0** |
| Feet orientation | `feetorientaion` | +3 × 0.02 = **+0.06** | `feetorientation` | **+3.0** |
| Post-ball upright | `postorientation` | +3 × 0.02 = **+0.06** | `postorientation` | **+3.0** |
| Post-ball ang vel | `postangvel` | +3 × 0.02 = **+0.06** | `postangvel` | **+3.0** |
| Post-ball lin vel | `postlinvel` | +1 × 0.02 = **+0.02** | `postlinvel` | **+1.0** |

### Regularization Rewards

| Reward | isaacgym | Scale (×dt) | mjlab | Weight |
|---|---|---|---|---|
| Angular velocity XY | `ang_vel_xy` | −0.1 × 0.02 = **−0.002** | *(not present)* | — |
| Joint acceleration | `dof_acc` | −2.5e-7 × 0.02 | *(not present)* | — |
| Action smoothness | `smoothness` (2nd order) | −0.1 × 0.02 | `action_rate_l2` (1st order) | **−0.1** |
| Torque magnitude | `torques` | −1e-5 × 0.02 | *(not present)* | — |
| Joint velocity | `dof_vel` | −5e-4 × 0.02 | *(not present)* | — |
| DOF position limits | `dof_pos_limits` | −3.0 × 0.02 | `joint_limit` | **−10.0** |
| DOF velocity limits | `dof_vel_limits` | −2.0 × 0.02 | *(not present)* | — |
| Torque limits | `torque_limits` | −3.0 × 0.02 | *(not present)* | — |
| Waist pitch deviation | `deviation_waist_pitch_joint` | −0.001 × 0.02 | *(not present)* | — |
| Sharp foot contact | `penalize_sharpcontact` | **−100 × 0.02 = −2.0** | *(not present)* | — |
| Knee height | `penalize_kneeheight` | **−100 × 0.02 = −2.0** | *(not present)* | — |
| Foot slippage | `feet_slippage` | +3 × 0.02 | *(not present)* | — |
| Post upper DOF | `postupperdofpos` | +1 × 0.02 | *(not present)* | — |
| Post waist DOF | `postwaistdofpos` | +1 × 0.02 | *(not present)* | — |
| Self-collisions | *(not present)* | — | `self_collisions` (force > 10 N) | **−10.0** |

---

## 2. Architectural Differences

### AMP (Adversarial Motion Prior)

**How it works:**

```
total_reward = 0.4 × AMP_reward + 0.6 × task_reward

AMP_reward (per env per step):
  - Sample 20 noisy perturbations of current DOF state (σ=0.3)
  - Evaluate discriminator D on all 20: reward = clamp(1 - 0.25 × min((D-1)²), 0)
  - Range: [0, 0.5]
  - Scaled by 0.5 at rollout → final range [0, 0.25]

Discriminator:
  - Input: 58-dim = [dof_pos_{t-1} | dof_pos_t] (29 joints × 2 timesteps)
  - Architecture: Linear(58→512)→ReLU→Linear(512→256)→ReLU→Linear(256→1)
  - 6 separate discriminators (one per motion class: lefthand/righthand/leftjump/rightjump/leftstep/rightstep)
  - Loss: LSGAN + gradient penalty (λ=5, scaled 0.1)
  - Trained jointly with actor/critic via shared Adam optimizer
```

The AMP reward is **implicit** — the agent has no direct penalty for specific pose errors; it simply learns "look like the expert dataset" through a binary real/fake signal.

**Motion data loading:**
- `.pt` files: `base_position`, `base_pose`, `joint_position`, `joint_velocity`, `link_position`
- 6 motion classes × multiple clips, indexed by end_region of each environment
- Expert samples: 2 consecutive frames at random temporal offset (fps interpolation)

---

### Motion Tracking (mjlab)

**How it works:**

```
total_reward = sum of all weighted terms (no mixing coefficient)

Motion tracking rewards use exp-decay kernels:
  reward = exp(-error² / (2σ²))
  - motion_global_root_pos:  σ=0.3, weight=10  → dominant signal
  - motion_body_pos:         σ=0.3, weight=10  → dominant signal
  - motion_global_root_ori:  σ=0.4, weight=5
  - motion_body_ori:         σ=0.4, weight=3
  - motion_body_lin_vel:     σ=1.0, weight=3
  - motion_body_ang_vel:     σ=3.14, weight=3
```

The motion tracking reward is **explicit** — the agent gets a real-valued penalty proportional to how far each body part is from the reference motion frame-by-frame. There is no learned discriminator.

**Motion data loading:**
- `.npz` files: per-body position, orientation, velocity, angular velocity at each timestep
- 14 tracked bodies: Trunk, hips, shanks, ankles, waist, arms, hands
- Each environment samples a random start frame from the motion clip
- Reference state advances in lockstep with simulation time

---

## 3. Observations During Training vs Inference

### isaacgym (AMP)

| Observation | Actor (inference) | Critic (training only) |
|---|---|---|
| Ball local position (torso frame) | ✓ | ✓ |
| Base angular velocity | ✓ | ✓ |
| Projected gravity | ✓ | ✓ |
| DOF positions (29) | ✓ | ✓ |
| DOF velocities (29) | ✓ | ✓ |
| Actions (29) | ✓ | ✓ |
| *10-step history of above* | ✓ (960-dim) | ✓ |
| **Base linear velocity** | — | ✓ (privileged) |
| **End region / motion class** | — | ✓ (privileged) |
| **Exact ball endpoint (target)** | — | ✓ (privileged) |
| **Ball velocity** | — | ✓ (privileged) |
| **Right hand position** | — | ✓ (privileged) |
| **Left hand position** | — | ✓ (privileged) |
| **Hand-to-target distance** | — | ✓ (privileged) |

Actor: **960-dim** (96 per step × 10 history)  
Critic: **113-dim**

The actor must **reconstruct** ball velocity, hand positions, and end-region from history alone. This is the "latent estimation" in HIM-PPO.

---

### mjlab (Motion Tracking)

| Observation | Actor (inference) | Critic (training only) |
|---|---|---|
| Base linear velocity | ✓ | ✓ |
| Base angular velocity | ✓ | ✓ |
| Joint positions (23) | ✓ | ✓ |
| Joint velocities (23) | ✓ | ✓ |
| Actions (23) | ✓ | ✓ |
| Ball position (base frame) | ✓ | ✓ |
| Ball velocity (base frame) | ✓ | ✓ |
| Left hand position | ✓ | ✓ |
| Right hand position | ✓ | ✓ |
| **All 14 body positions** (relative to motion ref) | — | ✓ (privileged) |
| **All 14 body orientations** (relative to motion ref) | — | ✓ (privileged) |

Actor: **87-dim** (no history, no motion reference terms)  
Critic: **87 + 14×(3+4) = 87 + 98 = 185-dim**

The actor sees ball position and velocity directly — it does not need to reconstruct them from history. Motion reference state is only accessible to the critic (as a training guide), not the deployed policy.

---

## 4. Hypothesis: Why One Might Learn Faster / Better

### Motion Tracking likely converges faster because:

1. **Dense, shaped signal**: Every step, each of the 14 bodies gets a real-valued reward proportional to pose error. The gradient is informative from step one — unlike AMP where the discriminator starts untrained and provides near-random signal.

2. **No chicken-and-egg problem**: AMP must train the discriminator and actor simultaneously. Early discriminator weights give poor signal, slowing actor learning. Motion tracking bypasses this by using ground-truth pose error.

3. **Simpler credit assignment**: The exp-decay kernel reward directly tells the agent *which body parts are wrong* (if you decompose the sum). AMP gives one scalar per step with no indication of where the motion diverges.

4. **No hyperparameter for mixing**: AMP requires tuning `amp_coef=0.4`. Set too high → ignores task reward; too low → ignores motion style. Motion tracking has no such knob — the weight ratio between motion and task rewards serves the same role but is more interpretable.

### AMP may produce more generalizable / robust motion because:

1. **Discriminator is harder to game**: A motion tracking reward can be minimized by finding an unusual pose that happens to match the reference numerically. The discriminator looks at temporal coherence and the full distribution of expert data.

2. **Covers unseen transitions**: Expert datasets contain natural recovery and in-between states. AMP generalizes from these implicitly. Motion tracking only rewards matching the specific frames in the clip.

3. **6 motion classes**: AMP supports lefthand/righthand/jump/step saves with a single agent by conditioning on the end_region. Motion tracking currently trains only on `lefthand_t1.npz` (one motion file).

4. **History-based actor**: The 960-dim actor obs (10-step history) allows the HIM-PPO policy to internally estimate state that's hidden at inference time (ball velocity, end region). This is more deployable — the real robot also lacks direct access to privileged state.

### Practical observation

Motion tracking with explicit rewards is the **standard approach for initial learning** of a specific motion: it converges in hundreds of iterations to a recognizable policy. AMP is the **standard approach for style generalization**: it produces more natural, less jerky motion at the cost of slower convergence and more hyperparameter sensitivity.

For a goalkeeper saving a single motion class (lefthand dive), motion tracking is likely the better starting point. For a full goalkeeper that handles all 6 directions naturally, AMP's multi-class discriminators are the right architecture.
