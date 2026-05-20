# Training Plan: singleMotionNonDomain → Sim2Real via Phased DR
**Date:** 2026-05-20  
**Branch:** `singleMotionNonDomain` (commit `07fcf53`)  
**Reference DR:** `master` (commit `c5fe974`)  
**Author:** Sonnet 4.6 (primary) + Haiku 4.5 (second opinion — appended below)

---

## Context

The `singleMotionNonDomain` branch is a working single-motion (lefthand_t1.npz) training setup. The robot correctly reaches its left hand toward the ball. This document:
1. Identifies non-DR gaps that should be fixed first
2. Defines a phased domain randomization plan building toward the full master DR config

---

## Part 1 — Non-DR Gaps

These are configuration correctness / completeness issues that should be fixed **before** adding DR. They are not DR themselves — they are bugs or missing baseline items.

---

### Gap 1: Push robot never fires (timing bug)

**File:** `tasks/goalkeeper_env_cfg.py`  
**Current:** `cfg.events["push_robot"].interval_range_s = (10.0, 15.0)`  
**Problem:** Episode length is 3.0 s. With a (10–15 s) interval, the push event never fires within a single episode. The robot effectively trains with zero perturbation force throughout, which means it learns nothing about recovering from unexpected pushes.  
**Fix:** Change interval to **(1.5, 3.0 s)** — this fires approximately once every 1–2 episodes (50–100% hit rate), giving the policy exposure to perturbation forces at minimal DR cost.  
**Note:** This is separate from upgrading to 6-DOF push (that's Phase 2 DR). Even 2D push at the right interval is valuable.

---

### Gap 2: No noise on IMU-derived observations

**File:** `tasks/goalkeeper_env_cfg.py`  
**Current:** `base_lin_vel` and `base_ang_vel` are replaced with direct sim reads but have **no noise** specified.  
**Problem:** Ball observations have explicit noise (±0.05 m, ±0.1 m/s), but proprioceptive velocity observations have none. This creates an inconsistency — the policy can trust IMU data with perfect precision which is unrealistic and creates a sim2real gap.  
**Fix:** Add `noise=Unoise(n_min=-0.05, n_max=0.05)` to `base_lin_vel` and `noise=Unoise(n_min=-0.02, n_max=0.02)` to `base_ang_vel` (smaller angular noise since gyros are more precise than accelerometers).

---

### Gap 3: No noise on hand position observations

**File:** `tasks/goalkeeper_env_cfg.py`  
**Current:** `left_hand_pos_b` and `right_hand_pos_b` have no noise.  
**Problem:** Hand position is computed via FK from joint encoders. Since joint encoders have noise (addressed in encoder_bias DR later), FK propagates that error to hand position. Before encoder_bias DR, a small baseline noise should be present to avoid a perfectly known hand pose.  
**Fix:** Add `noise=Unoise(n_min=-0.01, n_max=0.01)` to both hand position observations.

---

### Gap 4: Ball always spawns with zero angular velocity

**File:** `mdp/commands.py`, `_reset_ball()` method  
**Current:** `ball_ang_vel = torch.zeros((n, 3), device=self.device)` — ball has no spin at spawn.  
**Problem:** Real kicked balls have spin (top-spin, side-spin, back-spin) typically 5–15 rad/s. Zero spin makes the ball trajectory deterministic post-bounce, which doesn't generalize to hardware where ball spin affects deflection angle and bounce direction.  
**Fix:** This is a border case between non-DR and Phase 1 DR. Minimum fix: add small uniform angular velocity at spawn, e.g., each axis independently sampled ±8 rad/s. The ball's short flight time (~0.4–1.0 s) limits spin's effect on trajectory, but it affects the contact deflection physics.  
**Note:** Can also be deferred to Phase 1 ball DR alongside ball_mass and ball_friction.

---

### Gap 5: Foot friction geom name regex — verify it matches T1 URDF

**File:** `tasks/goalkeeper_env_cfg.py`  
**Current:** `cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_[12]$"`  
**Problem:** If this regex doesn't match any geom names in the T1 URDF, the foot friction event silently does nothing. The actual T1 foot geom names need to be verified against the URDF.  
**Action:** Run in play mode and check if foot_friction event fires (or grep the URDF for geom names matching the regex). If no match, update the regex to match actual T1 foot geoms.

---

### Gap 6: Observation delay floor of 0

**File:** `tasks/goalkeeper_env_cfg.py`  
**Current:** `delay_min_lag=0, delay_max_lag=2`  
**Problem:** With min=0, some envs train with zero observation delay, which is impossible on real hardware (always at least 1 control cycle = 20 ms). This means part of the policy population never learns to handle any delay.  
**Fix:** Set `delay_min_lag=1` so all envs train with at least 20 ms delay. This is a minor correctness fix, not DR expansion.

---

## Part 2 — Phased Domain Randomization Plan

### Transition Criteria (general)

Before advancing to the next phase, verify:
- Stopball success rate has plateaued or is improving stably
- Policy doesn't catastrophically regress (eereach reward doesn't drop >30%)
- Training curve has stabilized (no major oscillation over 200+ iterations)

Train duration guideline: each phase should be trained for **at minimum 2,000–3,000 iterations** before evaluating transition readiness, given 6144 envs × 100 steps.

---

### Phase 0 — Non-DR Fixes (apply before DR phases begin)

Apply all Gap fixes from Part 1 above. No new DR is added — these are correctness fixes.

| Fix | Where | Change |
|---|---|---|
| push_robot timing | env_cfg | interval (10–15s) → (1.5, 3.0s) |
| IMU noise | env_cfg | add Unoise to base_lin_vel, base_ang_vel |
| Hand pos noise | env_cfg | add Unoise ±0.01m to both hand obs |
| Obs delay floor | env_cfg | delay_min_lag: 0 → 1 |
| Foot geom names | env_cfg | verify regex matches T1 URDF |
| Ball spin (optional) | commands.py | add ±8 rad/s angular vel at ball reset |

**Training checkpoint:** Start from a fresh run from the `singleMotionNonDomain` baseline (or continue from model_3200.pt if curriculum allows). Verify the policy still dives and reaches after applying these fixes.

---

### Phase 1 — Ball Environment DR

**Rationale:** Ball DRs don't affect robot balance or actuator behavior — they only change the ball's physical properties. These are the lowest-risk DRs to add first because they keep the robot dynamics identical while making the contact interaction more diverse. The policy should adapt quickly since ball contact is already noisy (ball_pos noise ±0.05 m already present).

**Add:**
```python
# Ball mass: ±20% around FIFA standard 0.42 kg, resampled each episode
cfg.events["ball_mass"] = EventTermCfg(
    mode="reset",
    func=mjlab_dr.body_mass,
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "ranges": (0.8, 1.2),
        "operation": "scale",
    },
)

# Ball friction: covers wet/dry ball, gloves vs bare skin
cfg.events["ball_friction"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.geom_friction,
    params={
        "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
        "operation": "abs",
        "ranges": (0.2, 0.8),
    },
)
```

**Also:** Add ball angular velocity at reset if deferred from Phase 0.

**Expected impact:** Low. The robot's motion and balance rewards are unaffected. Contact quality and deflection physics become more variable. The stopball reward may slightly drop initially as the robot encounters varied ball weights.

---

### Phase 2 — Push Robot Upgrade (6-DOF)

**Rationale:** Phase 0 fixes the push timing so it fires. Phase 2 upgrades it to 6-DOF perturbation forces (adds z-linear + roll/pitch/yaw moments). This teaches the robot to maintain balance under realistic stumble forces, which is critical before adding actuator DR in Phase 3. Without balance robustness, actuator DR (Phase 3) would cause premature falls and destabilize training.

**Change:**
```python
cfg.events["push_robot"].interval_range_s = (3.0, 8.0)
cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}
```

**Note:** Interval (3–8 s) with 3 s episodes means push fires in ~30–50% of episodes — enough exposure without constant disruption.

**Expected impact:** Moderate. The robot must learn to recover from angular momentum changes, not just lateral pushes. The post-recovery rewards (postorientation, postangvel) become more important.

---

### Phase 3 — Encoder Bias + Observation Delay Upgrade

**Rationale:** After the robot is stable under physical perturbation (Phase 2), add sensor noise. Encoder bias simulates per-joint calibration offsets on real T1 hardware. Observation delay increase (0–2 → 2–8 steps) simulates realistic sensor latency. These go together because both affect the *quality of state information* the policy receives — training with both simultaneously teaches the policy to be robust to temporal and spatial observation errors.

**Add:**
```python
# Encoder bias: per-joint zero-offset error at startup
cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.encoder_bias,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
    },
)

# Obs delay upgrade
if not play:
    for term_cfg in cfg.observations["actor"].terms.values():
        term_cfg.delay_min_lag = 2
        term_cfg.delay_max_lag = 8
        term_cfg.delay_per_env = True
```

**Expected impact:** Moderate–high. Encoder bias shifts effective joint positions by ±0.015 rad, which is ~0.86°. For leg joints this is small, but for arm joints (reaching for ball) it changes effective hand position. The delay upgrade makes ball tracking significantly harder — the policy must predict forward in time. Expect stopball success rate to drop initially.

---

### Phase 4 — PD Gain Randomization

**Rationale:** PD gain scaling is the most impactful single DR in the actuator model. It changes the effective stiffness and damping of every joint by ±20%, which means the robot's effective torque output for a given position error varies significantly. This must come after encoder bias (Phase 3) — if the robot hasn't learned to handle position offsets, PD gain uncertainty compounds the problem unpredictably.

**Add:**
```python
cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.pd_gains,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
    },
)
```

**Expected impact:** High. The robot must learn to succeed across a 2.25× range of effective stiffness (0.8×0.8=0.64 to 1.2×1.2=1.44). This is the key sim2real bridge for actuator model mismatch. Expect notable training destabilization — may need to run 3,000+ iterations to restabilize.

**Contingency:** If training destabilizes (eereach/stopball collapse and don't recover), reduce range to (0.85, 1.15) first and expand later.

---

### Phase 5 — Ball Perception DR (Vanish Window + In-Flight Perturbation)

**Rationale:** These simulate camera detection latency and ball trajectory uncertainty from kicks. They are added last because:
1. They make the task fundamentally harder (policy must predict ball position during vanish window)
2. The robot needs robust physical DR (Phases 1–4) before adding perceptual uncertainty — otherwise two things fail simultaneously
3. Both require code changes in `mdp/observations.py` and `mdp/commands.py`

**Ball vanish window (in `mdp/observations.py`):**
```python
# ball_pos_b returns zeros for first 3–40 steps of episode
# Simulates camera detection latency (60–800 ms at 50 Hz)
# ball_vel_b is NOT zeroed (matches G1 exactly)
vanish_steps = initial_vanish + random_vanish  # 3–10 + 0–30
```

**Ball in-flight perturbation (in `mdp/commands.py`, `_update_command`):**
```python
# Every 25 steps (0.5 s at 50 Hz), add ±0.5 m/s to ball linear velocity
# Simulates wind, spin, bounce wobble
# Angular velocity NOT perturbed (matches G1)
```

**Expected impact:** Very high. The policy must develop implicit ball trajectory prediction. Training time likely needs to increase significantly (consider max_iterations 40k → 60k for this phase). The stopball reward will initially drop substantially as the policy encounters vanish windows.

---

### Phase 6 — Optional Advanced DR (Post-Deployment Hardening)

These are not in the current master reference but would further improve sim2real transfer:

| DR | What | When to add |
|---|---|---|
| Body link mass | Randomize limb masses ±15% | After Phase 4 is stable |
| Ground friction | Vary floor contact friction | After Phase 2 is stable |
| Action execution delay | Add 1–3 step action lag | After Phase 3 is stable |
| Ball radius | Vary radius ±10% (0.10–0.12 m) | Phase 1 or Phase 5 |
| Robot spawn yaw | Wider yaw range at reset | Can add any phase |

---

## Summary Table

| Phase | DR Added | Risk | Blocking prerequisite |
|---|---|---|---|
| 0 | Non-DR fixes (push timing, noise, delay floor) | Minimal | None — do first |
| 1 | Ball mass, ball friction, ball spin | Low | Phase 0 |
| 2 | Push 6-DOF upgrade | Moderate | Phase 0 (push timing fixed) |
| 3 | Encoder bias + obs delay 2–8 | Moderate–High | Phase 2 (balance robustness) |
| 4 | PD gain scaling (0.8–1.2) | High | Phase 3 (obs quality DR) |
| 5 | Ball vanish + in-flight perturbation | Very high | Phase 4 (full actuator DR) |
| 6 | Optional: link mass, ground friction, action lag | Variable | Phase 4 stable |

---

*Haiku 4.5 second opinion appended below — compare for additional insights.*

---

## Haiku 4.5 Second Opinion

### NON-DR GAPS (Haiku)

**Gap H1: Base lin/ang velocity — missing noise**  
`base_lin_vel` and `base_ang_vel` replaced with direct state reads (lines 181–186) but have NO noise. Haiku suggests ±0.5 m/s for lin_vel, ±0.2 rad/s for ang_vel — these are the values from the base tracking config. *(Note: Sonnet suggested more conservative ±0.05 m/s — the 10× difference is worth checking against G1's original obs noise.)*

**Gap H2: Observation delay 0–2 steps too short**  
Real hardware has 50–100 ms latency; current max is 40 ms. Haiku suggests delay_max_lag=5 (100 ms) as a non-DR fix before adding further DR. Agrees with Sonnet's gap 6.

**Gap H3: RSI sampling_mode="start" limits curriculum**  
Haiku argues switching to `sampling_mode="uniform"` would give a light curriculum (spawn mid-dive for faster hand-position learning). Sonnet did not flag this — and the commit history shows "uniform" was explicitly reverted to "start" in commit d163bc8 because of ball-arrival desync. Haiku's suggestion has a caveat: only safe if episodes < 3s and ball travel time 0.4–1.0s. **Flagged as debatable — uniform was reverted for a reason.**

**Gap H4: Biased encoder noise verification**  
Check that `joint_pos` has `biased=True` (constant per-episode offset) vs `biased=False` (per-step). Biased is more realistic. This is an inheritance/verification issue.

**Gap H5: PD gains + link mass disabled (critical)**  
Both disabled per DIVERGENCE doc. Haiku classifies these as Phase 1-critical gaps rather than Phase 4 (Sonnet's position). Key disagreement — see comparison below.

**Gap H6: Stopball threshold hardcoded at 2.0 m/s**  
For slow balls (~1 m/s approach) the threshold requires a larger proportional change. Not urgent but flagged as a tuning risk if training plateaus.

---

### PHASED DR PLAN (Haiku)

**Phase 1 — Foundation Stabilization:**  
PD gains (0.8–1.2), link mass (0.8–1.2), encoder bias (±0.015), delay expansion (0–100ms), base velocity noise, biased encoder verification, optional RSI uniform.  
*Haiku success criteria:* stopball > 0.5, eereach > 2.0, episode length > 80%, terminate < 10%.

**Phase 2 — Mid-Stage Robustness:**  
Push robot 6-DOF upgrade + interval (3–8s), foot friction expansion (0.1–2.0), ball mass (0.8–1.2), ball friction (0.2–0.8), extend delay to 8 steps.  
*More aggressive than Sonnet Phase 1+2 ordering.*

**Phase 3 — Late-Stage Realism:**  
Ball in-flight perturbation ±0.3–0.5 m/s every 25 steps, ball vanish window (3–40 steps), joint noise increase, actuator time delay (1–2 steps lag), delay to 8 steps.

---

### KEY DISAGREEMENTS — Sonnet vs Haiku

| Topic | Sonnet | Haiku | Verdict |
|---|---|---|---|
| **PD gains phase** | Phase 4 (last — most disruptive) | Phase 1 (first — most critical) | **Real debate.** Haiku's argument (it's the #1 real-world gap) is compelling, but Sonnet's argument (train stability first) is safer. Suggest Phase 2 as compromise. |
| **Base velocity noise magnitude** | ±0.05 m/s (conservative) | ±0.5 m/s (aggressive, matching base config) | Check what G1's original legged_gym used. Haiku's value is likely correct. |
| **RSI sampling_mode** | Keep "start" (was reverted for a reason) | Switch back to "uniform" for curriculum | Keep "start" unless investigation shows the d163bc8 desync issue is now fixed. |
| **Link mass** | Phase 6 optional | Phase 1 critical | Haiku more aggressive. Include in Phase 2 as compromise. |
| **Phase count** | 5 phases | 3 phases | Sonnet more granular; Haiku batches more together. Both valid — depends on how patient you want to be between training runs. |

---

### COMBINED RECOMMENDATION

Based on both analyses, the suggested ordering is:

**Phase 0 (Sonnet-led, both agree):** Non-DR fixes — push timing, missing noise, delay floor, foot geom verification.

**Phase 1 (compromise):** Ball DR (mass, friction, spin) + encoder bias + base velocity noise. Low disruption, addresses both ball and sensor gaps.

**Phase 2 (compromise):** Push 6-DOF upgrade + PD gains (moderate range 0.85–1.15 first) + obs delay 2–5 steps. Addresses real-world actuator and balance uncertainty.

**Phase 3:** PD gains full range (0.8–1.2) + link mass + obs delay 2–8 steps. Full actuator DR.

**Phase 4:** Ball vanish window + in-flight perturbation. Perceptual DR, hardest for the policy.

**Phase 5 (optional):** Actuator time delay, noise scaling, further extension.
