# Goalkeeper RL Improvement Plan — Booster T1 (mjlab)

**Date:** 2026-05-20  
**Status:** Non-DR fixes applied. DR phases not yet started.

---

## Non-DR Fixes (Applied 2026-05-20)

These were gaps versus the G1 upstream or the hardware-verified KaydenKnapik setup.

| Fix | File | What changed |
|-----|------|-------------|
| IMU velocity noise | `goalkeeper_env_cfg.py` | Added `Unoise(±0.1)` to `base_lin_vel`, `Unoise(±0.2)` to `base_ang_vel` (matched G1) |
| Per-joint torque limits | `rewards.py` `_T1_EFFORT_MAP` | Replaced universal 50 Nm cap with KaydenKnapik values per joint |
| Effort limits (actuators) | `robots/t1_constants.py` | Updated all effort_limit to KaydenKnapik hardware-verified values |
| XML force clamps removed | `assets/booster_t1/T1_serial_clean.xml` | Removed all `actuatorfrcrange` — Python effort_limit is now the only hard clamp |
| Joint velocity noise | `goalkeeper_env_cfg.py` | Increased from ±0.5 (base) to ±1.5 (KaydenKnapik value) |
| Actuator command delay | `robots/t1_constants.py` | Added `delay_min_lag=2, delay_max_lag=8` per actuator (KaydenKnapik) |

### KaydenKnapik Effort Limits (hardware-verified, deployed on real T1)

| Joint group | Old (our XML) | New (KaydenKnapik) |
|---|---|---|
| Arms (Shoulder/Elbow 4×2) | 18 Nm | **36 Nm** |
| Waist | 30 Nm | **40 Nm** |
| Hip Pitch | 45 Nm | **55 Nm** |
| Hip Roll / Hip Yaw | 30 Nm | **40 Nm** |
| Knee Pitch | 60 Nm | **65 Nm** |
| Ankle Pitch | 20 Nm | **50 Nm** |
| Ankle Roll | 15 Nm | **50 Nm** |

### Note: joint_pos noise
The base `tracking_env_cfg.py` already had `joint_pos` noise at ±0.01, which matches KaydenKnapik. No change needed.

---

## Domain Randomization Phases

### Why these phases?

Phase 1 adds exactly what KaydenKnapik used in their successfully-deployed T1 velocity policy. These are the conservative, hardware-tested baseline DR events. Phase 2 adds ball-specific DR that is goalkeeper-unique. Phase 3 adds heavier perturbations that require verification before committing.

---

### Phase 1 — KaydenKnapik Baseline DR (copy their exact setup)

**Goal:** Match the DR that was deployed on real T1 hardware. These are conservative values validated on the physical robot.

**Files to change:** `goalkeeper_env_cfg.py`

#### 1a. Push Robot (6-DOF, interval 1–3 s)

KaydenKnapik pushes more aggressively than our current setup. Add roll/pitch/yaw disturbances and shorten the interval.

```python
"push_robot": EventTermCfg(
    func=mjlab_dr.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
        "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.4, 0.4),
            "roll":  (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw":   (-0.78, 0.78),
        },
    },
),
```

**Implementation:** Find the `push_robot` event inherited from `make_tracking_env_cfg()`, override it with the above in `goalkeeper_env_cfg()`.

#### 1b. Foot Friction (startup, shared random)

Randomize friction at episode start so the policy doesn't assume a fixed coefficient.

```python
cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.geom_friction,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            geom_names=(r"^(left|right)_foot_[12]$",),
        ),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
    },
)
```

**Note on geom names:** KaydenKnapik uses `foot{1-4}_collision` (4 per foot). Our XML uses `left_foot_1`, `left_foot_2`, `right_foot_1`, `right_foot_2`. Use the regex above to target our naming.

#### 1c. Encoder Bias (startup)

Simulate joint encoder offsets — the single most impactful DR for sim2real on real T1:

```python
cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.encoder_bias,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
    },
)
```

#### 1d. Base COM Offset (startup)

Randomize center-of-mass of the trunk to simulate mass distribution uncertainty:

```python
cfg.events["base_com"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.body_com_offset,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
        "operation": "add",
        "ranges": {
            0: (-0.025, 0.025),
            1: (-0.025, 0.025),
            2: (-0.03,  0.03),
        },
    },
)
```

**How to verify Phase 1 works:**
- Training stays stable (no NaN, no early terminations)
- eereach reward still increases over training
- push_robot fires every ~2s average — robot learns to recover

---

### Phase 2 — Ball Domain Randomization

**Goal:** Make the policy robust to ball physics uncertainty. Goalkeeper-specific, no equivalent in KaydenKnapik's walking task.

#### 2a. Ball Mass Randomization

```python
cfg.events["ball_mass"] = EventTermCfg(
    mode="reset",
    func=mjlab_dr.body_mass_offset,
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "operation": "scale",
        "ranges": (0.7, 1.3),
    },
)
```

**Implementation note:** Check exact mjlab DR API name (`body_mass_offset`, `randomize_mass`, etc.) before adding. If scale is not supported, use absolute range 0.3–0.55 kg.

#### 2b. Ball Friction Randomization

```python
cfg.events["ball_friction"] = EventTermCfg(
    mode="reset",
    func=mjlab_dr.geom_friction,
    params={
        "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
        "operation": "scale",
        "ranges": (0.5, 2.0),
        "shared_random": True,
    },
)
```

**How to verify Phase 2 works:**
- `stopball` and `catch_success` rewards remain non-zero
- Training does not destabilize relative to Phase 1

---

### Phase 3 — Observation Delay Upgrade (Optional)

**Goal:** Add obs-level latency to simulate inference latency on the real robot's computer. KaydenKnapik handles delay at the actuator level only; actuator delay is already added (Phase 0 / non-DR fix). Obs delay is an additional measure.

```python
cfg.observations["actor"].delay_lag = (0, 3)
```

**When to add:** Only after Phase 1+2 are stable. Actuator delay (2–8 steps) + obs delay (0–3 steps) combined is heavy.

---

### Phase 4 (Optional / Advanced) — PD Gain Randomization

KaydenKnapik did NOT include this. Add only after Phase 1–3 are fully stable.

```python
cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=mjlab_dr.pd_gains,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "stiffness_range": (0.85, 1.15),
        "damping_range":   (0.85, 1.15),
    },
)
```

---

## DR Not Planned

- **Link mass randomization (other than Trunk COM):** KaydenKnapik did not use it.
- **Motor friction / damping:** Not in KaydenKnapik setup.
- **Terrain randomization:** Flat terrain by design.

---

## Verification Summary (Haiku subagent, 2026-05-20)

All KaydenKnapik values independently confirmed:

| Parameter | Value | Verified |
|---|---|---|
| Push interval | 1.0–3.0 s | ✓ |
| Push 6-DOF | x/y ±0.5, z ±0.4, roll/pitch ±0.52, yaw ±0.78 | ✓ |
| Foot friction | abs, (0.3, 1.2), shared_random=True | ✓ |
| Encoder bias | ±0.015 | ✓ |
| Base COM | x/y ±0.025, z ±0.03, add, Trunk body | ✓ |
| joint_pos noise | ±0.01 | ✓ |
| joint_vel noise | ±1.5 | ✓ |
| Actuator delay | DELAY_MIN=2, DELAY_MAX=8 per actuator | ✓ |
| PD gain DR | NOT used | ✓ |
| Link mass DR | NOT used | ✓ |
| Arm effort limit | 36 Nm | ✓ |
| Waist/HipRoll/Yaw effort | 40 Nm | ✓ |
| HipPitch effort | 55 Nm | ✓ |
| Knee effort | 65 Nm | ✓ |
| Ankle effort | 50 Nm | ✓ |
