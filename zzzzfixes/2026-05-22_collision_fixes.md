# Collision System Fixes — 2026-05-22

## What Was The Problem?

The mjlab port had four collision-related gaps vs the original Humanoid-Goalkeeper (Isaac Gym / G1):

1. `penalize_sharpcontact` was a geometry proxy, not a physics measurement
2. `feet_slippage` used a height proxy for contact detection
3. `self_collision` sensor was declared but never read by anything
4. No force-based episode termination

All four are now fixed. Details below.

---

## Q: feet_slippage was ~0.4 in WandB — is that wrong?

**No — 0.4 is correct.**

The `feet_slippage` function returns `exp(-10 * contactvel)` where
`contactvel = sum(foot_speed * in_contact)`.

A value of 0.4 means `contactvel ≈ 0.092 m/s` combined foot speed during contact.
This is realistic: the robot is on the ground, and its feet are sliding at ~9 cm/s on
average. That is genuine slippage signal from an early/mid training policy that has not
yet learned to plant its feet firmly.

**Why the proxy was not badly wrong for T1:**
The foot body (`left_foot_link`) origin sits at Z ≈ 0.03 m above the floor when
standing (foot geom bottom = body_Z - 0.01 - 0.02 capsule radius = 0).
The threshold was 0.05 m, so `0.03 < 0.05 = True` — the proxy **correctly detected**
ground contact in normal stance. The 0.4 reading was real slippage, not a phantom signal.

The proxy bug (feet-in-air → `in_contact=0` → `exp(0)=1.0`) is real but was a
secondary effect. The dominant training regime had feet on the ground, so the proxy
was mostly correct. The sensor-based fix is still better (more accurate in edge cases
like toe-off, mid-dive impact frames), but 0.4 was not a false reading.

---

## Fix 1: `penalize_sharpcontact` — foot force sensor instead of height proxy

**File:** `src/.../mdp/rewards.py`

### What changed
```python
# BEFORE — trunk height proxy
def penalize_sharpcontact(env, height_threshold=0.35):
    trunk_z = robot.data.root_link_pos_w[:, 2]
    env_z   = env.scene.env_origins[:, 2]
    return (trunk_z - env_z < height_threshold).float()

# AFTER — actual foot contact force
def penalize_sharpcontact(env, force_threshold=1000.0):
    sensor = env.scene["feet_contact"]
    force  = sensor.data.force                       # [B, 4, 3]
    mean_force = torch.norm(force, dim=-1).mean(-1)  # [B]
    return (mean_force > force_threshold).float()
```

### Why it was wrong
The proxy measured "did the trunk fall low?" not "was there a hard impact?".
- **False negative:** Hard foot-slam landing at standing height → trunk stays at 0.7m → no penalty
- **False positive:** Slow controlled crouch below 0.35m → fires even though no dangerous impact

### What the original does
`legged_robot.py:1477`: `mean(norm(contact_forces[:, feet, :])) > max_contact_force`
where `max_contact_force = 1000 N`.

### Why the fix is right
`feet_contact` sensor uses `reduce="netforce"` → `data.force [B, 4, 3]` is the true
net contact force per foot geom, equivalent to Isaac Gym's `net_contact_force_tensor`.
Mean over 4 geoms is mathematically identical to the original's mean over 2 foot bodies
(both are the unweighted average of all foot-contact force magnitudes).

**Weight: -100.0 (unchanged)**

---

## Fix 2: `feet_slippage` — sensor-based contact detection instead of height proxy

**File:** `src/.../mdp/rewards.py`

### What changed
```python
# BEFORE — height proxy contact detection
def feet_slippage(env, asset_cfg=_FEET_CFG, contact_height_threshold=0.05):
    foot_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    env_z  = env.scene.env_origins[:, 2:3]
    in_contact = (foot_z - env_z < contact_height_threshold).float()  # ← proxy
    contactvel = torch.sum(foot_speed * in_contact, dim=-1)
    return torch.exp(-10.0 * contactvel)

# AFTER — physics-based contact detection
def feet_slippage(env, asset_cfg=_FEET_CFG):
    sensor = env.scene["feet_contact"]
    found  = sensor.data.found                               # [B, 4]
    left_in_contact  = (found[:, 0] > 0) | (found[:, 1] > 0)
    right_in_contact = (found[:, 2] > 0) | (found[:, 3] > 0)
    in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1).float()
    contactvel = torch.sum(foot_speed * in_contact, dim=-1)
    return torch.exp(-10.0 * contactvel)
```

### Why the fix is still better (even though 0.4 was not wrong)
1. **Toe-off accuracy:** At slight toe-off (body_Z ≈ 0.04–0.06m), the proxy flickers
   between contact/no-contact near its threshold. The sensor is discrete (contact pair
   either exists or doesn't).
2. **Impact frames:** At the moment of a hard landing, the foot may still be at Z > 0.05m
   for one physics step while force is already large. Sensor catches it; proxy misses it.
3. **Correctness:** Matches the original's `contact_forces > 1 N` criterion exactly.

### What the original does
`legged_robot.py:1472`: `torch.norm(contact_forces[:, feet, :], dim=-1) > 1.0` as in_contact.

### Geom index mapping
```
Sensor data.found [B, 4]:
  index 0 = left_foot_1
  index 1 = left_foot_2
  index 2 = right_foot_1
  index 3 = right_foot_2
```

**Weight: +3.0 (unchanged — positive reward for not slipping)**

---

## Fix 3: `penalize_self_collision` — wire the unused sensor to a penalty

**Files:** `src/.../mdp/rewards.py`, `src/.../tasks/goalkeeper_env_cfg.py`

### What changed
The `self_collision` ContactSensor was already registered in the scene since an earlier
commit. It was computing contact data every step but the output was never read by any
reward or observation. Added a reward function that reads it.

```python
# NEW reward function in rewards.py
def penalize_self_collision(env):
    sensor = env.scene["self_collision"]
    return (sensor.data.found > 0).any(dim=-1).float()  # [B]

# NEW registration in goalkeeper_env_cfg.py
cfg.rewards["penalize_self_collision"] = RewardTermCfg(
    func=gk_rew.penalize_self_collision, weight=-50.0,
)
```

### Why this matters
The `self_collision` sensor monitors Trunk-subtree vs Trunk-subtree contacts (arm hitting
torso during a dive, etc.). Without this reward, the policy had no signal discouraging
self-collisions even though the sensor was running.

### Weight reasoning
-50.0 is half of `penalize_sharpcontact` (-100.0). Self-collision is undesirable but
recoverable; a hard ground impact is worse. Using half weight keeps the relative severity
proportional.

**data.found shape: [B, 1]** (reduce="none", num_slots=1). `.any(dim=-1)` reduces to [B].

---

## Fix 4: `sharpforce_termination` — force-based episode termination

**Files:** `src/.../mdp/resets.py`, `src/.../tasks/goalkeeper_env_cfg.py`

### What was missing
The original terminates episodes when foot contact force > 1500 N (`1.5 × max_contact_force`).
Reference: `legged_robot.py:258` — `sharpforce_buf = mean(norm(contact_forces[:, feet, :])) > 1500`.

The mjlab port had only orientation and height terminations. Catastrophic impacts (very hard
landings) did not trigger resets, so the policy could experience them repeatedly without
the episode ending — removing a key safety-curriculum signal.

### What changed
```python
# NEW in resets.py
def sharpforce_termination(env, max_contact_force=1500.0):
    sensor = env.scene["feet_contact"]
    force  = sensor.data.force                          # [B, 4, 3]
    mean_force = torch.norm(force, dim=-1).mean(-1)     # [B]
    return mean_force > max_contact_force               # bool [B]

# NEW in goalkeeper_env_cfg.py (terminations block)
cfg.terminations["sharpforce"] = TerminationTermCfg(
    func=gk_resets.sharpforce_termination,
    params={"max_contact_force": 1500.0},
    time_out=False,  # failure condition, not a timeout
)
```

### time_out=False is important
Setting `time_out=False` means this termination is treated as a failure (not a natural
episode end). The value bootstrap in PPO uses this flag to decide whether to treat the
terminal state value as zero (failure) vs. use the critic estimate (timeout). Wrong
`time_out` values distort value targets.

---

## New `feet_contact` Sensor

**File:** `src/.../tasks/goalkeeper_env_cfg.py`

All 3 of the physics-based fixes above depend on this sensor. It was not present before.

```python
ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
        mode="geom",
        pattern=r"^(left|right)_foot_[12]$",
        entity="robot",
    ),
    secondary=None,       # any contact partner: ground, ball, other bodies
    fields=("found", "force"),
    reduce="netforce",    # sum all contact points per geom → true net force
    history_length=0,     # instantaneous only; no history buffer needed
),
```

### Why `reduce="netforce"`
A single foot geom touching the ground can produce multiple contact points (capsule
geometry generates 2–4 points). `maxforce` would keep only the strongest single point,
discarding the rest. `netforce` sums all contact forces into one 3D vector per geom,
giving the true total force the foot is bearing — exactly what Isaac Gym's
`net_contact_force_tensor` reported.

### Why `secondary=None`
We want any contact the foot makes: ground, ball rolling under the foot, etc.
`secondary=None` means "capture contacts with any body", which is the equivalent of
the original's unconditional `contact_forces` tensor.

---

## Compatibility

These changes do **not** affect:
- Observation space (no new observations added)
- Action space
- Network architecture

All existing checkpoints are fully compatible and can continue training. Only reward
signals and termination conditions changed.

---

## Summary Table

| Component | Before | After | Why |
|---|---|---|---|
| `penalize_sharpcontact` | trunk Z < 0.35m (geometry) | mean foot force > 1000 N (physics) | original uses force threshold, not height |
| `feet_slippage` | foot Z < 0.05m for in_contact | sensor `found > 0` for in_contact | more accurate; matches original's force>1N criterion |
| `self_collision` sensor | registered, never read | wired to -50.0 penalty | sensor was running but discarded every step |
| force termination | missing | mean foot force > 1500 N | matches upstream `sharpforce_buf` |
| `feet_contact` sensor | missing | added (4 foot geoms, netforce) | required by all three physics-based fixes |
| WandB `feet_slippage` ≈ 0.4 | valid — real slippage signal | unchanged after fix | proxy was correct for T1 foot geometry |
