# Critical Bug Fix Complete — my_mjlab_project Training

**Date:** 2026-05-02  
**Status:** ✅ FIXED AND VERIFIED (Smoke tested, ready for full training)  
**Next Action:** Reboot laptop, then run full training on GPU

---

## Executive Summary

**Problem:** Every training episode terminated on step 1. All 1020 environments would crash immediately with `error_anchor_pos = 1.25 m` (way over the 0.25 m threshold).

**Root Cause:** Off-by-one body index bug in `convert.py` — the motion data was storing MuJoCo's worldbody (always at origin) at index 0, causing RSI (Reference State Initialization) to teleport the robot to the floor instead of the reference height.

**Solution:** Skip the worldbody when saving motion data. Changed 3 lines in `convert.py` and regenerated all 6 NPZ motion files.

**Result:** Smoke test confirms training now works — episodes last 7+ steps instead of terminating immediately.

---

## What Was Fixed

### Bug #1 (CRITICAL): Body Index Off-by-One in `convert.py`

**File:** `src/my_mjlab_project/motions/convert.py` lines 154–166

**Before (BUGGY):**
```python
num_bodies = model.nbody  # 31 (includes MuJoCo worldbody at index 0)
body_pos_w[t] = data.xpos.copy()        # Saves worldbody = (0,0,0) at NPZ[0]
body_quat_wxyz[t] = data.xquat.copy()
```

**After (FIXED):**
```python
num_bodies = model.nbody - 1  # 30 (skip worldbody)
body_pos_w[t] = data.xpos[1:].copy()        # Pelvis now at NPZ[0]
body_quat_wxyz[t] = data.xquat[1:].copy()
```

**Why it matters:**
- MuJoCo stores 31 bodies where body 0 = worldbody (always at 0,0,0)
- mjlab's `entity.body_names` is 0-indexed on robot bodies only (skips worldbody)
- MotionLoader reads `NPZ[0]` expecting robot pelvis, but was getting worldbody (0,0,0)
- RSI teleported robot to floor (z≈0) instead of reference height (z≈1.18m)
- Torso error: 1.18m >> 0.25m threshold → immediate termination

**Action taken:** Regenerated all 6 NPZ files with this fix:
- `lefthand.npz` (123 frames, 30 bodies)
- `righthand.npz` (246 frames, 30 bodies)
- `leftjump.npz` (254 frames, 30 bodies)
- `rightjump.npz` (200 frames, 30 bodies)
- `leftstep.npz` (194 frames, 30 bodies)
- `rightstep.npz` (197 frames, 30 bodies)

### Bug #2 (MINOR): Ball Z-Position Missing env_origins Offset

**File:** `src/my_mjlab_project/mdp/commands.py` line 239

**Before:**
```python
ball_pos_w[:, 2] = z_start  # Missing env z offset
```

**After:**
```python
ball_pos_w[:, 2] = origins[:, 2] + z_start  # Correct
```

**Impact:** Benign on flat terrain (z=0) but latent bug for non-zero terrain heights.

---

## Verification Results

### Smoke Test: 2 envs, 3 iterations, CPU

| Metric | Before Fix | After Fix | Status |
|--------|-----------|-----------|--------|
| `mean_episode_length` | 1.00 | 7.6 | ✅ FIXED |
| `error_anchor_pos` | 1.25 m | 0.14 m | ✅ FIXED |
| `anchor_pos terminations` | 100% (2/2) | 0% (0/2) | ✅ FIXED |
| Training iterations | Crashes | 3 complete | ✅ FIXED |
| W&B logging | N/A | Running | ✅ OK |

**Logs:** `wandb/run-20260502_212536-den9mpz2/`

---

## How to Resume Training After Reboot

### Step 1: Free GPU Memory (if needed)
```bash
nvidia-smi  # Check for stuck processes
kill -9 <PID>  # If any mjlab processes are stuck
```

### Step 2: Start Full Training on GPU
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'
```

**Expected Performance:**
- ~11k steps/second on RTX 3070 Laptop (1020 envs)
- 200k iterations default → ~18 hours full training
- W&B logging to: `https://wandb.ai/i-p-b-bouwmeester-eindhoven-university-of-technology/mjlab`

### Step 3: Monitor Training (Real-Time)
Watch W&B metrics as training runs:
- `mean_episode_length` should grow from ~7 → 20 → 50+
- `mean_reward` should improve (become less negative)
- `error_anchor_pos` should stay << 0.25m (no terminations)
- Tracking rewards should increase
- Goalkeeper rewards (eereach, catch_success) should improve from ~0

### Step 4: Visualize After Training (Optional)
After a few thousand iterations (1-2 hours), visualize a checkpoint:
```bash
uv run python -m mjlab.scripts.play goalkeeper \
  --checkpoint logs/rsl_rl/g1_goalkeeper/<run_id>/model_<iter>.pt \
  --num-envs 1
```

---

## If GPU Still Has Issues After Reboot

Use CPU instead (slower but reliable):
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[]'
```

**CPU Performance:**
- ~6-7 steps/second
- 200k iterations → ~8 days
- Still valid for training, just slower

---

## Files Changed

1. **src/my_mjlab_project/motions/convert.py**
   - Line 1-11: Updated docstring (31 → 30 bodies)
   - Line 152: `num_bodies = model.nbody - 1`
   - Line 164: `body_pos_w[t] = data.xpos[1:].copy()`
   - Line 165: `body_quat_wxyz[t] = data.xquat[1:].copy()`

2. **src/my_mjlab_project/mdp/commands.py**
   - Line 239: `ball_pos_w[:, 2] = origins[:, 2] + z_start`

3. **All 6 NPZ Motion Files** (regenerated)
   - Location: `src/my_mjlab_project/motions/data/`
   - Format: 30 bodies each (was 31)

---

## Architecture Assessment

### What Codex Got Right ✅
- All 6 motion files wired correctly via `MultiMotionCommandCfg`
- Ball physics with free joint and collision detection
- Joint mapping (21 Isaac Gym → 29 MuJoCo DOF) correct
- Observation space: 172-dim actor + 298-dim critic (comprehensive)
- Reward structure: 18 terms (tracking + goalkeeper + regularization)
- Domain randomization events (push, CoM, encoder bias, friction)
- MuJoCo contact buffers adequate (nconmax=100, njmax=500)

### What Was Wrong ❌
- Body index off-by-one (NOW FIXED)
- Ball z-offset missing (NOW FIXED)
- PORT_STATUS.md claimed "COMPLETE & VERIFIED" but training crashed (was inaccurate)

---

## Expected Training Trajectory

### Early iterations (0–1k)
- Episodes stabilize at 10–20 steps (RSI working correctly now)
- Rewards negative (policy untrained)
- Motion tracking improving

### Mid training (1k–50k)
- Episode length grows to 50+ steps
- Mean reward improves
- Motion tracking error decreases
- Goalkeeper rewards start appearing

### Late training (50k–200k)
- Episodes reach 200+ steps (approaching max)
- Stable performance with good tracking
- Policy learning to catch balls

---

## Key Metrics to Monitor in W&B

```
✓ Mean episode length > 5 (was 1.0 before fix)
✓ Mean reward improving (less negative)
✓ error_anchor_pos < 0.25m (no terminations)
✓ Episode_Termination/anchor_pos = 0
✓ Episode_Termination/ee_body_pos ~ 0-0.5 (some end-effector errors OK)
✓ Tracking rewards (motion_*) > 0
✓ Goalkeeper rewards (eereach, catch_success) increasing from ~0
```

---

## Commands Quick Reference

```bash
# Full GPU training (recommended after reboot)
cd ~/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'

# CPU fallback if GPU has issues
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[]'

# Visualize a checkpoint
uv run python -m mjlab.scripts.play goalkeeper \
  --checkpoint logs/rsl_rl/g1_goalkeeper/<run_id>/model_<iter>.pt \
  --num-envs 1

# Check GPU memory
nvidia-smi

# Kill stuck processes
pkill -9 python
```

---

## Timeline

- **2026-05-02 ~20:00** - Critical bug discovered during code review
- **2026-05-02 ~21:00** - Bug fixed, NPZ files regenerated
- **2026-05-02 ~21:25** - Smoke test verified: episodes now 7+ steps (was 1)
- **2026-05-02 ~21:30** - Full training attempted (GPU memory issue, minor)
- **NOW** - Ready for full training after reboot

---

## Summary

✅ **One critical bug fixed** (body index off-by-one)  
✅ **Verified with smoke test** (7.6 step episodes vs 1.0 before)  
✅ **All 6 motion files regenerated**  
✅ **Ready for full training**

**Next step:** Reboot laptop, run training command, monitor W&B for 18 hours while it completes 200k iterations.
