# Session Context — Booster T1 Goalkeeper RL

**Date:** 2026-05-09  
**Goal:** Adapt the G1 Humanoid-Goalkeeper pipeline to Booster T1 in mjlab.

---

## Current Status

Training has not yet been confirmed working on GPU. All code bugs are fixed. The only remaining blocker was an NVML/GPU error that is fixed by using `CUDA_VISIBLE_DEVICES=0`.

---

## Bugs Fixed This Session

### Bug 1 — Robot spawns underground (critical)
**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`  
**Property:** `body_pos_w`  
**Problem:** Code subtracted `0.39m` from z. This was copied from G1 where it's needed (G1 motion pelvis z ≈ 1.185m, standing height ≈ 0.795m → delta = 0.39m). T1 motion data Trunk z ≈ 0.69m is already correct after `convert_booster.py`, so subtracting 0.39 put the robot 0.39m underground.  
**Fix:** Removed `pos[:, :, 2] -= 0.39`. Now `body_pos_w` just adds env_origins with no z correction.

### Bug 2 — Overlay ghost wrong orientation
**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/commands.py`  
**Properties:** `body_quat_w`, `anchor_quat_w`  
**Problem:** Both properties applied a -90° yaw rotation. This was copied from G1 which has no yaw in its motion converter. But T1's `convert_booster.py` already applies +90° yaw to all quaternions. The -90° in commands.py undid that, misaligning the overlay.  
**Fix:** Removed -90° yaw rotation from both properties. They now just return `_gather()` / `_gather_anchor()` directly.

### Bug 3 — GPU IndexError during training
**Error:** `IndexError: list index out of range` in `mjlab/utils/gpu.py:70` inside `select_gpus()`  
**Cause:** `Can't initialize NVML` warning → `torch.cuda.device_count()` returns 0 → `available_gpus = []` → indexing `[0]` fails. NVML init was failing before PC restart.  
**Fix:** Prefix all commands with `CUDA_VISIBLE_DEVICES=0`. This bypasses the NVML path — mjlab reads the env var directly and sets `available_gpus = [0]`.

---

## Commands (also in commands.txt)

```bash
# Train (64 envs, smoke test — run this first to verify GPU works)
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1 && CUDA_VISIBLE_DEVICES=0 uv run python -m mjlab.scripts.train goalkeeper_booster_t1 --env.scene.num-envs 64

# Train (1020 envs, full run)
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1 && CUDA_VISIBLE_DEVICES=0 uv run python -m mjlab.scripts.train goalkeeper_booster_t1 --env.scene.num-envs 1020

# Play (with ghost overlay — shows reference motion)
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1 && CUDA_VISIBLE_DEVICES=0 uv run python -m mjlab.scripts.play goalkeeper_booster_t1_withoverlay --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-06_10-29-59/model_0.pt --motion-file src/my_mjlab_project_booster_t1/motions/data/lefthand_t1.npz

# Play (no overlay)
cd /home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1 && CUDA_VISIBLE_DEVICES=0 uv run python -m mjlab.scripts.play goalkeeper_booster_t1 --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-06_10-29-59/model_0.pt --motion-file src/my_mjlab_project_booster_t1/motions/data/lefthand_t1.npz
```

**Note:** After reboot, NVML should initialize correctly and the GPU error should be gone. Still keep `CUDA_VISIBLE_DEVICES=0` to be safe.

---

## Key File Summary

| File | Purpose | Modified? |
|------|---------|-----------|
| `mdp/commands.py` | MultiMotionCommand — controls robot spawn pos/ori and ghost overlay | YES (3 bugs fixed) |
| `robots/t1_constants.py` | T1 actuators, standing keyframe (pos=(0,0,0.67), rot=+90° yaw) | No |
| `motions/convert_booster.py` | Converts PKL → NPZ, applies +90° yaw + z=+0.079 offset | No |
| `motions/data/lefthand_t1.npz` | T1 motion reference data (123 frames, 24 bodies) | No |
| `tasks/goalkeeper_env_cfg.py` | Environment config, play/withoverlay task defs | No |
| `assets/booster_t1/T1_serial_clean.xml` | T1 MJCF model (nq=30, nbody=25) | No |

---

## T1 Motion Data Facts

- Source: `/home/isaak/BEPImitationlearning/Boosterversion/lefthand_booster_t1.pkl`
- After conversion: 123 frames, 24 bodies (excludes world body)
- Trunk (body 0) z: min=0.66m, max=0.74m, mean=0.71m
- Frame 0: Trunk z=0.690m, left foot z≈0.058m, right foot z≈0.042m
- Foot bottom min z ≈ 0.011–0.021m above ground (correctly near floor)
- +90° yaw already baked in by `convert_booster.py`

---

## Architecture Notes

- **G1 reference project:** `/home/isaak/BEPImitationlearning/my_mjlab_project_lefthand`
- **T1 project:** `/home/isaak/BEPImitationlearning/my_mjlab_project_booster_t1`
- G1 uses `unitree_g1_flat_tracking_env_cfg`; T1 uses `make_tracking_env_cfg()` (generic)
- G1 body_names: `pelvis, left_hip_pitch_link, ...` (12 bodies)
- T1 body_names: `Trunk, Hip_Roll_Left, Shank_Left, Ankle_Cross_Left, Hip_Roll_Right, Shank_Right, Ankle_Cross_Right, Waist, AL2, AL3, left_hand_link, AR2, AR3, right_hand_link` (14 bodies)
- T1 has 23 DOFs vs G1's 29 DOFs

---

## Next Steps After Reboot

1. Run 64-env smoke test train — should see reward/loss logs without errors
2. If it works, Ctrl+C and run 1020-env full train
3. Once a checkpoint exists, test play to verify robot no longer spawns underground
4. If robot still looks wrong in play, check `tasks/goalkeeper_env_cfg.py` for body index mismatches
