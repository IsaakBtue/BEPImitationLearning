# Divergence from Upstream (Humanoid-Goalkeeper)

This document tracks substantive changes where the Booster T1 adaptation deviates from the G1 tracking task pipeline.

## 2026-05-03 — Play config: disable motion files but keep command for ball reset

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** In `goalkeeper_play_env_cfg()`:
- Set `motion_files = ()` and `motion_file = ""` (empty: no motion files loaded)
- Removed 6 motion-dependent reward terms that require motion command attributes
- Removed 3 motion-dependent termination terms

**Why:** The play environment must:
1. Allow autonomous inference **without motion files** (no --motion-file argument needed)
2. Still call `_reset_ball()` which randomizes ball start position/velocity per env
3. Disable reward/termination terms that access command attributes (`anchor_pos_w`, etc.)

By keeping the motion command with empty motion files, `_reset_ball()` executes but never loads motion data.
This prevents both the "ball at feet" problem and AttributeError crashes.

**Impact:** 
- Ball spawns with proper random trajectory (3-5m away, random y/z, timed arc)
- No motion file required at play time
- Play mode runs stably without crashes

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
