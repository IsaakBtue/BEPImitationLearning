
---

## 2026-06-30 — Ghost overlay spawns in ground for DoubleStep/TripleStep motions

**What changed:** Added `body_pos_w` property override in `GhostMotionCommand` (`mdp/commands.py`) that adds +0.030 m to the Z coordinate of all ghost body positions.

**Why it was wrong:** `pkl_to_npz` stores NPZ `body_pos_w` with the foot capsule contact surface at Z = -0.030 m (`_CONTACT_Z_TARGET = -0.030`). The RSI system compensates at runtime by adding `_FOOT_CONTACT_BELOW_BODY = +0.030` to the root Z in `MotionResetManager._write_rsi_state()`. The ghost overlay (`GhostMotionCommand`) replays positions directly from the NPZ via the inherited `MotionCommand.body_pos_w` property without applying this offset, so the ghost robot spawned 0.030 m into the floor.

**Correct value:** Ghost `body_pos_w` = NPZ positions + env_origins + (0, 0, +0.030). Matches what RSI writes to sim.

**Evidence:** All 4 overlay clips (LeftDoubleStep, RightDoubleStep, LeftTripleStep, RightTripleStep) visually embedded in the floor.

---

## 2026-06-30 — Ghost overlay in ground (v2): baked Z offset into NPZ files

**What changed:**
- Added +0.030 m to `body_pos_w[:, :, 2]` in all 14 NPZ motion files in `src/simple_goalkeeper/motions/data/`
- Removed runtime `_FOOT_CONTACT_BELOW_BODY = 0.030` addition from `_write_rsi_state` in `events.py`
- Changed `_CONTACT_Z_TARGET` in `pkl_to_npz.py` from `-0.030` to `0.0` so future conversions output correct Z directly
- Reverted ghost command override in `commands.py` (previous fix was wrong approach)

**Why it was wrong:** The original fix tried to offset positions at runtime inside `GhostMotionCommand.body_pos_w`. The correct fix is to store the correct floor-relative Z in the NPZ files themselves, so all consumers (ghost overlay, RSI, AMP discriminator) see consistent data without any runtime patches.

**Evidence:** All 4 step motions (LeftDoubleStep, RightDoubleStep, LeftTripleStep, RightTripleStep) spawning in the floor during overlay visualization.

---

## 2026-07-03 — Post-save ball-obs release v2: G1 torso-edge + catch window replace the flag gate

**What changed:**
- `observations.py:ball_pos_xy_b`: removed `hide_when_behind` (v1, gated on the `_ball_is_behind` save flags); added `hide_behind_torso` (zero when ball `x_body < 0.05`, G1 flying-mask front edge, `legged_robot.py:401`), `hide_after_steps` (zero once episode step > window; G1 `catchstep > 0` analog), and `noise_scale` (uniform noise applied INSIDE the term BEFORE the mask, G1 `legged_robot.py:425-426` ordering)
- `goalkeeper_env_cfg.py` actor term: `hide_behind_torso=True`, `hide_after_steps=75`, `noise_scale=0.05`, manager `noise=None`; play block sets `noise_scale=0.0`
- Rewrote `tests/simple_goalkeeper/test_post_save_ball_release.py` for the v2 gate

**Why it was wrong:** v1 (fead241/fbd9e23) invented a mechanism G1 does not have. Audit of run 2026-07-02_22-56-40 (first and only v1 run): 17% of iterations diverged (mean reward to -9.5e5, value loss to 5e10, action_rate_l2 flailing) vs zero divergent iterations in the config-identical gate-free control run 20-18-38; play mode showed post-save walking instead of settling. Four gaps vs G1: (1) soft grazes below the 0.6 m/s flag threshold kept the ball visible, so footreach kept paying post-contact chasing; (2) the `x<0` trigger was un-latched and could re-toggle on a rebounding ball; (3) mjlab applies manager noise AFTER the term returns, so the "hidden" obs was actually a phantom ±5 cm ball at the robot's feet every step (G1 masks after noise → hidden = exact zeros); (4) no catch window — G1 policies spend the back ~2/3 of every episode with zeroed ball obs from iteration 1, which is where post-save standing is learned; v1 gave blind time only after registered saves.

**Correct value:** hidden iff `x_body < 0.05` OR `episode_step > 75`. Window resized from G1's 50 because SGK flight times are 0.7–1.3 s (G1: 0.4–1.0 s) — 75 steps (1.5 s) closes only after the latest possible arrival, so neither condition can fire while the ball is still en route (user constraint: no visibility reduction during the save). G1's warmup blackout, random vanish, and approach/cone checks intentionally not ported for the same reason.

**Evidence:** TensorBoard cross-run comparison (22-56-40 vs 20-18-38 vs 19-03-50 vs 01-14-33) + sgk_play observation of post-save walking + line-by-line read of `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:397-428,643,679`.

---

## 2026-07-03 — AMP dataset: re-added LeftStep/Rightstep near-standing motions

**What changed:** `goalkeeper_amp_cfg.py:_motion_files()` filter from `DoubleStep|TripleStep only` (4 files, 2026-07-02 experiment) to `everything except Safe*` (6 files: + `LeftStep_own`, `Rightstep_own`). Uniform sampling, no `motion_weights`.

**Why it was wrong:** The 4-motion dataset contained no standing/idle reference at all, so the discriminator rewarded stepping motion unconditionally — directly fighting the five `post*` recovery rewards after a save (`mean_amp_reward` dropped ~20 → ~11 and play showed the robot walking off post-save). G1's own dataset contains `leftstep.pt`/`rightstep.pt` alongside the save motions.

**Evidence:** run 2026-07-02_22-56-40 play behavior + `Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/` file listing.
