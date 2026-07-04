
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

---

## 2026-07-03 — Widened `y_end_range` ±0.9 → ±1.1 to force double-stepping

**What changed:** `y_end_range` (-0.9, 0.9) → (-1.1, 1.1) in BOTH the training and play `reset_ball` blocks (`goalkeeper_env_cfg.py`, parity rule).

**Why it was wrong:** The policy was not converging to double-step saves at ±0.9 — the widest balls remained reachable with a single lunge, so stepping never became necessary. G1 trains its low/feet target region out to ±1.8 m at max curriculum (`g1_29_config` `ranges_4.maxw`, expansion in `legged_robot.py:333-336`); ±1.1 stays conservative vs upstream. Deliberately NOT added: a stepping-specific reward (G1 has none; footwork emerges from target width + step motions in the prior, and a crafted step reward invites shuffle hacking) and the 4× double/triple AMP boost (double/triple already hold 77% of the 6-file dataset's frames; 4× would shrink the near-standing share to 7% and re-create the post-save AMP conflict fixed earlier today).

**Evidence:** play-mode observation (no double-steps at ±0.9) + G1 `assign_ball_states`/`command_ranges` read + NPZ frame-count audit (73/36/73/73/73/53 frames).

---

## 2026-07-03 — RSI: reinstated ball-conditioned NPZ tier branch (30/50/20 split)

**What changed:** `MotionResetManager.reset()` is now a three-way split decided by one draw per call: 30% ball-conditioned NPZ tier RSI (new `_tier_npz_reset`), 50% G1 `continue_keep` live-donor copy (unchanged literal port), 20% G1 randomized default standing (unchanged). `reset_ball_rolling` now stores the predicted goal-line crossing Y in `env._rsi_cross_y` (trajectory geometry, not laggy `ball.data`). The tier branch routes: |cy| < 0.20 → standing HOME; 0.20–0.40 → (side, double) = SafeMedium; 0.40–0.60 → (side, triple) = SafeFar; ≥ 0.60 → (side, wide) = DoubleStep+TripleStep, writing full RSI state (root + velocities + joints) via the still-intact `_write_rsi_state` pool system. Wrapper + `EventTermCfg` expose `tier_rsi_fraction=0.3`, `live_rsi_fraction=0.5`.

**Why it was wrong:** After the literal G1 port (d496aa3, 2026-07-01) nothing conditioned reset poses on ball trajectory, and the policy stopped converging toward double-stepping — wide balls were mostly met from generic donor poses. User-requested, deliberate divergence from the literal port; the combined RSI share (0.8) still matches G1's `torch.rand(1).item() > 0.2`.

**Evidence:** play-mode observation (no double-steps) + `tests/simple_goalkeeper/test_live_rsi.py` (24 tests: routing, partition boundaries, three-way statistics, crossing-Y storage) + live CUDA env exercise of the tier branch.

---

## 2026-07-03 — RSI: raised tier fraction 30/50/20 → 50/30/20

**What changed:** `MotionResetManager.reset()` and `reset_from_motion_data()` defaults: `tier_rsi_fraction` 0.3→0.5, `live_rsi_fraction` 0.5→0.3 (`events.py`). `goalkeeper_env_cfg.py`'s `reset_from_motion_data` params dict updated to match. `tests/simple_goalkeeper/test_live_rsi.py` updated (`test_reset_partition_boundaries_at_default_fractions`, `test_reset_three_way_split_statistics`, `test_reset_from_motion_data_passes_the_three_way_split_to_reset`, `test_reset_single_coin_flip_covers_the_whole_batch_at_once`).

**Why it was wrong:** Only the 30% tier branch deliberately seeds a double/triple-step pose for wide-crossing episodes. The other 50% (`continue_keep`) copies `dof_pos` from an arbitrary live env with no regard for that donor's own ball width — so 70% of wide-crossing resets got an unrelated donor pose or plain standing, diluting exposure to the harder behavior during the exploration phase that needed it most. `scripts/play.py`'s `--rsi` flag re-registers `reset_from_motion_data` with no params dict, so it was (and still is) relying on these function defaults directly — confirmed by reading `play.py:283-290`.

**Evidence:** direct investigation into why wide-ball saves converged to one continuous leap instead of double-stepping (this session) — traced to RSI dilution as one of several compounding causes, alongside the AMP-discriminator and `footreach`-target issues fixed in the entry below. User-directed fix.

---

## 2026-07-03 — footreach/foot_proximity: two-stage reach target on wide crossings

**What changed:** New helper `mdp.rewards._get_reach_target_y(env, ball_name, wide_threshold=0.6)`. For crossings where `|crossing_y - start_y| > 0.6`, the target fed into `footreach` and `foot_proximity` is the midpoint between the robot's stance and the true crossing point for the first half of the ball's flight time, then the full crossing point for the second half; narrow crossings always target the full point. `reset_ball_rolling` now caches the sampled flight time in `env._ball_t_flight` (previously a local variable only, discarded after deriving ball velocity). `scripts/play.py`'s `_patch_viewer_intercept_vis` draws a blue sphere at the midpoint during phase 1 and the existing green sphere at the full point otherwise, so the schedule is visible in `sgk_play`.

**Why it was wrong:** Widening `y_end_range` to ±1.1 (earlier today) did not produce double-stepping — the policy converged on one continuous maximal-speed leap straight to the final target. The AMP discriminator only ever sees `(s_t, s_{t+1})` 2-frame transitions (`beyondAMP/.../motion_dataset.py`), so it cannot judge global trajectory shape or step count — a smooth fast leap satisfies it as well as a genuine multi-step reference clip. `footreach`'s `vel_sigma` (up to 10×) rewards raw speed toward a single fixed target, not gait, so nothing in the reward pushed toward a paced approach. Reference-clip measurement confirmed the leap isn't a leg-reach trick: max lateral foot-extension relative to the trunk tops out ~0.28–0.32 m in every step/double/triple `.npz` clip, so genuine multi-step travel is required past that distance.

**Why this design over a new reward term:** reuses the existing, already-tuned `footreach`/`foot_proximity` reward instead of adding a new one-shot bonus with its own weight and a foot-contact state machine to guard against exploits. Self-correcting by construction — once the schedule flips to the full point, the sigmoid reach term and the up-to-10× `vel_sigma` multiplier both collapse for a foot still at the (now stale) midpoint, so lingering isn't rewarded and no separate exploit guard was needed. `is_left_ball` foot-side selection intentionally still uses the true crossing_y, not the staged target, so the assigned foot never flips mid-episode.

**Evidence:** direct investigation (this session): AMP transition-window read, reference-clip lateral-extension measurement, G1 upstream check (`legged_robot.py`, `g1_29_config.py` — no staged/waypoint mechanism exists there). `tests/simple_goalkeeper/test_reach_target_two_stage.py` (11 tests: narrow/wide gating, flight-time-half boundary, midpoint direction on both sides, missing-cache fallbacks, mixed batches) and `tests/simple_goalkeeper/test_footreach_two_stage_wiring.py` (4 tests: reward value drops for the same foot position once the schedule flips, narrow crossings unaffected, foot-side selection unaffected). User-directed design. Not yet validated against a training run — `model_0.pt` from a fresh run is being pushed alongside this fix so the play-mode blue→green marker timing can be checked visually before further training.

---

## 2026-07-04 — footreach/foot_proximity: landing gate on the two-stage schedule

**What changed:** `mdp.rewards._get_reach_target_y` gains a landing gate: the assigned foot (`_get_correct_foot_idx`) must be airborne (`feet_contact` sensor) at some point after reset, then land in ground contact within 0.3 m of the phase-1 midpoint ("blue ball") target on wide crossings (`|crossing_y - start_y| > 0.6`). Landing switches the target to the full crossing point ("green ball") immediately; the original `elapsed >= t_flight/2` time-based switch remains as a fallback for episodes that never land, so episodes cannot stall. New one-shot bonus reward `blue_ball_landed` (weight +10.0, no curriculum) pays out the first time the landing latch fires per episode. Landing state (`env._blue_was_airborne`, `env._blue_landed`) is only tracked when both a `robot` entity and `feet_contact` sensor are present in `env.scene`; absent in the lightweight fake envs used by the pre-existing `test_reach_target_two_stage.py`/`test_footreach_two_stage_wiring.py` suites, which remain fully unaffected. `mdp/rewards.py:_get_reach_target_y,blue_ball_landed`, `mdp/__init__.py`, `tasks/goalkeeper_env_cfg.py`.

**Why it was wrong:** the 2026-07-03 two-stage schedule (previous entry above) assumed the existing sigmoid reach term and up-to-10x `vel_sigma` multiplier would self-correct against lingering at the midpoint without needing a separate "did you actually step" check. In practice the policy has not learned the intended pause-at-blue-then-continue-to-green double-step motion — it can glide/leap through the midpoint region without ever placing a foot near it, and the time-based switch advances regardless, so nothing in the reward actually required the intermediate landing to happen.

**Why this design over alternatives:** rejected making it purely a soft bonus with the time-based schedule left unchanged, because that keeps the actual defect (schedule advances without requiring landing) unaddressed — a bonus alone doesn't stop the leap-through behavior, it just adds an extra incentive on top of it. Chose a hard gate with the existing time-based switch retained ONLY as a timeout fallback, so training can't stall on a robot that refuses to land. User-directed design; full design rationale in `docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md`.

**Evidence:** `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (7 tests). Not yet validated against a training run — next checkpoint should be compared against the prior run (`2026-07-03_18-53-55_phase1`) for whether double-stepping actually emerges, and checked in `sgk_play` for the blue→green marker now switching on landing rather than only on elapsed time.

---

## 2026-07-04 — blue-ball landing gate hardened to a pure hard gate; RSI rebalanced; landing diagnostics added

**What changed:** three coupled changes, shipped together in one training run (user's explicit choice):

1. `mdp.rewards._get_reach_target_y` loses its `elapsed >= t_flight/2` time-based fallback entirely. `phase1_active = wide & ~env._blue_landed` — no time term at all. A robot that never lands at the blue midpoint on a wide crossing now stays targeting it for the whole flight, until `footreach`/`foot_proximity`'s separate `ball_close < 0.5 m` live-ball override takes over (unaffected by this change, remains the true backstop against stalling). `scripts/play.py`'s sphere visualization, which had been independently re-deriving the now-removed time-based switch for rendering, was fixed to read the real `env._blue_landed` latch instead.
2. `blue_ball_landed`'s reward weight changed from flat +10.0 to curriculum-ramped 10→25 (reusing `reward_curriculum_ep_len`, identical formula/ceiling to `footreach_curriculum`), since it's no longer just an auxiliary bonus — it now gates whether the double-step choreography is reachable at all.
3. RSI three-way split rebalanced: `tier_rsi_fraction` 0.5→0.1, `live_rsi_fraction` 0.3→0.7 (standing unchanged at 0.2), across all three call sites (`MotionResetManager.reset()` defaults, module-level `reset_from_motion_data()` defaults — used by `sgk_play --rsi` — and `goalkeeper_env_cfg.py`'s training params dict).

Plus a new diagnostic: `env._blue_airborne_at_reset` (latched true if the assigned foot's first airborne transition happens within 2 steps of reset) and two new `cfg.metrics` entries, `blue_landed_genuine`/`blue_landed_rsi_assisted` (`mdp/metrics.py`, `Episode_Metrics/*`, no weight/dt scaling), classifying each landing as policy-driven or RSI-seeded.

**Why it was wrong:** the 2026-07-04 soft/timeout-gated landing mechanism (previous entry above) looked healthy in `Episode_Reward/blue_ball_landed` (~35-40% of all episodes, ~73-83% of wide crossings once correctly converted via this file's own documented one-shot-reward formula) but the user observed ~0% genuine landings in `sgk_play`. Root cause: `_tier_npz_reset` routes 50% of wide-crossing resets to a random frame from the DoubleStep/TripleStep motion clips, which can already have the assigned foot lifted and positioned near the blue target — satisfying the landing latch within a step or two of reset with zero credit due to the policy. The metric was measuring RSI seeding, not learned behavior.

**Why this design over alternatives:** considered testing the RSI rebalance alone first (keeping today's soft gate unchanged) to isolate whether it produces genuine learning before also hard-gating the schedule — user chose to bundle all three changes into one run instead, accepting the interpretability cost in exchange for speed; the new genuine/RSI-assisted diagnostic split is the agreed mitigation for that risk. Also notable: `tier_rsi_fraction` was raised 30%→50% on 2026-07-03 specifically because 30% wasn't enough exposure to produce double-stepping at all — dropping it back to 10% risks reintroducing that failure mode under a different symptom, documented as an accepted risk rather than resolved.

**Evidence:** `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (13 tests), rewritten tests in `tests/simple_goalkeeper/test_reach_target_two_stage.py`, updated fraction tests in `tests/simple_goalkeeper/test_live_rsi.py`. Full research and design: `docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md`. Not yet validated against a training run — the next run should be checked via `Episode_Metrics/blue_landed_genuine` vs. `blue_landed_rsi_assisted` (if genuine stays near zero while RSI-assisted accounts for most landings, the RSI rebalance risk above has materialized and `tier_rsi_fraction` may need to be walked back up) and visually in `sgk_play` for the blue→green marker now switching only on a genuine landing. Note that `_blue_airborne_at_reset` latches for any reset branch, not only `_tier_npz_reset` — with the rebalance above, 70% of resets now go through the `live_rsi` "continue_keep" donor branch (which copies another env's mid-motion joint pose onto a standing root and can likewise present an airborne foot near reset), so `blue_landed_rsi_assisted`'s magnitude is dominated by that 70%-share donor branch rather than the 10%-share tier branch and should not be read as "tier-RSI free credit" specifically; `blue_landed_genuine` staying near zero remains the reliable indicator that the accepted risk has materialized.
