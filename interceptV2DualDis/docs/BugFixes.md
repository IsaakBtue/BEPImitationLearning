
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

## 2026-07-05 — Ported SimpleGoalKeeper's post-save ball-visibility gate v2

**What changed:** `ball_pos_xy_b` (`mdp/observations.py`) gains `hide_behind_torso`, `hide_after_steps`, and `noise_scale` params. The actor's `ball_pos_b` term (`goalkeeper_env_cfg.py`) now sets `hide_behind_torso=True`, `hide_after_steps=75`, `noise_scale=0.05`, and the manager-level `Unoise` on that term is removed (noise now applied inside the term, before the mask). Play mode explicitly zeroes `noise_scale` since `enable_corruption=False` only disables manager-level noise.

**Why it was wrong (in SimpleGoalKeeper, the source of this port):** The actor's ball observation was fully visible for the entire episode with no post-save release, so the policy never learned to stop reacting once the ball left play — it kept tracking/chasing the ball after a save. An earlier attempt (`hide_when_behind`, a latched-flag gate) destabilized a training run (17% of iterations diverged). This project (interceptV2DualDis) never had a post-save release at all — its own `always_visible=True` came from reverting a *different, stricter* mistake (`8803b6e`: hiding the ball for the whole episode via G1's train-time vanish gate, not just post-save).

**Correct value:** Full visibility during the approach and save (ball visible until `x_body < 0.05` behind the torso, or until episode step 75 — sized for this project's 0.7–1.3 s flight-time range and 3 s episodes, identical to SimpleGoalKeeper's). Neither condition can fire while the ball is still en route.

**Evidence:** Not yet validated against a training run in this project (ported from SimpleGoalKeeper's `ce69f36`, which required and got a fresh run there). Next checkpoint here should be checked in play for post-save tracking/chasing behavior.

---

## 2026-07-05 — Ported SimpleGoalKeeper's `cleanstop` threshold tightening

**What changed:** `cleanstop`'s `speed_threshold` param (`goalkeeper_env_cfg.py`) lowered from `0.25` to `0.10` m/s.

**Why it was wrong:** After a deflection the ball is typically sliding, and translational friction during the slide-to-roll transition bleeds speed on its own — at 0.25 m/s this decay alone could cross the threshold with no genuine foot-trap, i.e. `cleanstop` could fire from friction rather than a real save.

**Correct value:** `speed_threshold=0.10`. Harder for pure friction decay to clear within the post-save window, still reachable by an actual foot-trap. Mechanism otherwise unchanged (one-shot bonus, weight 25.0, requires `softstop` already fired + correct-foot contact at that moment).

**Evidence:** Ported from SimpleGoalKeeper (`f974942`), not yet independently validated against a training run in this project.

---

## 2026-07-05 — WandB never actually enabled; missing SGK's full episode-metrics logging

**What changed:** Two separate bugs in `rsl_rl_multi/him_amp_on_policy_runner.py`, fixed together (user-reported: terminal output and the wandb.ai dashboard both looked far sparser than SimpleGoalKeeper's, despite this being "directly copied" from a working SGK setup):

1. `learn()` always constructed a plain `torch.utils.tensorboard.SummaryWriter`, never checking `cfg["use_wandb"]` — unlike the stock `AMPOnPolicyRunner` (`rsl_rl_amp`), which swaps in `WandbSummaryWriter` when the flag is set. The multi-disc config's `"use_wandb": True` was silently ignored; nothing was ever pushed to wandb.ai.
2. The rollout loop never collected `infos["episode"]`/`infos["log"]`, and `learn()` only ever logged 8 raw scalars via one terse `print()` line — vs. the stock runner's ~68-tag breakdown (every `Episode_Reward/*` term, every `Episode_Termination/*` cause, curriculum weights, AMP discriminator stats, policy noise, FPS/perf). This runner is a from-scratch rewrite for the 4-discriminator setup, not an actual copy of the stock runner, so that logging block was simply never written.

Also: `predict_region_routed_amp_reward` (`multi_disc_amp_ppo.py`) discarded the per-sample discriminator logits and raw AMP reward it already computed (`_, _`) — needed to log `Train/mean_amp_reward`/`Train/mean_discri_logits` like the stock runner. Now returns all three.

Separately (same request): `wandb_project` changed from `"SimpleGoalKeeper-MultiDisc"` to `"SimpleGoalKeeper"` (same project as SGK itself, not a separate one), and `experiment_name`/`run_name` gained an `intercept_` prefix (`intercept_simple_goalkeeper_multidisc` / `intercept_phase1`) so intercept's local log folders and W&B run names/groups are clearly distinguishable within that shared project.

**Why it was wrong:** Both gaps trace to the same root cause — this project's multi-discriminator training loop (`HimAMPOnPolicyRunner`) is a bespoke runner written to support 4 region-specific discriminators + HIM estimator heads, and never had either piece (W&B wiring, rich episode logging) ported over from the stock runner it otherwise parallels.

**Correct value:** `learn()`'s writer selection and `log()` method now mirror `AMPOnPolicyRunner`'s exactly (same `Episode/*`/`Loss/*`/`Policy/*`/`Perf/*`/`Train/*` tags and console table format), extended with this runner's own `Loss/est_ball`/`Loss/est_region` auxiliary losses.

**Evidence:** 48/48 existing tests still pass (no test covered this runner directly). Verified with a live smoke run (`--num-envs 64 --agent.max-iterations 3 --agent.use-wandb False`, throwaway experiment folder deleted afterward): full `Episode_Reward/*`/`Episode_Termination/*`/`Curriculum/*` breakdown printed and logged, exit code 0, no crash. A real run (`2026-07-05_12-08-20_intercept_phase1`) separately confirmed the W&B piece lands online (`run-20260705_120826-tw5dzkqb`, not an offline run folder) under the wiring fix alone, before this logging fix was added.

**Evidence:** Ported from SimpleGoalKeeper (`f974942`), not yet independently validated against a training run in this project.

---

## 2026-07-05 — `region_estimator` regressed to near-chance under `schedule="adaptive"`; two related runner bugs fixed

**What changed:** `goalkeeper_multidisc_amp_runner_cfg()`'s `"schedule"` changed from `"adaptive"` to `"fixed"` (`tasks/goalkeeper_multidisc_amp_cfg.py`). Also fixed in `him_amp_on_policy_runner.py`: `learn()` now sets `self.current_learning_iteration = it` right before each periodic checkpoint save (previously only updated once, after the whole `learn()` call returned, so every intermediate `model_{it}.pt`'s embedded `"iter"` field was always `0`); `load()` now restores `self.current_learning_iteration = loaded["iter"]` (previously never touched it at all, so any resume would silently restart iteration numbering at 0 and could overwrite/collide with existing checkpoint filenames in the same run directory).

**Why it was wrong:** User reported the region-conditioned track "again didn't learn to double step" after stopping run `2026-07-05_12-16-46_intercept_phase1` at `model_5750`. A live diagnostic (loaded `model_500.pt` and `model_5250.pt`, drove 512 envs with the trained policy 400 steps, compared `region_estimator`'s live argmax against `env._region_id`) found: `model_500` — 38.5% accuracy (chance=25%), real signal on all 4 classes; `model_5250` — 27.7% accuracy, confusion matrix shows near-total collapse onto `left_near`/`right_near` only, **unable to distinguish true left from true right at all** (~48%/52% split regardless of true side), "far" classes almost never predicted (6% vs true 50%). This is a regression, not "never learned" — and it happened *after* `ball_difficulty` curriculum reached 1.0 (step 2750), when near/far should have gotten easier to tell apart, not harder, ruling out task difficulty as the explanation.

Comparing against `Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py` (this runner's origin): G1 defaults to `schedule="fixed"` and `g1_29_config.py` never overrides it — the proven reference run used a constant LR throughout. This project's multi-disc config set `"adaptive"` instead, with no divergence-table entry — an undocumented departure from the proven baseline. Root cause: `region_arg` (`torch.argmax(estimate_region)`, identical construction to G1's) is a discrete value feeding the actor's own input; small logit shifts in the still-training `region_estimator` can flip this argmax between minibatch/epoch passes within one PPO update, inflating the *measured* policy KL divergence for reasons unrelated to real policy movement. `MultiDiscAMPPPO` uses **one Adam optimizer / one shared learning rate for the entire `actor_critic`** (actor, critic, history encoder, ball estimator, region estimator together — `multi_disc_amp_ppo.py:75`), so the adaptive-KL scheduler's repeated `lr /= 1.5` in response to that spurious KL froze capacity for every head, not just the policy. Confirmed empirically: `model_5750.pt`'s saved `optimizer_state_dict` shows `lr=7.59e-5` for every param group — 13x below the configured `1.0e-3`, sitting at the scheduler's floor. This also plausibly explains the double-step failure: the actor's own belief about which region it's in was unreliable, and by the time it mattered the whole network's effective learning rate had collapsed, leaving no room to learn a behavior (double-stepping on far balls) that depends on that signal.

The iteration-tracking bugs were fixed alongside because resuming training to test the schedule fix requires them: without the fix, a resume would restart `it` at 0 and silently overwrite `model_0.pt` onward in the same directory.

**Correct value:** `schedule="fixed"`, `learning_rate=1.0e-3` (unchanged base value, now never adaptively touched). Iteration counter now correctly tracked through save/load.

**Evidence:** Live probe confusion matrices (see above) and the checkpoint's own `optimizer_state_dict` lr values are the direct evidence for the collapse; the G1 default comparison is the basis for the fix. **Not yet independently confirmed as the fix** — training was resumed from `model_5750.pt` (fresh optimizer state, `load_optimizer=False`, to avoid inheriting Adam moments tuned for a near-zero LR regime) with `schedule="fixed"` to observe whether `Loss/est_region`/region accuracy recovers; results pending.

---

## 2026-07-05 — `schedule="fixed"` alone isn't resume-safe; region_estimator split into its own optimizer param group; a checkpoint got overwritten along the way

**What changed:** `MultiDiscAMPPPO.__init__` now builds `region_estimator`'s parameters as a separate optimizer param group (name `"region_estimator"`) with its own learning rate, decoupled from the shared `"actor_critic"` group (actor, critic, history_encoder, ball_estimator). New config key `region_estimator_learning_rate` (default `3.0e-3`) in `goalkeeper_multidisc_amp_runner_cfg()`. The adaptive-KL block in `update()` now skips the `"region_estimator"` group by name so it can never be silently overwritten if `schedule="adaptive"` is ever re-enabled.

**Why it was wrong:** The previous fix (`schedule="fixed"`, prior entry above) is correct for a *fresh* training run, but three attempts to resume the already-collapsed `2026-07-05_12-16-46_intercept_phase1` run with a single shared LR for the whole `actor_critic` all failed: (1) `load_optimizer=False` at the config's full `1.0e-3` (a 13x jump from the settled `~7.6e-5`) blew the policy up within 5 iterations — reward to -1e8, episode length ~12, every episode ending in `bad_orientation`/`base_height`/`shank_height`, almost certainly Adam's cold-start instability (near-zero `exp_avg_sq` denominator inflates the first few effective steps) compounding an oversized LR on an already-converged policy; (2) `load_optimizer=True` at a gentler `2.0e-4` (~2.5x) avoided the catastrophic blowup but left episode length stuck at ~22 (vs. pre-resume ~130) for 200 iterations with no recovery — even a modest shared-group bump was enough to knock the converged policy out of the narrow basin it had settled into under the slowly-decaying adaptive LR. Splitting `region_estimator` into its own group sidesteps this entirely: the main group can stay at its settled, safe LR (zero intended change to the already-good policy) while `region_estimator` gets real capacity to move, with no path for its gradient to touch any other module's weights.

**Separately, an operational mistake during attempt (1):** it resumed *in place* into the original run directory with `current_learning_iteration` manually set to exactly `5750` — a multiple of `save_interval=250` — so the very first loop iteration's periodic-save check fired immediately and **overwrote `model_5750.pt`** with the already-blown-up post-update state. Caught by checking the corrupted file's own saved `optimizer_state_dict` lr (`0.001`, matching attempt 1's setting, not the expected `~7.6e-5`) and its `"iter"` field (`5750`, only possible because this session's own iteration-tracking fix, from the prior entry, was already in place). Renamed on disk to `model_5750.pt.CORRUPTED-by-resume-attempt-do-not-use` rather than deleted. `model_5500.pt` (saved before that attempt started) and `model_5250.pt` (already pushed to git, commit `666f2f5`) are unaffected and were used for the successful resume instead.

**Correct value:** `region_estimator_learning_rate=3.0e-3` for `region_estimator`'s own param group; main group resumed at the settled `7.593750000000002e-05` read directly from `model_5500.pt`'s own `optimizer_state_dict` (not hardcoded, to avoid a repeat of the exact mismatch that caused attempt 3's blowup when it unknowingly read `1.0e-3` back out of the corrupted `model_5750.pt`). Any future resume must target a run directory distinct from the one being resumed from.

**Evidence:** Resumed from `model_5500.pt` into a separate directory (`..._continued_region_fix`): episode length recovered from ~22 to ~130+ within 7 iterations, reward back to ~29-30 within ~10 iterations, `Loss/est_region` moving from 1.386 to ~1.27 within 30 iterations — no instability observed through the smoke test (60 iterations) or in the subsequent long run. Whether region accuracy actually keeps improving (vs. plateauing again, just at a better point) is not yet confirmed — that requires watching the full run.

---

## 2026-07-06 — `reset_ball_rolling`'s y_end sampling ignored one-sided region ranges: ~50% wrong left/right, "far" regions landed near 75% of the time

**What changed:** `reset_ball_rolling` (`mdp/events.py`) now branches on whether `y_end_range` is one-sided (`y_end_range[0] * y_end_range[1] > 0`, i.e. doesn't straddle zero) vs the original two-sided/symmetric case. For one-sided ranges (all 4 region-conditioned calls from `reset_ball_rolling_by_region`), the sign is now taken directly from `y_end_range` (never randomized), and the sampled magnitude is lerped within that range's own `[lo, hi]` bounds (`lo=min(|bounds|)`, `hi=max(|bounds|)`; `inner=lo`, `outer=lo+(hi-lo)*d` at difficulty `d`) instead of the generic two-sided dead-zone constants. The original two-sided branch (used only by the single-disc task's plain `reset_ball`, `y_end_range=(-0.9,0.9)`) is unchanged.

**Why it was wrong:** `reset_ball_rolling` was written once for the generic, symmetric, either-side target (`(-0.9,0.9)`) and reused verbatim by `reset_ball_rolling_by_region` for the 4 per-region *one-sided* ranges (`left_near=(0.15,0.5)`, `left_far=(0.5,0.9)`, `right_near=(-0.5,-0.15)`, `right_far=(-0.9,-0.5)`) without adapting the sign/magnitude logic. Two independent defects stacked: (1) `side = torch.where(rand()>0.5, 1, -1)` always re-randomizes the sign with a 50/50 coin flip regardless of `y_end_range`'s actual sign — a "left_near" env's ball landed on the *right* side of the goal line roughly half the time, and vice versa for every region; (2) the magnitude bounds (`inner`/`outer`) were derived from the fixed generic constants `_Y_INNER=0.15`/`_Y_OUTER=0.35` and `max_half`, never clamped to the region's own `lo` bound — so a "far" region (`0.5`–`0.9`) sampled magnitudes as low as `0.1`, deep inside "near" territory, at every difficulty level once `d` was past the initial ramp.

This was caught while auditing whether interceptV2DualDis's region/ball-estimator mechanism matched `Humanoid-Goalkeeper`'s (per this project's standing rule to verify every mechanism against the G1 reference before trusting it) — G1's own 6-region split (`legged_robot.py:916-960`, `g1_29_config.py` `ranges_0`..`ranges_5`) uses **disjoint, non-overlapping height/width bounds per region with no re-randomized sign**, so this class of bug can't occur there; the SGK/interceptV2 region-conditioned ball spawn is a new mechanism without a direct G1 analogue, and its sign/magnitude logic was never checked against the region ranges it was actually being fed.

Verified empirically (20k-sample Monte Carlo of the exact pre-fix formula, at `d=1.0`, matching the checkpoint under test at the time — `model_3000.pt`, ball_difficulty curriculum saturated): `left_near` sampled negative (wrong side) 49.4% of the time and landed inside its own intended `[0.15,0.5]` band only 44.4% of the time; `left_far` sampled negative 49.8% of the time and landed inside its own intended `[0.5,0.9]` band only **25.1%** of the time (i.e. 75% of "far"-labeled balls were secretly easier near-magnitude shots). `right_near`/`right_far` showed the mirror pattern. This directly explains two previously-unresolved symptoms: the region_estimator's persistent, unresolved left/right confusion in the live probe (34.9% accuracy at `model_3000`, weak-to-nonexistent left/right separation despite the region_estimator LR-collapse fix already applied and holding) — because the ground-truth label (`env._region_id`) was statistically decorrelated from the ball's actual observable side for ~50% of samples — and the user-reported failure of far-side shots to converge on the double/triple-step AMP reference motions, since 75% of nominally-"far" episodes never actually presented a far shot for the policy to practice on.

**Correct value:** one-sided ranges now sample sign-locked, magnitude-lerped-within-own-bounds `y_end`; see fix description above. Verified via direct Monte Carlo of the post-fix formula: all 4 regions land inside their own intended range 100% of the time, at every difficulty level from `d=0` (deterministic at the region's inner edge) to `d=1` (full region span).

**Evidence:** Reproduction and fix verification scripts run standalone (not committed as tests — see gap noted below). Existing `tests/simple_goalkeeper/test_regions.py` did not catch this because it monkeypatches `reset_ball_rolling` entirely and only asserts on the `y_end_range` *parameter* passed to it, never exercising the actual sampling math inside — a real test-coverage gap; a follow-up should add a test that calls `reset_ball_rolling` (or an isolated helper) with a one-sided range and asserts the sign/bounds directly. Not yet confirmed against a full training run — this fix requires a fresh run (the in-flight `2026-07-06_15-13-34_region_fix_g1_match_2026-07-06` run's region_estimator and actor were trained against the corrupted ball-spawn distribution from iteration 0, so resuming it would carry the poisoned region conditioning forward; restarted fresh instead).

---

## 2026-07-07 — `reset_ball_rolling`'s `env._rsi_cross_y` used the wrong crossing-fraction formula (cos(angle) instead of the X-ratio)

**What changed:** the goal-line crossing-Y formula in `reset_ball_rolling` (`mdp/events.py`) changed from `y_start + (y_end - y_start) * (x_start + 0.3) / horiz_dist` to `y_start + (y_end - y_start) * (x_start / (x_start + 0.3))`.

**Why it was wrong:** the ball travels in a straight line from `(x_start, y_start)` to the aim point `(-0.3, y_end)` — 0.3 m *behind* the actual goal line. Parametrizing the line by `f ∈ [0,1]` (0 at spawn, 1 at the aim point), `X(f) = x_start - f*(x_start+0.3)`. The goal line is at `X=0`, not at the aim point, so the correct crossing fraction is `f = x_start / (x_start+0.3)`. The old formula instead computed `f = (x_start+0.3) / horiz_dist` where `horiz_dist = sqrt((x_start+0.3)² + (y_end-y_start)²)` — that's `cos(trajectory angle)`, a completely different quantity that happens to also land in `(0,1]` but has no geometric relationship to "how far along the path is the goal line." The two formulas disagree on every diagonal shot (e.g. `x_start=2.0, dy=0.5`: correct `f=0.870` vs buggy `f=0.977`), making the buggy version place `cross_y` closer to `y_end` than the true crossing point.

Found while investigating a user report that interceptV2DualDis's play viewer showed "near" balls incorrectly getting the blue/green two-stage split — traced the same bug independently to `SimpleGoalKeeper`'s `reset_ball_rolling` (see its `docs/BugFixes.md`, same date), which turned out to have the correct formula computed first and then a *second*, leftover duplicate block silently overwriting it with this exact wrong formula — almost certainly the source this project's single (buggy, never-corrected) copy was ported from. Proved mathematically that this bug is bounded and could not be the actual cause of the reported symptom: since `f ∈ (0,1]` for both formulas, `cross_y` can never leave the interval `[min(y_start,y_end), max(y_start,y_end)]` — a legitimately-narrow target (`|y_end| ≤ 0.5`, with `|y_start| ≤ 0.3`) can never produce a computed `cross_y` exceeding `0.5` regardless of which formula is used. Confirmed via two direct empirical tests (region-conditioned `play=True`/`play=False` configs, and a fresh 95-episode wide-flag consistency check on the sibling SGK project): `env._blue_wide`/the two-stage gate correctly separates near from far in every sampled episode, before and after this fix. The actual explanation for the reported symptom is a design fact, not a bug: "wide" is a purely *lateral* threshold (`|crossing_y - start_y| > 0.5`), independent of forward spawn distance — a physically close-spawning ball can legitimately have a large lateral target and correctly trigger the two-stage split.

**Correct value:** `env._rsi_cross_y[env_ids] = y_start + (y_end - y_start) * (x_start / (x_start + 0.3))`.

**Evidence:** 48/48 existing tests pass unchanged (no test asserted the exact numeric value of `_rsi_cross_y`, only the downstream `wide`/region classification, which this fix cannot affect — see bounded-impact argument above). This is a real accuracy fix for the footreach/foot_proximity/stopball/softstop target geometry on every diagonal shot, not a classification fix — did not warrant restarting the in-flight `blue_ball_gate_2026-07-06` run (its impact is bounded within the existing near/far envelope, unlike the section-12 sign/magnitude bug which flipped classifications outright).
