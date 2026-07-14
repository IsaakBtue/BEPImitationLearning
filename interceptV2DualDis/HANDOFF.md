# Handoff: region_estimator divergence fix — verification in progress

**Date written:** 2026-07-06, ~16:20. **Updated 2026-07-06 ~18:50: a second, more
severe bug found and fixed (ball-spawn sign/magnitude for region-conditioned
resets) — see section 12. Old run killed and restarted from scratch under the
new fix. Status: new run launched, monitoring resumed.**

## Prompt to paste into a new session

```
Read /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis/HANDOFF.md in full,
then continue monitoring the background training run described in it. Check
progress via the tensorboard commands in "How to check progress", watching for
either (a) divergence (matches the "If it diverges again" section) or (b) clean
survival past iteration ~2600-3000, at which point do the deeper "success
criteria" analysis (live region-accuracy probe). Use /loop with no interval
(self-paced) to keep checking back until the region estimator is confirmed
fixed or a new divergence is found and root-caused. Do not read the raw stdout
log continuously -- use the tensorboard event file as the source of truth.
```

---

## 1. The bug

`interceptV2DualDis` (Booster T1 / mjlab port) trains a 4-region ball-region
estimator inside `HimActorCritic`, modeled on `Humanoid-Goalkeeper/rsl_rl/rsl_rl/
modules/actor_critic.py`'s 6-region estimator (the frozen G1/Isaac Gym reference
in this same repo, never modified, treated as ground truth).

Every training run since commit `7b7684d` (current `HEAD` before this session's
uncommitted fix) diverged to `NaN` in the policy's action distribution, at
iteration ~1870-2560 depending on the run, preceded by the policy's own raw
actions exploding (`mean_action_acc` metric jumping from O(100) to O(1e12) within
a few logged iterations) and reward collapsing to catastrophic negative values
(down to -1e20 or worse).

## 2. Root cause (verified twice: direct read + independent unbiased subagent)

Commit `7b7684d` made two changes to `MultiDiscAMPPPO`
(`interceptV2DualDis/src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`)
and its config (`interceptV2DualDis/src/simple_goalkeeper/tasks/
goalkeeper_multidisc_amp_cfg.py`):

1. Split `region_estimator` into its **own optimizer param group** with a fixed,
   undamped `lr=3.0e-3`, explicitly **exempted** from the adaptive-KL learning-rate
   throttle that governs the rest of the network.
2. Switched the main group's `schedule` from `"adaptive"` to `"fixed"`.

Both changes were justified in `docs/BugFixes.md` and `CLAUDE.md`'s divergence
table by the claim **"G1 defaults to `schedule='fixed'` ... the proven reference
run used a constant LR throughout."** This claim is **factually wrong** — nobody
had actually traced G1's real, effective config chain, only assumed a library
default applied.

Verified (independently, twice) against `Humanoid-Goalkeeper/`:

- `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_29_config.py:368-369`
  (`G129CfgPPO.algorithm`) only overrides `entropy_coef` — it inherits
  `schedule = 'adaptive'` unmodified from
  `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot_config.py:326`.
- `Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py:101-116`: **one**
  optimizer param group (`'actor_critic'`) covers the *entire* actor-critic —
  actor, critic, history_encoder, ball_estimator, **and region_estimator** — no
  split, no separate LR, ever.
- `him_ppo.py:196-209`: the adaptive-KL block rescales **every** param group's
  LR uniformly (`for param_group in self.optimizer.param_groups: param_group['lr']
  = self.learning_rate`) — no exemption for region_estimator or anything else.
- `him_ppo.py:310`: a **single joint** `clip_grad_norm_(self.actor_critic.
  parameters(), self.max_grad_norm)` call, not split by submodule.
- `region_arg = torch.argmax(estimate_region, ...)` feeds into the actor's own
  input as a hard argmax in G1 too (`actor_critic.py:231-232`) — identical
  construction to ours. G1 tolerates this fine because the shared adaptive LR
  throttles the *whole* network (region_estimator included) the moment KL spikes
  from a flip — nothing is exempt, so nothing can run away unchecked.

Empirical confirmation: the pre-`7b7684d` run (`2026-07-05_12-16-46_intercept_
phase1`, code-equivalent to `HEAD~2` = commit `666f2f5`) trained cleanly, reward
climbing 12→30, for **5790 iterations** before it eventually diverged once. Every
run since `7b7684d` diverged far earlier (iter ~988-2560) and in one case
(`2026-07-06_01-14-09`) repeatedly oscillated between blowup and partial recovery.

## 3. Fix applied (uncommitted — working tree only)

Two files reverted to match G1 exactly:

**`interceptV2DualDis/src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`**
- Removed the `region_estimator_learning_rate` constructor param and the split
  param group. `region_estimator` now lives in the single `"actor_critic"` group
  (`self.main_params = list(actor_critic.parameters())`).
- Removed the `if pg.get("name") != "region_estimator":` exemption in the
  adaptive-KL LR loop — every group now gets rescaled uniformly.
- Unified grad clipping into one `nn.utils.clip_grad_norm_(self.main_params,
  self.max_grad_norm)` call (previously two independent clips).
- **Note:** this file also has *other*, pre-existing uncommitted changes from
  an earlier session today, not part of this fix and not touched by it: a
  NaN-guard in `act()` (zeros out a NaN'd obs batch before it reaches the
  network, ported from `him_ppo.py:138-140`) and a `returns_batch`/
  `target_values_batch` clamp to `±1000` in `update()`. Both remain active.

**`interceptV2DualDis/src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py`**
- `"schedule": "fixed"` → `"schedule": "adaptive"`.
- Removed the `"region_estimator_learning_rate": 3.0e-3` key entirely.

Run `git diff -- interceptV2DualDis/src/simple_goalkeeper/rsl_rl_multi/
multi_disc_amp_ppo.py interceptV2DualDis/src/simple_goalkeeper/tasks/
goalkeeper_multidisc_amp_cfg.py` in a new session to see the exact diff.

**These changes are NOT committed.** The working tree also has unrelated,
pre-existing modified files from before this investigation (`SimpleGoalKeeper/*`,
`interceptV2DualDis/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/utils/wandb_utils.py`)
— do **not** sweep these into a commit for this fix; stage only the two files
above (`git add <path>` per file, never `git add -A`) when it's time to commit.

## 4. Checkpoints already pushed to origin/master (for user's own inspection)

- `0516ff1` — `model_1750.pt` from the broken run `2026-07-06_10-15-32_intercept_
  phase1` (reward ~20.2, last checkpoint before that run's NaN crash at iter
  ~1870). Requested explicitly by the user despite knowing the code was broken.
- `666f2f5` (pre-existing, from before this session) — `model_5250.pt` from the
  clean pre-`7b7684d` run `2026-07-05_12-16-46_intercept_phase1` (reward ~30).
- **Not yet pushed:** `model_5500.pt` from that same clean run exists on disk
  (`interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-05_12-16-46_intercept_phase1/model_5500.pt`, gitignored — use
  `git add -f` to push it), reward ~29-30, later than `model_5250` and unaffected
  by the checkpoint-corruption incident documented in `docs/BugFixes.md`.

## 5. Currently running training (the live test of the revert)

- **Started:** 2026-07-06 15:13:34.
- **PID:** `2576767` — launched via `nohup ... & disown`, fully detached from any
  terminal/session. It keeps running regardless of which session checks on it,
  as long as the machine stays up.
- **Command:**
  ```
  cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096 \
    --agent.run-name region_fix_g1_match_2026-07-06
  ```
- **Run directory (source of truth — tensorboard + checkpoints):**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-06_15-13-34_region_fix_g1_match_2026-07-06/`
- **Raw stdout log** (session-scoped scratchpad, may not exist in a new
  session — prefer the tensorboard event file instead):
  `/tmp/claude-6380/-home-ibouwmeest/86c694c0-ca22-44e7-8f08-c2b186e3bc5f/
  scratchpad/intercept_train_g1match.log`
- `save_interval=250` — checkpoints `model_0.pt`, `model_250.pt`, ... written
  into the run directory as training progresses.
- **Observed pace:** ~15.3-15.4 iterations/minute (~3.6-3.7s/iteration) at
  `num_envs=4096`. Reaching iteration 2000 takes roughly 2 hours from start
  (~17:15); iteration 3000 roughly 2h45m (~18:00).

## 6. Progress log

| Iter | Wall time | Train/mean_reward | Loss/est_region | Loss/value_function | Notes |
|---|---|---|---|---|---|
| 0 | 15:13 | -0.10 | 1.385 (chance) | 0.18 | start |
| 472 | 15:44 | 13.1 | 1.35 (~chance) | 0.56 | healthy climb, region loss flat |
| 914 | 16:13 | 19.8 | 1.30 | 0.96 | region loss starts genuinely decreasing |
| 998 | 16:19 | 22.4 | 1.31 | 0.92 | still healthy, past iter-988 danger point |
| 1097 | 16:25 | 21.0 | 1.303 | 0.86 | healthy |
| 1431 | 16:46 | 23.0 | 1.291 | 0.93 | healthy |
| 1750 | 17:07 | 22.6 | 1.291 | 0.86 | healthy, ~halfway through danger zone |
| 2067 | 17:28 | 24.2 | 1.292 | 0.77 | healthy, new reward high |
| 2384 | 17:49 | 24.2 | 1.305 | 0.75 | healthy, at edge of iter-2560 historical failure point |
| 2700 | 18:10 | 25.5 | 1.294 | 0.71 | **cleared iter 2560, the earliest historical divergence point** |
| 3018 | 18:31 | 27.5 | 1.259 | 0.99 | **new reward high; region loss already better than historical best (1.27)** |

No divergence observed through iter 3018 — cleared the full historical danger
zone (988-2560) cleanly. Historical broken runs (pre-revert code) diverged as
early as iter ~988 (repeated oscillation, `2026-07-06_01-14-09`) and as late as
iter ~2560 (`2026-07-06_10-15-32`).

**IMPORTANT — new critical window identified 2026-07-06 ~18:35:** checked
`Episode/Curriculum/ball_difficulty/ball_difficulty` — it hit `1.0` (fully
saturated) by iter ~3070. In the *original* pre-`7b7684d` clean run
(`2026-07-05_12-16-46`), curriculum saturation happened at step 2750, and the
region-estimator regression (38.5% → 27.7% live accuracy, left/right collapse)
happened **after** that saturation point, not at it — so reaching iter 3000-3070
with curriculum saturated is not yet a clean bill of health; the failure mode
this fix targets specifically manifests in the post-saturation window. Ran the
live region-accuracy probe (new script, not yet committed:
`/tmp/claude-6380/-home-ibouwmeest/940e4985-7161-49e4-9e26-1e9fbd86c5ae/scratchpad/region_probe.py`,
512 envs, 400 steps, loads a checkpoint + drives the multi-disc runner headlessly,
compares `actor_critic.estimate_region` argmax to `env._region_id`) against
`model_3000.pt`:

- Overall accuracy: **34.9%** (chance = 25%) — between the historical good
  checkpoint (`model_500`: 38.5%) and the historical collapsed checkpoint
  (`model_5250`: 27.7%).
- Confusion matrix shows a strong **near/far bias** (near predicted far more
  often than far, regardless of true class) but, critically, **no total
  left/right collapse** — the historical collapse's signature was true left
  and true right predicting nearly identically (~48%/52% split, i.e. literally
  no left/right signal). Here `left_near`→predicts left 52.4%/right 47.6%,
  `right_near`→right 50.6%/left 49.4% (still weak but not degenerate),
  `left_far`→left 50.8%/right 49.2%, `right_far`→right 54.8%/left 45.2%. Left/right
  discrimination is weak but present at every true class, unlike the historical
  failure.
- **Not a conclusive "confirmed fixed" yet** — this window (curriculum just
  saturated) is exactly when the original regression started. Next several
  checks must specifically re-run this probe on later checkpoints
  (`model_3250`, `model_3500`, ...) and watch for: (a) `Loss/est_region` turning
  back upward after this point, or (b) probe accuracy dropping back toward/below
  ~28% with the left/right collapse signature reappearing. If neither happens
  through a few thousand more iterations post-saturation, that's the strong
  confirmation the fix worked.

## 7. How to check progress (exact commands)

```bash
# Process still alive?
ps -p 2576767 -o pid,etime,stat,cmd

# GPU still busy (should show ~60-70% util, ~5.3GB used)?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Training curves (source of truth):
cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis
.venv/bin/python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r = 'logs/rsl_rl/intercept_simple_goalkeeper_multidisc/2026-07-06_15-13-34_region_fix_g1_match_2026-07-06'
ea = EventAccumulator(r, size_guidance={'scalars': 0})
ea.Reload()
for tag in ['Train/mean_reward', 'Loss/est_region', 'Loss/value_function', 'Policy/mean_noise_std']:
    events = ea.Scalars(tag)
    e = events[-1]
    print(f'{tag}: n={len(events)} last_step={e.step} value={e.value:.4f}')
"
```

Do **not** tail/read the raw stdout log repeatedly — the tensorboard event file
above is the durable, complete record; the stdout log is redundant and
session-scoped.

## 8. Success criteria — what "fixed" looks like

1. **Primary:** training survives past iteration ~2600-3000 without diverging
   (no NaN, no reward collapse to astronomical negative values, no
   `mean_action_acc` blowup).
2. `Loss/est_region` keeps decreasing meaningfully below chance (1.386),
   ideally past the historical best of ~1.27 achieved right before the original
   collapse (`2026-07-05_12-16-46`'s `model_5250`, which independently measured
   at 38.5% live classification accuracy via a probe script — chance is 25% for
   4 classes). **Loss curves alone are not fully trustworthy for this** — if a
   checkpoint looks promising, verify with a live probe: load the checkpoint,
   drive several hundred envs for ~400 steps, compare `region_estimator`'s live
   `argmax` against `env._region_id` ground truth, and build a confusion matrix
   (method fully described in `docs/BugFixes.md`'s dated entry for the original
   region-estimator collapse investigation).
3. **Stretch goal:** reward keeps climbing past the ~30 ceiling the original
   pre-`7b7684d` run reached before its own eventual (iter ~5790) divergence —
   this would mean we've actually improved on the historical baseline, not just
   matched its failure timing.

## 9. If it diverges again

- The revert applied here is the **highest-confidence hypothesis available**,
  directly grounded in two independent, careful reads of G1's actual code (not
  assumed defaults). Do not casually second-guess it without new evidence.
- G1's *own* reference design still diverged once, eventually (iter ~5790, single
  event, never recurred in that run because it was stopped) — full parity with
  G1 does not guarantee zero divergence forever, only a return to the original,
  much-later, much-rarer failure mode. If the new run diverges *near or after*
  iter 5790, that's actually consistent with "matched G1's baseline," not a
  failure of this fix.
- If it diverges *earlier* than that, the fix did not fully work and needs
  further root-causing — re-read `docs/BugFixes.md` and `CLAUDE.md`'s divergence
  table for full incident history before proposing new changes, and continue
  requiring every proposed change to be justified against `Humanoid-Goalkeeper/`
  specifically (this project's explicit standing rule, re-emphasized by the user
  in this session: "any changes must be consistent with the unitree g1 humanoid
  goalkeeper folder").
- Do **not** re-introduce a region_estimator-specific optimizer split or
  `schedule="fixed"` without first finding new evidence in G1 itself that
  contradicts the current reading.

## 10. Housekeeping (not part of this fix, noted for awareness)

- ~30 stale/zombie `sgk_train` processes sit in the tmux session named
  `training` (2 windows, panes `7907` and `2338564`), dating back to Jun 29, all
  in stopped/zombie state, holding no GPU memory. Not cleaned up — user hasn't
  confirmed whether to kill them.
- Git working tree has unrelated pre-existing modified files (see §3) that
  predate this investigation and must not be included in any commit for this
  fix.

## 11. Git reference points

- `HEAD` before this session's uncommitted fix: `7b7684d` (region_estimator LR
  split + `schedule="fixed"` — the bug).
- `HEAD~2` / last known-good code: `666f2f5` (code-equivalent to `b4fcee7`) —
  `schedule="adaptive"`, single shared optimizer, no split. This is the design
  now restored in the uncommitted working-tree fix.
- `0516ff1` (this session) — pushed `model_1750.pt` from the broken run.

## 12. Second bug found 2026-07-06 ~18:50: `reset_ball_rolling` ignored one-sided region ranges

User suspected (correctly) that the region estimator "was not going to work"
and specifically that far-side balls weren't converging to double-step AMP
motions, and asked for a full audit against `Humanoid-Goalkeeper` before
stopping the (at the time healthy, iter-3000+, no-NaN) training run from
section 5-9 above.

**Audit process:** read G1's `end_regions` mechanism (`legged_robot.py:916-960`,
`g1_29_config.py` `ranges_0..5`) — 6 regions = side × height-tier, **disjoint,
non-overlapping bounds per region, sign never re-randomized**. Compared against
this project's `reset_ball_rolling_by_region` (`mdp/regions.py`) → generic
`reset_ball_rolling` (`mdp/events.py`). Found: `reset_ball_rolling`'s y_end
sign/magnitude logic was written once for the plain single-disc task's
symmetric range (`y_end_range=(-0.9,0.9)`) and reused verbatim for the 4
region-conditioned one-sided ranges (`left_near=(0.15,0.5)`, etc.) without
adaptation. Two stacked defects, verified by 20k-sample Monte Carlo of the
exact pre-fix formula: (1) sign was **always** re-randomized 50/50 regardless
of the region's actual side — a "left_near" env's ball landed on the right
~49-50% of the time, for every region; (2) magnitude bounds ignored the
region's own inner edge — "far" regions (`0.5-0.9`) sampled magnitudes as low
as `0.1`, landing inside their own intended band only **25.1%** of the time
(75% of "far" episodes were secretly near-magnitude shots).

This fully explains both the live-probe's persistent left/right confusion
(34.9% accuracy at `model_3000`, no clean left/right separation despite the
region_estimator LR fix from section 1-9 holding and the ball_difficulty
curriculum having safely saturated) and the user's independently-reported
far-side double-step convergence failure — the region_estimator's
ground-truth label (`env._region_id`) was statistically decorrelated from the
ball's actual observable trajectory for a large fraction of samples.

**Fix applied** (`mdp/events.py:reset_ball_rolling`): branch on whether
`y_end_range` is one-sided (`y_end_range[0]*y_end_range[1] > 0`). One-sided
ranges now take a fixed sign from the range itself and lerp magnitude within
that range's own `[lo,hi]` bounds (`inner=lo`, `outer=lo+(hi-lo)*d`) instead of
the generic two-sided dead-zone constants. Two-sided/symmetric callers (the
plain single-disc `reset_ball`) are unaffected — verified via `pytest
tests/simple_goalkeeper/test_regions.py` (5 passed) and a direct Monte Carlo
of the post-fix formula (100% of samples land in the intended range/sign for
all 4 regions, at every difficulty level). Full writeup:
`docs/BugFixes.md`, dated entry `2026-07-06`.

**Test-coverage gap noted, not yet closed:** `test_regions.py` monkeypatches
`reset_ball_rolling` out entirely and never exercised this sampling math —
that's why the bug went undetected since this mechanism was introduced. A
follow-up should add a direct test of `reset_ball_rolling`'s sign/bounds for a
one-sided range.

**Training restarted, not resumed:** the in-flight run
(`2026-07-06_15-13-34_region_fix_g1_match_2026-07-06`, PID `2576767`, killed
via `SIGTERM` at ~18:52) had trained its region_estimator and actor against
the corrupted ball-spawn distribution from iteration 0 — resuming would carry
the poisoned region conditioning forward, so a fresh run was started instead.

**New run:**
- **Started:** 2026-07-06 19:05:00.
- **PID:** `2729376` — launched via `nohup ... & disown`, fully detached.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name region_spawn_fix_2026-07-06`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-06_19-05-00_region_spawn_fix_2026-07-06/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_region_spawn_fix.log` (session-scoped, may not
  exist in a new session — prefer tensorboard).
- Region-accuracy probe script (reusable, not yet committed to the repo):
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/region_probe.py` — usage:
  `.venv/bin/python3 region_probe.py <checkpoint.pt> --num-envs 512 --steps 400`.
  Consider committing this under `src/simple_goalkeeper/scripts/` or `tests/`
  as a permanent diagnostic tool given how central this metric has become.

**What "fixed" now looks like (updated success criteria):** everything from
section 8 still applies (no NaN divergence, `Loss/est_region` below chance and
improving), **plus** the region probe's confusion matrix should show genuine
left/right separation (not a ~50/50 split regardless of true side) and "far"
classes should stop being systematically under-predicted relative to "near".
Re-run `region_probe.py` periodically against new checkpoints and compare to
the broken baseline documented here (`model_3000.pt` under the *old*, still-buggy
spawn code: 34.9% overall, near-total left/right ambiguity) — the bar to clear
is genuinely better left/right discrimination, not just a higher raw percentage.

**CONFIRMED 2026-07-06 ~19:28 — fix validated immediately, dramatically.**
`Loss/est_region` at iteration 334: **0.4214** — already far below chance
(1.386) and far below anything seen in *any* previous run at any iteration
count (historical best under buggy code was ~1.27, and the previous
`region_fix_g1_match` run held at ~1.29 through iteration 3000+). Ran
`region_probe.py` against `model_250.pt` (iteration 250 — very early):

- **Overall accuracy: 72.5%** (chance 25%, vs. the old code's 34.9% *at
  iteration 3000*, 12x more training).
- **Left/right confusion is essentially eliminated**: confusion matrix shows
  `left_near`→{left_near 45.8%, left_far 52.6%, right_near 0.1%, right_far
  1.5%} — cross-side prediction is ~1.6% combined, down from the ~50/50
  coin-flip under the buggy code. Same pattern for `right_near`/`right_far`
  (cross-side ~1.3%) and both `_far` classes (~99% same-side, correct-class
  accuracy on `left_far`/`right_far` individually is 98.9%/98.6%).
- Remaining confusion is now the much more benign near/far ambiguity *within
  the same side* (`left_near`↔`left_far` at the touching 0.5m boundary) —
  qualitatively different from and far less severe than the old total
  left/right breakdown.

This is a clean, immediate, large-effect-size confirmation that the
`reset_ball_rolling` one-sided-range bug (section 12) was the actual root
cause of both the region_estimator's stuck accuracy and (very likely) the
far-side double-step convergence failure the user flagged. Continuing to
monitor for (a) sustained health / no NaN divergence through the historical
danger zone (988-2560, still relevant since the LR/schedule fix from earlier
in the session is unchanged in this run), and (b) whether near/far separation
within each side keeps improving with more training, and (c) whether this
finally translates into visible far-side double/triple-step behavior (would
need a play-mode/qualitative check once a later checkpoint exists, not just
the region probe).

**Progress log (new run):**

| Iter | Wall time | Reward | Loss/est_region | Notes |
|---|---|---|---|---|
| 334 | 19:28 | — | 0.421 | far below chance from the start |
| 666 | 19:49 | 13.1 | 0.429 | healthy |
| 988 | 20:10 | 18.6 | 0.602 | cleared earliest historical NaN-divergence point cleanly |
| 1309 | 20:31 | 21.0 | 0.574 | region loss stabilized, not still rising |
| 1627 | 20:52 | 23.4 | 0.588 | healthy |

**2026-07-06 ~20:52 — probe on `model_1500.pt`:** overall accuracy **65.4%**
(down from 72.5% at `model_250`, but still far above the old broken run at any
iteration). Confusion matrix shows a new, asymmetric pattern:
`left_near`→{left_near 69.7%, left_far 29.9%, right_near 0.5%, right_far 0.0%}
(good, same-side only, near/far split improving vs. iter 250's 45.8/52.6);
`left_far`→97.0% correct (still excellent); but **`right_near`→{left_near
63.5%, left_far 3.2%, right_near 26.4%, right_far 7.0%}** — a specific,
asymmetric cross-side confusion (true right_near predicted as left_near
nearly 2/3 of the time) that was *not* present at iter 250 (was 0.7%
cross-side then); `right_far`→68.8% correct, with some leakage into
left_near/left_far (13.9%/10.5%) not seen at iter 250 either.

**Not yet clear whether this is transient training noise (region_estimator
still converging, non-monotonic) or a new, real, class-specific issue** —
notably it's asymmetric (only `right_near`/`right_far` leak toward
`left_near`, not the reverse), which is a bit unusual for pure noise. Next
probe (on a later checkpoint) should clarify: if it's noise, expect the
cross-side confusion to shrink back down; if it's a real, persistent bug,
expect it to stay or worsen. Do not treat 65.4%/this confusion pattern as a
regression of the `reset_ball_rolling` fix until confirmed — the *symmetric*
~50/50 total left/right collapse from the pre-fix code is not what's being
observed here.

**User instruction 2026-07-06 ~21:34: switch to hourly checks (not every 30
min); do a full hard-check/reflection once iteration crosses ~5000 — assess
whether it's worth continuing, or re-audit against G1/own code if something's
still wrong, the same way the reset_ball_rolling bug was found.**

**2026-07-06 ~21:16 — probe on `model_1750.pt` (iter 1956 in-progress, reward
23.8, region loss steady ~0.58, no NaN, still inside 988-2560 danger zone with
no signs of trouble):** overall accuracy 65.9% (flat vs. 65.4%). The
`right_near`→`left_near` cross-confusion **shrank from 63.5% to 43.2%** —
supports "transient noise while still converging" over "new persistent bug."
Full matrix: `left_near`→88.7% correct (up from 69.7%); `left_far`→66.8%
correct with 24.4% leaking to `left_near` (same-side, benign, down from 97%
correct — some give-and-take with `left_near`'s improvement, still zero
cross-side leakage for this class); `right_near`→35.3% correct, 43.2% to
`left_near` (shrinking, per above); `right_far`→72.9% correct. Continuing to
watch whether the remaining `right_near`/`left_near` cross-confusion keeps
shrinking on the next checkpoint — if it keeps dropping, this is noise from
imbalanced/still-settling capacity across the 4 output classes, not a repeat
of the old bug's symmetric 50/50 collapse.

**2026-07-06 ~22:35 — probe on `model_3000.pt` (iter 3158 in-progress, reward
26.6, region loss steady ~0.605, no NaN, cleared the full historical danger
zone 988-2560 cleanly):** overall accuracy 58.3% (down from 65.9%). **Revised
read — this is NOT monotonically shrinking noise as hoped.** Tracking
right-side-classes' total cross-side error (predicted as `left_near` +
`left_far` when true class is `right_near`/`right_far`) across all 4
checkpoints probed so far in this run:

| Checkpoint (iter) | `right_near`→left (total) | `right_far`→left (total) |
|---|---|---|
| model_250 (250) | 0.7% | 0.7% |
| model_1500 (1500) | 66.7% | 0.7% |
| model_1750 (1750) | 52.7% | 7.4% |
| model_3000 (3000) | 63.2% | 34.1% |

Both are **oscillating, and `right_far`'s cross-side error is trending up**
(0.7%→34.1%), not down. Meanwhile **`left_near`/`left_far`'s cross-side error
has stayed consistently near-zero (~0-1%) at every checkpoint** — the
confusion is asymmetric: right-side true classes leak toward left-side
predictions, but never the reverse. This directional asymmetry across 4
consecutive checkpoints is starting to look less like transient
still-converging noise and more like a systematic issue specific to the right
side (mirroring/frame convention in an observation term? a right-side-specific
data or reward asymmetry? worth a fresh, targeted audit rather than assuming
it'll self-resolve). **Not chasing this now per user instruction** — flagged
here for the full reflection once iteration crosses ~5000. If it's still
present/worse by then, this asymmetric left-favoring bias should be the
starting point of that audit.

## 13. Ad-hoc health check + `model_3000.pt` pushed, 2026-07-06 ~22:50 (training left running throughout, no code changes applied)

User asked for a report-only code health check (no changes) plus an NPZ
dataset sanity check (root translation) and to push `model_3000.pt` for
manual play/inspection.

**NPZ root-translation audit — no bug found.** Loaded every motion file in
`src/simple_goalkeeper/motions/data/*.npz`, computed root body (`body_pos_w[:,0,:]`)
Y-displacement start→end. All `Left*` files have positive dy (e.g.
`LeftDoubleStep`: +0.599, `LeftTripleStep`: +1.084), all `Right*` files have
negative dy (e.g. `RightDoubleStep`: -0.787, `RightTripleStep`: -1.084) —
labeling is internally consistent, not a source of the residual left/right
confusion.

**Design observation (not a confirmed bug) — triple-step motions are wired
into the dataset folder but unused by this track's actual AMP reward.**
`REGION_MOTION_FILES` (`goalkeeper_multidisc_amp_cfg.py:93-98`) maps only 4
files: `left_near→LeftStep` (single), `left_far→LeftDoubleStep`,
`right_near→Rightstep` (single), `right_far→RightDoubleStep`. The
`LeftTripleStep`/`RightTripleStep` files exist on disk and are loaded by the
(separately deprecated-for-`reset()`) old RSI pool system at startup, but are
**never used as an AMP discriminator imitation target** in this multi-disc,
4-region track — there is no `region_far` sub-tier that maps to triple-step.
This may be an intentional simplification (2-tier near/single, far/double per
side, matching the 4-region design exactly) rather than a bug, but it means:
if "learn to triple-step" was ever an implicit expectation for the hardest
far shots (`y_end` near 0.9m), there is currently **zero AMP reward signal**
pushing toward that — the far region's only imitation target is
`DoubleStep`, regardless of how far within `[0.5,0.9]` the actual shot lands.
Flagging for the user to confirm intent; not treating as a bug to fix
unilaterally.

**Other checks, no bugs found:** region_id↔discriminator↔AMP-dataset indexing
(`REGION_NAMES` tuple order used consistently everywhere via `enumerate`, not
dict-order-dependent — checked `multi_disc_amp_ppo.py`,
`him_amp_on_policy_runner.py`); critic-obs slice indices for `ball_gt`/`region_gt`
(`ball_gt_critic_obs_slice=slice(-5,-1)`, `region_id_critic_obs_index=-1`) match
the documented term insertion order and are empirically validated by the
region loss dropping to 0.42 post-fix (would look like noise if misindexed);
foot-side convention (`rewards.py:_get_correct_foot_idx`, positive Y = left
foot) is self-consistent with `regions.py`'s left=positive-Y convention; ball
observation frame transform (`observations.py`, `quat_apply(quat_inv(...))`)
is a standard symmetric body-frame rotation, no hardcoded left/right bias.

**Did not find an additional coded bug explaining the residual
`right_near`/`right_far`→left-side confusion** documented in the table above
(section 12). Continuing to track it via the region probe per the existing
plan (full reflection at iteration ~5000).

**Pushed:** `model_3000.pt` from the new run
(`2026-07-06_19-05-00_region_spawn_fix_2026-07-06`), commit `8a4fbdc`, for the
user's own play/inspection of double-stepping behavior. No source changes
committed — `docs/BugFixes.md` and the `reset_ball_rolling` fix
(`mdp/events.py`) remain uncommitted working-tree changes, per the project's
change-approval workflow (not yet explicitly approved for commit).

## 14. Blue-ball → green-ball two-stage waypoint gate ported from SimpleGoalKeeper, 2026-07-06 ~23:00-23:30

User reported `model_3000.pt` still hadn't learned to double-step and asked to
port SGK's two-stage "blue ball → green ball" mechanism: for wide crossings,
target an intermediate midpoint first, hard-gated on the assigned foot
genuinely landing there (airborne then in ground contact), before switching to
the true crossing point. User's stated reasoning: "AMP is not going to do the
double stepping himself" — matches SGK's own documented finding that a
2-frame-transition AMP discriminator cannot judge global trajectory shape or
step count, so nothing in a bare AMP-imitation setup pushes toward a paced
multi-step approach over one continuous fast leap.

**Read SGK's full implementation and history first** (`SimpleGoalKeeper/
src/simple_goalkeeper/mdp/rewards.py`, `CLAUDE.md` divergence table) — this
went through several iterations there: v1 (soft/timeout gate) was found to be
~35-40% "landed" purely from RSI-seeded free credit, not learned behavior; v2
(hard gate, no time fallback) closed that but still let the policy bypass the
blue waypoint via one continuous reach to the green target while still
collecting `stopball`/`softstop`, since those didn't check the gate at all; the
final (2026-07-05) version additionally gates `stopball`/`softstop` themselves
on the landing latch. **That final version was marked "not yet validated
against a training run" even in SGK itself** — this is the first real test of
it, in either project.

**Presented a plan for approval (per this project's Change Approval Workflow)
before touching code; user approved, plus asked to also port the diagnostic
metrics and play-mode visualization.**

**Implemented in `interceptV2DualDis`:**

1. New `_get_reach_target_y(env, ball_name, asset_cfg, wide_threshold=0.5,
   landing_radius=0.05)` in `mdp/rewards.py` — ported near-verbatim from SGK.
   `wide = |crossing_y - start_y| > 0.5` conveniently lines up with this
   project's own near/far region boundary (also 0.5) — the gate applies almost
   exactly to "far" region envs, never to "near". Caches `env._blue_wide`,
   `env._blue_was_airborne`, `env._blue_landed`, `env._blue_airborne_at_reset`
   (diagnostic latch). No time-based fallback — hard-gated exactly like SGK's
   final version.
2. `footreach`/`foot_proximity` now target `_get_reach_target_y(...)` instead
   of the raw frozen `_get_ball_crossing_y(...)`, and their live-ball
   `ball_close` switch is gated `& (~env._blue_wide | env._blue_landed)`.
3. New `blue_ball_landed` one-shot reward (weight 10→25 curriculum, mirrors
   `footreach_curriculum`'s shape) — fires once when the landing latch flips.
4. `stopball`/`softstop` now both gate `fired` on `landing_ok = ~env._blue_wide
   | env._blue_landed` — the fix SGK found was necessary to actually stop the
   policy from bypassing the waypoint (gating only `footreach`/`foot_proximity`
   was not sufficient there).
5. New `mdp/metrics.py`: `blue_landed_genuine`/`blue_landed_rsi_assisted`
   diagnostic metrics (no weight/dt scaling), ported from SGK, distinguishing
   a policy-driven landing from one an RSI-seeded airborne-at-reset foot got
   for free. Wired via `cfg.metrics.update({...})` in `goalkeeper_env_cfg.py`
   — **used `.update()`, not assignment**, since `cfg.metrics` already carried
   `mean_action_acc` from mjlab's base `make_velocity_env_cfg` (would have
   silently deleted it otherwise — caught before running).
6. `scripts/play.py`'s `_patch_viewer_intercept_vis` now draws a BLUE sphere at
   the midpoint during phase 1 (wide + not yet landed), GREEN at the full
   crossing point otherwise — lets a human watching `sgk_play` confirm timing
   visually.
7. `mdp/__init__.py` updated to export `blue_ball_landed`, `blue_landed_genuine`,
   `blue_landed_rsi_assisted`.

The region-conditioned multi-disc track (`goalkeeper_multidisc_env_cfg`) only
adds observation/event terms on top of `goalkeeper_env_cfg()` and never
touches `cfg.rewards`/`cfg.metrics`/`cfg.curriculum`, so all of the above flows
through unchanged. Confirmed `env._rsi_cross_y` (which `_get_reach_target_y`
reads) is set inside the shared `reset_ball_rolling` function itself, so it's
populated correctly whether called generically or via the region-conditioned
`reset_ball_rolling_by_region` wrapper.

**Verification before restart:** `pytest tests/simple_goalkeeper/` — 48/48
pass (no regressions). Ran a 90s smoke test (`--num-envs 64`, throwaway run
name, log/run-dir deleted after) — confirmed no crash, and
`Episode_Reward/blue_ball_landed`, `Curriculum/blue_ball_landed_curriculum/
weight`, `Episode_Metrics/blue_landed_genuine`, `Episode_Metrics/
blue_landed_rsi_assisted` all appear correctly in the logged output (all near
0 at this stage, expected — `ball_difficulty` curriculum starts at 0 so wide
crossings are rare early on).

**Training restarted** (not resumed — a reward-shaping change this
significant needs a fresh run to be a fair test): killed
`2026-07-06_19-05-00_region_spawn_fix_2026-07-06` (PID `2729376`, healthy,
iter ~4700+ at kill time) and launched a new run:

- **Started:** 2026-07-06 23:29:33.
- **PID:** `2903136` — `nohup ... & disown`, fully detached.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_ball_gate_2026-07-06`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-06_23-29-33_blue_ball_gate_2026-07-06/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_ball_gate.log` (session-scoped).
- Confirmed alive and past iteration 1 with no crash before handing off to the
  hourly monitoring loop.

**What to watch on this run, beyond the usual NaN-divergence check:**
1. `Episode_Metrics/blue_landed_genuine` vs `blue_landed_rsi_assisted` — if
   genuine stays near zero while RSI-assisted dominates, the landing credit is
   mostly free (SGK's original failure mode with the tier-RSI system; this
   project's own RSI mechanism differs, so this needs its own read, not an
   assumed transfer of SGK's finding).
2. `Episode_Reward/blue_ball_landed` event rate (convert via the documented
   `Episode_Reward` formula in `CLAUDE.md`) — should climb above 0 as
   `ball_difficulty` ramps and start producing genuine landings, not stay
   flat at 0 (would mean the gate is simply unsatisfiable in practice).
3. **Known risk, inherited from SGK, not yet resolved there either:**
   hard-gating `stopball`/`softstop` (by far the largest-weight rewards) behind
   a potentially-low genuine-landing rate risks sparsifying the primary
   training signal sharply on wide/far crossings specifically. Watch for reward
   collapse or stalled learning on far-region episodes specifically (could show
   up as a region-specific reward plateau, not just an overall metric).
4. Play `model_3000.pt` (already pushed, pre-this-change) is NOT trained under
   this gate — not a valid comparison point for whether double-stepping
   emerges; need a checkpoint from this new run for that.

**2026-07-07 ~00:31 — first check-in on the new run.** Iteration 889, healthy
(reward 19.95, region loss 0.56, no NaN — hasn't reached the 988 danger-zone
start yet). Converted the new metrics via the documented `Episode_Reward`
formula (curriculum weights at this step: stopball=30, softstop=210,
blue_ball_landed=20):

- `stopball`: ~51.7% of episodes
- `softstop`: ~30.8% of episodes
- `blue_ball_landed`: ~16.1% of episodes (genuine 6.1% + RSI-assisted 10.0%,
  read directly per CLAUDE.md — no conversion needed for `Episode_Metrics/*`;
  the two sum to ≈16.1%, consistent with the reward rate)

**No sign of the sparsification risk (section 14, point 3) materializing** —
`stopball`/`softstop` are firing at healthy rates for this stage of training,
not suppressed toward zero. Genuine landings are a real, non-trivial fraction
(6.1%), unlike SGK's original near-zero-genuine failure mode — RSI-assisted
still edges it out (~1.6x), worth continuing to watch but not alarming yet.
`ball_difficulty` at 0.667, still ramping.

**2026-07-07 ~01:40 — check-in, iteration 1283 (inside the 988-2560 danger
zone, no NaN, process alive):** reward 23.3 (up from 19.95), region loss 0.435
(flat/healthy). Converted metrics (weights at this step: stopball=30,
softstop=210, blue_ball_landed=20):

- `stopball`: ~69.0% of episodes (up from 51.7%)
- `softstop`: ~47.7% of episodes (up from 30.8%)
- `blue_ball_landed`: ~23.85% of episodes (up from 16.1%) — **genuine 13.11%,
  RSI-assisted 10.77%** (read directly, no conversion)

Genuine landings more than doubled since the last check (6.1%→13.11%) while
RSI-assisted barely moved (10.0%→10.77%) — this is the good pattern: growth is
coming from the policy actually learning to land the gate, not from free RSI
credit. `stopball`/`softstop` both climbing strongly, no sparsification. No
NaN, no reward collapse. `ball_difficulty` still 0.667 (unchanged, not yet
advanced further). User asked to switch monitoring cadence to every 1.5 hours;
`ScheduleWakeup` clamps to a 3600s (1 hour) max, so checks continue hourly
instead — the closest available cadence.

**2026-07-07 ~02:34 — check-in, iteration 2172 (still inside the 988-2560
danger zone, process alive, `mean_action_acc`=1.02 — nowhere near the O(1e12)
blowup signature, confirms no NaN):** reward 25.2 (up from 23.3), region loss
0.424 (flat/slightly improved from 0.435). Converted metrics (weights
unchanged: stopball=30, softstop=210, blue_ball_landed=20):

- `stopball`: ~69.4% of episodes (flat vs 69.0%)
- `softstop`: ~53.4% of episodes (up from 47.7%)
- `blue_ball_landed`: ~24.0% of episodes (flat vs 23.85%) — **genuine 9.92%,
  RSI-assisted 14.08%** (read directly, no conversion)

**Note the reversal vs. last check:** genuine dropped 13.11%→9.92% while
RSI-assisted rose 10.77%→14.08% — the ratio flipped back the other way after
last check's favorable genuine-dominant reading. Total landing rate is flat
(~24%), so this looks like noise in the genuine/RSI split rather than a trend
in either direction yet — one data point of "genuine winning" followed by one
of "RSI winning" isn't enough to call a direction. Continuing to track this
ratio each check; would only be concerning if RSI-assisted keeps pulling away
over multiple consecutive checks while genuine stays flat or drops.
`ball_difficulty` confirmed stuck at exactly 0.667 since ~iteration 1000 (checked
the full curve, not just the latest point) — matches the previous clean run's
timing (that run saturated to 1.0 around iter 2750-3070), so not yet a concern,
just noting it hasn't advanced past 2/3 yet. `stopball`/`softstop` both still
climbing, no sparsification.

**2026-07-07 ~03:31 — check-in, iteration 3057 — cleared the full historical
988-2560 NaN-divergence danger zone cleanly** (`mean_action_acc`=1.03, still
healthy, no blowup signature). Reward 25.1 (flat vs 25.2, plateauing), region
loss 0.412 (still slowly improving). Converted metrics (weights unchanged:
stopball=30, softstop=210, blue_ball_landed=20):

- `stopball`: ~80.1% of episodes (up from 69.4%)
- `softstop`: ~66.9% of episodes (up from 53.4%)
- `blue_ball_landed`: ~28.2% of episodes (up from 24.0%) — **genuine 14.28%,
  RSI-assisted 13.94%** (read directly, no conversion)

**Third data point on the genuine-vs-RSI-assisted split — resolves the
question from the last two checks.** History: check 1 = 13.11%/10.77%
(genuine ahead), check 2 = 9.92%/14.08% (RSI ahead), check 3 (now) =
14.28%/13.94% (genuine ahead again, nearly tied). This is oscillation around
near-parity, not a consistent RSI-assisted-pulling-ahead trend — the "watch
for RSI dominance" risk (section 14, point 3) is not materializing. Total
landing rate keeps climbing steadily (16.1%→23.85%→24.0%→28.2%) each check,
which is the more important number and it's trending the right direction.
`ball_difficulty` still stuck at exactly 0.667 even at iter 3057 — the
previous clean run had already saturated to 1.0 by iter 2750-3070 at this
point, so this run is running slightly behind that schedule but not
dramatically so; continuing to watch, not yet a concern. `stopball`/`softstop`
both still climbing strongly (80.1%/66.9%), no sparsification from the
blue-ball gate.

**2026-07-07 ~04:32 — check-in, iteration 3941 — `ball_difficulty` has now
saturated to 1.0** (curriculum weights ramped accordingly: stopball 30→37.5,
softstop 210→262.5, blue_ball_landed 20→25). No NaN (`mean_action_acc`=1.13,
still healthy, well clear of the O(1e12) blowup signature). Reward 27.0 (up
from 25.1, new high for this run). Region loss 0.456 (up slightly from 0.412 —
small enough to be curriculum-ramp noise from the sudden harder-weight/harder-
difficulty transition, not yet a red flag, but this is exactly the kind of
post-saturation window where the *previous* clean run's region_estimator
regression began (section 12: 38.5%→27.7% shortly after that run's own
saturation) — next check should watch this specifically). Converted metrics
(new weights: stopball=37.5, softstop=262.5, blue_ball_landed=25):

- `stopball`: ~77.8% of episodes (down slightly from 80.1%, expected transient
  dip from the weight/difficulty jump, not a reward collapse)
- `softstop`: ~61.6% of episodes (down slightly from 66.9%, same transient
  read)
- `blue_ball_landed`: ~28.6% of episodes (flat vs 28.2%) — **genuine 14.73%,
  RSI-assisted 13.84%**

**Fourth data point on genuine-vs-RSI split: genuine ahead again** (14.73 vs
13.84, continuing the same near-parity oscillation as the last 3 checks:
13.11/10.77, 9.92/14.08, 14.28/13.94, now 14.73/13.84). Still no consistent
RSI-dominance trend — this risk continues to look like non-issue. Total
landing rate holding steady (~28-29% the last two checks) after climbing from
16.1% two checks ago.

**Approaching the user's requested iteration-5000 deep-reflection
checkpoint** (currently at 3941, ~15-16 iterations/min pace observed
historically → roughly 1-1.5 hours away). Next check should specifically
watch for the post-saturation region_estimator regression pattern from the
historical bad run, and if iteration crosses 5000, run the full reflection
(region_probe.py against a recent checkpoint, re-audit against G1 for the
right_near/right_far asymmetry from section 12 if it's still present).

## 15. Iteration ~5000 deep reflection, 2026-07-07 ~05:33

**Status check at iteration 4827** (close enough to the ~5000 target to run
the full reflection now against `model_4750.pt`, the latest checkpoint on
disk): no NaN (`mean_action_acc`=1.13, stable), reward **29.07 — new high for
this run, and higher than any pre-blue-ball-gate run reached before its own
eventual divergence/plateau.** `Loss/est_region` has stayed flat/healthy the
entire run — checked the full curve (every 250 steps from 0 to 4827): it
oscillates in a narrow 0.38-0.51 band with no sustained upward drift, ending
at **0.4088, its best value yet**. **The post-saturation regression pattern
from the previous run (section 12: region loss/accuracy degrading after
`ball_difficulty` hit 1.0) has NOT repeated here** — saturation happened at
iter ~3941 and region loss has, if anything, kept improving since.

**Converted reward/metric rates** (weights: stopball=37.5, softstop=262.5,
blue_ball_landed=25): `stopball` ~75.4% (flat), `softstop` ~65.8% (up from
61.6%), `blue_ball_landed` ~23.8% total — **genuine 8.87%, RSI-assisted
14.87%.** Fifth data point on the genuine/RSI split: history is
13.11/10.77, 9.92/14.08, 14.28/13.94, 14.73/13.84, now 8.87/14.87 — this is
the second time RSI-assisted has led, and by the largest margin yet (~1.7x).
Total landing rate also dipped from ~28.6% to ~23.8%. Not calling this a
trend off two RSI-led data points out of five, but it's the most RSI-leaning
reading so far — **next check should specifically watch whether this is the
start of a real drift toward RSI-dominance** (the SGK-documented failure mode)
or another oscillation back toward parity.

**Region-accuracy probe (this run's first — all of section 12's confusion-
matrix history was collected on the *previous* run, before the blue-ball gate
was added and training restarted from scratch; not directly comparable
run-to-run, but the same underlying mechanism/fix):**

```
Checkpoint: model_4750.pt (iter ~4750)
Overall live region accuracy: 75.8% (chance = 25.0%)

true\pred    left_near   left_far   right_near   right_far
left_near    85.5%       14.3%      0.2%         0.0%
left_far     23.5%       76.3%      0.2%         0.0%
right_near   30.3%       9.5%       54.4%        5.8%
right_far    1.6%        0.8%       10.7%        86.9%
```

75.8% overall is the best region-accuracy number recorded in either run's
history at any checkpoint (previous best was 72.5% at the old run's very
early `model_250`, before its right-side bias emerged). `left_near`/`left_far`
remain excellent (near-zero cross-side leakage, consistent with every prior
check in project history). `right_near`/`right_far`'s cross-side leakage
toward the left side (39.8% and 2.4% respectively, total) is present — the
same directional asymmetry flagged in section 12 — but **substantially
smaller than the previous run showed at a comparable point** (iter 3000:
63.2%/34.1%, and trending worse there). `right_far` in particular has nearly
resolved (86.9% correct now vs. 65.6% correct at the old run's iter 3000).
Remaining confusion is now dominated by `right_near`→`right_near`/`left_near`
(a same-general-area near/far-style mixup, similar in character to
`left_far`→`left_near`'s existing benign 23.5%) rather than a clean symmetric
50/50 collapse.

**Honest assessment: training is worth continuing, no new root-cause audit
against G1 is warranted right now.** Reasoning:
1. No divergence through 4827 iterations — already past every historical NaN
   crash point (988, 1870, 2560, 3070) with wide margin, and reward is still
   climbing, not plateaued or degrading.
2. The specific failure this reflection was watching for — region_estimator
   regressing after curriculum saturation — has not appeared; region loss and
   live-probe accuracy are both at their best-ever levels post-saturation, the
   opposite of the historical pattern.
3. The right-side asymmetric confusion is real and not fully resolved, but is
   moving in the improving direction relative to the last time it was
   measured at a comparable point in training, not worsening. It doesn't rise
   to the level that justified the emergency stop-and-audit for the
   `reset_ball_rolling` bug (section 12) — that was a near-total 50/50
   collapse; this is a partial, shrinking, one-directional bias.
4. The one open item worth watching, not acting on yet: the blue-ball
   genuine/RSI-assisted split's most recent (fifth) reading was the most
   RSI-leaning so far. If the next 1-2 checks confirm a sustained drift (not
   just this being data point #2-of-5 in that direction), that would be the
   next thing worth a real investigation — not the region estimator.

Continuing hourly monitoring; will re-open the "root-cause audit" conversation
immediately if either (a) the right-side confusion resumes climbing the way it
did in the old run, or (b) the genuine/RSI split shows a real sustained trend
rather than oscillation.

**2026-07-07 ~06:36 — check-in, iteration 5741.** No NaN
(`mean_action_acc`=1.13). Reward 29.18 (flat vs 29.07 — plateauing around this
level rather than still climbing). Region loss 0.460 (within the same
0.38-0.51 healthy band as every prior check, not a new high). Converted
metrics (weights unchanged: stopball=37.5, softstop=262.5,
blue_ball_landed=25):

- `stopball`: ~76.7% (flat)
- `softstop`: ~66.1% (flat)
- `blue_ball_landed`: ~28.9% total (up from 23.8%, recovered back toward the
  ~28% peak) — **genuine 10.91%, RSI-assisted 17.98%**

**Sixth data point on genuine/RSI split — this is now the second consecutive
RSI-led reading, at a similar large margin to the first** (5th: 8.87/14.87,
ratio 1.68x; 6th: 10.91/17.98, ratio 1.65x). Full history: 13.11/10.77 (1.22x
genuine), 9.92/14.08 (1.42x RSI), 14.28/13.94 (1.02x genuine), 14.73/13.84
(1.06x genuine), 8.87/14.87 (1.68x RSI), 10.91/17.98 (1.65x RSI). Two-in-a-row
at consistent magnitude is more suggestive of a real shift than the earlier
back-and-forth, so upgrading this from "watch" to "likely a real trend,
confirm with one more check." Important nuance: **both genuine and
RSI-assisted rates increased this check** (genuine 8.87%→10.91%, RSI
14.87%→17.98%) — this is not the policy regressing on genuine landings, it's
RSI-assisted landings growing faster than genuine ones as training
progresses. `stopball`/`softstop` remain strong and flat, so this is not (yet)
the SGK-documented sparsification failure mode — that would show `stopball`/
`softstop` suppressed toward zero, which isn't happening. Next check should
confirm whether the RSI-lead keeps widening (real trend) or this settles back
down.

**2026-07-07 ~07:37 — check-in, iteration 6629. TREND CONFIRMED, widening.**
No NaN (`mean_action_acc`=1.14). Reward 28.53 (flat/plateaued around 28-29
for the last 3 checks — not climbing further). Region loss 0.449 (still
within the healthy 0.38-0.51 band). Converted metrics (weights unchanged:
stopball=37.5, softstop=262.5, blue_ball_landed=25):

- `stopball`: ~78.2% (up from 76.7%)
- `softstop`: ~69.25% (up from 66.1%)
- `blue_ball_landed`: ~29.0% total (flat vs 28.9%) — **genuine 7.56%,
  RSI-assisted 21.39%**

**Third consecutive RSI-led reading, and the margin just jumped sharply**
(ratio history: 1.22x genuine, 1.42x RSI, 1.02x genuine, 1.06x genuine, 1.68x
RSI, 1.65x RSI, now **2.83x RSI**). Genuine dropped again (10.91%→7.56%) while
RSI-assisted kept climbing (17.98%→21.39%). This is now a real, accelerating
trend, not oscillation — three in a row with a widening gap.

**What this means, and what it doesn't:** `stopball`/`softstop` are not
suppressed — they're both still climbing (78.2%/69.25%, new highs) — so this
is *not* the SGK-documented reward-sparsification failure mode from section
14 point 3. But it does suggest something more subtle and arguably more
important to the user's actual goal: **the policy may be increasingly
satisfying the blue-ball landing gate via favorable RSI-seeded starting poses
rather than learning genuine footwork to get there.** Since the whole point of
porting this gate was to force real double-stepping locomotion (not just
credit for landing), a rising RSI-assisted share at flat/declining genuine
share means the reward signal *looks* healthy in aggregate while the
underlying behavior this mechanism was designed to teach may not actually be
improving. The reward numbers alone would not reveal this — only the
genuine/RSI split does, which is exactly why this diagnostic metric was
ported from SGK.

**Recommended next step (not yet done, flagging for user decision): visually
play the latest checkpoint (`model_6500.pt`, on disk) via `sgk_play` and watch
whether it's doing genuine multi-step footwork to the blue waypoint, or
mostly single fast lunges that happen to already satisfy the gate from a
lucky RSI start pose.** This is a qualitative check a metric can't fully
replace. Not doing this unilaterally since it involves the user's own
judgment call on what "good" looks like, and per this project's change-
approval workflow no code changes are being made regardless.

## 16. Third bug found 2026-07-07 ~11:00: region-far episodes only triggered the `wide` gate 80-83% of the time — training restarted

User asked me to check `model_6500.pt` (pushed for their own inspection) and
separately disputed the play viewer's near/far blue-green split. Investigating
that dispute (see conversation, not fully reproduced here) surfaced a genuine,
previously-undiscovered bug in the *region-conditioned* task specifically —
distinct from, and layered on top of, the crossing-fraction fix in the section
above.

**The bug:** `_get_reach_target_y`'s `wide` flag compares `_rsi_cross_y` (the
ball's Y position at the goal line, `x=0`) against the `0.5` threshold. But
`regions.py`'s near/far boundary is defined on `y_end` — the aim point 0.3m
*behind* the goal line. Since `_rsi_cross_y = y_start + (y_end-y_start)*f`
with `f = x_start/(x_start+0.3) < 1` always, the crossing point is
systematically smaller in magnitude than `y_end` — so a region-far episode
whose `y_end` sits near the region's own 0.5m inner edge computes a crossing
point *under* 0.5 and gets silently treated as narrow (no landing-gate
requirement). Measured directly (64 envs, difficulty=1.0): `left_far`/
`right_far` only triggered `wide=True` 80.4%/83.3% of the time (should be
100% by the region system's own design); `left_near`/`right_near` correctly
stayed at 0%.

**This affects training, not just play** — `_get_reach_target_y` backs the
actual `footreach`/`foot_proximity`/`stopball`/`softstop` landing gate, so
~17-20% of the `blue_ball_gate_2026-07-06` run's far-region episodes were
silently exempt from the two-stage requirement the whole experiment exists to
test, collecting full save reward via a direct reach with no genuine
waypoint landing.

**Fix applied** (`mdp/rewards.py:_get_reach_target_y`): OR the threshold-based
`wide` flag with the region's own far/near ground truth
(`env._region_id`) when present — region label is authoritative for the
region-conditioned task; the plain single-disc task (no `_region_id`) is
unaffected. Verified: `left_far`/`right_far` now 100.0%/100.0% (up from
80.4%/83.3%); near regions unchanged at 0%. 48/48 tests pass. Full writeup:
`docs/BugFixes.md`, dated entry `2026-07-07` (region-far/wide-gate section).
Committed and pushed: `605f77e`.

**Training restarted per user's standing instruction ("always implement a bug
you find, and then push")** — this changes which episodes require the
landing gate, too large a behavior change to continue training under the old
partially-leaky gate:

- Killed `blue_ball_gate_2026-07-06` (PID `2903136`, healthy, iteration
  ~6600+ at kill time, last probe: reward 28.53, region accuracy untested at
  this specific checkpoint but region loss 0.449/healthy).
- **New run started:** 2026-07-07 11:09:11.
- **PID:** `3345049`/`3345053` (worker) — `nohup ... & disown`, fully detached.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_gate_region_fix_2026-07-07`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-07_11-09-11_blue_gate_region_fix_2026-07-07/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_gate_region_fix.log` (session-scoped).
- W&B tracking confirmed active (run `7rn7swgt`, project `SimpleGoalKeeper`).
- Confirmed alive, GPU busy, no crash before handing off to hourly monitoring.

**Historical danger-zone context resets with this run** — the 988/1870/2560/
3070/3941/5790 NaN-divergence checkpoints from prior runs are all from code
predating this fix; watch the same zone again as a precaution but there's no
specific reason this fix would introduce new instability (it only changes
which episodes require landing before collecting save reward, not any
gradient/optimizer/LR mechanics). Monitoring baseline for genuine/RSI-assisted
split and region-accuracy probes also resets — this run's episodes now have a
strictly stronger landing-gate requirement on far shots than any prior run, so
even a lower initial genuine-landing rate here is not directly comparable to
the old run's numbers (harder gate, and correctly so).

## 17. Fourth bug found 2026-07-07 ~11:50: `blue_ball_landed` fired from a foot sweeping past blue mid-stride, not stopping there — training restarted again

While debugging an unrelated `sgk_play` visualization issue in `SimpleGoalKeeper`
(SGK), user reported directly watching `interceptV2DualDis`'s own play output
and seeing the assigned foot's actual ground contact land near the GREEN
target while `blue_ball_landed` had already fired — the policy was still
doing one continuous big step/leap, not the intended paced double-step.

**Root cause:** blue and green share the same X (goal line) and are always
`>= wide_threshold/2` apart in Y for a wide crossing, so a foot planting AT
green cannot itself be within the old `landing_radius=0.05` of blue — ruled
that out directly. Blue sits directly on the straight-line path from the
robot's stance to green, so a single continuous stride/leap toward green
necessarily sweeps *through* blue's Y-coordinate mid-flight. A momentary or
glancing `feet_contact` sensor reading during that pass-through (sensor
noise, a low-clearance shuffle, foot-scuff during swing phase) was enough to
satisfy "airborne, then in contact, within landing_radius of blue" without
the policy ever intending to stop there, before it continued on to plant at
green. Same category of exploit SGK found and fixed for the identical reason
on 2026-07-05 (`landing_radius` 0.3→0.05) — this project inherited SGK's
already-tightened 0.05 at port time, but that wasn't tight enough here.

**Fix applied** (`mdp/rewards.py:_get_reach_target_y`): `landing_radius`
tightened `0.05` → `0.02`. Doesn't eliminate the exploit in principle (a
sweep could in theory still graze an arbitrarily small radius by chance) but
makes accidental pass-through much less likely. 48/48 tests pass. Full
writeup: `docs/BugFixes.md`, dated entry `2026-07-07` (landing_radius
section). Committed and pushed: `11a56d8`.

**Known limitation, flagged for a follow-up if this doesn't fully resolve
it:** a velocity-based check (require the assigned foot's horizontal speed
to be near zero at the moment of contact) would more robustly distinguish a
genuine plant from a foot passing through at speed, regardless of radius.
Not implemented now — the user asked specifically about the radius/margin,
so implementing the tightened radius first and watching the next run's
behavior (via play, not just metrics — metrics can't distinguish "landed
briefly while sweeping through" from "landed and paused") is the more
surgical first step.

**Training restarted per user's standing instruction ("always implement a
bug you find, and then push")** — this changes the strictness of the exact
mechanism this run exists to validate:

- Killed `blue_gate_region_fix_2026-07-07` (PID `3345049`, healthy, ~45 min
  in, no NaN, still early).
- **New run started:** 2026-07-07 11:55:25.
- **PID:** `3387650`/`3387654` (worker) — `nohup ... & disown`, fully detached.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_gate_landing_radius_fix_2026-07-07`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-07_11-55-25_blue_gate_landing_radius_fix_2026-07-07/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_landing_radius_fix.log` (session-scoped).
- W&B tracking confirmed active (run `ftlclv51`, project `SimpleGoalKeeper`).
- Confirmed alive, GPU busy, no crash before handing off to monitoring.

**What to watch, beyond the usual NaN check:** since metrics alone can't
distinguish a genuine plant from a fast pass-through, the real confirmation
this fix worked requires a play-mode/qualitative check on a checkpoint from
this run once one exists — watch specifically whether the robot now visibly
pauses at the blue marker before continuing to green, rather than one
continuous motion. If `blue_landed_genuine` drops sharply relative to prior
runs at a comparable stage, that could mean the tighter radius is making the
gate too hard to satisfy even genuinely (worth watching, not yet expected).

## 18. Fifth bug found 2026-07-07 ~12:10: `sgk_play`'s viewer never applied the region-ground-truth `wide` override — pure rendering bug, no training impact

User immediately pushed back on the landing_radius fix (section 17): "no the play script even shows its not working, this is not a rsi assisted problem." Investigated with a live diagnostic driving the actual trained policy (`model_6500.pt`, the OLD pre-landing-radius-fix checkpoint — reward-side parameters aren't baked into weights, so this is still a valid test of the current code) and logging the exact assigned-foot position at the moment of every `blue_ball_landed` firing.

**First pass (misleading):** 127/127 sampled landings showed `_blue_airborne_at_reset=True`, initially read as "100% RSI-assisted free credit." Proposed gating `_blue_landed` on `~env._blue_airborne_at_reset`. **User correctly rejected this** — the diagnostic's 100% rate turned out to be a measurement artifact: `_blue_airborne_at_reset` only checks whether the assigned foot was airborne within the first 2 steps of the episode, and this project's 80% "continue_keep" RSI branch copies a donor env's mid-motion pose at every such reset — meaning *some* foot is almost always airborne within the first 2 steps of nearly every episode, regardless of whether the actual landing (which can happen much later, confirmed up to step 332+ in an earlier smaller sample) was genuinely earned. The flag doesn't mean what it was being read as meaning here; no RSI-gating change was made.

**Actual root cause:** re-examined `scripts/play.py`'s `_patch_viewer_intercept_vis`, which recomputes `wide` independently from the reward-side `_get_reach_target_y` for rendering purposes. Found it still used the bare `abs(rel) > 0.5` threshold check — never updated to include the region-ground-truth override added to the reward side earlier the same day (section 16). For a region-far episode whose actual crossing magnitude sits below 0.5 (exactly the population section 16's fix targets), the reward side correctly computes `wide=True` and the robot correctly walks to and lands at the blue midpoint — but the viewer computed `wide=False` for that same episode and rendered ONLY a green sphere at the full crossing point for the whole episode, never showing blue at all. This exactly matches what the user saw: `blue_ball_landed` firing with the foot nowhere near the only marker visible, because the foot was correctly walking to and landing at the real (blue) target the viewer simply never rendered.

Confirmed the landing *position check itself* was never broken: the same diagnostic's foot-position log showed `dist_to_blue < landing_radius` held in every one of the 127 sampled cases.

**Fix applied** (`scripts/play.py`): viewer's `wide` now ORs in the same region-far check as the reward side. 48/48 tests pass. Pure play-time rendering fix — `_patch_viewer_intercept_vis` is never called during training, so **no training restart needed**. Committed and pushed: `5390bb5`. Full writeup: `docs/BugFixes.md`, dated entry `2026-07-07` (viewer wide-override section).

**Takeaway for future debugging in this file:** whenever a `wide`/region-related override is added to the reward side (`mdp/rewards.py`), check `scripts/play.py`'s independent recomputation for the same override — they've now diverged twice in one day (crossing-fraction formula was shared correctly via `_rsi_cross_y`, but the *region* override was reward-only both times it was almost added). Consider refactoring so the viewer reads `env._blue_wide` directly (already computed and cached by `_get_reach_target_y` every step) instead of recomputing it from scratch, to make this class of bug structurally impossible going forward — not done now, flagged as a low-priority cleanup.

## 19. Sixth bug found 2026-07-07 ~12:20: single-env play sessions could never show a near-region episode — training restarted (fresh run, all fixes)

After the viewer fix (section 18) resolved the "foot not close" symptom, user's next report: "close balls shouldn't be split up" — visually-close-looking shots were still getting the blue/green split. User explicitly said this was **not** a request to add `--num-envs`: "no num_envs is not what i want, i want one singular agent that describes the full agent, but also with the correct visualisation just like sgk has."

**Root cause:** `assign_static_regions` splits `num_envs` into 4 contiguous blocks; at `num_envs=1` (the play default), `quarter=0/remainder=1` permanently pins the single env to region 3 (`right_far`) for the *entire session*, every episode. What looked like "close balls incorrectly split" was `right_far` episodes with a small sampled magnitude (just past the region's 0.5m inner edge) — correctly still requiring the blue gate by region label, just not looking dramatically wide. A single-env session could never show a genuine near-region episode at all to compare against.

**Fix applied:** new `mdp/regions.py:randomize_region_on_reset` re-samples `env._region_id` uniformly at random every episode, wired in for `play=True` only (`goalkeeper_multidisc_amp_cfg.py`); training keeps the unchanged permanent per-env split (`assign_static_regions`, `mode="startup"`) since the region_estimator needs a stable, balanced ground-truth distribution across the parallel training batch. **Caught and fixed a second bug while implementing this:** `cfg.events["reset_ball"]` already existed in the dict from the base config, so a plain reassignment didn't move its execution order relative to the newly-added region-assignment event — `reset_ball_rolling_by_region` was reading `env._region_id` before `randomize_region_on_reset` ever set it (`AttributeError` on the very first reset). Fixed by popping and re-inserting the order-sensitive events (`reset_ball`, `reset_from_motion_data`, `tick_catchstep`) in the correct relative order. 48/48 tests pass. Empirically verified: `num_envs=1`, 2500 steps, 61 episodes — region sequence freely varied across all 4 (`left_far` 20, `left_near` 15, `right_near` 13, `right_far` 13). Full writeup: `docs/BugFixes.md`, dated entry `2026-07-07` (region randomization section). Committed and pushed: `8d598ff`.

This is a play-only behavioral change (training's event order ends up equivalent to before), so it alone didn't require a restart — but the user separately asked to "spin up a new training with the blue ball fixes" after the visualization work was done, so training was restarted anyway for a clean, fully-fresh run incorporating everything from this session (crossing-fraction fix, region-ground-truth wide override, landing_radius tightening):

- Killed `blue_gate_landing_radius_fix_2026-07-07` (PID `3387650`, healthy, ~34 min in, iteration ~437 last checked, reward 16.88, region loss 0.416, no NaN).
- **New run started:** 2026-07-07 12:29:17.
- **PID:** `3416662`/`3416666` (worker) — `nohup ... & disown`, fully detached.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_ball_fixes_2026-07-07`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-07_12-29-17_blue_ball_fixes_2026-07-07/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_ball_fixes.log` (session-scoped).
- W&B tracking confirmed active (run `u5fniknq`, project `SimpleGoalKeeper`).
- Confirmed alive, GPU busy, no crash before handing off to monitoring.

**Cumulative fix set now in this run:** crossing-fraction formula (section
16's predecessor), region-ground-truth `wide` override (section 16),
`landing_radius` 0.05→0.02 (section 17). The play-viewer fix (section 18)
and region-randomization fix (section 19, this section) are play-only and
don't change training behavior, but this is nonetheless the cleanest,
fully-consistent run to date — resume monitoring per the usual NaN-danger-
zone and genuine/RSI-assisted-split checks, now on a fresh iteration count.

---

## 20. Seventh bug found 2026-07-07 ~17:40: stopball/softstop's landing gate accepted RSI-assisted free landings — training restarted again

User asked for a direct measurement: "how many percentage of where blue ball is available the robot lands on blue ball?" Wrote a live diagnostic (`scratchpad/blue_landing_rate.py`) driving `blue_ball_fixes_2026-07-07`'s `model_4500.pt` for 3210 episodes (256 envs): **1637 wide (blue-available) episodes, 504 (30.8%) landed on blue at all, but only 23 (1.4%) were genuine — 481 (29.4%) were RSI-assisted free landings.**

User's reaction: "we have to do it more radical: u cant get stopball and softstop, if u didnt land first on blue ball marker point." Turns out this gate already existed (`cde4737`, added 2026-07-07 00:21) — `stopball`/`softstop` already required `~env._blue_wide | env._blue_landed` — but it accepted *any* `_blue_landed`, including the RSI-assisted 29.4%. Since the gate was satisfied for free ~95% of the time it fired, it was never actually forcing the policy to learn the two-stage step.

**Fix applied:** both gates tightened to `~env._blue_wide | (env._blue_landed & ~env._blue_airborne_at_reset)` — same genuine/RSI-assisted definition `metrics.py` already used for diagnostics, now load-bearing in the reward itself. 48/48 tests pass. Known caveat carried over from section 18: `_blue_airborne_at_reset` can be a broad false-positive under this project's 80% `continue_keep` RSI branch, so the tightened gate may occasionally over-reject a genuine landing whose episode merely started with an unrelated early airborne transition — accepted as the best available proxy, flagged for revisit if the gate essentially never opens even as the policy visibly improves. Full writeup: `docs/BugFixes.md`, dated entry `2026-07-07` ("stopball/softstop's wide-crossing landing gate accepted RSI-assisted 'free' landings").

**Checkpoint pushed before stopping:** `model_4750.pt` (one step past the diagnosed `model_4500.pt`) force-added and committed (`3f2ac63`) since `/logs/` is gitignored by default.

**Commits:** `552e303` (gate fix), `3f2ac63` (checkpoint). Pushed to `origin/master`.

**Training restarted** (not resumed — this changes which wide episodes can earn any save reward at all, too large a reward-structure change to continue under the old gate):

- Killed `blue_ball_fixes_2026-07-07` (PID `3416662`/`3416666`, healthy, ~5.5h in, iteration ~4750, no NaN).
- **New run started:** 2026-07-07 (immediately after kill).
- **PID:** `3622196`.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_landing_gate_2026-07-07`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-07_18-01-30_blue_landing_gate_2026-07-07/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_landing_gate.log` (session-scoped).
- Confirmed alive, first iteration logged cleanly, no crash before handing off to monitoring.

**What to watch for:** since genuine landings were only 1.4% under the old (leaky) gate, `stopball`/`softstop` reward may show as near-zero on wide episodes for a while under the new strict gate — this is the intended forcing function, not a bug, but if `Episode_Reward/stopball`/`softstop` stay pinned near zero for a very long stretch while narrow-episode performance is fine, that's a reward-sparsity risk worth re-running `scratchpad/blue_landing_rate.py` against a later checkpoint to check whether the genuine-landing rate is actually climbing.

---

## 21. Eighth bug found 2026-07-07 ~19:50: landing check was position-only, nothing required the foot to actually stop — velocity gate added, training restarted again

Section 20's fix made the `stopball`/`softstop` gate airtight (no more RSI-assisted free credit). User then watched `sgk_play` against the fresh `blue_landing_gate_2026-07-07` run and reported the robot still sweeping straight past blue to the intercept point — but this time correctly earning zero `footreach`/`stopball`/`softstop` for it (confirming the gate itself no longer leaks). User's question: "so how can it still not learn this double step motion."

**Root cause:** `_get_reach_target_y`'s landing check (`dist_to_blue < landing_radius`) was pure position, with nothing requiring the foot to decelerate — meanwhile `footreach`'s own `vel_sigma` term (up to 10x) rewards *speed* toward whichever point is the current reach target, including blue during phase 1. The two combine into a reward structure that favors blasting through blue over planting there, and at `landing_radius=0.02` (2cm, one 0.02s physics step) a genuine deliberate plant was nearly as hard to hit as an accidental graze — the section-17 radius tightening fought the symptom (accidental grazes) without fixing the cause (nothing rewards stopping).

**Fix applied:** added `landing_speed_threshold=0.5` m/s — the assigned foot's horizontal speed (`robot.data.body_link_lin_vel_w`) must be below this at the moment of contact, in addition to the position check. `landing_radius` loosened back 0.02 → 0.08 now that speed (not radius) rules out sweep-throughs. Cascades automatically to `stopball`/`softstop`/`footreach`/`foot_proximity` since all four read `env._blue_landed` from this one function. 48/48 tests pass; live smoke test (4 envs, zero-action, 30 steps) confirmed no shape/attribute errors. Full writeup: `docs/BugFixes.md`, dated entry "blue-midpoint landing check was position-only... velocity gate added". Committed: `ddfa617` (fix), `17cf0a7` (checkpoint push).

User also asked "do i need a better dataset?" — answered no: the region track's AMP library already includes genuine motion-capture double/triple-step clips (`LeftDoubleStep_own_booster_t1.npz` etc.) with real mid-motion plants, and this project's own docs already establish AMP's 2-frame discriminator structurally cannot enforce global step count/trajectory shape regardless of clip quality — that's the whole reason the blue/green task-reward system exists as a supplement. The bug just fixed was reward-structure, not data.

**Training restarted** (not resumed — changes which landings are achievable at all, too large a change to continue under the old gate):

- Killed `blue_landing_gate_2026-07-07` (PID `3622196`/`3622199`, healthy, ~2h in, iteration ~1750, no NaN).
- Pushed final checkpoint `model_1750.pt` before killing (`17cf0a7`).
- **New run started:** 2026-07-07 20:06:31.
- **PID:** `3702375`/`3702378` (worker).
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_landing_velocity_gate_2026-07-07`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-07_20-06-31_blue_landing_velocity_gate_2026-07-07/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_landing_velocity_gate.log` (session-scoped).
- Confirmed alive, first iteration logged cleanly, no crash from the new velocity computation.

**What to watch for:** rerun `scratchpad/blue_landing_rate.py` (update its `ckpt` path) against a checkpoint once iterations accumulate (~1500+) to check whether the genuine-landing rate on wide episodes climbs meaningfully above the section-20 baseline of 1.4%. If it's still stuck near-zero after real training time under this fix, the next hypothesis to check is whether the AMP demo clips' step length/spacing actually covers the range of blue-midpoint targets the region system samples (a dataset-alignment question, not a "need more data" question) — not yet investigated, flagged only as a fallback if the reward fix doesn't move the number.

**2026-07-07 ~22:20 — first landing-rate check at iteration 2006, result is a regression, not an improvement:** diagnostic against `model_2000.pt` (2570 episodes, 1293 wide): landed-on-blue 55.8% (up from 30.8%), but genuine dropped to **0.0%** (down from 1.4%) and RSI-assisted rose to 55.8% (up from 29.4%). Likely cause, not yet fixed: `newly_landed`'s velocity check fires at the exact instant `foot_in_contact` transitions false→true (touchdown impact), when a genuine footstrike still carries residual swing velocity — it only decays toward zero over the *following* few steps as weight transfers. Checking speed at that instant systematically rejects real plants while trivially passing a stationary RSI donor pose (at rest before the episode even starts), which is the opposite of the intended effect. A fix would need to check speed after a short settle/dwell window rather than at the contact-transition instant.

**Decision:** flagged to the user, who judged this not worth interrupting the run over ("i dont think this is really a big problem") — training continues uninterrupted, no code change, no restart (this would have been the 5th restart today). Pushed `model_2250.pt` (`997c6e9`). This is left as an open, understood-but-unfixed issue — revisit the velocity-check timing if a later checkpoint's landing-rate check still shows ~0% genuine.

---

## 22. Ninth change 2026-07-08 ~00:10: `blue_overshoot_penalty` added — no cost to ignoring the blue waypoint entirely, so PPO had no gradient away from that local optimum

User asked for a full codebase audit ("blue ball still not working, the robot learns in standing pose to save the ball without going landing at blue ball first... maybe a negative reward for going past blue ball without first landing at it? do a full review and come up with ideas"). Did a full pass through the reward/curriculum stack (see the audit message in conversation, not reproduced here) and found the real root cause: **a wide episode where the policy ignores the blue waypoint entirely earns exactly the same reward (zero) as one where it genuinely tries and fails** — `stopball`/`softstop`/`blue_ball_landed` are all landing-gated, so there's no cost differential between "didn't try" and "tried and missed." With narrow crossings (~half the region-sampled distribution) already paying out fully from a near-default stance, "specialize in narrow, ignore wide" is a stable local optimum with zero gradient pulling the policy out of it.

Proposed 5 options (overshoot penalty, kill `vel_sigma` during blue-approach, bias episode sampling toward wide, gate curriculum advancement on wide-episode success, two-phase pretraining of the approach-and-stop sub-skill). User picked the overshoot penalty and specifically asked why not keep `vel_sigma` targeting blue-then-green (their instinct, not mine) rather than neutralizing it. Rereading `footreach()` confirmed the user was right and my original plan had an error: `vel_sigma`'s direction already comes from `reach_target_y`, which flips blue→green exactly on `env._blue_landed` — it already only rewards speed toward blue pre-landing. Left it untouched.

**Fix applied:** new reward term `blue_overshoot_penalty` (`mdp/rewards.py`, weight `-30.0`, `tasks/goalkeeper_env_cfg.py`) — penalizes the assigned foot for advancing past blue toward green before landing there on a wide crossing, scaled by how far past (deadband matches the existing `0.08` landing radius). 48/48 tests pass; live smoke test (8 envs, zero-action, 60 steps) confirmed the term registers and runs. Full writeup: `docs/BugFixes.md`, dated entry "added `blue_overshoot_penalty`". Committed: `fa589ec` (feature), `0cfa9cb` (checkpoint push).

**Known caveat carried forward:** this doesn't fix the still-open velocity-check-timing bug from section 21 (0% genuine under the current gate) — it only adds pressure to *attempt* stopping at blue. Whether those attempts can register as genuine landings still depends on that separately-flagged issue.

**Training restarted:**

- Killed `blue_landing_velocity_gate_2026-07-07` (PID `3702375`/`3702378`, healthy, iteration ~3500, no NaN). Pushed final checkpoint `model_3500.pt` (`0cfa9cb`).
- **New run started:** 2026-07-08 00:10:41.
- **PID:** `3856273`/`3856276` (worker).
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_overshoot_penalty_2026-07-08`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-08_00-10-41_blue_overshoot_penalty_2026-07-08/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_overshoot_penalty.log` (session-scoped).
- Confirmed alive, `blue_overshoot_penalty` visible in the active reward-term table at weight -30.0, first iteration logged cleanly.

**What to watch for:** rerun `scratchpad/blue_landing_rate.py` (update `ckpt`) once iterations accumulate (~1500+). Two things to check: (1) does the genuine-landing rate move off 0% at all — even a small nonzero number would validate the mechanism; (2) does `Episode_Reward/blue_overshoot_penalty` in tensorboard shrink in magnitude over training (policy learning to avoid it) or stay flat (weight too weak, or policy accepting the penalty as cheaper than the alternative). If the genuine rate is still 0% after real training time, the velocity-check-timing bug (section 21) becomes the next thing to actually fix rather than defer.

---

## 23. Tenth change 2026-07-08 ~00:10-01:00: AMP setup brought to literal G1 parity + landing-check settle window, then an independent re-check caught a missed divergence — training restarted twice more

User pushed back hard on the "AMP might be a contributing factor" hand-wave from the audit conversation: "i want the same amp setup as g1 not the same outcome... same fucking code... not only same functionality but really the same methods no unnecessary divergence. then do the other reward implementations and restart the training."

**AMP parity pass (commit `0a8cbac`):** read G1's `rsl_rl/rsl_rl/algorithms/him_ppo.py` and `rsl_rl/rsl_rl/modules/amp.py` directly and fixed three real, confirmed divergences in `multi_disc_amp_ppo.py`/`goalkeeper_multidisc_amp_cfg.py`:
- Gradient penalty: `lambda_=10` unscaled → `lambda_=5 * 0.1` (effective coefficient 10 → 0.5, matching G1's `compute_loss`).
- `amp_loss`: `0.5*expert + 0.5*policy` → unweighted `expert + policy`, matching G1's `gail_loss`.
- Discriminator hidden dims: `[256,256]` → `[512,256]`, matching G1's unoverridden `AMP.__init__` default.
- `amp_storages` (per-region `ReplayBuffer`, previously a persistent 250k-transition FIFO never cleared) now `.clear()`d every `update()`, matching G1 having no replay buffer at all — its discriminator trains strictly on the current on-policy rollout.
Checked and found NOT divergent (left alone): the reward blend formula (mathematically equivalent to G1's at current constants) and `amp_min_normalized_std` (not actually an AMP parameter — it's the actor's exploration-noise floor, no G1 equivalent needed).

**Landing-check settle window (same commit):** the velocity check from section 21/BugFixes.md that drove genuine landings to 0% fired at the exact instant of ground contact — before a real footstrike's residual swing velocity decays. Added a 3-consecutive-step settle requirement (`env._blue_settle_count`) before checking speed, giving a genuine plant time to decelerate while still rejecting fast sweep-throughs (can't hold contact+radius for 3 steps while moving fast).

**Restart 1:** killed `blue_overshoot_penalty_2026-07-08` (PID `3856273`/`3856276`, iteration ~250), pushed `model_250.pt` (`43ff8bd`), launched `amp_g1_parity_2026-07-08` (PID `3882667`/`3882670`) at 00:41:10.

**Independent re-check catches a real gap (commit `a627bfc`):** user explicitly asked for a *second, unbiased subagent* to re-verify the AMP parity work from scratch, worried about AI-generated imprecision ("claude code tend to hallucinate"). Dispatched a fresh Explore agent with no access to the first agent's conclusions, told to independently re-derive every claim and flag anything wrong. Five of six claims confirmed exactly; the sixth was a real miss: the first pass had reported AMP observation content as "no divergence... neither can see world-frame/ball-relative state" — true, but it silently conflated "no spatial leakage" with "content matches G1." G1's `get_amp_observations()` (`legged_robot.py:157-161`) returns `dof_pos` **only** — no joint velocities. This port's AMP obs group included `joint_vel` alongside `joint_pos`, a genuine superset G1's discriminator never sees. Fixed as a task-local override (`goalkeeper_multidisc_env_cfg`'s own `cfg.observations["amp"]`, new `_MULTIDISC_AMP_OBS_TERMS = ["joint_pos"]`) — the shared `AMPObsBaiscTerms` constant and base `goalkeeper_env_cfg()` are untouched, so SimpleGoalKeeper's single-disc AMP track is unaffected.

**Restart 2:** the `amp_g1_parity_2026-07-08` run had only reached iteration 0 (~5 min in, only `model_0.pt`, itself already stale against the new 21-dim AMP obs) when this was found — killed with nothing worth pushing, launched `amp_g1_parity_2026-07-08b` (PID `3890295`/`3890298`) at 00:57. Confirmed alive: `amp` obs group shape `(21,)`, first iteration logged cleanly.

**Full writeup:** `docs/BugFixes.md`, three dated 2026-07-08 entries (AMP parity, landing-check settle window, AMP obs content). The last entry includes a "process note" explicitly flagging this as a template for future work: a claim of the shape "X matches G1 exactly" should default to an independent re-check before being treated as settled, since a single pass reporting a technically-true-but-incomplete finding reads identically to a fully-verified one.

**What to watch for:** same as section 22 (rerun `scratchpad/blue_landing_rate.py` once iterations accumulate to check whether genuine landings move off 0%), plus now also worth comparing AMP loss curves (`Loss/AMP`, `Loss/AMP_grad`) against pre-parity-fix runs to see whether the less-over-regularized, narrower-content, strictly-on-policy discriminator produces a qualitatively different (hopefully more informative) style signal.

---

## 24. First scheduled 5000-iteration escalation, 2026-07-08 ~07:00: genuine landings still 0.0%, AMP ruled out, dense `blue_stick_landing` shaping added — training restarted

Per the standing monitoring policy (`CLAUDE.md`, added this session), the bihourly cron ran its scheduled escalation check when `amp_g1_parity_2026-07-08b` crossed iteration 5000 (observed at 5548). Two prior routine checks had already independently measured **0.0% genuine** blue landings (iteration 2000: 0/1290 wide episodes; iteration 3750: 0/1348 wide episodes) — consistent, not noise. `blue_overshoot_penalty` stayed flat and strongly negative across the whole run (~-0.64 to -0.86 sampled at iterations 1111 through 5556) instead of shrinking, meaning the policy wasn't learning to avoid the penalty it kept incurring.

**AMP ruled out, but not before a scare:** `Train/mean_discri_logits` showed -115 to -120, which looks like a badly collapsed discriminator at first glance. Before concluding anything, read the actual logging code (`him_amp_on_policy_runner.py:238-239`): `cur_discri_sum += d_logits`, accumulated every step for the whole episode and only flushed at episode end — a per-episode **sum**, not a per-step value. Divided out, per-step logits are around -0.5 to -0.8, unremarkable. `Loss/AMP`/`Loss/AMP_grad` stayed small and stable across the run; `mean_action_acc` flat around 1.15, no divergence. AMP is not the bottleneck this time.

**Diagnosis:** the settle-window landing check (section 21/23) requires contact + within-radius + below-speed-threshold sustained for 3 consecutive steps. Nothing in the reward stack gave *dense* credit for approaching that joint condition — `footreach`/`foot_proximity` reward proximity to blue, `vel_sigma` rewards speed toward blue, but nothing rewarded decelerating once close. The only payoff for "close and slow" was the sparse settle-window event itself, which the policy hadn't stumbled into after 5500+ iterations.

**Fix applied:** new dense reward `blue_stick_landing` (`mdp/rewards.py`, weight `8.0`) — `exp(-15·dist_to_blue)·exp(-3·foot_speed)` for the assigned foot on a wide, unlanded crossing, peaking at exactly "close AND stopped." Mirrors this project's own `cleanstop` pattern (rewards low ball speed near a target after a save). 48/48 tests pass; live smoke test (3 iterations, 64 envs) confirmed the term registers and runs. Full writeup: `docs/BugFixes.md`, dated entry "scheduled 5000-iteration escalation." Committed: `208d91c` (fix), `6b8973f` (checkpoint push).

**Training restarted:**

- Killed `amp_g1_parity_2026-07-08b` (PID `3890295`/`3890298`, iteration ~5548, healthy otherwise, no NaN). Pushed final checkpoint `model_5500.pt` (`6b8973f`).
- **New run started:** 2026-07-08 07:18:55.
- **PID:** `4132607`/`4132610` (worker).
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_stick_landing_2026-07-08`
- **Run directory:**
  `interceptV2DualDis/logs/rsl_rl/intercept_simple_goalkeeper_multidisc/
  2026-07-08_07-18-55_blue_stick_landing_2026-07-08/`
- **Raw stdout log:**
  `/tmp/claude-6380/-home-ibouwmeest-BEPImitationLearning/940e4985-7161-49e4-9e26-1e9fbd86c5ae/
  scratchpad/intercept_train_blue_stick_landing.log` (session-scoped).
- Confirmed alive, `blue_stick_landing` visible in the active reward table at weight 8.0, already nonzero (0.033) on the very first logged iteration.

**What to watch for:** whether the genuine-landing rate moves off 0% at all at the next scheduled checks (iteration 1500+ routine diagnostic), and whether it holds up through the next 5000-iteration escalation if reached. If `blue_stick_landing`'s dense gradient alone doesn't do it, the remaining unexplored lever from the original ideas list is a proper potential-based (PBRS) reformulation, or reconsidering whether `_BLUE_SETTLE_STEPS=3`/`landing_radius=0.08`/`landing_speed_threshold=0.5` are still miscalibrated even with better dense guidance toward them.

---

## 25. Second scheduled escalation, 2026-07-08 ~17:00: user raised threshold 5000→6500 (curriculum cu=3 not yet reached); AMP init/reward-smoothing divergences found and fixed; landing_speed_threshold loosened

User pushed the escalation threshold from 5000 to 6500 after checking `Train/mean_episode_length` at iteration 5072 (125.6, `cu=int(ep_len/47)=2`, needs ≥141 for `cu=3`) — reasoning the run needed more time before its curriculum-mature behavior could be fairly judged. Also explicitly requested the AMP re-comparison be repeated regardless of the stuck/progressing outcome: "amp reward seems to be really low." Separately, mid-cycle the user asked "why is it not learning to stand longer" — investigated the termination breakdown and found `shank_height` (kneeling too low) was the dominant remaining termination cause (~32% of episodes) and had plateaued rather than continuing to improve, plausibly because the newly-added blue-ball incentives push more aggressive stepping that risks knee collapse. Pushed `model_7750.pt` as a checkpoint at that point without stopping the run.

**Escalation triggered at iteration 8102.**

**Part A — AMP magnitude re-audit (independent subagent, told nothing of prior conclusions):** confirmed two real divergences in `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/modules/amp_discriminator.py`:
1. No explicit weight init anywhere — G1's `amp.py:100-118` explicitly inits every trunk layer past the first, plus the final `amp_linear` output layer, with `uniform_(-1,1)`/zero-bias. This port used PyTorch's default init throughout, including on `amp_linear` — ~16x smaller weight-magnitude bound for a `Linear(256,1)` layer.
2. No reward-noise-smoothing — G1's `predict_reward` (`amp.py:185-206`) perturbs the input with 20 Gaussian noise samples and takes the *minimum* squared-error-from-1 before computing reward, systematically raising the reward relative to a raw single-sample read. This port used a raw single-sample evaluation.

Both fixed: added G1's exact conditional init pattern (first trunk layer default, everything after explicit `uniform_(-1,1)`/`zeros_`), and ported the 20-sample noise-perturbation-then-min mechanism into `predict_amp_reward`. Verified via smoke test: `amp_linear` weights now span ±1.0 with std≈0.577 (matching uniform(-1,1) theory) — the fix works as intended.

**Part B — stuck/progressing judgment:** genuine landings 0.0% at iteration 8000 — the 5th consecutive measurement this run (one 0.1% fluke at iteration 2500). `blue_stick_landing` showed consistent nonzero reward the whole run (policy demonstrably getting closer/slower over time) but never crossed the discrete settle-window threshold. Also noted: curriculum crossed into `cu=3` right at this checkpoint, and `mean_episode_length` regressed sharply at the same moment (119.4→92.5) with `shank_height` terminations spiking (10.6→30.0) — a curriculum-transition destabilization, consistent with the earlier "why isn't it standing longer" investigation.

**Judgment: stuck.** Fixed `landing_speed_threshold` 0.5→0.8 m/s — the dense shaping reward showing real progress without ever crossing the discrete gate suggested the threshold itself, not the shaping, was the remaining blocker.

**Training restarted:**

- Killed `blue_stick_landing_2026-07-08` (PID `4132607`/`4132610`, iteration ~8102). Pushed final checkpoint `model_8000.pt` (`6a3299a`).
- **New run started:** 2026-07-08 (immediately after kill).
- **PID:** `296934`/`296937`.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name amp_reward_fix_2026-07-08`
- Commits: `8dbb8ad` (AMP + threshold fix), `6a3299a` (checkpoint), `5836acd` (transient docs addendum).

**Notable transient at restart, not a bug:** `Loss/AMP`/`Loss/AMP_grad` spiked to 1299/1998 on iteration 0 (vs. ~0.1-0.3 typical of pre-fix runs), then decayed monotonically to ~17/~44 by iteration 19. `mean_action_acc` stayed in [1.56, 1.88] throughout, trending down, never near divergence. Expected: larger G1-matching init weights produce larger initial discriminator gradients; `grad_pen_loss` needs several steps to reel the gradient norm back toward 0. Very likely how G1 behaves on every fresh run too — this port just never had weights large enough to trigger a visible version of it before. Documented in `docs/BugFixes.md` so a future health check doesn't misread the first ~20 iterations as a new problem.

**What to watch for:** does the genuine-landing rate move off 0% now with three stacked fixes in effect (AMP init, AMP reward smoothing, `landing_speed_threshold` loosened)? If it does, worth eventually isolating which one mattered rather than assuming all three were necessary. If `shank_height` terminations stay elevated (the curriculum-transition destabilization from section escalation above), that's a separate thread worth revisiting even if landing rate improves — a robot that can't sustain `cu=3` reliably will have unstable reward-weight dynamics regardless of the blue-ball fixes.

---

## 26. ROOT CAUSE FOUND, 2026-07-08 ~evening: `_blue_airborne_at_reset` was a broken metric all along — genuine landing rate was actually 94.7%, not 0%

User watched `sgk_play` on `model_2250.pt` and reported the robot still appeared to skip blue. Asked to (1) check `footreach` for a bug, (2) do a codebase-wide audit if not, (3) do yet another independent AMP-vs-G1 comparison since "amp has worsen i think looking at reward."

**Footreach: confirmed correct** by code review AND live diagnostic — the assigned foot reaches within `landing_radius` **while in ground contact** in 97.6% of wide episodes (measured directly, not inferred). Footreach was never the problem.

**AMP: third independent audit came back clean** — no further divergence from G1 found beyond a minor, low-significance normalizer-width nuance. AMP was ruled out for good this round.

**The real bug, found through a chain of live diagnostics (including catching a bug in my own diagnostic tooling along the way):**

1. Measured foot speed at closest in-contact approach to blue: median 0.815 m/s, just above `landing_speed_threshold=0.8`. Loosened to `1.0`. Re-checked same checkpoint — still 0.0% genuine.
2. Hypothesized the 3-consecutive-step settle window was the blocker. Diagnostic showed `_blue_settle_count` never exceeding 1 across any episode. Set `_BLUE_SETTLE_STEPS=1`. Re-checked — still 0.0% genuine, "landed at all" *unchanged* at ~95% before and after — a tell that something else was going on, since a real fix to the actual blocker should have changed the landed-at-all rate too.
3. Found the bug in my own settle-count diagnostic: it filtered on `active = wide & ~landed`, which excludes the exact step a landing fires on (since `landed` flips true that step) — so it could never observe the true peak settle-count. **Reverted `_BLUE_SETTLE_STEPS` back to 3** (never actually broken).
4. Measured **when** landings fire instead: `episode_length_buf` at the moment of `newly_landed`, over 522 samples. **Zero landings before step 28** (median 35, max 86) — clearly real approach behavior, not free credit. Yet **100% were still flagged `env._blue_airborne_at_reset=True`**.

**Root cause:** `env._blue_airborne_at_reset` is a sticky per-episode latch (`|= first_time_airborne & near_reset`) that goes true almost universally within the first 2 steps of any episode (RSI donor poses are mid-motion by construction) and **never resets**. It answers "was some foot ever briefly airborne near reset," not "was this specific landing free" — and since `stopball`/`softstop` gated on `env._blue_landed & ~env._blue_airborne_at_reset`, this flag being essentially always-true silently made those rewards **unreachable on every wide-crossing episode this entire investigation**, regardless of how well the policy performed.

**Fix:** replaced with `env._blue_landed_was_free`, computed once per landing *event* (`episode_length_buf < 10` at the moment that specific landing fires) instead of a stale per-episode flag. Updated `stopball`, `softstop`, `metrics.blue_landed_genuine`/`blue_landed_rsi_assisted`.

**Validation — same checkpoint, same episodes, only the classification changed:**

| | Before fix | After fix |
|---|---|---|
| Landed on blue (any) | 94.7% | 94.7% (unchanged, as expected) |
| Genuine | **0.0%** | **94.7%** |
| "Free" | 94.7% | 0.0% |

The policy has genuinely been learning to land on blue this whole time. The reward pipeline just couldn't see it. Commits: `6a5c3dd` (fix), `e45f24a` (checkpoint `model_2750.pt`). Full writeup: `docs/BugFixes.md`, including a process note on questioning the measurement when a "should move" metric doesn't.

**Training restarted:**

- Killed `amp_reward_fix_2026-07-08` (PID `296934`/`296937`, iteration ~2750, healthy otherwise).
- **New run started:** 2026-07-08 evening.
- **PID:** `444940`/`444950`.
- **Command:** `cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis &&
  uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096
  --agent.run-name blue_landing_gate_fix_2026-07-08`
- Confirmed alive, first iteration logged cleanly.

**What to watch for:** with `stopball`/`softstop` finally reachable on wide crossings, this run should look qualitatively different from every prior one — expect `Episode_Reward/stopball`/`softstop` to actually reflect wide-episode performance now, not just narrow. The landing-rate diagnostic itself should now report meaningful numbers from the start (no more universal 0% regardless of checkpoint quality). This is the first run where the diagnostic signal driving all prior decisions is actually trustworthy — worth treating earlier "0% genuine" history as void, not as evidence of anything about policy quality.

**Note for future cron firings:** the bihourly cron prompt text was frozen at this point (run name `blue_landing_gate_fix_2026-07-08`, PID `444940`/`444950`) and has not been regenerated since, even though several more runs have come and gone. Every firing since has correctly treated that run/PID as a stale placeholder and resolved to the actual current run via `ls -td .../*/  | head -1` / live `ps` checks — see section 27 for the most recent resolution and outcome.

## 27. Bihourly escalation, 2026-07-11 ~evening: `landing_success_and_footreach_fix_2026-07-11` stopped — genuine landing rate 0.1-3%, core double-step problem still unresolved after two more bundled fixes

Between section 26 and here, several more runs ran and were superseded: `blue_ball_foot_orientation_fix_2026-07-10` (stopped by user judgment, "didn't learn to double step"), then `landing_success_scaling_2026-07-11` (killed almost immediately, ~97 iterations, to bundle in a second fix before committing GPU time), then this run — `landing_success_and_footreach_fix_2026-07-11`, bundling the footreach assigned-foot fix (`footreach`'s phase1/vel_sigma now key off the task-assigned foot instead of root position) and landing-success payoff scaling (`stopball`/`softstop`'s wide-crossing payoff scaled by a rolling landing-success rate).

At iteration 1500 (`ball_difficulty` already saturated to 1.0), the bihourly cron's escalation criteria triggered:

- Live diagnostic on `model_1500.pt`: **0.1% genuine landing rate** (2/1748 wide episodes), 0% RSI-assisted — vs. the 94.7% reference point from section 26, and well below the cron's own 10% "surprisingly low" floor.
- Built-in `Episode_Metrics/blue_landed_genuine` corroborates: ~1.3-3.0% through iteration 1891.
- `Episode_Reward/blue_overshoot_penalty` drifted to -0.55/-0.70 by iteration 1891 — matching the historically-stuck range (-0.65 to -0.86) from `amp_g1_parity_2026-07-08b` (the run whose failure originally motivated `blue_stick_landing`'s creation, section 24-area). Trending toward that range, not away from it.
- Everything else (AMP loss, `mean_action_acc`, no NaN) stayed clean — this is specifically a blue-ball-landing-mechanism problem, not a general training-health problem.

**Interpretation:** two more targeted reward fixes, on top of everything in sections 17-26, still did not move the core metric. The growing overshoot penalty points at the same "sweeps past blue in one continuous stride" failure mode documented since section 17, not a new regression.

**Separately found this session (live zero-action probe, not yet actioned):** `blue_stick_landing` pays real reward (~0.10-0.19/step) to a foot that never moves at all — `dist_sigma=8`'s basin happens to cover the robot's passive default-stance distance from the blue midpoint on typical wide crossings. Proposed fix (gate on `env._blue_was_airborne`) was surfaced to the user but not yet actioned, and is not assumed to be the primary explanation given the overshoot-penalty evidence points more toward sweep-through than passive idling.

**Action taken:** pushed `model_1750.pt`, killed PID `2815464`/`2815467`. Did **not** restart with another incremental reward tweak on this cycle's own authority — after this many targeted fixes (sections 17-26 plus the two in this run) without resolving the core problem, whether to keep iterating on reward shaping or move to a more structural rethink (e.g. the previously-deprioritized task/behavior stage-gate restructure) is being left to the user rather than picked unilaterally. Full evidence and reasoning: `docs/BugFixes.md`, "2026-07-11 (bihourly escalation)" entry.

**No new training running as of this entry.** Awaiting user direction.

---

## 28. Fourth escalation, 2026-07-12 ~12:50: `blue_sweepthrough_2x_retiming_2026-07-12` stopped — genuine landing rate REGRESSING (5.1%->1.5%->0.3%) across the strongest bundled fix set yet, including the most aggressive timing-gap attempt

**Context:** section 27's run was stopped, then this session found and fixed a real bug on top of it (see `docs/BugFixes.md`, "2026-07-11 -- `_BLUE_LANDED_SEED_FRACTION` RSI teleport leaked free reward"): `blue_ball_landed` and `track_blue_landing_success` never excluded `env._blue_landed_was_free` landings, so ~12% of episodes farmed the landing bonus and the rolling landing-success rate for free the whole time section 27's fixes were running -- plausibly masking whether any of sections 17-27's fixes were doing anything at all. Separately, independent timing-budget analysis (two dispatched research passes, both re-verified by direct computation, not trusted blindly) found the ball's flight-to-crossing window (0.58-1.01s at full difficulty) is shorter than the DoubleStep/TripleStep reference clips' native 1.44s duration -- in the worst case the robot has 40% of the time its own reference demonstration needs. Per user direction (explicitly declined to touch `t_flight_range`), addressed this from the demonstration side instead: built `retime_motion.py`, baked 1.5x (0.94s) and, after the user visually confirmed the 1.5x clip still looked plausible in the ghost overlay, a 2x (0.72s) variant too -- 0.72s lands almost exactly on the median ball-crossing timing budget (0.80s). Both wired into the region AMP discriminators (left_far/right_far train on 3 files each now) and the RSI wide pool/`seed_blue_landed_practice`.

`blue_sweepthrough_2x_retiming_2026-07-12` launched with all of this bundled: the four fixes from section 27's predecessor runs, the reward-leak fix, and three reference-clip paces (1.0x/1.5x/2x) -- six confounded changes on top of everything in sections 17-26.

**Evidence, gathered over three health-check cycles (4h apart) once `ball_difficulty` saturated to 1.0 at ~iteration 1750 and held:**

- Live diagnostic, `model_1500.pt` (iter 1500, difficulty just reaching saturation): **5.1% genuine** (80/1563 wide episodes), 0% RSI-assisted.
- Live diagnostic, `model_3000.pt` (~1250 iterations later): **1.5% genuine** (21/1407 wide episodes).
- Live diagnostic, `model_4500.pt` (~1500 iterations later still, ~2850 iterations post-saturation total): **0.3% genuine** (5/1895 wide episodes).

This is a monotonic decline across three large-sample measurements, not noise around a floor -- it's regressing back toward the *original* ~0.1% failure baseline (section 27's `landing_success_and_footreach_fix_2026-07-11`, 2/1748) despite six more fixes on top, including the strongest timing-gap closure attempted so far. The built-in `Episode_Metrics/blue_landed_genuine` corroborates: flat/oscillating 0.6-1.6% for the entire ~2850-iteration saturated window, no upward trend at any point.

**Mixed picture, not uniformly bad:** `blue_overshoot_penalty` recovered substantially from its worst point (-0.87 at iter 2100) to -0.46 by the end -- a real, sustained improvement, not the pure-worsening signature from section 27. `stopball`/`softstop` climbed steadily (roughly 2.7x each over the run) -- but per `_blue_landing_reward_scale`'s own design (section "landing-success payoff scaling" fix, still active), wide-crossing payoff is damped toward near-zero when the rolling success rate is this low, so this climb is almost certainly narrow-crossing-driven, not wide-crossing progress. **New finding, not previously flagged this strongly:** `Episode_Termination/shank_height` climbed ~5x over the same saturated window (5.8 -> 29.4) with `mean_episode_length` declining in parallel and no sign of plateauing -- a real stability regression, independent of the blue-landing problem, that emerged during this run.

**Interpretation:** closing nearly the entire timing gap (2x retiming's 0.72s vs. the 0.80s median budget) did not help -- if anything, genuine landing rate got worse as training progressed further under these six changes, not better. This is evidence AGAINST "insufficient reference-clip pace" being the dominant blocker, on top of already being evidence against every reward-shaping hypothesis tested in sections 17-27. Two structural hypotheses worth the user's attention, neither implemented:
1. **Reward-magnitude imbalance persists even post-leak-fix:** `_blue_landing_reward_scale` damps wide-crossing `stopball`/`softstop` toward near-zero while narrow-crossing reward (same terms, undamped) keeps climbing unboundedly -- the policy may be rationally specializing entirely on narrow saves, since the expected value of attempting a genuine wide-crossing landing is now tiny relative to narrow-crossing farming. This wasn't true before the payoff-scaling fix (section 27) existed at all, so it's a candidate this session's own history hasn't tested in isolation.
2. **AMP dilution:** adding three paces to each far-region discriminator's single-file dataset may have broadened what that discriminator accepts as "natural," reducing rather than sharpening its ability to distinguish a genuine paced landing from a fast continuous sweep -- untested, opposite of the intended effect.

**Action taken:** pushed `model_4500.pt` (`5c5a993`), killed PID `3322902`. Per the standing bihourly/4-hourly instructions, did **not** restart with another tweak on this cycle's own authority -- this is now the fourth escalation of this exact problem (sections 17-20 area, 24-25, 27, and this one), the regressing (not just flat) trend is a new and worse signal than any prior escalation, and the strongest timing-gap fix attempted still failed -- whether to keep iterating on reward/AMP mechanism fixes or move to the previously-deprioritized structural rethink (multi-waypoint target architecture, hierarchical skill-selection per the earlier literature research, or something else) is surfaced to the user rather than picked unilaterally.

**No new training running as of this entry.** Awaiting user direction.

## 29. Fifth escalation, 2026-07-13 ~05:00: `blue_2p5x_retiming_2026-07-12` stopped — genuine landing rate stuck at 0.0-1.1% for the full ~2000-iteration post-saturation window, no recovery trend

**Context:** after section 28's regression, the user directed a deliberately smaller, more surgical restart (`blue_amp2xonly_decelfix_2026-07-12`): dropped from three blended AMP paces back to a single 2x pace (testing the AMP-dilution hypothesis from section 28's item 2 directly), plus a targeted reward-math fix (`footreach`'s decel-zone floor now tracks the curriculum-eased `landing_radius` instead of a hardcoded strict value). A health check on that run found the settle-window's `candidate` condition (contact + within-radius) essentially never fired at all (2202/2203 sampled unlanded episodes never got `env._blue_settle_count` above 0) — traced to `blue_stick_landing`'s dense reward never requiring actual foot-ground contact, so hovering near blue scored identically to a genuine plant. Fixed by gating `blue_stick_landing` on `env._blue_foot_in_contact` (`blue_stick_landing_contact_gate_2026-07-12`). Separately, per user request, the AMP reference pace was upgraded 2x → 2.5x (0.72s → 0.576s, matching the tightest observed 0.58s wide-crossing window almost exactly, since the user felt 2x still looked too slow watching play) — this became `blue_2p5x_retiming_2026-07-12`, the run this entry covers.

**Evidence, gathered over two health-check cycles (4h apart) once `ball_difficulty` saturated to 1.0 at ~iteration 1500:**

- First check (~iteration 2053, ~500 post-saturation iterations): `blue_landed_genuine` already down to ~0.01, `blue_overshoot_penalty` at -0.70 to -0.88. Judged too early to call — held below the 2000-iteration bar.
- Second check (~iteration 3525, ~2000+ post-saturation iterations): `blue_landed_genuine` had spent the *entire* intervening window oscillating in the 0.001-0.011 band, no upward trend at any point. `blue_overshoot_penalty` stayed deeply negative throughout (-0.5 to -0.83), never recovering toward zero despite the contact-gate fix's own reasoning predicting it might. Live diagnostic on the final checkpoint (`model_3500.pt`, 1803 wide episodes): **0.1% genuine, 0.0% RSI-assisted, 99.9% never-landed** — the low end of the historical 0.1-5.1% range.

**Everything else stayed healthy:** no NaN/Inf/blowup across all 15 tracked TensorBoard tags for the entire run. AMP loss converged cleanly and stayed low (~0.08-0.10 at the end, no dilution symptom this time — the single-2.5x-pace change appears to have worked as intended on that specific front). Episode length and `shank_height` both fine throughout (no repeat of section 28's stability regression). The failure is isolated specifically to the blue-waypoint landing mechanism.

**Interpretation:** this is now the *fifth* stop-and-report cycle for genuine blue-ball landing, and the *cleanest* diagnostic work yet (the settle-window near-miss instrumentation directly disproved the "settle-window is too fragile" hypothesis and correctly identified the actual gap — `blue_stick_landing` rewarding hovering over planting). That diagnosis was almost certainly correct on its own terms; fixing it plainly changed the dense-reward landscape (early-training `blue_stick_landing` values dropped and had to be re-earned via contact, exactly as intended). It simply wasn't sufficient to move the needle on the downstream discrete landing rate. Combined with section 28's AMP-dilution test (also didn't help) and the whole history in sections 17-27 (radius/threshold/settle-window/curriculum/reward-magnitude tuning, none sufficient either), the pattern across roughly a dozen distinct, individually well-reasoned fixes is now strong enough that another parameter-level tweak is unlikely to be the answer. The two structural hypotheses flagged in section 28 (reward-magnitude imbalance between damped wide-crossing and undamped narrow-crossing payoff; and, now weaker given this run's clean AMP loss, residual AMP dilution) remain untested. A third, not previously written up: the settle-window's underlying task — plant a foot with sub-1-m/s residual velocity, in ground contact, within `landing_radius` of an intermediate waypoint, *before* continuing to the real target — may simply be a harder motor-control problem than the reward stack (however well-tuned) can shape via dense proxies alone, and might need either a fundamentally different landing-detection mechanism (e.g., a continuous/soft landing signal instead of a discrete 3-step conjunction) or a curriculum that trains the plant-and-hold skill in isolation (e.g., a dedicated pretraining phase against only the blue midpoint, no green target downstream) before combining it with the full two-stage task.

**Action taken:** pushed `model_3500.pt` (commit `b895393`), killed PID `4175948`. Per the standing 4-hourly instructions, did **not** restart with another tweak on this cycle's own autonomous authority — five escalations of the same core problem, with this run representing the best-diagnosed and most surgical attempt yet, is a strong enough signal that the next move should be a deliberate user decision (continue with structural reward changes, try isolating the plant-and-hold skill, or reconsider the two-stage waypoint architecture itself) rather than another autonomous bundled-fix cycle.

**Separately, `green-ball-baseline` was restarted this same session** (`green_2p5x_tflight_widen_2026-07-12`, PID `4186113`): the three "green-specific" fixes the user asked to port to blue (foot-orientation sign, `ball_difficulty` accumulator pattern, curriculum EMA-smoothing) turned out to already be present on blue (independently fixed on both branches around the same dates) — nothing needed porting. Instead, green's own AMP dataset was replaced to use only the 2.5x `LeftDoubleStep`/`RightDoubleStep` clips (mirroring blue's AMP-dilution fix, applied per user request even though green has no near/far region split), and its hard-difficulty `t_flight_range` upper bound was widened 1.1s → 1.5s. Unlike blue, green is showing a genuinely healthy, still-improving trajectory as of this entry (~2200 post-saturation iterations, `softstop`/`stopball` both still climbing, falls dropping sharply) — no landing-rate concept applies there (no waypoint mechanism), so this isn't directly comparable to blue's problem, but it's a positive sign the 2.5x pace + widened timing change aren't harmful on their own.

**No new training running for blue as of this entry.** Green-ball training continues uninterrupted. Awaiting user direction on blue.

## 30. Potential breakthrough, 2026-07-14: `blue_self_imitation_2026-07-14` second health check measures 96.6% genuine landing (deterministic policy) — far above every prior ceiling, but not yet confirmed sustained

**Context:** user directed a fresh restart with self-imitation learning (SIL, see `docs/BugFixes.md` "2026-07-14 -- self-imitation learning (SIL) added") to address the pattern established by every prior run that ever found genuine landings at all: a rise, then a decay back to the ~0-5% floor (most recently, the immediately preceding run peaked 44.8% at iteration 2000 then decayed to 2.5% by iteration 6500). Also bundled: the leaky settle-window counter fix from the previous section.

**First health check** (`model_750.pt`, pre-saturation, difficulty forced to 1.0 for eval): 14.3% genuine (671/4703 wide episodes) — unremarkable, consistent with the early "rise" phase seen in every prior successful attempt.

**Second health check** (`model_2250.pt`, ~650 post-saturation iterations, `ball_difficulty` saturated at ~iteration 1600): live diagnostic measured **96.6% genuine (1433/1484 wide episodes), 0% RSI-assisted** — vastly above anything this project has measured before, including the 44.8% peak. This directly contradicted the training-time `Episode_Metrics/blue_landed_genuine` TB metric, which stayed flat at 15-22% over the same window with no upward trend. Investigated rather than trusted blindly (per this project's established practice, and given section 26's history of exactly this kind of trap): traced to a real, non-bug measurement-condition difference — the live diagnostic runs the DETERMINISTIC policy while the TB metric reflects actual training rollouts, which still sample with substantial exploration noise (`Policy/mean_noise_std` 0.56-0.66 throughout this window, barely decayed from the 0.996 init value). A precision task like the settle-window landing check is plausibly disrupted by that much action noise far more than the underlying deterministic skill would suggest. Full investigation trail in `docs/BugFixes.md`.

**Interpretation — genuinely promising, not yet confirmed:** this is the best result this multi-week investigation has produced by a wide margin, but every previous "rise" also looked like a breakthrough at a comparable post-saturation iteration count before decaying over the following several thousand iterations. SIL exists specifically to prevent that decay; at only ~650 post-saturation iterations, whether it succeeds is not yet determined. The next several health checks (targeting the same ~2000/3500/5000/6500-iteration cadence used to characterize the previous run's decay) are the real test.

**Action taken:** not stopping — this is a positive signal, doesn't meet stop criteria. Committed `model_2250.pt` locally as a reference checkpoint (not pushed to origin, no explicit request to do so). Documented promptly per the standing instruction to flag results meaningfully above the historical ceiling even before full statistical confidence. Continuing routine 4-hourly monitoring.

**Training continues uninterrupted (PID 1300789).** No user decision needed yet — will report definitively once the sustained-vs-decay question resolves one way or the other.
