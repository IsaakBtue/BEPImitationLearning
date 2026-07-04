# Hard-gate the blue-ball landing, rebalance RSI, add landing diagnostics

**Date:** 2026-07-04
**Track:** SimpleGoalKeeper (`src/simple_goalkeeper/`)
**Status:** Approved, implementing
**Supersedes/extends:** `docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md` (the soft/timeout-gated landing mechanism this design hardens)

## Problem

The 2026-07-04 landing gate (`_get_reach_target_y`, prior spec above) made the blue→green
target switch fire early on a genuine foot landing, but kept the original
`elapsed >= t_flight/2` switch as a fallback for episodes that never land. In practice this
means the schedule advances to the green target on a timer regardless of whether landing
ever happens, so a robot that never learns to land at blue still gets the green-target
reward shaping every episode — the fallback was never actually forcing the behavior.

Research done before this design (see conversation, summarized here for the record):

1. **Reward-log math correction.** `Episode_Reward/blue_ball_landed` values must be
   converted via this repo's own documented formula (`SimpleGoalKeeper/CLAUDE.md`,
   "Reading TensorBoard / WandB Episode Reward Metrics": `rate = logged_value × 150 / weight`
   for a one-shot term). Applied to the live run `2026-07-04_03-47-17`, the actual
   `blue_ball_landed` firing rate is **~35-40% of all episodes**, not the ~2-3% first
   (incorrectly) reported.
2. **Wide-crossing fraction.** Monte Carlo of the actual `reset_ball_rolling` sampling
   code at the current max curriculum difficulty (`_ball_difficulty = 1.0`, confirmed from
   the log since iter ~3000) gives **~48% of episodes are wide crossings**
   (`|crossing_y - start_y| > 0.6`). So the landing rate *conditioned on being wide* is
   roughly 35-40% / 48% ≈ **73-83%** under today's soft/timeout gate — looks healthy.
3. **But this metric is confounded by RSI.** `_tier_npz_reset` (`events.py`) routes ~50% of
   wide-crossing resets to a random frame from the DoubleStep/TripleStep motion clips —
   clips that are themselves mid-step reference motions for reaching a wide target. A
   randomly sampled frame from such a clip plausibly already has the assigned foot lifted
   and positioned near the blue target, satisfying `_blue_was_airborne`/`_blue_landed`
   within a step or two of reset with zero credit due to the policy. This is consistent
   with the user's direct observation in `sgk_play`: **0% visible genuine landings**, despite
   the ~73-83%-of-wide metric. It's also consistent with project history: `tier_rsi_fraction`
   was raised 30%→50% on 2026-07-03 specifically because at 30% the policy was **not**
   converging on double-stepping at all — i.e. there's already documented evidence the
   policy struggles to produce this behavior unassisted.

Net: the existing soft gate's "success" is largely explained by free RSI credit, not
learned behavior. Removing the timeout fallback outright (making landing mandatory to ever
reach the green target) is the intended fix, but doing that alone, on top of a 50%
RSI-inflated metric, risks looking like it works while still not teaching anything.

## Goal

1. Make the blue→green switch depend **only** on a genuine landing event, no time-based
   escape hatch.
2. Reduce the RSI "free credit" confound so the policy is forced to learn the behavior
   from a normal starting state in most episodes.
3. Scale the `blue_ball_landed` bonus with existing curriculum machinery, since it's no
   longer just an auxiliary signal — it now gates whether the double-step choreography is
   even reachable.
4. Add logging that can tell a genuine landing from an RSI-assisted one, since three
   variables are changing in the same run and post-hoc interpretation needs a way to
   separate their effects.

All four changes ship together in one training run (user's explicit choice, weighed
against the alternative of testing the RSI rebalance in isolation first — rejected as
slower; the diagnostic in (4) is the agreed mitigation for that risk).

## Design

### 1. Schedule: remove the time-based fallback

`_get_reach_target_y` (`mdp/rewards.py:143-243` currently) currently computes:

```python
elapsed = env.episode_length_buf.to(full_y.dtype) * env.step_dt
first_half = elapsed < (t_flight / 2.0)
...
phase1_active = wide & first_half & ~env._blue_landed
```

New:

```python
phase1_active = wide & ~env._blue_landed
```

`elapsed`, `first_half`, and the `t_flight = getattr(env, "_ball_t_flight", None); if t_flight
is None: return full_y` guard are all removed — `_get_reach_target_y` no longer depends on
`_ball_t_flight` at all. `_blue_was_airborne`/`_blue_landed` latch logic (airborne-then-
contact-within-0.3m detection) is unchanged. The only other override remains
`footreach`/`foot_proximity`'s existing `ball_close = ball_x_local < 0.5` switch to the live
ball position — that backstop is untouched, so episodes still cannot stall forever; a
robot that never lands just gets no green-target shaping until the ball is nearly on top
of it, instead of getting it for free at `t_flight/2`.

### 2. `blue_ball_landed`: flat weight → curriculum

Reuse `reward_curriculum_ep_len` (`mdp/events.py`), the same generic mechanism already
driving `footreach_curriculum`/`stopball_curriculum`/`softstop_curriculum` — not a new
curriculum class. Add to `goalkeeper_env_cfg.py`, alongside the other `reward_curriculum_ep_len`
entries (currently ~line 159-172):

```python
cfg.curriculum["blue_ball_landed_curriculum"] = CurriculumTermCfg(
    func=gk_mdp.reward_curriculum_ep_len,
    params={
        "reward_name": "blue_ball_landed",
        "base_weight": 10.0,     # unchanged base; now ramps like footreach
        "update_interval": 500,
        "ep_len_divisor":  47,
    },
)
```

Weight ramps 10→25 across `_curriculumupdate` 0→3 (shared episode-length-driven counter),
same ceiling as `footreach`, since the two are coupled through the same reach-target
subsystem. The existing `RewardTermCfg(func=gk_mdp.blue_ball_landed, weight=10.0, ...)`
entry is unchanged — `weight=10.0` there is just the seed value; `reward_curriculum_ep_len`
overwrites `env.reward_manager.get_term_cfg("blue_ball_landed").weight` on its first call
(fires immediately, `_last_update` initialized to `-update_interval`).

### 3. RSI fraction rebalance

`tier_rsi_fraction`: 0.5 → 0.1, `live_rsi_fraction`: 0.3 → 0.7 (standing stays 0.2; sums
still to 1.0). Three call sites must all change together (the 2026-07-03 fraction change
history shows this file/config pattern):

- `MotionResetManager.reset()` defaults, `mdp/events.py:269-270`.
- Module-level `reset_from_motion_data()` defaults, `mdp/events.py:367-368` — **this is
  the one `sgk_play --rsi` actually uses** (it re-registers the event with no params
  dict, per the existing 2026-07-03 note in `CLAUDE.md`, so it silently relies on these
  function defaults). Missing this site would leave play mode testing the *old* 50/30/20
  split while training runs the new one.
- `goalkeeper_env_cfg.py:563`, the explicit `params={"tier_rsi_fraction": 0.5,
  "live_rsi_fraction": 0.3}` dict passed to the training config's `reset_from_motion_data`
  event.

Known tension, documented rather than resolved: `tier_rsi_fraction` was raised 30%→50% on
2026-07-03 because 30% wasn't enough exposure to produce double-stepping at all. Dropping
to 10% risks reintroducing that failure mode, just observed through a different symptom
(no double-stepping at all, vs. today's illusory landing metric). This is the acknowledged
and accepted risk of bundling the RSI change with the hard gate in one run rather than
testing it in isolation first (user's explicit choice — see diagnostic in (4) as the
mitigation).

### 4. Diagnostic: genuine vs. RSI-assisted landing

SimpleGoalKeeper does not currently use mjlab's `cfg.metrics`/`MetricsTermCfg` mechanism
(checked — no `cfg.metrics` assignment anywhere in `goalkeeper_env_cfg.py`; the one
pre-existing `Episode_Metrics/mean_action_acc` tag comes from a different task's default,
not anything SGK wired). `MetricsTermCfg` is the right fit here over a `RewardTermCfg`
hack: metrics have no weight/dt scaling (`mjlab/managers/metrics_manager.py` — "Unlike
rewards, metrics have no weight, no dt scaling, and no normalization by episode length"),
avoiding exactly the conversion-formula mistake made earlier in this same investigation.
`reduce="last"` reports the final-step value of the episode rather than a mean — correct
for a one-shot per-episode 0/1 classification.

**New state in `_get_reach_target_y`:** alongside `_blue_was_airborne`/`_blue_landed`, add
`env._blue_airborne_at_reset` (bool, latched once per episode): true if
`_blue_was_airborne` transitions False→True while `episode_length_buf <= 2` — one step
wider than the `just_reset = episode_length_buf <= 1` convention used elsewhere in this
function, to give the physics a single settle step post-reset before treating an airborne
reading as meaningful. This flags "the foot was already up essentially at reset" — i.e.
plausibly an RSI artifact, not something the policy did.

**New metric functions** (`mdp/metrics.py`, new file — mirrors the `MetricsTermCfg`
pattern from `mjlab/tasks/velocity/velocity_env_cfg.py`):

- `blue_landed_genuine(env, ball_name, asset_cfg) -> torch.Tensor`: `env._blue_landed &
  ~env._blue_airborne_at_reset`, cast to float.
- `blue_landed_rsi_assisted(env, ball_name, asset_cfg) -> torch.Tensor`: `env._blue_landed
  & env._blue_airborne_at_reset`, cast to float.

Both call `_get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)` first (same
freshness-guard idiom as `blue_ball_landed` itself) before reading the latches.

**Config wiring** (`goalkeeper_env_cfg.py`, new `cfg.metrics = {...}` block):

```python
cfg.metrics = {
    "blue_landed_genuine": MetricsTermCfg(
        func=gk_mdp.blue_landed_genuine,
        params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        reduce="last",
    ),
    "blue_landed_rsi_assisted": MetricsTermCfg(
        func=gk_mdp.blue_landed_rsi_assisted,
        params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        reduce="last",
    ),
}
```

Logged as `Episode_Metrics/blue_landed_genuine` and `Episode_Metrics/blue_landed_rsi_assisted`
— read directly as per-episode rates, no formula needed, unlike the reward-log values that
caused the earlier miscalculation in this investigation.

## Out of scope

- No change to the 0.3m landing radius or the airborne-then-contact detection logic itself.
- No change to `footreach`/`foot_proximity`'s `ball_close < 0.5m` live-ball override — this
  remains the true backstop against episodes stalling.
- No change to the standing-pose branch (20%) of the RSI three-way split.
- No change to `is_left_ball` foot-side assignment (still keyed off true `crossing_y`).
- No retroactive fix to the currently-running, already-diverging training run
  (`2026-07-04_03-47-17`) — this design requires a fresh run regardless, since the reward/
  schedule/RSI code all change.

## Testing

Existing tests that assert the **old** time-based fallback must be updated, not left to
fail:

- `tests/simple_goalkeeper/test_reach_target_two_stage.py`: `test_wide_crossing_second_half_targets_full_point`,
  `test_flight_time_half_boundary_is_second_half_not_first`, `test_missing_t_flight_falls_back_to_full_target`,
  and the second-half assertion inside `test_mixed_batch_wide_narrow_and_both_flight_halves_independently`
  all currently assert "elapsed past t_flight/2 ⇒ full target regardless of landing." These
  must change to assert the target **stays at the midpoint** past that point when landing
  hasn't occurred (only `wide & ~landed` matters now, elapsed time no longer does).
- `tests/simple_goalkeeper/test_blue_ball_landing_gate.py`:
  `test_landing_gate_fallback_advances_at_half_flight_time_when_never_landed` must flip —
  it should now assert the target does **not** advance when landing never happens, even
  well past `t_flight/2`.

New tests needed:

- RSI fraction: extend the existing three-way-split statistical tests
  (`test_live_rsi.py`, per the 2026-07-03 update history) to the new 10/70/20 split.
- New `_blue_airborne_at_reset` latch: fires when airborne is first true within the
  2-step grace window; does not fire when airborne first becomes true later in the
  episode (i.e. the policy visibly lifted the foot mid-episode).
- `blue_landed_genuine`/`blue_landed_rsi_assisted`: mutually exclusive given `_blue_landed`
  true (exactly one fires, never both, never either without `_blue_landed`); both zero when
  `_blue_landed` is false.

## Documentation

- `SimpleGoalKeeper/CLAUDE.md`: update the existing "landing gate" divergence-table row
  (2026-07-04) to reflect the fallback removal, add a new row for the RSI fraction change
  (mirroring the format of the existing 2026-07-03 fraction-change row), and add the two
  new metrics to the "Reward Design" section (or a new "Diagnostics" section, since they
  aren't rewards).
- `SimpleGoalKeeper/docs/BugFixes.md`: dated entry per the mandatory fix-log rule, covering
  all three code changes and citing this spec.

## Post-plan note (not a task)

This is a code-only change; it requires a fresh training run to evaluate. Given the design
explicitly accepts the risk of bundling three variables in one run, the two new
`Episode_Metrics` (genuine vs. RSI-assisted landing) plus a `sgk_play` visual check should
be the first things reviewed once that run has enough iterations to produce wide-crossing
episodes past the current difficulty ramp-up (~iter 3000, per the current run's
`ball_difficulty` curve). If `blue_landed_genuine` stays near zero while
`blue_landed_rsi_assisted` accounts for most landings, that's the reintroduced
non-convergence risk materializing, and `tier_rsi_fraction` may need to be walked back up.
