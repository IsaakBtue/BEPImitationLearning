# Orange-ball trailing-foot waypoint mirror

**Date:** 2026-08-08

## Problem

During interception, the trailing (non-assigned) foot is sometimes stationary and does
not lift/step forward to complete the double-stepping motion. The leading (assigned)
foot has a rich positional-target mechanism — the "blue ball" waypoint
(`_get_reach_target_y`, `rewards.py:340-510`) — that gives it a concrete Y-axis target
to approach and land on during wide (far-region, multi-step) crossings, feeding
`footreach`, `foot_proximity`, `blue_ball_landed`, `blue_overshoot_penalty`, and
`blue_stick_landing`. The trailing foot has no positional target at all: only
`trailing_foot_forward_continuous` (orientation-only, always active) and `foot_clearance`
(weight +2.0, `max(height)` **across both feet** — fully satisfied once the leading foot
alone lifts, giving the trailing foot zero pressure of its own to lift).

Separately, an ongoing sideways-leading-foot-landing investigation is out of scope for
this change (tracked independently — see `docs/BugFixes.md`'s 2026-08-07
`footyawspinfix` entries).

## Goal

Give the trailing foot its own positional target ("orange") and a landing-focused
reward subset mirroring blue's, so it has a concrete reason to move during a
double/triple-step approach, plus its own viewer marker so training behavior can be
visually verified. Landing-focused subset only, deliberately narrower than a full
mirror of blue's entire reward stack (see "Explicitly out of scope" below) — this
project's own history (`docs/BugFixes.md`) shows every prior addition here landed best
as a small, single change validated by a live training run before the next one was
layered on.

## Target formula

Let `start_y = env.scene.env_origins[:, 1]` (robot stance Y) and
`full_y = _get_ball_crossing_y(env, ball_name)` (the true/"green" ball-crossing Y, same
value blue already computes). Define `delta = full_y - start_y` (signed — negative for
right-side crossings).

```
shrunk   = sign(delta) * max(|delta| - 0.30, 0.0)   # 30cm off the top, sign-safe
orange_y = start_y + shrunk / 2.0
```

Confirmed with the user via worked examples:

| delta | blue_y (existing) | orange_y (new) |
|---|---|---|
| +1.00 m | start+0.50 m | start+0.35 m |
| +0.40 m | start+0.20 m | start+0.05 m |
| +0.20 m | start+0.10 m | start+0.00 m (collapses to start_y) |
| -1.00 m | start-0.50 m | start-0.35 m (sign preserved, not flipped) |

The `sign(delta) * max(|delta|-0.30, 0)` shrink (rather than a plain `delta - 0.30`) is
required for correctness on right-side crossings — a plain subtraction would push the
right-side target further out instead of pulling it in, the opposite of the intent.

This is implemented as a new pure function, `_get_orange_reach_target_y`, parallel to
(and calling `_get_ball_crossing_y` the same way as) `_get_reach_target_y`, but:

- keyed to the **trailing foot**: `trailing_idx = 1 - _get_correct_foot_idx(env, ball_name)`
  instead of the leading foot,
- uses the shrink formula above instead of blue's plain `(full_y - start_y) / 2.0`,
- reuses `env._blue_wide` directly for the wide/narrow gate — **no new wide/region
  computation** is added, since "wide" is a property of the ball's crossing geometry
  (is this a far-region, multi-step crossing), not which foot is being tracked. Blue
  already computes and caches this every step.
- copies blue's curriculum-eased `landing_radius` (0.20→0.15 m) and
  `landing_speed_threshold` (2.0→1.0 m/s) lerps, and the leaky-settle-count mechanism,
  verbatim in shape — these are blue's own best-evidenced mechanisms (see
  `_get_reach_target_y`'s docstring, "44.8% genuine landing rate" finding), not
  something to re-derive from scratch.
- maintains its own state namespace, entirely separate from blue's: `_orange_was_airborne`,
  `_orange_landed`, `_orange_settle_count`, `_orange_landed_was_free`,
  `_orange_landed_genuine`, `_orange_last_settle_step`, `_orange_landing_radius_current`.
  Blue's own `_blue_*` state is never read or written by this mechanism except
  `_blue_wide` (read-only).

## New reward functions (landing-focused subset)

| Function | Mirrors | Shape |
|---|---|---|
| `orange_foot_proximity` | `foot_proximity` | Dense `exp(-sigma·dist)` pull of the trailing foot toward the orange target. |
| `orange_ball_landed` | `blue_ball_landed` | One-shot bonus when the trailing foot genuinely lands at orange (same settle-window + speed-gated logic). |
| `orange_overshoot_penalty` | `blue_overshoot_penalty` | Penalty for the trailing foot advancing past orange, toward the true crossing point, before landing there. |
| `orange_stick_landing` | `blue_stick_landing` | Dense "close AND slow" bonus near orange, same anti-oscillation rationale as blue's. |

Each is a straight structural copy of its blue counterpart with: (a) the foot index
swapped to `trailing_idx`, (b) the target sourced from `_get_orange_reach_target_y`
instead of `_get_reach_target_y`, (c) the phase-active gate using
`env._blue_wide & ~env._orange_landed_genuine` in place of
`env._blue_wide & ~env._blue_landed_genuine`.

### Explicitly out of scope for this change

- `footreach`'s full phase1 (lateral pre-position)/phase2 (sigmoid reach × vel_sigma)/
  overshoot-kill-flag/blue-decel-zone machinery — the most complex and most
  ball-interception-specific piece of blue's stack. Not mirrored yet.
- `near_stick_reach` (narrow-crossing anti-oscillation) — not needed since orange is
  wide-only (see gating below).
- `blue_trunk_drive` (whole-body lateral drive toward the target) — not mirrored yet.

These can be added later as a follow-up if the landing-focused subset alone doesn't
resolve the stationary-trailing-foot symptom, each validated by its own training run
per this project's established pattern.

## Gating

Orange rewards fire only when `env._blue_wide` is true — i.e. only on the same
far-region, multi/double/triple-step crossings blue already targets. Per this project's
own RSI region-routing design (single-step motions are used for narrow/near crossings;
double/triple-step motions for wide/far ones — `docs/BugFixes.md`), the trailing foot
only has genuine double-stepping work to do on wide crossings in the first place, so no
new wide/narrow logic is needed for orange — it reuses `env._blue_wide` as computed by
`_get_reach_target_y` this same step.

## Weights (conservative first pass)

Half of blue's current curriculum weights, since this is a brand-new, unvalidated
mechanism and the trailing foot's job is secondary to the leading foot's actual ball
interception — avoids it competing too strongly against already-tuned leading-foot/ball
rewards for gradient share until there's live evidence to tune from.

| Term | Weight |
|---|---|
| `orange_ball_landed` | +5.0 → 10.0 (curriculum, mirrors `blue_ball_landed_curriculum`) |
| `orange_overshoot_penalty` | -15.0 → -30.0 (curriculum) |
| `orange_stick_landing` | +4.0 → 8.0 (curriculum) |
| `orange_foot_proximity` | +2.5 (flat) |

A new `orange_ball_landed_curriculum` `CurriculumTermCfg` entry mirrors
`blue_ball_landed_curriculum` (`goalkeeper_env_cfg.py:331`).

## Visualization (`play.py`)

Extend `_patch_viewer_intercept_vis` (`play.py:397-500`) to also draw an **ORANGE**
sphere (rgba `[1.0, 0.55, 0.0, 0.75]`) at the current orange target, using the exact
same `_add_sphere`/`_add_line` helpers already defined in that function. Visible under
the same condition as blue's own sphere (`env._blue_wide`), independent of
`env._orange_landed` (no color-switch-on-landed / no "graduate to a second target" the
way blue graduates to green — the landing-focused subset has no live-ball-tracking
phase for the trailing foot to graduate into). Reads `env._orange_landed`/the orange
target the same cache-based way the existing code reads `env._blue_wide`/`env._blue_landed`,
per that function's own documented rationale ("so the marker can't drift out of sync
with what [the rewards] are actually gating on").

This is a viewer-only change (no training effect) and directly reflects the real reward
target, satisfying this project's Visualization Honesty Rule.

## Documentation

- A dated 2026-08-08 entry in `docs/BugFixes.md`, in the style of the existing
  2026-07-23 blue-ball-waypoint entry, describing the mechanism and explicitly listing
  what was/wasn't ported (mirroring that entry's own "NOT ported" section convention).
- A new row in `CLAUDE.md`'s "Divergences from G1 Upstream" table (no G1 equivalent —
  same justification class as blue's own "Two-stage blue/green waypoint mechanism" row)
  and new rows in the "Reward Design" table for the four new terms.
- Each new function's docstring and the BugFixes.md entry will state "Not yet validated
  against a live training run," consistent with this project's convention for every
  other new reward term.

## Testing

A lightweight fake-env unit test for `_get_orange_reach_target_y`'s shrink-then-halve
formula, mirroring `tests/simple_goalkeeper/test_landing_speed_threshold_curriculum.py`'s
pattern (a minimal `_FakeEnv`/`_Scene` with no real `robot`/`feet_contact` scene entries
needed, since the target-Y math is computed before that branch). Cases:

- Positive `delta` well above 0.30 m (matches the worked-example table above).
- Negative `delta` (sign safety — must shrink toward zero, not flip past it).
- `|delta| <= 0.30` m (floors to 0 — target collapses to `start_y`).

## Non-goals / risks carried forward

- Not a fix for the leading-foot sideways-landing issue (separate, already-tracked
  investigation).
- Like nearly every other reward addition in this project's history, this is not yet
  validated against a live training run — a fresh run is needed to confirm the
  trailing-foot-stationary symptom actually improves, and the conservative (halved)
  weights may need retuning once there's live evidence.
