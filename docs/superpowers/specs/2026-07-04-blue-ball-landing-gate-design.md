# Blue-ball landing gate for the two-stage footreach schedule

**Date:** 2026-07-04
**Track:** SimpleGoalKeeper (`src/simple_goalkeeper/`)
**Status:** Approved, implementing

## Problem

The two-stage footreach/foot_proximity target (`_get_reach_target_y`, added 2026-07-03,
commit `0d1f2e6`) schedules an intermediate "blue ball" waypoint (midpoint between stance
and the true crossing point) for wide crossings (`|crossing_y - start_y| > 0.6`), switching
to the full "green ball" crossing point once `elapsed >= t_flight / 2`.

This switch is **purely time-based** — nothing checks that the robot's foot actually
reached the blue ball before the schedule advances. In practice the policy has not learned
the intended pause-at-blue-then-continue-to-green double-step motion: it can glide/leap
through without ever placing a foot near the intermediate point, and the schedule advances
anyway once time elapses.

## Goal

Gate the phase-1 → phase-2 transition on an actual foot-landing event at the blue ball,
detected via the existing `feet_contact` `ContactSensor`, so genuinely landing there becomes
a precondition (not just a time-elapsed side effect) for reaching the green ball's
higher-value phase-2 reward.

## Design

### Landing definition

The assigned foot (`_get_correct_foot_idx` — same foot `footreach`/`airborne_at_save`/
`cleanstop` already track) must, within one episode:

1. Be **airborne** at some point (`feet_contact.found == 0` for that foot) — prevents a
   robot that never lifts the foot from satisfying the gate for free.
2. Subsequently register **ground contact** (`found > 0`) while within **0.3 m** (matches
   `footreach`'s existing `reach_th`) of the phase-1 target point
   `(goal_x_w, half_y, floor_z + 0.10)` — the exact point the blue sphere is drawn at in
   `play.py`.

### State

Two new per-env latched `torch.bool` buffers, cached on `env` following the existing
`_ball_crossing_y`/`_ball_t_flight` idiom (lazily created, reset on
`episode_length_buf <= 1`):

- `env._blue_was_airborne` — OR'd true once the assigned foot is ever off the ground.
- `env._blue_landed` — OR'd true the first step where `_blue_was_airborne` is already
  true, the foot is now in contact, and it's within radius. Latched for the rest of the
  episode once true.

Both computed inside `_get_reach_target_y` (extending it directly — same function already
owns the two-stage schedule, `_ball_crossing_y`, and `t_flight` lookups), so every caller
observes consistent state with no new cross-module wiring.

### Schedule change

```python
phase1_active = wide & first_half & ~env._blue_landed
return torch.where(phase1_active, half_y, full_y)
```

Landing before `t_flight / 2` advances the target early. Never landing falls back to
exactly today's behavior — forced advance at `t_flight / 2`. This is a hard gate with a
timeout fallback: landing is required to advance early, but episodes can never stall
waiting for a landing that doesn't come.

`is_left_ball`/foot-side assignment in `footreach`/`foot_proximity` is unaffected — it
already keys off the true `crossing_y`, not the staged target.

### New reward: `blue_ball_landed`

One-shot bonus, same flag idiom as `cleanstop`/`airborne_at_save`:

```python
def blue_ball_landed(env, ball_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG) -> torch.Tensor:
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_landed is fresh this step
    if not hasattr(env, "_blue_landed_bonus_flag"):
        env._blue_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_landed_bonus_flag[just_reset] = False
    fired = env._blue_landed & ~env._blue_landed_bonus_flag
    env._blue_landed_bonus_flag |= fired
    return fired.float()
```

The explicit `_get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)` call guards against
reward-term evaluation order in the config — this function must not silently read a stale
flag if it happens to run before `footreach`/`foot_proximity` in a given step. `asset_cfg` is
threaded through (not left as the module default) so it resolves to the same config-wired
`SceneEntityCfg` (with populated `body_ids`) that `footreach`/`foot_proximity` already use —
added during implementation (Task 1's rewrite of `_get_reach_target_y`), not present in this
earlier draft; corrected here 2026-07-04 per final-review finding of spec/implementation drift.

**Weight: +10.0 flat, no curriculum.** Auxiliary shaping signal for an intermediate
sub-goal, not a primary outcome like `stopball` (100→250) or `cleanstop` (+25) — comparable
in scale to `footreach`'s base weight (10→20), but flat since curriculum-scaling an
already-secondary bonus adds tuning surface for little benefit. Only relevant on wide
crossings (same `>0.6` threshold that creates the blue target); narrow crossings are
entirely unaffected (no blue target exists for them).

### Config wiring

New `RewardTermCfg(func=mdp.blue_ball_landed, weight=10.0, params={"ball_name": BALL_NAME})`
in `goalkeeper_env_cfg.py`'s reward terms, placed after `footreach`/`foot_proximity` in the
dict (belt-and-suspenders alongside the explicit refresh call above).

## Out of scope

- No change to narrow-crossing behavior (never had a blue target).
- No change to `is_left_ball` foot-side assignment logic.
- No play.py visualization change (blue/green sphere logic is unaffected by this gate;
  could be revisited separately if useful to see landing events visually).
- No curriculum on the new bonus weight.

## Testing

New tests alongside the existing `test_reach_target_two_stage.py` /
`test_footreach_two_stage_wiring.py`, covering:
- Landing gate never fires without a prior airborne transition.
- Landing gate fires on contact within radius after airborne, advances target early.
- Fallback still advances at `t_flight/2` when landing never occurs.
- `blue_ball_landed` fires exactly once per episode, gated by the same latch.
- Narrow crossings unaffected (no latch state changes phase1/phase2 selection since `wide`
  is false).

## Documentation

- `SimpleGoalKeeper/CLAUDE.md`: new row in "Divergences from G1 Upstream" table (G1 has no
  equivalent mechanism, same as the parent two-stage-target row) + new row in the reward
  table.
- `SimpleGoalKeeper/docs/BugFixes.md`: dated entry per the mandatory fix-log rule.
