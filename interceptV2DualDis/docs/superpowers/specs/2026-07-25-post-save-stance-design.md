# Post-save recovery pose: retarget to a straight-leg, 45°-arms-out stance

**Date:** 2026-07-25
**Status:** approved, not yet implemented

## Problem

After a save, the robot settles into an unstable end pose. `postupperdofpos`
(arms) and `postlegdofpos` (legs) currently reward returning to
`robot.data.default_joint_pos` — the same crouched, bent-knee stance used for
`HOME_KEYFRAME` (episode reset / ready-to-defend). That stance is a deliberate
"ready to move" pose (bent legs = ready to spring into a dive), and per user
report, training does not reliably converge to it post-save either. A bent
crouch requires continuous active torque to hold; it's plausibly harder for
the policy to settle into stably right after the chaotic dynamics of a
dive/step than a straight-legged stance would be.

## Non-goals

- `HOME_KEYFRAME` (the episode-reset pose) is NOT changed. It's confirmed
  fine for its purpose (ready-to-defend stance at reset / during approach).
- No change to `postwaistdofpos`, `postorientation`, `postangvel`,
  `postlinvel`, or the `_ball_is_behind()` gating mechanism all of these
  (and the two functions being changed) already share.
- No new reward term, no new gating/timing mechanism, no curriculum change.

## Design

Retarget `postupperdofpos` and `postlegdofpos` from `default_joint_pos` to a
new hardcoded stance constant: legs straight, arms out to the sides at 45°.

Target values (derived from `t1_headless.xml`'s own joint zero-conventions —
e.g. `Knee_Pitch` range `[0, 2.34]` confirms 0 = straight leg; `Shoulder_Roll`
is the dedicated ab/adduction axis):

| Joint | Current (`default_joint_pos`) | New stance target |
|---|---|---|
| `L/R_Hip_Pitch` | -0.3 | 0.0 |
| `L/R_Knee_Pitch` | 0.6 | 0.0 |
| `L/R_Ankle_Pitch` | -0.3 | 0.0 |
| `L/R_Shoulder_Pitch` | -0.21 | 0.0 |
| `Left_Shoulder_Roll` | -0.41 (~-23°) | -0.785 (-45°) |
| `Right_Shoulder_Roll` | +0.41 | +0.785 (+45°) |
| `L/R_Elbow_Pitch` | -0.13 | 0.0 |
| `Left_Elbow_Yaw` | -0.21 | 0.0 |
| `Right_Elbow_Yaw` | +0.21 | 0.0 |

Hip_Roll/Hip_Yaw/Waist are unaffected (already 0 in `default_joint_pos`,
untouched by this change since `postwaistdofpos` is not being modified).

### Implementation approach

Add a new joint-name → angle map (same pattern as the existing
`_T1_VEL_LIMIT_MAP`/`_T1_KP_MAP` in `rewards.py`), e.g.
`_POST_SAVE_STANCE_MAP: dict[str, float]`. Resolve it once against
`robot.joint_names` into a joint-ordered tensor, cached on `env` (same
lazy-cache pattern those two existing maps use: `if not hasattr(env, "_post_save_stance_target"): ...`).

`postupperdofpos`/`postlegdofpos` then compute
`delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - env._post_save_stance_target[asset_cfg.joint_ids]`
instead of reading `default_joint_pos`. Reward shape (`exp(-k * sum_sq_err) * behind`),
weights, and curriculum are all unchanged — only the comparison target moves.

### Rejected alternative

Modifying `HOME_KEYFRAME` itself (or adding a second `EntityCfg.InitialStateCfg`)
and swapping which pose `default_joint_pos` resolves to at runtime based on a
post-save flag. Rejected: `default_joint_pos` is a fixed buffer set once at
articulation init, not meant to be swapped dynamically; would also risk
accidentally changing the reset pose, which is explicitly out of scope.

## Testing / validation plan

- No existing tests cover `postupperdofpos`/`postlegdofpos` directly (checked
  `tests/simple_goalkeeper/`). Add unit coverage for the new stance-target
  resolution (joint-name map → correct per-joint-id tensor values) at
  minimum, following `superpowers:test-driven-development`.
- Not validated against a live training run as part of this spec — this is a
  reward-shaping change to an existing, already-curriculum-scheduled term,
  same validation posture as other same-day fixes in `docs/BugFixes.md`
  (syntax + existing test suite green, live training validation deferred to
  the next run).
- Visual sanity check via the mjlab viewer (e.g. a throwaway script setting
  `robot.data.joint_pos` to the new target and rendering) recommended before
  committing to a training run, to confirm the pose looks like the intended
  "standing straight, arms at 45°" stance and doesn't violate any joint limit
  (all target values are within the ranges checked above).

## Open questions for the implementation plan

- Exact cached-tensor resolution pattern to follow (mirror `_t1_vel_limits`/
  `_t1_kp_inv` construction in `rewards.py` verbatim).
- Whether to also log a `docs/BugFixes.md` entry at implementation time per
  this project's fix-logging rule (yes, per `CLAUDE.md` — every reward
  change needs one).
