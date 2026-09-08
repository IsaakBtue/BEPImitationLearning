---
name: reward-shaping-scene-entity-cfg
description: Use whenever writing, debugging, or reviewing ANY reward-shaping change in this project -- not just `SceneEntityCfg` wiring. Covers (1) `SceneEntityCfg`/asset_cfg resolution gotchas, (2) designing a decaying/shrinking-target reward (division-by-target instability, distance-metric conflation, geometric-constant overlap, shared-state contamination across multiple bodies, unaccounted geometry baselines, duplicate/overlapping reward mechanisms), and (3) the MANDATORY verification workflow (graph the design, build a teleport probe script, build a live `sgk_play` agent) before any reward-shaping change can be considered done. Use this before touching any reward function, and before claiming a reward-shaping change works.
---

# Reward Shaping Pitfalls (mjlab / `RewardManager`)

## Overview

`SceneEntityCfg` is NOT self-resolving. It's an inert container of `*_names`
until something calls `.resolve(scene)` on it, which populates the matching
`*_ids` fields. Before `.resolve()` runs, every `*_ids` field silently reads
as `slice(None)` -- i.e. "select everything" -- not an error, not `None`,
not empty. A reward that's supposed to be scoped to 4 joints can end up
silently scoped to all 21 with zero warning, zero crash, and plausible-
looking output the whole time.

This bit `interceptV2DualDis` twice in the same day (2026-08-06,
`postupperdofpos`'s "shoulder-scope wiring" saga -- see `docs/BugFixes.md`).
Both times the code LOOKED correct by every normal reading. This skill exists
so the next asset-scoped reward doesn't take a third round to catch.

This skill has grown beyond `SceneEntityCfg` specifically -- Part 2 and Part
3 below cover a SEPARATE class of mistakes (reward-shape math and
verification discipline) found the same day building `leading_foot_lift`'s
shrinking-target-near-blue mechanism (2026-09-08, 6 real bug-fix passes in
one session -- see `docs/BugFixes.md`). Load this skill for EITHER class,
not just `SceneEntityCfg` wiring.

## When to Use

- Writing a new reward/observation/event function with a `SceneEntityCfg`
  parameter that's meant to select a SUBSET of joints/bodies (not the whole
  entity).
- Registering that function's `RewardTermCfg`/`ObservationTermCfg`/
  `EventTermCfg` in `goalkeeper_env_cfg.py`.
- A reward's measured behavior looks like it's reading MORE joints/bodies
  than you scoped it to (e.g. error magnitude looks like a whole-body sum,
  not a 4-joint one), or looks identical regardless of which `asset_cfg` you
  pass.
- You're about to remove an explicit `asset_cfg` override from `params`,
  intending "let the function's own default apply."
- You're indexing into `asset_cfg.body_ids`/`joint_ids` positionally and
  assuming it matches the order you wrote in `body_names=(...)`/
  `joint_names=(...)`.
- **(Part 2)** You're designing ANY reward whose target/threshold/weight
  changes as a function of distance, time, or another reward's state (a
  decaying target, a fade, a gate that gets stricter near some point) --
  read Part 2 before writing the formula, not after it misbehaves.
- **(Part 3)** You're about to say a reward-shaping change is "done" or
  "working" -- read Part 3 first. Graphing the design and building a live
  probe are not optional polish, they're how every mistake in Part 2 was
  actually caught.

**Don't use for:** contact-sensor geom-matching bugs (see
`debugging-mujoco-contact-sensors` -- a related but distinct failure class:
that skill covers sensors firing on the wrong body part; this one covers
`SceneEntityCfg` never resolving to the joints/bodies you think it did).

## Part 1: `SceneEntityCfg` Resolution

### Pitfall 1: an explicit `params["asset_cfg"]` ALWAYS wins over the function's own default

`mjlab`'s reward/observation/event managers call `term_cfg.func(env,
**term_cfg.params)`. Standard Python keyword-argument semantics: if
`asset_cfg` is a key in `params`, it overrides the function's default no
matter what that default is -- even if the default was JUST changed to fix a
bug. This is the first wiring bug this project hit: `postupperdofpos`'s
default `asset_cfg` was narrowed from 8 joints to 4 (2026-08-03), but
`goalkeeper_env_cfg.py`'s registration still explicitly passed the OLD
8-joint cfg via `params` -- silently reverting the fix. The registration and
the function's default drifted out of sync, and nothing caught it because
both are individually valid Python.

**Check:** whenever a function's default `asset_cfg` changes, grep every
`RewardTermCfg`/etc. registration for that function name and confirm
`params` either doesn't set `asset_cfg` at all, or sets it to something that
matches the NEW intended scope.

### Pitfall 2: NOT passing `asset_cfg` in `params` does not mean the function's default gets resolved

This is the deeper, easier-to-miss bug -- the "fix" for Pitfall 1 above
walked straight into it. `mjlab`'s `ManagerBase._resolve_common_term_cfg`
(`manager_base.py`) is the ONLY code path that calls `.resolve()` on a
`SceneEntityCfg`, and it only does so for objects it finds in
`term_cfg.params.values()`:

```python
def _resolve_common_term_cfg(self, term_name, term_cfg):
    for value in term_cfg.params.values():
        if isinstance(value, SceneEntityCfg):
            value.resolve(self._env.scene)
```

A function's own default argument is invisible to this loop -- it's not in
`params`, it's baked into the function's signature. Removing an
`asset_cfg` override from `params` so "the function's own default applies"
is true in plain Python (the default value IS what gets bound at call time)
but FALSE in the sense that matters here: that default object's `joint_ids`/
`body_ids` NEVER get resolved, so they stay `slice(None)` forever --
selecting every joint/body on the entity, not the subset the default's
`joint_names`/`body_names` describe.

Concretely, this is what `postupperdofpos` hit on 2026-08-06: the Pitfall-1
fix removed the `params["asset_cfg"]` override entirely, intending the
function's 4-joint default to apply -- and it silently ran on all 21 joints
instead, for the rest of that day, undetected until a second engineer wired
up a sibling reward and happened to print `.joint_ids` after real
`env.step()` calls.

**The only real fix:** always pass a `SceneEntityCfg` explicitly via
`params`, even when its value is identical to the function's own default.
This is why every OTHER asset-scoped reward in this codebase
(`postwaistdofpos`, `postlegdofpos`, `penalize_arm_above_shoulder`,
`arm_dof_vel`, ...) already does this -- it's not stylistic, it's the only
path that makes `.resolve()` run. Prefer importing the SAME object the
function uses as its own default (e.g. `gk_mdp.rewards._ARM_JOINT_CFG`)
rather than constructing a second, independent copy in
`goalkeeper_env_cfg.py` -- a second copy is a second source of truth that
can drift (this project's older convention duplicates
`_ARM_HEIGHT_CFG`-style configs in both files; importing directly avoids
that risk for new terms).

### Pitfall 3: `SceneEntityCfg.body_ids`/`joint_ids` do NOT preserve declaration order

Once genuinely resolved, `body_ids`/`joint_ids` come back in the model's own
kinematic-tree index order -- NOT the order you wrote `body_names=(...)`/
`joint_names=(...)` in. `penalize_arm_above_shoulder` hit this 2026-07-30:
`_ARM_HEIGHT_CFG`'s declared `(AL2, AR2, left_hand_link, right_hand_link)`
resolved as `(AL2, left_hand_link, AR2, right_hand_link)`, because the
model's kinematic tree fully declares the left-arm chain before the right
arm begins -- so a naive positional `asset_cfg.body_ids[0]`/`[1]`/`[2]`/`[3]`
read would have silently paired the LEFT hand against the RIGHT shoulder.

**Fix pattern:** never index resolved ids positionally against your own
declared order. Use `robot.find_bodies([...])`/`robot.find_joints([...])`,
which returns BOTH the ids and their matching names, and build a
name-keyed lookup dict from that pair -- see `penalize_arm_above_shoulder`
(`rewards.py`) for the exact pattern (`env._arm_above_shoulder_body_idx`,
computed once and cached).

### Verifying a Reward Is Actually Scoped Correctly

Don't trust a standalone `python -c "..."` import-and-call test for this --
it constructs a fresh, never-resolved `SceneEntityCfg` unless you manually
call `.resolve()` yourself, which proves nothing about what the REAL
registered term does inside training. Verify through the real env/manager
instead:

```python
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
env.reset()
term_cfg = env.reward_manager.get_term_cfg("your_reward_name")
asset_cfg = term_cfg.params["asset_cfg"]  # KeyError here means Pitfall 2 -- fix it
print(asset_cfg.joint_ids, asset_cfg.joint_names)
assert asset_cfg.joint_ids != slice(None), "still unresolved -- selecting everything"
```

If `"asset_cfg"` isn't a key in `term_cfg.params` at all, that's the bug by
itself -- stop there, don't bother checking `.joint_ids`, the function's own
default is guaranteed unresolved regardless of what it looks like in source.

See `verify_reward_scoping.py` (same directory) for a reusable script that
walks EVERY registered reward/observation/event term and flags any
`SceneEntityCfg` whose `joint_ids`/`body_ids` stayed an unresolved
`slice(None)` while it was meant to be scoped to a subset -- run this after
adding or touching any asset-scoped reward, not just the one you think you
changed (a scope change to a shared default, like `_ARM_JOINT_CFG`, can
silently affect every consumer of that same object).

**Caveat found while writing that script (a 4th subtlety, not a 4th
production bug):** `joint_ids == slice(None)` is NOT by itself proof of
Pitfall 2 -- `SceneEntityCfg.resolve()` deliberately encodes "every joint/body
on the entity matched" as `slice(None)` too (its own docstring: "optimizes to
slice(None) if all selected"), and this project's own `_ALL_JOINTS_CFG =
SceneEntityCfg("robot", joint_names=(".*",))` genuinely, correctly resolves
that way. A naive `bool(joint_names) and joint_ids == slice(None)` check
false-positives on every one of those. The real test is whether the
EXPANDED `joint_names`/`body_names` (which `resolve()` populates with
concrete names regardless of whether the input was a regex, an explicit
subset, or nothing) count as a proper SUBSET of the entity's total joint/
body count -- compare against `len(robot.joint_names)`/`len(robot.body_names)`,
not against `slice(None)` alone. `verify_reward_scoping.py` does this
correctly; an earlier draft of it did not, and reported 6 real registrations
as broken when they were actually fine.

## Part 2: Decaying / Shrinking-Target Reward Design

All six sub-pitfalls below came from ONE mechanism, `leading_foot_lift`
(`rewards.py`) -- a reward whose target height decays toward zero as the
leading foot approaches the "blue" waypoint on a wide crossing, restoring
once genuinely landed. It took 6 separate bug-fix passes IN ONE SESSION
(2026-09-08) to get right, each one a different pitfall below. Read this
before writing (or reviewing) any reward whose target, threshold, or weight
is a function of distance/time/another reward's state, not a fixed constant.

### Pitfall 4: dividing by a target that can shrink toward zero

A common "reach a target height/distance" kernel looks like
`tanh(steepness * value / target)` (steeper rise = faster saturation, scaled
by how big the target is). This is fine as long as `target` is fixed. The
moment `target` itself is the thing decaying toward 0 (e.g. "want a lower
foot as you approach"), that division blows up: `value / target` grows
without bound for ANY nonzero `value`, no matter how small, once `target`
is small enough -- `tanh` saturates near 1 immediately, and a kernel meant
to REWARD a low height instead scores a barely-lifted foot (a few cm, often
just sensor/geometry noise) as if it had wildly overshot a near-zero target.
Concretely: `tanh(3 * 0.03 / 0.001)` = `tanh(90)` ≈ `1.0` -- a foot at 3cm
reads as fully "at target" against a 1mm target, the opposite of intended.

**Fix pattern:** decouple the kernel's STEEPNESS from the CURRENT (possibly
shrinking) target. Calibrate the steepness against a fixed reference (the
reward's own standard/undecayed target) always: `tanh((steepness /
STANDARD_target) * value)`. Let only the well-behaved half of the kernel (a
Gaussian falloff / squared-error term, which uses SUBTRACTION not DIVISION)
track the shrinking target. A value of exactly 0 then always scores exactly
0 regardless of how far the target has shrunk (`tanh(0)=0`), which is
usually the actual invariant you want ("never inflate reward for something
genuinely at rest/zero").

### Pitfall 5: an unsigned/combined distance doesn't detect a directional condition

"Has the foot passed point X" is a DIRECTIONAL question (signed progress
along one axis past a line). "How far is the foot from point X" is an
UNSIGNED, often multi-axis, distance. These are not the same thing, and
using the wrong one to gate a decay silently breaks it in one direction:
`leading_foot_lift`'s first attempt used the raw combined X+Y distance to
detect "close to blue," which meant a foot that had already overshot blue's
Y position while still far from the goal line in X (X dominates the
combined 2D distance) never registered as "close" at all -- the decay never
fired even well past the point of no return.

**Fix pattern:** when the real question is "has X crossed a line," compute
the SAME signed/directional metric an already-correct sibling reward uses
for that exact condition (here, `blue_overshoot_penalty`'s own
`signed_progress = direction * (foot_y - line_y)`) rather than inventing a
new distance check. If two reward terms need to agree on "have we passed
this point," they should share the literal formula, not two independently
plausible-looking ones.

### Pitfall 6: an outer-zone/tolerance constant that overlaps an unrelated fixed geometry

Two "reasonable-looking" numeric constants, each independently chosen and
individually correct in isolation, can silently interact if their
geometries actually overlap. `leading_foot_lift`'s decay used a 0.30m outer
zone (copied from an older, since-removed mechanism); separately, the
`orange` and `blue` waypoints are ALWAYS exactly 0.25m apart on a wide
crossing (both derived from the same underlying `delta`, confirmed live:
`gap = 0.2500` exactly, every time). Since 0.30m > 0.25m, the decay zone
silently extended backward past `orange`'s own fixed position -- the foot
merely walking through `orange` (unrelated to `blue`) already triggered
part of the decay meant only for approaching `blue`.

**Fix pattern:** before picking (or reusing) an outer-zone/tolerance
constant, grep for every OTHER fixed distance/geometry constant already in
the same reward system and compute their actual pairwise relationships (not
just "does this number sound reasonable on its own"). A constant copied from
an unrelated, older mechanism is not guaranteed to still be compatible with
newer geometry added since.

### Pitfall 7: one shared aggregate signal silently couples two independent things

`leading_foot_lift` originally computed `max(both feet)`'s height and
applied ONE fade/decay to that combined value, scoped conceptually to the
LEADING foot's approach to blue. But the TRAILING foot's own, completely
unrelated incentive (reaching for `orange`) fed into that SAME `max()` --
so whenever the leading foot was near/past blue (fade active), the trailing
foot's genuine, correct behavior got silently suppressed too, since both
shared one scalar gate.

**Fix pattern:** if a reward aggregates over multiple independent bodies
(`max`, `sum`, `mean` across two feet, two joints, etc.), any gate/decay
that's conceptually scoped to only ONE of them must be applied to that
one's contribution BEFORE the aggregation, never to the combined result.
Also check whether a SEPARATE reward already exists for the other body's
version of the same behavior (see Pitfall 9) -- `trailing_foot_lift`
already existed; the fix was to stop `leading_foot_lift` from covering the
trailing foot at all, not to fix the aggregation.

### Pitfall 8: an unaccounted geometry baseline breaks "should be zero at rest"

A body-link origin frequently does NOT coincide with the true physical
contact surface (collision geometry sits some fixed offset away from the
link's own reference frame). Reading "height above floor" as a raw
coordinate difference silently measures this offset, not the physical
quantity a human means by "the foot is on the ground." Confirmed live: a
genuinely flat, fully-grounded foot (`feet_contact` sensor reads full
contact) still measures `~0.03m` above `floor_z`, not `0`. Feeding this raw
value into a kernel that assumes "0 = at rest" produces a nonzero baseline
reward for a state that should score exactly zero.

**Fix pattern:** before trusting a "height/distance above surface X" read,
check whether this project already has a named constant for this exact
class of geometry gap (`_FOOT_CONTACT_BELOW_BODY` in `events.py`, the
ghost-overlay `+0.030m` fix, both the SAME 0.03m foot-geometry offset) --
if one exists, reuse it; if measuring fresh, confirm live with a contact
sensor read at a known-grounded state before assuming the raw number means
what it looks like it means.

### Pitfall 9: a new mechanism silently duplicates an existing one

Before adding logic to reward A that also shapes "what body B should do,"
check whether reward B already exists and already covers it.
`leading_foot_lift`'s `max(both feet)` design (Pitfall 7) meant it was
ALSO, accidentally, a second reward for the trailing foot's height -- on
top of the already-existing, purpose-built `trailing_foot_lift`. Two
overlapping mechanisms for the same physical behavior create hidden
double-credit (or, if their gates disagree, hidden contradiction) that's
invisible from reading either function alone.

**Fix pattern:** grep the reward module and the env config's registration
list for the OTHER body/behavior before writing new logic that touches it,
even in passing. If a purpose-built sibling term exists, scope your new
term to exclude what the sibling already owns, don't recreate it.

## Part 3: Mandatory Verification -- Graph It, Then Build Probes

A reward-shaping change whose intended behavior is a SHAPE (a curve, a
gate, a decay) is not verified by "the math looks right" or "no exception
was thrown." Every pitfall in Part 2 above passed a plain code-read and
still shipped wrong the first time. Verification is not optional polish for
this class of change -- it is how every one of those mistakes was actually
caught, and skipping any of the three steps below is how a broken shape
ships anyway.

### Step 1: graph the intended design BEFORE writing the implementation

Two phrasings that sound similar in conversation -- "fade the reward
toward zero near the target" vs. "shrink the TARGET toward zero" -- describe
genuinely different reward landscapes (one goes silent near the target with
no opinion on height; the other actively prefers a low height there). This
project shipped the wrong one once (`leading_foot_lift` pass 4) because the
distinction was never made explicit before code existed to argue about.
Render the candidate curve(s) (a quick matplotlib PNG is enough --
`Read`-ing it displays inline) and get explicit confirmation on the SHAPE,
not just the goal in words, before writing any implementation. When there's
more than one plausible shape (e.g. "exponential" is ambiguous between "fast
then flat" and "flat then fast"), graph 2-3 candidates side by side rather
than guessing which one someone meant.

### Step 2: build (or reuse) a static teleport probe script

A standalone `python -c "..."` snippet with synthetic torch tensors proves
the FORMULA is self-consistent -- it proves nothing about the REAL
registered reward, because it can't catch env-attribute wiring bugs (e.g.
`_get_reach_target_y` silently preferring `env._rsi_cross_y` over
`env._ball_crossing_y` for its wide/narrow classification -- overriding
only the latter is a complete no-op). Build a script that constructs the
REAL env, rigidly teleports the relevant body to controlled test positions
(root translation is enough for most cases -- no IK needed, an unnatural
pose is fine for a diagnostic script), and calls the REAL registered reward
function at each position. See `scripts/probe_leading_foot_lift.py` for the
reference pattern -- reuse its documented teleport technique rather than
re-deriving it.

### Step 3: build (or reuse) a live, animated `sgk_play` agent

A table of numbers from Step 2 does not always communicate a SHAPE the way
watching an animated curve does, and the person who asked for the change
needs to be able to watch it, not just read your table. Add (or reuse) a
`--agent <name>` in `scripts/play.py`, following the existing
`scripted_yaw`/`scripted_lean`/`scripted_blue_approach` pattern
(monkeypatched `env.reset`/`env.step`, a zero-action policy, the mechanism
under test re-applied every step) so the reward's own P-panel plot animates
live while the demo runs.

**Test-harness pitfalls hit building both kinds of probe (none are reward
bugs -- all are mistakes in the PROBE, easy to re-make):**

- **Stale root-pose reuse:** re-read the robot's CURRENT root pose fresh
  immediately before every teleport. Caching a `root_pose` snapshot once at
  the start and reusing it for every subsequent teleport silently breaks
  every teleport after the first (the delta gets computed against a base
  that no longer reflects where the robot actually is).
- **Wrong cached attribute overridden:** know which cached env attribute a
  helper ACTUALLY reads before overriding a different one that merely looks
  related (`_rsi_cross_y` vs. `_ball_crossing_y`, Pitfall-5-adjacent).
- **State leaking between test rows/phases:** a stateful counter (a settle
  counter, an accumulated flag) from one scripted scenario can silently
  trigger an unintended transition on the very NEXT one, even after
  explicitly resetting the flag you think is the relevant one. List out
  EVERY stateful attribute the mechanism touches and reset all of them, not
  just the boolean that seems obviously relevant.
- **Memoization guards keyed on a clock that isn't advancing:** a per-tick
  memoization guard (comparing against `episode_length_buf`) needs that
  clock to actually change between calls. In a probe that doesn't step real
  physics, bump `episode_length_buf` manually to defeat the guard; in a live
  animated agent, real stepping already advances it.
- **Unrelated real physics keeps running underneath the scripted part:** in
  a LIVE animated probe (not a one-shot static call), everything else in
  the sim keeps simulating for real. A ball will keep rolling and can
  permanently flip an unrelated sticky "behind"/"done" flag partway through
  the demo, silently zeroing the exact reward being demonstrated, well
  before the demo even finishes one cycle. Park/freeze anything not
  directly relevant to the mechanism under test (the same fix
  `scripted_yaw`/`scripted_lean` already needed for the ball).
- **Hardcoded phase-length constants that don't scale:** if a demo exposes
  a configurable total period/duration, express every internal phase
  boundary as a FRACTION of that period, never an absolute step count --
  otherwise shortening or lengthening the period silently breaks (or
  degenerates) the later phases.

## Quick Reference

| Question | How to answer it |
|---|---|
| Does this reward's `params` need an explicit `asset_cfg`? | Yes, always, if the function's default is scoped to anything narrower than the whole entity -- there is no other way `.resolve()` runs. |
| I changed a function's default `asset_cfg` scope -- what else needs to change? | Grep every registration of that function; if any passes `asset_cfg` explicitly, update it to match (Pitfall 1) or remove the override in favor of passing the SAME default object explicitly (Pitfall 2). |
| Is `asset_cfg.body_ids[i]` the body I declared at `body_names[i]`? | Not guaranteed. Resolve via `robot.find_bodies(names)`'s returned `(ids, names)` pair and look up by name, never by position (Pitfall 3). |
| How do I know a `SceneEntityCfg` is genuinely resolved, not just "looks fine in source"? | Read `term_cfg.params["asset_cfg"].joint_ids`/`.body_ids` off the REAL `reward_manager`/`observation_manager`/`event_manager` after real `env.step()` calls -- never trust a fresh standalone import. |
| A reward's measured error/behavior looks like it spans way more joints than intended -- what's the first thing to check? | Whether `asset_cfg` is actually in that term's registered `params` at all (Pitfall 2) before assuming the reward math itself is wrong. |
| My kernel divides `value` by a `target` that I'm about to make decay toward 0 -- is that safe? | No. Decouple: calibrate the steepness against the FIXED standard target always; only the subtraction-based half of the kernel (Gaussian falloff, squared error) should track the shrinking target (Pitfall 4). |
| I need to detect "has this crossed a line" -- can I just use distance-to-point? | Only if the point's other axes are also pinned. If not, use a SIGNED, directional progress metric along the one axis that matters -- ideally the same formula a sibling reward already uses for the identical condition (Pitfall 5). |
| I'm picking a new outer-zone/tolerance radius -- what else do I need to check? | Every OTHER fixed distance/geometry constant in the same reward system, and their actual pairwise gaps -- not just whether this number looks reasonable alone (Pitfall 6). |
| My reward aggregates (`max`/`sum`/`mean`) over two bodies, but I only want to gate ONE of them -- where does the gate go? | On that one body's contribution, before the aggregation -- never on the combined result (Pitfall 7). Also check whether a reward for the OTHER body's version of this already exists (Pitfall 9). |
| A "height/distance above surface" reads nonzero even though the body looks genuinely at rest -- what's wrong? | Probably nothing wrong with your code -- the body-link origin likely doesn't coincide with the true contact surface. Check for an existing project-wide offset constant for this before assuming a raw sensor value means what it looks like (Pitfall 8). |
| Is my reward-shaping change actually done? | Not until you've graphed the intended shape and gotten it confirmed (Part 3, Step 1), built or reused a static teleport probe (Step 2), and built or reused a live animated `sgk_play` agent (Step 3) -- a passing test suite and clean `ast.parse` are necessary, not sufficient, for a shape-based reward change. |
