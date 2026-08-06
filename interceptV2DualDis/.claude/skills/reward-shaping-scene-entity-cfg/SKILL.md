---
name: reward-shaping-scene-entity-cfg
description: Use when writing or debugging any mjlab reward/observation/event function that takes a `SceneEntityCfg` parameter (asset_cfg, or any argument scoped to specific joints/bodies) -- especially when a reward seems to affect more or fewer joints/bodies than intended, or you're about to remove/add an explicit `asset_cfg` from a RewardTermCfg's `params`. Also use before trusting `SceneEntityCfg.body_ids`/`joint_ids` ordering to match the order names were declared in.
---

# Reward Shaping: `SceneEntityCfg` / `RewardManager` Wiring Gotchas

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

**Don't use for:** contact-sensor geom-matching bugs (see
`debugging-mujoco-contact-sensors` -- a related but distinct failure class:
that skill covers sensors firing on the wrong body part; this one covers
`SceneEntityCfg` never resolving to the joints/bodies you think it did).

## Core Pattern

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

## Verifying a Reward Is Actually Scoped Correctly

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

## Quick Reference

| Question | How to answer it |
|---|---|
| Does this reward's `params` need an explicit `asset_cfg`? | Yes, always, if the function's default is scoped to anything narrower than the whole entity -- there is no other way `.resolve()` runs. |
| I changed a function's default `asset_cfg` scope -- what else needs to change? | Grep every registration of that function; if any passes `asset_cfg` explicitly, update it to match (Pitfall 1) or remove the override in favor of passing the SAME default object explicitly (Pitfall 2). |
| Is `asset_cfg.body_ids[i]` the body I declared at `body_names[i]`? | Not guaranteed. Resolve via `robot.find_bodies(names)`'s returned `(ids, names)` pair and look up by name, never by position (Pitfall 3). |
| How do I know a `SceneEntityCfg` is genuinely resolved, not just "looks fine in source"? | Read `term_cfg.params["asset_cfg"].joint_ids`/`.body_ids` off the REAL `reward_manager`/`observation_manager`/`event_manager` after real `env.step()` calls -- never trust a fresh standalone import. |
| A reward's measured error/behavior looks like it spans way more joints than intended -- what's the first thing to check? | Whether `asset_cfg` is actually in that term's registered `params` at all (Pitfall 2) before assuming the reward math itself is wrong. |
