# Orange-Ball Trailing-Foot Waypoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the trailing (non-assigned) foot its own positional target ("orange"), mirroring the leading foot's existing "blue ball" waypoint mechanism (landing-focused subset), so it has a concrete reason to move during a wide-crossing double/triple-step approach instead of staying stationary.

**Architecture:** A new, self-contained parallel mechanism in `rewards.py`: one target-formula function (`_get_orange_reach_target_y`) plus four reward functions, wired into `goalkeeper_env_cfg.py`'s reward/curriculum config, with a matching viewer marker in `play.py`. Reuses `env._blue_wide` (read-only) for gating; never reads or writes any `env._blue_*` state otherwise. Blue's own code is untouched.

**Tech Stack:** Python, PyTorch, mjlab (MuJoCo-Warp), this project's existing `ManagerBasedRlEnv`/`RewardTermCfg`/`CurriculumTermCfg` config pattern.

**Spec:** `docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md` (read this first — it has the full worked-example derivation for every formula/weight used below).

## Global Constraints

- Landing-focused subset only: `_get_orange_reach_target_y`, `orange_foot_proximity`, `orange_ball_landed`, `orange_overshoot_penalty`, `orange_stick_landing`. Do **not** mirror `footreach`'s phase1/vel_sigma/overshoot-kill/decel-zone machinery, `near_stick_reach`, or `blue_trunk_drive` in this plan.
- Target formula: `delta = full_y - start_y` (signed); `shrunk = sign(delta) * max(|delta| - 0.30, 0.0)`; `orange_y = start_y + shrunk / 2.0`. Sign-safe — must not flip direction for right-side (negative-delta) crossings.
- Gating: orange terms are active only when `env._blue_wide` is true. No new wide/region computation — read `env._blue_wide` directly (it's computed every step by `_get_reach_target_y`, which every existing wide-gated reward already calls first in the reward-manager's term order).
- Keyed to the trailing foot: `trailing_idx = 1 - _get_correct_foot_idx(env, ball_name)`.
- Weights are half of blue's corresponding `base_weight`/flat weight (conservative first pass — confirmed with user): `orange_ball_landed` base_weight `5.0` (blue: `10.0`), `orange_overshoot_penalty` base_weight `-30.0` (blue: `-60.0`), `orange_stick_landing` base_weight `4.0` (blue: `8.0`), `orange_foot_proximity` flat weight `2.5` (blue's `foot_proximity`: `5.0`, no curriculum).
- `reward_curriculum_ep_len` (`mdp/events.py:613`) computes `weight = base_weight * (1 + 0.5*cu)`, `cu` in `[0,3]` — so each curriculum'd term's true peak is `2.5x` its base_weight, not `2x`. Use this exact formula when writing peak values into documentation; do not copy the (informally rounded) "`X→2X`" shorthand some older rows in `CLAUDE.md` use.
- **Change Approval Workflow (this project's `CLAUDE.md`):** the exact functions, formulas, and weights below were already presented to the user as an explicit before/after list during the brainstorming/design phase (`docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md`, approved) and the user separately said "start it" to begin implementation — this satisfies the workflow's list-and-confirm requirement for the `rewards.py`/`goalkeeper_env_cfg.py` changes in Tasks 2–3. Do **not** re-ask the user to re-approve these same values. If a task's implementer needs to deviate from any concrete value in this plan for a technical reason, that deviation (not the already-approved values) is what requires a fresh explicit confirmation before editing.
- Test suite baseline: `env -u PYTHONPATH uv run pytest tests/ -q` currently passes 76/76. Every task that touches Python code must end with this passing (76 + however many new tests that task added).
- No G1-equivalent lookup is needed for this change — it's a same-project mirror of an already-G1-justified-as-N/A mechanism (blue's own waypoint), not a new port from upstream.

---

### Task 1: Orange target-formula function + unit tests

**Files:**
- Modify: `src/simple_goalkeeper/mdp/rewards.py` (add `_get_orange_reach_target_y`, placed directly after `_get_reach_target_y`, i.e. after its closing `return torch.where(phase1_active, half_y, full_y)` at line 687, before `def footreach(` at line 690)
- Test: `tests/simple_goalkeeper/test_orange_reach_target_y.py` (new file)

**Interfaces:**
- Produces: `_get_orange_reach_target_y(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG, landing_radius=0.15, landing_speed_threshold=1.0) -> torch.Tensor` (shape `(N,)`, the orange target Y in world frame). Also creates/updates env state: `_orange_was_airborne`, `_orange_landed`, `_orange_settle_count`, `_orange_landed_was_free`, `_orange_landed_genuine`, `_orange_last_settle_step`, `_orange_landing_radius_current`, `_orange_landing_speed_threshold_current` (all `(N,)` tensors, same dtypes as their `_blue_*` counterparts in `_get_reach_target_y`). Tasks 2+ read `env._orange_landed_genuine` and call this function to get the target Y.
- Consumes: `_get_ball_crossing_y` (existing, `rewards.py:310`), `_get_correct_foot_idx` (existing, `rewards.py:259`), `env._blue_wide` (existing, set by `_get_reach_target_y` — read via `getattr(env, "_blue_wide", ...)` with an all-`False` fallback so this function never raises if called before `_get_reach_target_y` in some edge case).

- [ ] **Step 1: Write the failing tests**

Create `tests/simple_goalkeeper/test_orange_reach_target_y.py`:

```python
"""Tests for _get_orange_reach_target_y's target-position formula.

NEW 2026-08-08: trailing-foot ("orange") mirror of _get_reach_target_y's
midpoint targeting, using a different formula -- shrink |delta| by 0.30m
(sign-safe) before halving, instead of blue's plain halve. See
docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md for the
full worked-example derivation these expected values come from.
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_orange_reach_target_y


class _Scene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeEnv:
    """Deliberately omits 'robot'/'feet_contact' scene entries so
    _get_orange_reach_target_y's try/except falls into the robot=None branch --
    the target-Y formula is computed unconditionally before that branch, so no
    robot/contact-sensor mocking is needed to test it in isolation. Mirrors
    tests/simple_goalkeeper/test_landing_speed_threshold_curriculum.py's _FakeEnv."""

    def __init__(self, num_envs: int, crossing_delta: float):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(num_envs)
        self._rsi_cross_y = torch.full((num_envs,), crossing_delta)
        self.scene = _Scene(num_envs)


def _orange_y(crossing_delta: float) -> float:
    env = _FakeEnv(num_envs=4, crossing_delta=crossing_delta)
    result = _get_orange_reach_target_y(env, "ball")
    return result[0].item()


def test_orange_target_shrinks_positive_delta_by_030_then_halves():
    # delta=+1.00m -> shrunk=0.70 -> orange_y=0.35 (blue's own midpoint would be 0.50)
    assert abs(_orange_y(1.0) - 0.35) < 1e-6


def test_orange_target_shrinks_moderate_positive_delta():
    # delta=+0.40m -> shrunk=0.10 -> orange_y=0.05
    assert abs(_orange_y(0.4) - 0.05) < 1e-6


def test_orange_target_floors_at_start_y_when_delta_below_030():
    # delta=+0.20m -> shrunk clamped to 0.0 -> orange_y collapses to start_y (0.0)
    assert abs(_orange_y(0.2) - 0.0) < 1e-6


def test_orange_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> shrunk=-0.70 -> orange_y=-0.35 (NOT -0.65, which a naive
    # `delta - 0.30` without sign handling would produce)
    assert abs(_orange_y(-1.0) - (-0.35)) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/simple_goalkeeper/test_orange_reach_target_y.py -v`
Expected: FAIL with `ImportError: cannot import name '_get_orange_reach_target_y'`

- [ ] **Step 3: Implement `_get_orange_reach_target_y`**

Insert into `src/simple_goalkeeper/mdp/rewards.py` directly after `_get_reach_target_y`'s closing `return torch.where(phase1_active, half_y, full_y)` (line 687), before `def footreach(`:

```python
def _get_orange_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.15,
    landing_speed_threshold: float = 1.0,
) -> torch.Tensor:
    """Trailing-foot ("orange") mirror of _get_reach_target_y -- see that
    function's docstring for the full blue-ball-waypoint mechanism history
    this reuses (leaky settle-count decrement, curriculum-eased landing
    radius/speed, free-landing classification).

    Target formula (confirmed with user via worked examples, 2026-08-08
    design spec):
        delta = full_y - start_y                          (signed)
        shrunk = sign(delta) * max(|delta| - 0.30, 0.0)    (30cm off the top, sign-safe)
        orange_y = start_y + shrunk / 2.0

    Equivalently: blue's own midpoint, 15cm short of it in delta-magnitude
    terms -- NOT a plain `delta - 0.30` (that would push the target further
    OUT, not in, for right-side/negative-delta crossings).

    Reuses env._blue_wide (set by _get_reach_target_y, which every existing
    wide-gated reward already calls first in the reward-manager's term
    order) directly as the wide/narrow gate -- "wide" is a property of the
    ball's crossing geometry, not which foot is being tracked, so no
    separate wide/region computation is added here. Defensive fallback to
    all-False if _blue_wide isn't set yet (should not happen in the real
    term order).

    Keyed to the TRAILING foot (1 - _get_correct_foot_idx), not the leading
    one. Unlike blue, this function does NOT graduate to a second/live-ball
    target once landed -- the landing-focused subset gives the trailing foot
    one target for the whole wide-crossing window, no live-ball-tracking
    phase (see design spec's "Explicitly out of scope" section).

    Maintains its own state namespace (env._orange_*), entirely separate
    from env._blue_*'s -- never reads or writes any env._blue_* field except
    the read-only env._blue_wide gate above.

    See docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    delta = full_y - start_y
    wide = getattr(env, "_blue_wide", torch.zeros_like(delta, dtype=torch.bool))

    shrunk = torch.sign(delta) * (delta.abs() - 0.30).clamp(min=0.0)
    orange_y = start_y + shrunk / 2.0

    n = env.num_envs
    if not hasattr(env, "_orange_was_airborne"):
        env._orange_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_settle_count = torch.zeros(n, dtype=torch.int64, device=env.device)
        env._orange_landed_was_free = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_landed_genuine = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._orange_last_settle_step = torch.full((n,), -1, dtype=torch.int64, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._orange_was_airborne[just_reset] = False
    env._orange_landed[just_reset] = False
    env._orange_settle_count[just_reset] = 0
    env._orange_landed_was_free[just_reset] = False
    env._orange_landed_genuine[just_reset] = False

    d = float(min(max(getattr(env, "_ball_difficulty", 1.0), 0.0), 1.0))
    landing_radius = 0.20 + (landing_radius - 0.20) * d
    env._orange_landing_radius_current = landing_radius

    _EASY_LANDING_SPEED_THRESHOLD = 2.0
    landing_speed_threshold = (
        _EASY_LANDING_SPEED_THRESHOLD + (landing_speed_threshold - _EASY_LANDING_SPEED_THRESHOLD) * d
    )
    env._orange_landing_speed_threshold_current = landing_speed_threshold

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
        foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                      # (N,)
        trailing_idx = 1 - foot_idx
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]                # (N, 3)
        assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]                # (N, 3)

        found = feet_contact.data.found                                      # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(trailing_idx == 0, left_in_contact, right_in_contact)

        currently_airborne = ~foot_in_contact
        env._orange_was_airborne |= currently_airborne

        goal_x_w = env.scene.env_origins[:, 0]
        target_point_xy = torch.stack([goal_x_w, orange_y], dim=-1)          # (N, 2)
        dist_to_orange = torch.norm(assigned_foot_pos[:, :2] - target_point_xy, dim=-1)
        foot_speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

        candidate = wide & env._orange_was_airborne & foot_in_contact & (dist_to_orange < landing_radius)
        is_first_call_this_tick = env.episode_length_buf != env._orange_last_settle_step
        env._orange_last_settle_step = env.episode_length_buf.clone()
        _ORANGE_SETTLE_STEPS = 3
        env._orange_settle_count = torch.where(
            candidate,
            torch.where(is_first_call_this_tick, env._orange_settle_count + 1, env._orange_settle_count),
            torch.where(is_first_call_this_tick, (env._orange_settle_count - 1).clamp(min=0), env._orange_settle_count),
        )
        newly_landed = (
            (env._orange_settle_count >= _ORANGE_SETTLE_STEPS)
            & (foot_speed < landing_speed_threshold)
            & ~env._orange_landed
        )
        env._orange_landed |= newly_landed

        _ORANGE_LANDING_FREE_STEP_THRESHOLD = 10
        env._orange_landed_was_free = torch.where(
            newly_landed,
            env.episode_length_buf < _ORANGE_LANDING_FREE_STEP_THRESHOLD,
            env._orange_landed_was_free,
        )

    env._orange_landed_genuine = env._orange_landed & ~env._orange_landed_was_free

    return orange_y
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/simple_goalkeeper/test_orange_reach_target_y.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/ -q`
Expected: 80 passed (76 existing + 4 new)

- [ ] **Step 6: Commit**

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
git add src/simple_goalkeeper/mdp/rewards.py tests/simple_goalkeeper/test_orange_reach_target_y.py
git commit -m "$(cat <<'EOF'
feat(rewards): add orange trailing-foot target formula

_get_orange_reach_target_y mirrors _get_reach_target_y's leaky-settle-count
landing mechanism for the trailing foot, using a shrink-then-halve target
formula (30cm off the crossing delta before halving, sign-safe). Not yet
wired into any reward term or the reward manager.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Orange reward functions

**Files:**
- Modify: `src/simple_goalkeeper/mdp/rewards.py` (add 4 functions directly after `blue_trunk_drive`'s closing — find the line after that function ends and before `def foot_clearance(`, so the new orange functions sit adjacent to their blue counterparts)
- Modify: `src/simple_goalkeeper/mdp/__init__.py` (add the 4 new names to the `from .rewards import (...)` block, line 18, next to the existing `blue_ball_landed, blue_overshoot_penalty, blue_stick_landing, blue_trunk_drive,` line)
- Test: manual live-env smoke check (see Step 4) — this project's own convention for reward-function-level verification (per `docs/BugFixes.md`) is a live check on a real `ManagerBasedRlEnv`, not mocked unit tests, since these functions need real scene entities (robot bodies, contact sensors) that aren't practical to fake.

**Interfaces:**
- Consumes: `_get_orange_reach_target_y` (Task 1), `_get_correct_foot_idx`, `_get_ball_crossing_y`, `_ball_is_behind` (all existing in `rewards.py`).
- Produces: `orange_foot_proximity(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG, sigma=5.0) -> torch.Tensor`, `orange_ball_landed(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG) -> torch.Tensor`, `orange_overshoot_penalty(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG, landing_radius=0.08, max_overshoot=0.5) -> torch.Tensor`, `orange_stick_landing(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG, dist_sigma=8.0, speed_sigma=1.5) -> torch.Tensor` — all `(N,)` tensors, same signatures as their blue counterparts. Task 3 registers these as `RewardTermCfg`s.

- [ ] **Step 1: Implement the 4 reward functions**

Insert into `src/simple_goalkeeper/mdp/rewards.py`, directly after `blue_trunk_drive`'s function body ends (before `def foot_clearance(`):

```python
def orange_foot_proximity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Trailing-foot mirror of foot_proximity -- dense exp(-sigma*dist) pull of
    the TRAILING (non-assigned) foot toward the orange target. Wide-crossings
    only (env._blue_wide) -- unlike foot_proximity, which has no such gate
    because _get_reach_target_y itself switches targets on narrow crossings;
    _get_orange_reach_target_y never switches, so this function gates
    explicitly instead. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
    """
    robot: Entity = env.scene[asset_cfg.name]
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z = env.scene.env_origins[:, 2]
    target_point = torch.stack([goal_x_w, orange_y, env_z + 0.10], dim=-1)   # (N, 3)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), trailing_idx]
    dist = torch.norm(foot_pos_active - target_point, dim=-1)

    behind = _ball_is_behind(env, ball_name)
    return torch.exp(-sigma * dist) * env._blue_wide.float() * (~behind).float()


def orange_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_ball_landed -- one-shot bonus when the
    trailing foot genuinely lands at the orange target. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
    """
    _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _orange_landed_genuine is fresh

    if not hasattr(env, "_orange_landed_bonus_flag"):
        env._orange_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._orange_landed_bonus_flag[just_reset] = False

    fired = env._orange_landed_genuine & ~env._orange_landed_bonus_flag
    env._orange_landed_bonus_flag |= fired
    return fired.float()


def orange_overshoot_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.08,
    max_overshoot: float = 0.5,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_overshoot_penalty -- penalizes the
    trailing foot for advancing past the orange target, toward the true
    crossing point, before landing there. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
    """
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    direction = torch.sign(full_y - start_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, trailing_idx, 1]            # (N,)

    signed_progress = direction * (assigned_foot_y - orange_y)
    overshoot = torch.clamp(signed_progress - landing_radius, min=0.0, max=max_overshoot)

    phase1_active = env._blue_wide & ~env._orange_landed_genuine
    return overshoot * phase1_active.float()


def orange_stick_landing(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 8.0,
    speed_sigma: float = 1.5,
) -> torch.Tensor:
    """Trailing-foot mirror of blue_stick_landing -- dense reward for the
    trailing foot being simultaneously CLOSE to and SLOW near the orange
    target on a wide, unlanded crossing. See
    docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
    """
    orange_y = _get_orange_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, orange_y], dim=-1)              # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)
    trailing_idx = 1 - foot_idx
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]              # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, trailing_idx]              # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    phase1_active = env._blue_wide & ~env._orange_landed_genuine
    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * phase1_active.float()
```

- [ ] **Step 2: Export the new functions from `mdp/__init__.py`**

In `src/simple_goalkeeper/mdp/__init__.py`, change line 18 from:

```python
    blue_ball_landed, blue_overshoot_penalty, blue_stick_landing, blue_trunk_drive,
```

to:

```python
    blue_ball_landed, blue_overshoot_penalty, blue_stick_landing, blue_trunk_drive,
    orange_foot_proximity, orange_ball_landed, orange_overshoot_penalty, orange_stick_landing,
```

- [ ] **Step 3: Static check — module imports cleanly**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run python -c "from simple_goalkeeper.mdp import orange_foot_proximity, orange_ball_landed, orange_overshoot_penalty, orange_stick_landing; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Live smoke check on a real env**

Run (adjust checkpoint/env count as needed, but this must use a real `ManagerBasedRlEnv`, not a mock, per this project's own verification convention in `docs/BugFixes.md`):

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
env -u PYTHONPATH uv run python -c "
import torch
import simple_goalkeeper.tasks  # registers task ids with mjlab's registry
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from simple_goalkeeper.mdp import rewards as gk_rewards

env_cfg = load_env_cfg('Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc', play=True)
env_cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=env_cfg, device='cpu')
env.reset()
for _ in range(10):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))

for name in ['orange_foot_proximity', 'orange_ball_landed', 'orange_overshoot_penalty', 'orange_stick_landing']:
    fn = getattr(gk_rewards, name)
    out = fn(env.unwrapped, 'ball')
    assert torch.isfinite(out).all(), f'{name} produced non-finite values: {out}'
    print(name, out.shape, out.tolist())
print('all finite, ok')
"
```

Expected: all four print finite tensors of shape `(4,)`, then `all finite, ok`. This is the same `load_env_cfg`/`ManagerBasedRlEnv` construction pattern `scripts/play.py:run_play` (lines 1036, 1126) already uses.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/ -q`
Expected: 80 passed (unchanged from Task 1 — this task adds no new pytest file)

- [ ] **Step 6: Commit**

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
git add src/simple_goalkeeper/mdp/rewards.py src/simple_goalkeeper/mdp/__init__.py
git commit -m "$(cat <<'EOF'
feat(rewards): add orange trailing-foot reward functions

orange_foot_proximity/orange_ball_landed/orange_overshoot_penalty/
orange_stick_landing mirror their blue leading-foot counterparts, retargeted
to the trailing foot via _get_orange_reach_target_y. Not yet registered in
any RewardTermCfg -- wiring is the next commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Config wiring + documentation

**Files:**
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (add 4 `RewardTermCfg` entries + 3 `CurriculumTermCfg` entries)
- Modify: `docs/BugFixes.md` (new dated entry)
- Modify: `CLAUDE.md` (new row in "Divergences from G1 Upstream" table + 4 new rows in "Reward Design" table)

**Interfaces:**
- Consumes: `orange_foot_proximity`, `orange_ball_landed`, `orange_overshoot_penalty`, `orange_stick_landing` (Task 2, already exported from `simple_goalkeeper.mdp` as `gk_mdp.orange_*`).
- Produces: 4 live reward terms in the reward manager plus 3 curriculum terms — this is the point where the mechanism becomes training-affecting.

- [ ] **Step 1: Add the 4 `RewardTermCfg` entries**

In `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, insert directly after the existing `"blue_trunk_drive": RewardTermCfg(...)` entry (ends at line 735) and before the `"foot_clearance"` entry's preceding comment (`# --- active stepping: reward lifting feet during approach ---`, line 736):

```python
        # --- trailing-foot ("orange") mirror of the blue waypoint above,
        # landing-focused subset (2026-08-08). See rewards.py's
        # _get_orange_reach_target_y/orange_ball_landed/orange_overshoot_penalty/
        # orange_stick_landing docstrings and
        # docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
        # Weights are half of blue's own (conservative first pass, unvalidated). ---
        "orange_foot_proximity": RewardTermCfg(
            func=gk_mdp.orange_foot_proximity,
            weight=2.5,
            params={"ball_name": BALL_NAME, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        "orange_ball_landed": RewardTermCfg(
            func=gk_mdp.orange_ball_landed,
            weight=5.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        "orange_overshoot_penalty": RewardTermCfg(
            func=gk_mdp.orange_overshoot_penalty,
            weight=-30.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
        "orange_stick_landing": RewardTermCfg(
            func=gk_mdp.orange_stick_landing,
            weight=4.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
```

- [ ] **Step 2: Add the 3 `CurriculumTermCfg` entries**

In the same file, insert directly after the existing `blue_trunk_drive_curriculum` entry's closing (the block starting at line 365 — find its closing `)` and insert after it):

```python
        # NEW 2026-08-08: same curriculum shape as the blue_* terms above,
        # base_weight halved (conservative first pass) -- see
        # docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
        cfg.curriculum["orange_ball_landed_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "orange_ball_landed",
                "base_weight": 5.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        cfg.curriculum["orange_overshoot_penalty_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "orange_overshoot_penalty",
                "base_weight": -30.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
        cfg.curriculum["orange_stick_landing_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "orange_stick_landing",
                "base_weight": 4.0,
                "update_interval": 500,
                "ep_len_divisor":  50,
            },
        )
```

(`orange_foot_proximity` gets no curriculum entry, matching `foot_proximity`, which also has none — it stays flat at `2.5`.)

- [ ] **Step 3: Verify the config loads and the reward manager resolves all 4 new terms**

Run:

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
env -u PYTHONPATH uv run python -c "
import torch
import simple_goalkeeper.tasks  # registers task ids with mjlab's registry
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

env_cfg = load_env_cfg('Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc', play=True)
env_cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=env_cfg, device='cpu')
env.reset()
for name in ['orange_foot_proximity', 'orange_ball_landed', 'orange_overshoot_penalty', 'orange_stick_landing']:
    term_cfg = env.reward_manager.get_term_cfg(name)
    print(name, term_cfg.weight)
env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
print('step ok')
"
```

(Same `load_env_cfg`/`ManagerBasedRlEnv` construction pattern as Task 2 Step 4 and `scripts/play.py:run_play`.)

Expected: prints the 4 term names with weights `2.5, 5.0, -30.0, 4.0` and `step ok` with no exceptions.

- [ ] **Step 4: Add the `docs/BugFixes.md` dated entry**

Append to the end of `docs/BugFixes.md` (after the existing final `---` separator):

```markdown
## 2026-08-08 -- orange trailing-foot waypoint (landing-focused subset)

**Context:** user reported the trailing (non-assigned) foot is sometimes stationary during interception and doesn't lift/step to fully complete the double-stepping motion. Investigated first (via a research subagent): the leading/assigned foot has a rich positional-target mechanism -- the "blue ball" waypoint (`_get_reach_target_y`, `rewards.py:340`) -- feeding `footreach`/`foot_proximity`/`blue_ball_landed`/`blue_overshoot_penalty`/`blue_stick_landing`. The trailing foot had no positional target at all: only `trailing_foot_forward_continuous` (orientation-only, always active) and the shared `foot_clearance` (weight +2.0, `max(height)` **across both feet** -- fully satisfied once the leading foot alone lifts, giving the trailing foot zero pressure of its own).

**Fix:** added a parallel "orange" mirror, landing-focused subset only (deliberately NOT the full blue stack -- see below):

- `_get_orange_reach_target_y` (`rewards.py`, new): target formula `orange_y = start_y + sign(delta)*max(|delta|-0.30, 0)/2` where `delta = full_y - start_y` -- blue's own midpoint formula, shrunk 30cm off the top before halving (confirmed with user via worked examples: delta=+1.00m gives orange_y=start+0.35m vs blue's start+0.50m; sign-safe so right-side/negative-delta crossings shrink toward the robot, not away from it). Reuses `env._blue_wide` directly (read-only) for the wide/narrow gate -- no new wide/region computation, since wide crossings are exactly where this project's own RSI region routing already uses double/triple-step motions (single-step motions are used for narrow crossings). Keyed to the trailing foot (`1 - _get_correct_foot_idx(...)`). Copies blue's curriculum-eased `landing_radius` (0.20->0.15m)/`landing_speed_threshold` (2.0->1.0 m/s) lerp and leaky-settle-count mechanism verbatim in shape (blue's own best-evidenced design, not re-derived). Own state namespace (`env._orange_*`), never touches `env._blue_*` except the read-only wide gate. Unlike blue, does NOT graduate to a live-ball target once landed -- no second phase for the trailing foot to converge into in this subset.
- `orange_foot_proximity`, `orange_ball_landed`, `orange_overshoot_penalty`, `orange_stick_landing` (`rewards.py`, new): structural copies of `foot_proximity`/`blue_ball_landed`/`blue_overshoot_penalty`/`blue_stick_landing` respectively, retargeted to the trailing foot and `_get_orange_reach_target_y`.
- Registered in `goalkeeper_env_cfg.py`: 4 new `RewardTermCfg` entries + 3 new `CurriculumTermCfg` entries (`orange_foot_proximity` has no curriculum, matching `foot_proximity`). Weights are half of blue's own current values (conservative first pass, confirmed with user) -- `orange_ball_landed` base_weight 5.0 (blue: 10.0), `orange_overshoot_penalty` base_weight -30.0 (blue: -60.0), `orange_stick_landing` base_weight 4.0 (blue: 8.0), `orange_foot_proximity` flat 2.5 (blue's `foot_proximity`: 5.0).

**Explicitly NOT ported in this change** (see the design spec's "Explicitly out of scope" section): `footreach`'s full phase1 (lateral pre-position)/phase2 (sigmoid reach x vel_sigma)/overshoot-kill-flag/blue-decel-zone machinery (the most complex, most ball-interception-specific piece of blue's stack), `near_stick_reach` (narrow-crossing anti-oscillation -- not needed since orange is wide-only), `blue_trunk_drive` (whole-body lateral drive). This project's own history shows every prior addition here landed best as a small, single change validated by a live training run before the next one was layered on -- these three can be added later as follow-ups if the landing-focused subset alone doesn't resolve the stationary-trailing-foot symptom.

**No G1 equivalent** -- same justification class as blue's own "Two-stage blue/green waypoint mechanism" row in `CLAUDE.md`'s Divergences table (G1 has no intermediate-waypoint concept at all, for either foot).

**Verification:** `env -u PYTHONPATH uv run pytest tests/ -q` -- 80/80 pass (76 existing + 4 new `_get_orange_reach_target_y` formula tests). Live check on a real `ManagerBasedRlEnv` (4 envs, cpu, 10 zero-action steps): all 4 new reward functions return finite `(4,)` tensors when called directly; reward manager resolves all 4 registered terms with the expected initial weights (2.5, 5.0, -30.0, 4.0) and a full env `.step()` completes with no exceptions. **Not yet validated against a live training run** -- next checkpoint after a fresh run should be compared against a pre-change run for whether the trailing-foot-stationary symptom actually improves, and the conservative (halved) weights may need retuning once there's live evidence, consistent with this project's convention for every other new reward term.

**See also:** `docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md` (full design), `docs/superpowers/plans/2026-08-08-orange-ball-trailing-foot.md` (implementation plan).

---
```

- [ ] **Step 5: Add the `CLAUDE.md` Divergences table row**

In `CLAUDE.md`'s "Divergences from G1 Upstream" table, insert a new row directly after the existing "Two-stage blue/green waypoint mechanism" row:

```markdown
| Trailing-foot positional targeting ("orange ball") | N/A (no G1 equivalent, same justification class as the "Two-stage blue/green waypoint mechanism" row above — G1 has no intermediate-waypoint concept for either foot) | **NEW 2026-08-08 (landing-focused subset).** `_get_orange_reach_target_y`, `orange_foot_proximity`, `orange_ball_landed`, `orange_overshoot_penalty`, `orange_stick_landing` (`rewards.py`) — trailing-foot mirror of the leading-foot blue-ball waypoint. | **FIX 2026-08-08 (user request):** trailing foot was sometimes stationary during double-stepping — `trailing_foot_forward_continuous` (orientation-only) and the shared `foot_clearance` (max height across both feet, fully satisfied by the leading foot alone) gave it no positional target of its own. New target formula (confirmed with user via worked examples): `orange_y = start_y + sign(delta)*max(\|delta\|-0.30,0)/2` where `delta = full_y - start_y` — blue's own midpoint formula, shrunk 30cm off the top before halving, sign-safe for both left- and right-side crossings. Reuses `env._blue_wide` directly (no new wide/region computation). Weights are a conservative first pass, half of blue's own curriculum base weights. Deliberately does NOT mirror `footreach`'s phase1/vel_sigma/overshoot-kill/decel-zone machinery, `near_stick_reach`, or `blue_trunk_drive` yet — see `docs/BugFixes.md`, 2026-08-08. `rewards.py`, `goalkeeper_env_cfg.py`, `scripts/play.py` (orange viewer sphere, viewer-only). Not yet validated against a live training run. See `docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md`. |
```

- [ ] **Step 6: Add the `CLAUDE.md` Reward Design table rows**

In `CLAUDE.md`'s "Reward Design" table, insert 4 new rows directly after the existing `blue_trunk_drive` row:

```markdown
| `orange_foot_proximity` | +2.5 (flat, half of `foot_proximity`'s +5.0) | **NEW 2026-08-08.** Trailing-foot mirror of `foot_proximity` — dense `exp(-sigma*dist)` pull toward the orange target. Wide-crossings only (`env._blue_wide`). See Divergences table. |
| `orange_ball_landed` | base_weight 5.0, curriculum peak 12.5 at cu=3 (half of `blue_ball_landed`'s base_weight 10.0 / peak 25.0) | **NEW 2026-08-08.** One-shot bonus when the trailing foot genuinely lands at the orange target. |
| `orange_overshoot_penalty` | base_weight -30.0, curriculum peak -75.0 at cu=3 (half of `blue_overshoot_penalty`'s base_weight -60.0 / peak -150.0) | **NEW 2026-08-08.** Penalty for the trailing foot advancing past orange while unlanded. |
| `orange_stick_landing` | base_weight 4.0, curriculum peak 10.0 at cu=3 (half of `blue_stick_landing`'s base_weight 8.0 / peak 20.0) | **NEW 2026-08-08.** Dense "close AND slow" bonus near orange. |
```

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/ -q`
Expected: 80 passed

- [ ] **Step 8: Commit**

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
git add src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py docs/BugFixes.md CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(rewards): wire orange trailing-foot terms into reward manager

Registers orange_foot_proximity/orange_ball_landed/orange_overshoot_penalty/
orange_stick_landing as RewardTermCfg entries (weights half of blue's own,
conservative first pass) plus 3 matching CurriculumTermCfg entries. Documents
the mechanism in docs/BugFixes.md and CLAUDE.md's Divergences/Reward Design
tables. Not yet validated against a live training run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Orange viewer sphere (viewer-only, no training effect)

**Files:**
- Modify: `src/simple_goalkeeper/scripts/play.py` (extend `_patch_viewer_intercept_vis`, `play.py:397-500`)
- Modify: `docs/BugFixes.md` (amend the 2026-08-08 entry from Task 3 with a short Visualization note)

**Interfaces:**
- Consumes: the same local variables `_patched_update` already computes (`wide`, `cross_y`, `goal_x`, `floor_z`, `sphere_z`, `origins`) — no new env attributes needed, since (matching how blue's own `mid_y` is handled) the target is recomputed inline rather than read from a cached attribute, so the marker can't drift out of sync with the reward code.
- Produces: an additional MuJoCo decor sphere+line drawn each render frame. No return value / no interface other tasks depend on.

Per this project's `CLAUDE.md` Change Approval Workflow, viewer/diagnostic-only code (this task) does not require a pre-edit approval round-trip — only `rewards.py`/`goalkeeper_env_cfg.py` changes (Tasks 2–3) did.

- [ ] **Step 1: Add the orange sphere block**

In `src/simple_goalkeeper/scripts/play.py`, inside `_patch_viewer_intercept_vis`'s `_patched_update` function, insert directly after the existing `if wide and not landed: ... else: ...` block (i.e. after the `else` branch's closing `_add_line(...)` call, before `native_viewer._update_debug_visualizers = _patched_update`):

```python
        # NEW 2026-08-08: orange sphere -- trailing-foot ("orange") mirror of
        # the blue midpoint above, recomputed inline the same way blue's own
        # mid_y is (not read from a cached env attribute) so this marker can't
        # drift out of sync with what orange_foot_proximity/orange_ball_landed/
        # orange_overshoot_penalty/orange_stick_landing (rewards.py) actually
        # target. See rewards.py:_get_orange_reach_target_y and
        # docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
        # No color-switch-on-landed (unlike blue->green): the landing-focused
        # subset has no live-ball-tracking phase for the trailing foot to
        # graduate into, so a single static orange sphere is shown for the
        # whole wide-crossing window regardless of env._orange_landed.
        if wide:
            start_y = float(origins[1])
            delta = cross_y - start_y
            sign = 1.0 if delta >= 0 else -1.0
            shrunk = sign * max(abs(delta) - 0.30, 0.0)
            orange_y = start_y + shrunk / 2.0
            _add_sphere(goal_x, orange_y, sphere_z, 0.08, [1.0, 0.55, 0.0, 0.75])
            _add_line(
                np.array([goal_x, orange_y, floor_z], dtype=np.float64),
                np.array([goal_x, orange_y, sphere_z], dtype=np.float64),
                0.008, [1.0, 0.55, 0.0, 0.6],
            )
```

- [ ] **Step 2: Static check — file parses**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run python -c "import ast; ast.parse(open('src/simple_goalkeeper/scripts/play.py').read())" && echo ok`
Expected: `ok`

- [ ] **Step 3: Live visual check**

Run play with a real checkpoint (the plan's author should substitute whatever checkpoint the user wants verified, e.g. the `model_17250.pt` mentioned at the start of this feature request) and confirm visually in the viewer that an orange sphere appears alongside the blue/green one during a wide crossing, at a position visibly between the robot's stance and the blue sphere (per the worked-example table in the design spec):

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --checkpoint-file logs/rsl_rl/intercept_simple_goalkeeper_multidisc/2026-08-07_20-46-08_6144_shoulderscale1_2026-08-07/model_17250.pt
```

This step requires a human (the user) to visually confirm the sphere's presence/position — report back what you see rather than assuming success from the command exiting cleanly.

- [ ] **Step 4: Amend the `docs/BugFixes.md` entry**

Add this paragraph to the end of the 2026-08-08 entry added in Task 3 (before its closing `**See also:**` line):

```markdown
**Visualization:** `_patch_viewer_intercept_vis` (`play.py`) extended to draw an ORANGE sphere (rgba `[1.0, 0.55, 0.0, 0.75]`) at the current orange target whenever `env._blue_wide` (same visibility window as blue's own sphere), recomputed inline from the same formula rather than read from a cached env attribute, matching blue's own established pattern for this marker. No color-switch-on-landed (unlike blue's blue->green graduation) since this subset has no live-ball-tracking phase for the trailing foot. Viewer-only, no training effect.
```

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `cd /home/isaak/BEPImitationlearning/interceptV2DualDis && env -u PYTHONPATH uv run pytest tests/ -q`
Expected: 80 passed

- [ ] **Step 6: Commit**

```bash
cd /home/isaak/BEPImitationlearning/interceptV2DualDis
git add src/simple_goalkeeper/scripts/play.py docs/BugFixes.md
git commit -m "$(cat <<'EOF'
feat(play): add orange trailing-foot target sphere to viewer

Viewer-only addition to _patch_viewer_intercept_vis -- draws an orange
sphere at the current orange target (rewards.py:_get_orange_reach_target_y)
alongside the existing blue/green leading-foot markers, recomputed inline
from the same formula so it can't drift out of sync with the reward code.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Not in this plan (deliberately deferred)

- Pushing any of these commits to `origin/v2-blue-ball-waypoint` — ask the user explicitly before pushing, per general repo-safety practice (this plan intentionally stops at local commits).
- Launching a training run to validate the mechanism, and the "ask about scheduled monitoring" follow-up `CLAUDE.md` describes for any new training run — do this only after the user reviews the 4 commits above.
- Any follow-up on the separately-tracked leading-foot sideways-landing investigation (`docs/BugFixes.md`, 2026-08-07 `footyawspinfix` entries) — explicitly out of scope for this plan per the user's own sequencing ("orange-ball first").
- `footreach`'s full phase1/vel_sigma/overshoot-kill/decel-zone mirror, `near_stick_reach`, `blue_trunk_drive` — explicitly deferred per the approved design's scope decision; revisit only if a live training run shows the landing-focused subset alone doesn't resolve the stationary-trailing-foot symptom.
