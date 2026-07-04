# Blue-Ball Landing Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate the two-stage footreach/foot_proximity schedule's blue→green target switch on an actual foot-landing event (not elapsed time alone), plus a one-shot bonus reward for the landing itself.

**Architecture:** Extend `rewards.py::_get_reach_target_y` with two new episode-scoped latched buffers (`env._blue_was_airborne`, `env._blue_landed`) tracking whether the assigned foot has left the ground and then landed within radius of the blue-ball target; the existing time-based switch becomes a fallback for episodes where landing never happens. A new `blue_ball_landed` reward function reads the resulting latch for a one-shot bonus. Both are wired into `goalkeeper_env_cfg.py`.

**Tech Stack:** Python 3.11, PyTorch, mjlab (MuJoCo-Warp), pytest.

## Global Constraints

- Landing radius: 0.3 m (matches `footreach`'s existing `reach_th`).
- Fallback deadline: unchanged `elapsed >= t_flight / 2` (matches current behavior exactly when landing never occurs).
- New reward `blue_ball_landed` weight: +10.0 flat, no curriculum.
- Only affects wide crossings (`|crossing_y - start_y| > 0.6`) — narrow crossings have no blue target and must be unaffected.
- Must not break the existing 11 tests in `tests/simple_goalkeeper/test_reach_target_two_stage.py` (fake envs there have no `robot`/`feet_contact` in scene) or the 4 tests in `test_footreach_two_stage_wiring.py` (fake env there has `robot`/`ball` but no `feet_contact`) — landing detection must silently no-op (`KeyError` → skip) when either is absent from `env.scene`, giving identical behavior to before this change.
- Per `SimpleGoalKeeper/CLAUDE.md`: every divergence from G1 upstream must be documented in the "Divergences from G1 Upstream" table (G1 has no equivalent mechanism, same as the parent two-stage-target row), and every fix/feature must get a dated entry in `SimpleGoalKeeper/docs/BugFixes.md`.
- Design source of truth: `docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md`.

---

### Task 1: Landing gate in `_get_reach_target_y`

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py:143-193` (the `_get_reach_target_y` function)
- Test: `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (new file)

**Interfaces:**
- Consumes: `_get_ball_crossing_y(env, ball_name)`, `_get_correct_foot_idx(env, ball_name)`, `_DEFAULT_FEET_CFG` (all already in `rewards.py`); `ContactSensor` (already imported in `rewards.py` from `mjlab.sensor`).
- Produces: `_get_reach_target_y(env, ball_name, wide_threshold=0.6, asset_cfg=_DEFAULT_FEET_CFG, landing_radius=0.3) -> torch.Tensor` (signature gains two new keyword params with defaults — existing positional callers unaffected). Side effects: sets `env._blue_was_airborne` (bool tensor, shape `(N,)`) and `env._blue_landed` (bool tensor, shape `(N,)`) as episode-scoped latches, consumed by Task 2's `blue_ball_landed`.

- [ ] **Step 1: Write the failing tests**

Create `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py`:

```python
"""Landing gate on the two-stage footreach schedule (2026-07-04, user-directed design).

_get_reach_target_y now requires the assigned foot to physically land near the
blue (midpoint) target before advancing early to the green (full) target on wide
crossings, rather than advancing on elapsed time alone. See rewards._get_reach_target_y
and docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md.

Uses the same fake env/robot/ball approach as test_footreach_two_stage_wiring.py,
extended with a fake "feet_contact" ContactSensor (.data.found, shape (N, 8)).
"""
import types

import torch

from simple_goalkeeper.mdp.rewards import _get_reach_target_y, _get_ball_crossing_y


class _EntityData:
    pass


class _Entity:
    def __init__(self, **kwargs):
        self.data = _EntityData()
        for k, v in kwargs.items():
            setattr(self.data, k, v)


class _Scene(dict):
    def __init__(self, entities, env_origins):
        super().__init__(entities)
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, robot, ball, feet_contact, env_origins, episode_step, rel_cross_y, t_flight, step_dt=0.02):
        n = env_origins.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene({"robot": robot, "ball": ball, "feet_contact": feet_contact}, env_origins)
        self.episode_length_buf = torch.full((n,), episode_step, dtype=torch.long)
        self.step_dt = step_dt
        self._rsi_cross_y = rel_cross_y
        self._ball_t_flight = t_flight
        self._ball_crossing_y = env_origins[:, 1] + rel_cross_y


def _feet_cfg():
    cfg = types.SimpleNamespace()
    cfg.name = "robot"
    cfg.body_ids = [0, 1]
    return cfg


def _make_contact(found_left: bool, found_right: bool):
    # 8 geoms: 0-3 left, 4-7 right (feet_contact geom layout; see feet_slippage/
    # penalize_sharpcontact in rewards.py for the same convention).
    found = torch.zeros(1, 8)
    found[:, :4] = 1.0 if found_left else 0.0
    found[:, 4:] = 1.0 if found_right else 0.0
    return _Entity(found=found)


def _make_env(foot_y: float, rel_cross_y: float, episode_step: int,
              found_left: bool, found_right: bool, t_flight: float = 1.0):
    n = 1
    env_origins = torch.zeros(n, 3)  # start_y=0, goal_x_w=0, floor_z=0

    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, foot_y, 0.0], [0.0, foot_y, 0.0]]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        root_link_lin_vel_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    ball = _Entity(root_link_pos_w=torch.tensor([[2.0, rel_cross_y, 0.11]]))
    feet_contact = _make_contact(found_left, found_right)

    rel = torch.tensor([rel_cross_y])
    t_flight_t = torch.tensor([t_flight])
    return _FakeEnv(robot, ball, feet_contact, env_origins, episode_step, rel, t_flight_t)


def test_landing_never_fires_without_prior_airborne():
    # rel_cross_y=0.9 (wide) -> crossing_y(0.9) > origin_y(0.0) -> assigned foot = left (idx 0).
    # Foot planted exactly at the midpoint (0.45) AND in contact from the very
    # first check -- it was never airborne, so landing must not fire.
    env = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=2, found_left=True, found_right=False)
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = (0.0 + full_y) / 2.0
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_landing_fires_after_airborne_then_contact_within_radius():
    # Step A: foot airborne (found=False), far from target.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_was_airborne[0].item()
    assert not env._blue_landed[0].item()

    # Step B (same env, later step): foot now in contact, planted at the midpoint.
    env.episode_length_buf[:] = 6  # elapsed = 0.12s, still < t_flight/2 = 0.5s
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot now in contact

    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_landed[0].item()
    # Landing switches to the full target EARLY, even though elapsed < t_flight/2.
    assert torch.allclose(target, full_y)


def test_landing_gate_fallback_advances_at_half_flight_time_when_never_landed():
    # Foot stays airborne the whole time (never lands near the target) -- the
    # schedule must still fall back to the full target once elapsed >= t_flight/2,
    # exactly like the pre-landing-gate behavior.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=30, found_left=False, found_right=False)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, full_y)


def test_landing_far_from_target_does_not_count():
    # Foot goes airborne then lands, but far from the blue target -- must not count.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    env.episode_length_buf[:] = 6
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]])  # far away
    env.scene["feet_contact"].data.found[:, :4] = 1.0
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = (0.0 + full_y) / 2.0
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_reach_target_y_without_robot_or_sensor_in_scene_is_unaffected():
    # Backward compatibility: envs with no "robot"/"feet_contact" in scene (the
    # existing test_reach_target_two_stage.py fake envs) must behave exactly as
    # before -- landing detection silently no-ops via the KeyError guard.
    class _BareScene(dict):
        def __init__(self, env_origins):
            super().__init__()
            self.env_origins = env_origins

    class _BareEnv:
        def __init__(self):
            n = 1
            env_origins = torch.zeros(n, 3)
            env_origins[:, 1] = 10.0
            self.num_envs = n
            self.device = "cpu"
            self.scene = _BareScene(env_origins)
            self._rsi_cross_y = torch.tensor([0.9])
            self._ball_t_flight = torch.tensor([1.0])
            self._ball_crossing_y = env_origins[:, 1] + 0.9
            self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
            self.step_dt = 0.02

    env = _BareEnv()
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    expected_mid = 10.0 + (full_y[0].item() - 10.0) / 2.0
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: FAIL — `_get_reach_target_y() got an unexpected keyword argument 'asset_cfg'` (signature doesn't accept it yet), and `AttributeError`/`assert` failures on `env._blue_landed`/`env._blue_was_airborne` (attributes don't exist yet).

- [ ] **Step 3: Implement the landing gate**

Replace the full body of `_get_reach_target_y` in `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` (currently lines 143-193) with:

```python
def _get_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    wide_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.3,
) -> torch.Tensor:
    """Two-stage reach target for wide crossings (2026-07-03, extended 2026-07-04).

    For crossings where |crossing_y - start_y| > wide_threshold (same threshold
    _tier_npz_reset uses to seed a double/triple-step RSI pose, events.py), the
    target is the MIDPOINT between the robot's stance and the true crossing point
    for the first half of the ball's flight time, then switches to the full
    crossing point for the second half. Narrow crossings always target the full
    point, unchanged from before this feature.

    2026-07-04 landing gate (user-directed design): the switch to the full point
    now ALSO fires early, as soon as the assigned foot (_get_correct_foot_idx)
    physically lands near the midpoint -- it must be airborne at some point after
    reset, then come into ground contact (feet_contact sensor) within
    landing_radius of the midpoint target. This replaces the pure-time switch
    with a hard gate + timeout fallback: landing before t_flight/2 advances the
    target early; never landing still falls back to the exact original
    elapsed >= t_flight/2 behavior, so episodes can never stall. Motivation: the
    original pure-time schedule let the policy glide/leap through the midpoint
    without ever placing a foot near it, since nothing checked it actually
    arrived -- see docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md.

    Landing state (env._blue_was_airborne, env._blue_landed) is only tracked when
    both a "robot" entity and a "feet_contact" sensor are present in env.scene --
    absent in the lightweight fake envs used by test_reach_target_two_stage.py,
    which exercise the time-based schedule in isolation and are unaffected by
    this extension (KeyError on either lookup silently skips the landing check,
    matching the exact original behavior).

    Root XY is pinned to env.scene.env_origins at every reset (reset_base,
    goalkeeper_env_cfg.py; _write_rsi_state, events.py, only ever overwrites Z),
    so "robot start Y" can always be read live with no new per-episode cache.

    Does not affect the separate live-ball switch already in footreach/
    foot_proximity (ball_x_local < 0.5 m), which still takes priority once the
    ball is genuinely close — this only changes the FROZEN target fed into that
    existing logic.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y

    rel = getattr(env, "_rsi_cross_y", None)
    lateral = rel if rel is not None else (full_y - start_y)
    wide = lateral.abs() > wide_threshold

    t_flight = getattr(env, "_ball_t_flight", None)
    if t_flight is None:
        return full_y

    elapsed = env.episode_length_buf.to(full_y.dtype) * env.step_dt
    first_half = elapsed < (t_flight / 2.0)

    half_y = start_y + (full_y - start_y) / 2.0

    # --- landing gate (2026-07-04) ---
    n = env.num_envs
    if not hasattr(env, "_blue_was_airborne"):
        env._blue_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_was_airborne[just_reset] = False
    env._blue_landed[just_reset] = False

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                    # (N,)
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                  # (N, 3)

        found = feet_contact.data.found                                    # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(foot_idx == 0, left_in_contact, right_in_contact)  # (N,)

        env._blue_was_airborne |= ~foot_in_contact

        goal_x_w = env.scene.env_origins[:, 0]
        floor_z_w = env.scene.env_origins[:, 2]
        target_point = torch.stack([goal_x_w, half_y, floor_z_w + 0.10], dim=-1)  # (N, 3)
        dist_to_blue = torch.norm(assigned_foot_pos - target_point, dim=-1)

        newly_landed = wide & env._blue_was_airborne & foot_in_contact & (dist_to_blue < landing_radius)
        env._blue_landed |= newly_landed

    phase1_active = wide & first_half & ~env._blue_landed
    return torch.where(phase1_active, half_y, full_y)
```

Then update the two existing call sites to pass `asset_cfg` through so the resolved (config-wired) `SceneEntityCfg` is used, not the unresolved module default:

In `footreach` (currently line ~231): change
```python
    reach_target_y = _get_reach_target_y(env, ball_name)               # (N,)
```
to
```python
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,)
```

In `foot_proximity` (currently line ~312): apply the identical change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the existing two-stage tests to confirm no regression**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_reach_target_two_stage.py tests/simple_goalkeeper/test_footreach_two_stage_wiring.py -v`
Expected: PASS (all 15 previously-passing tests, unchanged)

- [ ] **Step 6: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py
git commit -m "feat(sgk): gate two-stage footreach schedule on actual foot landing at the blue ball"
```

---

### Task 2: `blue_ball_landed` one-shot bonus reward

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` (add new function after `foot_proximity`)
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py` (export the new function)
- Test: `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (append to the file from Task 1)

**Interfaces:**
- Consumes: `_get_reach_target_y(env, ball_name, asset_cfg=...)` and `env._blue_landed` (both from Task 1).
- Produces: `blue_ball_landed(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG) -> torch.Tensor`, exported from `simple_goalkeeper.mdp` as `gk_mdp.blue_ball_landed` for Task 3's config wiring.

- [ ] **Step 1: Write the failing tests**

Append to `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py`:

```python
from simple_goalkeeper.mdp.rewards import blue_ball_landed


def test_blue_ball_landed_fires_once_per_episode():
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    r0 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r0.item() == 0.0

    env.episode_length_buf[:] = 6
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0
    r1 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r1.item() == 1.0  # fires exactly on the landing step

    env.episode_length_buf[:] = 7
    r2 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r2.item() == 0.0  # already paid, does not fire again


def test_blue_ball_landed_resets_on_new_episode():
    env = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=6, found_left=True, found_right=False)
    env._blue_landed = torch.ones(1, dtype=torch.bool)
    env._blue_was_airborne = torch.ones(1, dtype=torch.bool)
    env._blue_landed_bonus_flag = torch.ones(1, dtype=torch.bool)  # already paid last episode

    env.episode_length_buf[:] = 1  # new episode (reset step)
    r = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed_bonus_flag[0].item()
    assert not env._blue_landed[0].item()  # never airborne THIS episode -- must not land for free
    assert r.item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v -k blue_ball_landed`
Expected: FAIL with `ImportError: cannot import name 'blue_ball_landed'`

- [ ] **Step 3: Implement `blue_ball_landed`**

Add to `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py`, directly after the `foot_proximity` function:

```python
def blue_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """One-shot bonus when the assigned foot lands at the blue (midpoint) target.

    Fires once per episode the first time _get_reach_target_y's landing latch
    (env._blue_landed) becomes true -- the assigned foot was airborne at some
    point, then came down in contact with the ground within landing_radius of
    the phase-1 midpoint target on a wide crossing. See _get_reach_target_y for
    the detection mechanism (2026-07-04, user-directed design: the robot must
    physically land at the intermediate waypoint before advancing to the
    green/full target, rather than the schedule advancing on elapsed time alone).
    Weight: +10.0.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_landed is fresh this step

    if not hasattr(env, "_blue_landed_bonus_flag"):
        env._blue_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_landed_bonus_flag[just_reset] = False

    fired = env._blue_landed & ~env._blue_landed_bonus_flag
    env._blue_landed_bonus_flag |= fired
    return fired.float()
```

In `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py`, add `blue_ball_landed` to the `from .rewards import (...)` tuple:

```python
from .rewards import (
    ball_vx_reduction, posture, ang_vel_xy_l2, ang_vel_z_l2,
    stayonline, noretreat, feetorientation, foot_ang_vel_xy, deviation_waist_joint,
    footreach, foot_proximity, blue_ball_landed, stopball, softstop, single_foot_save, cleanstop, foot_clearance,
    airborne_at_save, inner_face_orientation_save, foot_inner_face_continuous,
    penalize_kneeheight, dof_vel_limits,
    postorientation, postangvel, postlinvel,
    torques_normalized_l2, torque_limits,
    postupperdofpos, postwaistdofpos,
    penalize_sharpcontact, penalize_self_collision, feet_slippage,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: PASS (7 tests total: 5 from Task 1 + 2 from this task)

- [ ] **Step 5: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py
git commit -m "feat(sgk): add blue_ball_landed one-shot bonus reward"
```

---

### Task 3: Config wiring

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py:343-347` (reward terms dict, right after `foot_proximity`)

**Interfaces:**
- Consumes: `gk_mdp.blue_ball_landed` (Task 2), `_FEET_CFG` (already defined at `goalkeeper_env_cfg.py:36`), `BALL_NAME` (already defined at `goalkeeper_env_cfg.py:34`).
- Produces: `cfg.rewards["blue_ball_landed"]` entry, active in both training and play configs (this file builds both via the same `cfg.rewards` dict).

- [ ] **Step 1: Add the reward term**

In `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, immediately after the `foot_proximity` entry (currently ending at line 347):

```python
        "foot_proximity": RewardTermCfg(
            func=gk_mdp.foot_proximity,
            weight=5.0,
            params={"ball_name": BALL_NAME, "sigma": 5.0, "asset_cfg": _FEET_CFG},
        ),
        # --- landing gate bonus: rewards physically landing at the blue-ball
        # midpoint target before the two-stage schedule advances to the green
        # target (2026-07-04, see _get_reach_target_y) ---
        "blue_ball_landed": RewardTermCfg(
            func=gk_mdp.blue_ball_landed,
            weight=10.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _FEET_CFG},
        ),
```

- [ ] **Step 2: Verify the config imports and builds without error**

Run: `cd SimpleGoalKeeper && uv run python -c "from simple_goalkeeper.tasks.goalkeeper_env_cfg import *" 2>&1 | tail -20`

If the module exposes a config factory function (check the file for `def make_*_cfg` or similar at the bottom), also run that factory once to confirm `cfg.rewards["blue_ball_landed"]` resolves without raising. Expected: no exceptions, no `AttributeError`/`KeyError` about `gk_mdp.blue_ball_landed` or `_FEET_CFG`.

- [ ] **Step 3: Run the full SimpleGoalKeeper test suite**

Run: `cd SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: PASS — all previously-passing tests plus the 7 new ones from Tasks 1-2, zero failures.

- [ ] **Step 4: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat(sgk): wire blue_ball_landed into the reward config"
```

---

### Task 4: Documentation

**Files:**
- Modify: `SimpleGoalKeeper/CLAUDE.md` (Divergences from G1 Upstream table + Reward Design table)
- Modify: `SimpleGoalKeeper/docs/BugFixes.md` (new dated entry)

**Interfaces:** None (docs only).

- [ ] **Step 1: Add a divergence-table row**

In `SimpleGoalKeeper/CLAUDE.md`, in the "Divergences from G1 Upstream" table, add a new row directly after the existing `footreach`/`foot_proximity` two-stage-target row:

```markdown
| `footreach`/`foot_proximity` reach target: landing gate on the two-stage schedule | N/A — G1 has no staged/waypoint target mechanism at all (see the parent two-stage-target row above); a landing-gated version of a mechanism that doesn't exist upstream has no G1 equivalent either. | **Landing gate (2026-07-04, user-directed design):** the blue→green switch on wide crossings now ALSO fires early as soon as the assigned foot is airborne at some point then lands in ground contact within 0.3 m of the blue midpoint target (`feet_contact` sensor). The original pure-time switch (`elapsed >= t_flight/2`) remains as the fallback for episodes where landing never happens, so episodes cannot stall. New one-shot bonus `blue_ball_landed` (weight +10.0) pays out the first time the landing latch fires. `mdp/rewards.py:_get_reach_target_y,blue_ball_landed`. | **Deliberate divergence:** the 2026-07-03 two-stage schedule assumed the sigmoid reach term and vel_sigma multiplier would "self-correct" against lingering at the midpoint without a separate landing check — in practice the policy glided/leapt through the midpoint without ever placing a foot near it, since nothing checked it actually arrived before the time-based switch advanced anyway. Gating the switch on a genuine airborne-then-contact event forces the intermediate step to actually happen. See `docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md`. Covered by `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (7 tests: no-landing-without-airborne, landing advances early, fallback unchanged when never landed, out-of-radius landing doesn't count, backward compatibility with the pre-existing two-stage tests' bare fake envs, one-shot bonus fires once, bonus resets on new episode). Not yet validated against a training run. |
```

Then add a row to the "Reward Design" table (Phase 1 reward structure), directly after the `foot_proximity` row:

```markdown
| `blue_ball_landed` | +10.0 | One-shot bonus when the assigned foot lands (airborne-then-contact) within 0.3 m of the blue-ball midpoint target on a wide crossing, gating the two-stage schedule's early advance to the green target. |
```

- [ ] **Step 2: Add a BugFixes.md entry**

Append to `SimpleGoalKeeper/docs/BugFixes.md`:

```markdown

---

## 2026-07-04 — footreach/foot_proximity: landing gate on the two-stage schedule

**What changed:** `mdp.rewards._get_reach_target_y` gains a landing gate: the assigned foot (`_get_correct_foot_idx`) must be airborne (`feet_contact` sensor) at some point after reset, then land in ground contact within 0.3 m of the phase-1 midpoint ("blue ball") target on wide crossings (`|crossing_y - start_y| > 0.6`). Landing switches the target to the full crossing point ("green ball") immediately; the original `elapsed >= t_flight/2` time-based switch remains as a fallback for episodes that never land, so episodes cannot stall. New one-shot bonus reward `blue_ball_landed` (weight +10.0, no curriculum) pays out the first time the landing latch fires per episode. Landing state (`env._blue_was_airborne`, `env._blue_landed`) is only tracked when both a `robot` entity and `feet_contact` sensor are present in `env.scene`; absent in the lightweight fake envs used by the pre-existing `test_reach_target_two_stage.py`/`test_footreach_two_stage_wiring.py` suites, which remain fully unaffected. `mdp/rewards.py:_get_reach_target_y,blue_ball_landed`, `mdp/__init__.py`, `tasks/goalkeeper_env_cfg.py`.

**Why it was wrong:** the 2026-07-03 two-stage schedule (previous entry above) assumed the existing sigmoid reach term and up-to-10x `vel_sigma` multiplier would self-correct against lingering at the midpoint without needing a separate "did you actually step" check. In practice the policy has not learned the intended pause-at-blue-then-continue-to-green double-step motion — it can glide/leap through the midpoint region without ever placing a foot near it, and the time-based switch advances regardless, so nothing in the reward actually required the intermediate landing to happen.

**Why this design over alternatives:** rejected making it purely a soft bonus with the time-based schedule left unchanged, because that keeps the actual defect (schedule advances without requiring landing) unaddressed — a bonus alone doesn't stop the leap-through behavior, it just adds an extra incentive on top of it. Chose a hard gate with the existing time-based switch retained ONLY as a timeout fallback, so training can't stall on a robot that refuses to land. User-directed design; full design rationale in `docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md`.

**Evidence:** `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (7 tests). Not yet validated against a training run — next checkpoint should be compared against the prior run (`2026-07-03_18-53-55_phase1`) for whether double-stepping actually emerges, and checked in `sgk_play` for the blue→green marker now switching on landing rather than only on elapsed time.
```

- [ ] **Step 3: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/CLAUDE.md SimpleGoalKeeper/docs/BugFixes.md
git commit -m "docs(sgk): document the blue-ball landing gate divergence and fix"
```

---

## Post-plan note (not a task)

This change alone does not retrain anything — it only changes the reward/schedule code. A fresh training run is needed to see whether the landing gate actually produces the intended double-step behavior; that run should be launched and monitored as a separate, explicit step after this plan lands, not folded into it.
