# Multi-Discriminator AMP (HIM-style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4-region (left/right × near/far, split at the existing 0.5 m crossing-distance
threshold), 4-discriminator AMP system with HIM-style observation-history + ball/region estimator
heads to `interceptV2DualDis`, ported from `Humanoid-Goalkeeper/rsl_rl/rsl_rl/{modules/actor_critic.py,modules/amp.py,algorithms/him_ppo.py,runners/him_on_policy_runner.py}`.

**Architecture:** Static per-env-slot region partition (4 contiguous blocks of `num_envs`, assigned
once at startup, ball-spawn range resolved per block) feeds a region id into the critic
observation. A forked actor-critic (`HimActorCritic`) adds three small MLPs — `history_encoder`,
`ball_estimator`, `region_estimator` — consuming a 10-step observation history (via mjlab's native
`ObservationGroupCfg(history_length=...)`), trained against ground truth stashed in the critic obs.
A forked algorithm (`MultiDiscAMPPPO`) holds 4 independent `AMPDiscriminator` instances and 4
independent `MotionDataset` instances (one motion file each), routes both the rollout-time style
reward and the update-time GAIL loss by each sample's (static, known) region id — ported from
`HIMPPO`/`him_on_policy_runner.py`'s `motion_ids`-masking pattern.

**Tech Stack:** mjlab (MuJoCo-Warp), `beyondAMP`/`rsl_rl_amp` (forked, not edited in place), PyTorch, pytest.

## Global Constraints

- Everything in this plan lives under `interceptV2DualDis/` only. `SimpleGoalKeeper/` and the
  shared `beyondAMP` upstream clone are never modified — new classes are added as new files, not
  edits to existing `beyondAMP` source.
- 4 regions: `left_near`, `left_far`, `right_near`, `right_far` (indices 0-3 in that order).
  Split point: `|cross_y| < 0.5` → near, `>= 0.5` → far (mirrors the existing pool-threshold
  convention already used elsewhere in this codebase).
- Motion assignment per region: `left_near` → `LeftStep_own_booster_t1.npz` only, `left_far` →
  `LeftDoubleStep_own_booster_t1.npz` only, `right_near` → `Rightstep_own_booster_t1.npz` only,
  `right_far` → `RightDoubleStep_own_booster_t1.npz` only. `LeftTripleStep_own_booster_t1.npz` /
  `RightTripleStep_own_booster_t1.npz` are excluded from every buffer — never loaded by this task.
- Ball estimator output is 2D (`pos_x, pos_y, vel_x, vel_y` — 4 dims), not G1's 3D/6-dim, matching
  SGK's existing XY-only deployment convention (`ball_pos_xy_b`).
- Region estimator is 4-way (not G1's 6-way).
- History window: 10 steps (`actor_history_length = 10`), matching G1's `num_actor_history`
  default — SGK runs the same `dt = 0.02`.
- New task id: `Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc`, registered alongside (not replacing) the
  existing `Mjlab-BeyondAMP-Goalkeeper-T1` task, so the existing single-discriminator task keeps
  working unmodified.
- Reference: `docs/superpowers/specs/2026-07-02-multi-discriminator-amp-design.md` (the approved
  design spec this plan implements).

---

## File Map

New files (all under `interceptV2DualDis/`):
- `src/simple_goalkeeper/mdp/regions.py` — static region assignment, region-conditioned ball-spawn dispatcher, critic-obs ground-truth term functions.
- `src/simple_goalkeeper/rsl_rl_multi/__init__.py`
- `src/simple_goalkeeper/rsl_rl_multi/him_actor_critic.py` — `HimActorCritic`.
- `src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py` — `MultiDiscAMPPPO`.
- `src/simple_goalkeeper/rsl_rl_multi/him_amp_on_policy_runner.py` — `HimAMPOnPolicyRunner`.
- `src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py` — `goalkeeper_multidisc_amp_runner_cfg()`, `MultiDiscAMPRunnerCfg`.
- `tests/simple_goalkeeper/test_regions.py`
- `tests/simple_goalkeeper/test_him_actor_critic.py`
- `tests/simple_goalkeeper/test_multi_disc_amp_ppo.py`

Modified files:
- `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` — add `assign_static_regions` startup event, replace `reset_ball` event with region-conditioned dispatcher, add `region_gt`/`ball_gt` critic obs terms, add `actor_history` obs group.
- `src/simple_goalkeeper/mdp/__init__.py` — export new `regions.py` functions.
- `src/simple_goalkeeper/tasks/__init__.py` — register the new task id.

---

## Task 1: Static region assignment + region-conditioned ball spawn

**Files:**
- Create: `src/simple_goalkeeper/mdp/regions.py`
- Modify: `src/simple_goalkeeper/mdp/__init__.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py:513-530` (the `init_motion_loader` / `reset_ball` event block)
- Test: `tests/simple_goalkeeper/test_regions.py`

**Interfaces:**
- Produces: `assign_static_regions(env, env_ids) -> None` — sets `env._region_id: torch.Tensor[int64]` shape `(num_envs,)`, values 0-3, called once at `mode="startup"`.
- Produces: `REGION_NAMES: tuple[str, ...] = ("left_near", "left_far", "right_near", "right_far")`.
- Produces: `reset_ball_rolling_by_region(env, env_ids, ball_name, dist_range, t_flight_range, spawn_z) -> None` — region-conditioned wrapper around the existing `reset_ball_rolling` (imported from `events.py`, signature unchanged: `reset_ball_rolling(env, env_ids, ball_name, dist_range, y_start_range, y_end_range, t_flight_range, spawn_z)`).
- Consumes (from existing `events.py`): `reset_ball_rolling` (unmodified).

- [ ] **Step 1: Write the failing test for region assignment**

Create `tests/simple_goalkeeper/test_regions.py`:

```python
"""Tests for static region assignment and region-conditioned ball spawn."""
import types

import torch

from simple_goalkeeper.mdp.regions import (
    REGION_NAMES,
    assign_static_regions,
    reset_ball_rolling_by_region,
)


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"


def test_region_names_are_four_in_order():
    assert REGION_NAMES == ("left_near", "left_far", "right_near", "right_far")


def test_assign_static_regions_splits_into_four_equal_contiguous_blocks():
    env = _FakeEnv(num_envs=12)
    assign_static_regions(env, env_ids=None)
    assert env._region_id.shape == (12,)
    assert env._region_id.dtype == torch.int64
    expected = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3, dtype=torch.int64)
    assert torch.equal(env._region_id, expected)


def test_assign_static_regions_handles_non_multiple_of_four():
    # 10 envs: quarter=2, remainder 2 envs go to the last block (right_far).
    env = _FakeEnv(num_envs=10)
    assign_static_regions(env, env_ids=None)
    assert env._region_id.shape == (10,)
    counts = torch.bincount(env._region_id, minlength=4)
    assert counts[0].item() == 2
    assert counts[1].item() == 2
    assert counts[2].item() == 2
    assert counts[3].item() == 4  # 2 base + 2 remainder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd interceptV2DualDis && source .venv/bin/activate && python3 -m pytest tests/simple_goalkeeper/test_regions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_goalkeeper.mdp.regions'`

- [ ] **Step 3: Implement `assign_static_regions` and `REGION_NAMES`**

Create `src/simple_goalkeeper/mdp/regions.py`:

```python
"""Static 4-way region partitioning and region-conditioned ball spawn.

Region assignment is a permanent, one-time split of the parallel env batch
into 4 contiguous blocks — mirrors Humanoid-Goalkeeper's `end_regions`
mechanism (legged_gym/legged_gym/envs/base/legged_robot.py:916-924), which
splits `num_envs` into 6 fixed blocks at startup and never reassigns them.
Here it's 4: left_near, left_far, right_near, right_far.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .events import reset_ball_rolling

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

REGION_NAMES: tuple[str, ...] = ("left_near", "left_far", "right_near", "right_far")

# Per-region ball-spawn y_start_range / y_end_range. Side sign matches the
# existing convention: positive Y crossing = left, negative Y crossing =
# right (see rewards.py:_get_correct_foot_idx). |cross_y| < 0.5 = near,
# >= 0.5 = far, matching the design spec's threshold.
_REGION_Y_START_RANGE: dict[int, tuple[float, float]] = {
    0: (0.0, 0.3),     # left_near
    1: (0.0, 0.3),     # left_far
    2: (-0.3, 0.0),    # right_near
    3: (-0.3, 0.0),    # right_far
}
_REGION_Y_END_RANGE: dict[int, tuple[float, float]] = {
    0: (0.15, 0.5),    # left_near: crosses on the left, under 0.5 m
    1: (0.5, 0.9),     # left_far: crosses on the left, at/above 0.5 m
    2: (-0.5, -0.15),  # right_near
    3: (-0.9, -0.5),   # right_far
}


def assign_static_regions(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: split env.num_envs into 4 fixed contiguous region blocks.

    Sets env._region_id (int64, shape (num_envs,), values 0-3). Called once
    with mode="startup" — env_ids is ignored (region assignment always
    covers the full batch and is never reassigned on reset).
    """
    n = env.num_envs
    quarter = n // 4
    remainder = n - quarter * 4
    counts = [quarter, quarter, quarter, quarter + remainder]
    region_id = torch.cat([
        torch.full((counts[r],), r, dtype=torch.int64, device=env.device)
        for r in range(4)
    ])
    env._region_id = region_id


def reset_ball_rolling_by_region(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 3.5),
    t_flight_range: tuple[float, float] = (0.7, 1.1),
    spawn_z: float = 0.12,
) -> None:
    """Region-conditioned ball spawn: calls reset_ball_rolling once per region
    subset of env_ids, using that region's y_start_range/y_end_range so the
    spawned ball actually produces that region's category of shot.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    region_id = env._region_id[env_ids]
    for r in range(4):
        mask = region_id == r
        if not mask.any():
            continue
        reset_ball_rolling(
            env,
            env_ids[mask],
            ball_name,
            dist_range=dist_range,
            y_start_range=_REGION_Y_START_RANGE[r],
            y_end_range=_REGION_Y_END_RANGE[r],
            t_flight_range=t_flight_range,
            spawn_z=spawn_z,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_regions.py -v`
Expected: 3 passed (the ball-spawn dispatcher test is added in Step 5 below before this final run).

- [ ] **Step 5: Write the failing test for the region-conditioned ball spawn dispatcher**

Add to `tests/simple_goalkeeper/test_regions.py`:

```python
def test_reset_ball_rolling_by_region_calls_reset_ball_rolling_per_region(monkeypatch):
    calls = []

    def fake_reset_ball_rolling(env, env_ids, ball_name, **kwargs):
        calls.append((tuple(env_ids.tolist()), kwargs["y_start_range"], kwargs["y_end_range"]))

    import simple_goalkeeper.mdp.regions as regions_mod
    monkeypatch.setattr(regions_mod, "reset_ball_rolling", fake_reset_ball_rolling)

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)
    reset_ball_rolling_by_region(env, env_ids=None, ball_name="ball")

    # 4 regions, 2 envs each (8 // 4 = 2) -> 4 calls, one per region.
    assert len(calls) == 4
    called_env_ids = {c[0] for c in calls}
    assert called_env_ids == {(0, 1), (2, 3), (4, 5), (6, 7)}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_regions.py -v`
Expected: 4 passed

- [ ] **Step 7: Wire the new events into goalkeeper_env_cfg.py**

In `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, this task only touches the
**new multi-disc task's** env cfg builder (added in Task 6/7 below) — do NOT modify
the existing `goalkeeper_env_cfg()` function used by the single-discriminator task.
No edit here yet; Task 6 adds a new `goalkeeper_multidisc_env_cfg()` function that
calls `goalkeeper_env_cfg()` then overrides the `reset_ball` and adds the
`assign_static_regions` event. Skip to Task 2.

- [ ] **Step 8: Export new functions from mdp/__init__.py**

Read `src/simple_goalkeeper/mdp/__init__.py` first to see the current export line, then
add `assign_static_regions, reset_ball_rolling_by_region` to the existing
`from .regions import ...` style import list (mirror how `events.py` functions are
already exported in that file).

- [ ] **Step 9: Commit**

```bash
git add src/simple_goalkeeper/mdp/regions.py src/simple_goalkeeper/mdp/__init__.py tests/simple_goalkeeper/test_regions.py
git commit -m "feat(multidisc): add static 4-region assignment and region-conditioned ball spawn"
```

---

## Task 2: Critic-observation ground-truth fields (ball state + region id)

**Files:**
- Modify: `src/simple_goalkeeper/mdp/regions.py` (add observation term functions)
- Modify: `src/simple_goalkeeper/mdp/__init__.py`
- Test: `tests/simple_goalkeeper/test_regions.py`

**Interfaces:**
- Produces: `ball_state_gt(env, ball_name) -> torch.Tensor` shape `(N, 4)` — `(pos_x, pos_y, vel_x, vel_y)` in robot body frame (reuses the same `quat_apply(quat_inv(...))` pattern already used by `observations.py:ball_pos_b`/`ball_vel_b`).
- Produces: `region_id_gt(env) -> torch.Tensor` shape `(N, 1)` — `env._region_id` as float32, unsqueezed.
- Consumes: `env._region_id` (set by `assign_static_regions`, Task 1).

- [ ] **Step 1: Write the failing test**

Add to `tests/simple_goalkeeper/test_regions.py`:

```python
def test_region_id_gt_returns_float_column_vector():
    from simple_goalkeeper.mdp.regions import region_id_gt

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)
    out = region_id_gt(env)
    assert out.shape == (8, 1)
    assert out.dtype == torch.float32
    assert torch.equal(out.squeeze(-1), env._region_id.float())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_regions.py::test_region_id_gt_returns_float_column_vector -v`
Expected: FAIL with `ImportError: cannot import name 'region_id_gt'`

- [ ] **Step 3: Implement `region_id_gt` and `ball_state_gt`**

Append to `src/simple_goalkeeper/mdp/regions.py`:

```python
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, quat_inv


def region_id_gt(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Ground-truth region id for the region_estimator's cross-entropy target.
    Shape (N, 1), float32 (cross-entropy target is cast to long by the caller).
    """
    return env._region_id.float().unsqueeze(-1)


def ball_state_gt(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ground-truth ball state for the ball_estimator's MSE target.

    Shape (N, 4): (pos_x, pos_y, vel_x, vel_y) in robot body frame. Same
    frame convention as observations.py:ball_pos_b/ball_vel_b (always
    visible here — this is privileged critic-only info, not gated).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    quat_i = quat_inv(robot.data.root_link_quat_w)
    pos_b = quat_apply(quat_i, ball.data.root_link_pos_w - robot.data.root_link_pos_w)
    vel_b = quat_apply(quat_i, ball.data.root_link_lin_vel_w)
    return torch.cat([pos_b[:, :2], vel_b[:, :2]], dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_regions.py -v`
Expected: 5 passed

- [ ] **Step 5: Export from mdp/__init__.py**

Add `ball_state_gt, region_id_gt` to the same import list edited in Task 1 Step 8.

- [ ] **Step 6: Commit**

```bash
git add src/simple_goalkeeper/mdp/regions.py src/simple_goalkeeper/mdp/__init__.py tests/simple_goalkeeper/test_regions.py
git commit -m "feat(multidisc): add ball/region ground-truth terms for critic obs"
```

---

## Task 3: `HimActorCritic` — history encoder + ball/region estimator heads

**Files:**
- Create: `src/simple_goalkeeper/rsl_rl_multi/__init__.py`
- Create: `src/simple_goalkeeper/rsl_rl_multi/him_actor_critic.py`
- Test: `tests/simple_goalkeeper/test_him_actor_critic.py`

**Interfaces:**
- Produces: `HimActorCritic(num_one_step_obs, actor_history_length, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, history_latent_dim=16, estimate_ball_dim=4, num_regions=4, **kwargs)`.
- Produces methods matching `rsl_rl_amp.modules.ActorCritic`'s contract so it drops into `AMPPPO`/`MultiDiscAMPPPO` unchanged: `act(obs_history) -> actions`, `act_inference(obs_history) -> action_mean`, `evaluate(critic_obs) -> value`, `get_actions_log_prob(actions)`, `.std`, `.distribution`, `.action_mean`, `.action_std`, `.entropy`, `is_recurrent = False`, `.fixed_std` (needed by `AMPPPO.update()`'s `min_std` clamp — set `False`).
- Produces (new, beyond base `ActorCritic`): `self.estimate_ball: torch.Tensor` and `self.estimate_region: torch.Tensor` set on every `act()`/`act_inference()` call (read by `MultiDiscAMPPPO.update()` to compute `est_loss`/`region_loss`).
- Consumes: nothing external — pure `nn.Module`, no env dependency. `num_one_step_obs` is the per-step actor obs dim (computed by the caller from the "actor" obs group's term shapes — see Task 6).

**Note on input layout vs G1:** mjlab's native `ObservationGroupCfg(history_length=N)` flattens
history **term-major** (`[termA_t0..t9, termB_t0..t9, ...]`), not G1's time-major
(`[obs_t0, obs_t1, ..., obs_t9]` each a full one-step vector). This means "the current step" is
**not** a contiguous slice of the flattened history tensor the way it is in G1
(`obs_history[:, -num_one_step_obs:]`). To keep the current-step slice simple and correct, the
actor consumes **two separate observation tensors**: `obs_current` (current one-step obs, shape
`(N, num_one_step_obs)` — the existing "actor" obs group, unchanged) and `obs_history` (the new
10-step-stacked "actor_history" group, shape `(N, num_one_step_obs * 10)`, used only to compute
`history_latent`/`estimate_ball`/`estimate_region`). This is a deliberate, documented adaptation
from G1's single-tensor slicing approach — functionally equivalent for a learned MLP consumer.

- [ ] **Step 1: Write the failing test**

Create `tests/simple_goalkeeper/test_him_actor_critic.py`:

```python
"""Tests for HimActorCritic: shapes, estimator heads, act/evaluate contract."""
import torch

from simple_goalkeeper.rsl_rl_multi.him_actor_critic import HimActorCritic


def _make_model(num_one_step_obs=20, history_length=10, num_critic_obs=40, num_actions=21):
    return HimActorCritic(
        num_one_step_obs=num_one_step_obs,
        actor_history_length=history_length,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
    )


def test_estimator_head_output_shapes():
    model = _make_model()
    assert model.history_encoder[-1].out_features == 16
    assert model.ball_estimator[-1].out_features == 4
    assert model.region_estimator[-1].out_features == 4


def test_actor_input_dim_matches_composition():
    num_one_step_obs = 20
    model = _make_model(num_one_step_obs=num_one_step_obs)
    # actor input = last raw one-step obs (20) + history_latent (16) + ball (4) + region argmax (1)
    expected = num_one_step_obs + 16 + 4 + 1
    assert model.num_actor_input == expected
    assert model.actor[0].in_features == expected


def test_act_sets_estimate_ball_and_estimate_region_with_correct_shapes():
    num_envs = 5
    num_one_step_obs = 20
    history_length = 10
    model = _make_model(num_one_step_obs=num_one_step_obs, history_length=history_length)
    obs_current = torch.randn(num_envs, num_one_step_obs)
    obs_history = torch.randn(num_envs, num_one_step_obs * history_length)
    actions = model.act(obs_current, obs_history)
    assert actions.shape == (num_envs, 21)
    assert model.estimate_ball.shape == (num_envs, 4)
    assert model.estimate_region.shape == (num_envs, 4)


def test_act_inference_returns_deterministic_action_mean():
    num_envs = 3
    num_one_step_obs = 20
    history_length = 10
    model = _make_model(num_one_step_obs=num_one_step_obs, history_length=history_length)
    obs_current = torch.randn(num_envs, num_one_step_obs)
    obs_history = torch.randn(num_envs, num_one_step_obs * history_length)
    mean1 = model.act_inference(obs_current, obs_history)
    mean2 = model.act_inference(obs_current, obs_history)
    assert torch.equal(mean1, mean2)
    assert mean1.shape == (num_envs, 21)


def test_evaluate_returns_scalar_value_per_env():
    num_envs = 4
    model = _make_model(num_critic_obs=40)
    critic_obs = torch.randn(num_envs, 40)
    value = model.evaluate(critic_obs)
    assert value.shape == (num_envs, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_him_actor_critic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_goalkeeper.rsl_rl_multi'`

- [ ] **Step 3: Implement `HimActorCritic`**

Create `src/simple_goalkeeper/rsl_rl_multi/__init__.py` (empty file).

Create `src/simple_goalkeeper/rsl_rl_multi/him_actor_critic.py`:

```python
"""HIM-style actor-critic: observation-history encoder + ball/region estimator
heads feeding the actor, ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/
actor_critic.py, adapted for SGK's 2D (XY-only) ball convention and 4 regions
instead of G1's 3D/6 regions. See docs/superpowers/specs/2026-07-02-multi-
discriminator-amp-design.md section B.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
    return {
        "elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
        "crelu": nn.ReLU(), "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }[act_name]


class HimActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_one_step_obs: int,
        actor_history_length: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        history_latent_dim: int = 16,
        estimate_ball_dim: int = 4,
        num_regions: int = 4,
        fixed_std: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(f"HimActorCritic.__init__ got unexpected arguments, ignored: {list(kwargs.keys())}")
        super().__init__()
        act = get_activation(activation)

        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.history_latent_dim = history_latent_dim
        self.estimate_ball_dim = estimate_ball_dim
        self.num_regions = num_regions
        self.fixed_std = fixed_std

        mlp_input_dim_h = num_one_step_obs * actor_history_length
        self.num_actor_input = num_one_step_obs + history_latent_dim + estimate_ball_dim + 1

        self.history_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, history_latent_dim),
        )
        self.ball_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, estimate_ball_dim),
        )
        self.region_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_regions),
        )

        actor_layers = [nn.Linear(self.num_actor_input, actor_hidden_dims[0]), act]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]), act]
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), act]
        self.critic = nn.Sequential(*critic_layers)

        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution: Normal | None = None
        self.estimate_ball: torch.Tensor | None = None
        self.estimate_region: torch.Tensor | None = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _build_actor_input(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        history_latent = self.history_encoder(obs_history)
        self.estimate_ball = self.ball_estimator(obs_history)
        self.estimate_region = self.region_estimator(obs_history)
        region_arg = torch.argmax(self.estimate_region, dim=-1, keepdim=True).float()
        return torch.cat([obs_current, history_latent, self.estimate_ball, region_arg], dim=-1)

    def update_distribution(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> None:
        actor_input = self._build_actor_input(obs_current, obs_history)
        mean = self.actor(actor_input)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, obs_current: torch.Tensor, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs_current, obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_current: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        actor_input = self._build_actor_input(obs_current, obs_history)
        return self.actor(actor_input)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_him_actor_critic.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/rsl_rl_multi/ tests/simple_goalkeeper/test_him_actor_critic.py
git commit -m "feat(multidisc): add HimActorCritic with history+ball+region estimator heads"
```

---

## Task 4: `MultiDiscAMPPPO` — 4 discriminators, region-routed loss + reward

**Files:**
- Create: `src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`
- Test: `tests/simple_goalkeeper/test_multi_disc_amp_ppo.py`

**Interfaces:**
- Produces: `MultiDiscAMPPPO(actor_critic: HimActorCritic, discriminators: dict[str, AMPDiscriminator], amp_datasets: dict[str, MotionDataset], amp_normalizer, num_learning_epochs=1, num_mini_batches=1, ..., device='cpu', min_std=None)`.
- Produces methods matching the call sites `HimAMPOnPolicyRunner` (Task 5) needs: `init_storage(...)`, `act(obs_current, obs_history, critic_obs, amp_obs)`, `process_env_step(rewards, dones, infos, amp_obs)`, `compute_returns(last_critic_obs)`, `update() -> (mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_est_loss, mean_region_loss, mean_policy_pred, mean_expert_pred)`.
- Produces: `predict_region_routed_amp_reward(amp_obs, next_amp_obs, region_id, task_reward) -> torch.Tensor` — used by the runner at rollout time (mirrors `him_on_policy_runner.py:161-178`'s masked `predict_reward` loop, but using each region's `AMPDiscriminator.predict_amp_reward` since that's the discriminator class actually in use here, not G1's `AMP.predict_reward`).
- Consumes: `HimActorCritic` (Task 3) — `act(obs_current, obs_history)` two-arg signature, `.estimate_ball`, `.estimate_region`. Consumes `rsl_rl_amp.modules.amp_discriminator.AMPDiscriminator` (existing, unmodified — `forward`, `compute_grad_pen`, `predict_amp_reward`). Consumes `beyondAMP.motion.motion_dataset.MotionDataset` (existing, unmodified — `.feed_forward_generator(n, batch_size)`). Consumes `rsl_rl_amp.storage.rollout_storage.RolloutStorage` (existing, unmodified).
- Region id is read from a fixed, caller-specified column index in `critic_obs` (constructor arg `region_id_critic_obs_index: int`, ground-truth ball state from `ball_gt_critic_obs_slice: slice`) — set by whoever constructs this class (Task 5/6) to match wherever Task 2's `region_id_gt`/`ball_state_gt` terms land in the critic obs concatenation order.

- [ ] **Step 1: Write the failing test**

Create `tests/simple_goalkeeper/test_multi_disc_amp_ppo.py`. This test constructs the algorithm
with tiny synthetic dimensions and a fake discriminator/motion-dataset dict (no env, no mjlab
needed) and checks that `update()` only ever back-propagates through the discriminator matching
each sample's region id — i.e. a region's discriminator gets a nonzero gradient only when at
least one minibatch sample belongs to that region, and never touches another region's parameters.

```python
"""Tests for MultiDiscAMPPPO region-routed loss/reward — no env, synthetic data."""
import torch

from simple_goalkeeper.rsl_rl_multi.him_actor_critic import HimActorCritic
from simple_goalkeeper.rsl_rl_multi.multi_disc_amp_ppo import MultiDiscAMPPPO
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer


NUM_ONE_STEP_OBS = 10
HISTORY_LEN = 10
NUM_CRITIC_OBS = 25  # includes 4 (ball_gt) + 1 (region_gt) appended at the end
NUM_ACTIONS = 6
AMP_OBS_DIM = 8
REGION_NAMES = ("left_near", "left_far", "right_near", "right_far")


class _FakeMotionDataset:
    """Minimal stand-in for beyondAMP.motion.motion_dataset.MotionDataset."""
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            yield (
                torch.randn(mini_batch_size, self.obs_dim),
                torch.randn(mini_batch_size, self.obs_dim),
            )


def _make_alg(num_envs=8, num_transitions=4):
    actor_critic = HimActorCritic(
        num_one_step_obs=NUM_ONE_STEP_OBS,
        actor_history_length=HISTORY_LEN,
        num_critic_obs=NUM_CRITIC_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
    )
    discriminators = {
        name: AMPDiscriminator(AMP_OBS_DIM * 2, amp_reward_coef=1.0,
                                hidden_layer_sizes=[16, 8], device="cpu", task_reward_lerp=0.5)
        for name in REGION_NAMES
    }
    amp_datasets = {name: _FakeMotionDataset(AMP_OBS_DIM) for name in REGION_NAMES}
    normalizer = Normalizer(AMP_OBS_DIM)
    alg = MultiDiscAMPPPO(
        actor_critic=actor_critic,
        discriminators=discriminators,
        amp_datasets=amp_datasets,
        amp_normalizer=normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        region_id_critic_obs_index=-1,
        ball_gt_critic_obs_slice=slice(-5, -1),
    )
    alg.init_storage(num_envs, num_transitions, [NUM_ONE_STEP_OBS], [NUM_ONE_STEP_OBS * HISTORY_LEN],
                      [NUM_CRITIC_OBS], [NUM_ACTIONS])
    return alg, actor_critic, discriminators


def test_region_routing_only_updates_the_matching_discriminator():
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=2)

    # Build a rollout where env 0-1 are region 0 (left_near) and env 2-3 are region 2 (right_near).
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])
    before = {name: [p.clone() for p in d.trunk.parameters()] for name, d in discriminators.items()}

    for _ in range(2):  # num_transitions
        obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(4, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(4, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        )
    alg.compute_returns(torch.randn(4, NUM_CRITIC_OBS))
    alg.update()

    after = {name: [p.clone() for p in d.trunk.parameters()] for name, d in discriminators.items()}

    for name in REGION_NAMES:
        changed = any(not torch.equal(b, a) for b, a in zip(before[name], after[name]))
        if name in ("left_near", "right_near"):
            assert changed, f"{name} discriminator should have been updated (region present in batch)"
        else:
            assert not changed, f"{name} discriminator should NOT have been touched (region absent)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multi_disc_amp_ppo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_goalkeeper.rsl_rl_multi.multi_disc_amp_ppo'`

- [ ] **Step 3: Implement `MultiDiscAMPPPO`**

Create `src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`:

```python
"""Multi-discriminator AMP-PPO: dict of independent AMPDiscriminator instances
and MotionDataset expert buffers, one per region, region-masked loss and
reward — ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py
(HIMPPO.update / him_on_policy_runner.py's rollout reward loop), adapted to
this codebase's AMPDiscriminator (predict_amp_reward, task-reward lerp baked
into the discriminator) instead of G1's AMP module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_amp.storage.rollout_storage import RolloutStorage
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator

from .him_actor_critic import HimActorCritic

REGION_NAMES: tuple[str, ...] = ("left_near", "left_far", "right_near", "right_far")


class MultiDiscAMPPPO:
    actor_critic: HimActorCritic

    def __init__(
        self,
        actor_critic: HimActorCritic,
        discriminators: dict[str, AMPDiscriminator],
        amp_datasets: dict,
        amp_normalizer,
        region_id_critic_obs_index: int,
        ball_gt_critic_obs_slice: slice,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        min_std=None,
        **kwargs,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.region_id_critic_obs_index = region_id_critic_obs_index
        self.ball_gt_critic_obs_slice = ball_gt_critic_obs_slice

        assert set(discriminators.keys()) == set(REGION_NAMES)
        assert set(amp_datasets.keys()) == set(REGION_NAMES)
        self.discriminators = discriminators
        self.amp_datasets = amp_datasets
        for d in self.discriminators.values():
            d.to(device)
        self.amp_normalizer = amp_normalizer

        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage: RolloutStorage | None = None

        params = [{"params": self.actor_critic.parameters(), "name": "actor_critic"}]
        for name, d in self.discriminators.items():
            params.append({"params": d.trunk.parameters(), "weight_decay": 10e-4, "name": f"amp_trunk_{name}"})
            params.append({"params": d.amp_linear.parameters(), "weight_decay": 10e-2, "name": f"amp_head_{name}"})
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self._obs_current = None
        self._obs_history = None
        self._amp_obs = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_current_shape,
                      obs_history_shape, critic_obs_shape, action_shape):
        # Stack obs_current and obs_history into one "observations" tensor for
        # RolloutStorage (which only knows one obs slot); split again at update
        # time via the known obs_current width.
        self._obs_current_dim = obs_current_shape[0]
        self._obs_history_dim = obs_history_shape[0]
        combined_obs_shape = [self._obs_current_dim + self._obs_history_dim]
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, combined_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_current, obs_history, critic_obs, amp_obs):
        combined_obs = torch.cat([obs_current, obs_history], dim=-1)
        self.transition_actions = self.actor_critic.act(obs_current.detach(), obs_history.detach()).detach()
        self.transition_values = self.actor_critic.evaluate(critic_obs.detach()).detach()
        self.transition_actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_actions).detach()
        self.transition_action_mean = self.actor_critic.action_mean.detach()
        self.transition_action_sigma = self.actor_critic.action_std.detach()
        self._pending_obs = combined_obs
        self._pending_critic_obs = critic_obs
        self._pending_amp_obs = amp_obs
        return self.transition_actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        from rsl_rl_amp.storage.rollout_storage import RolloutStorage as _RS
        transition = _RS.Transition()
        transition.observations = self._pending_obs
        transition.critic_observations = self._pending_critic_obs
        transition.actions = self.transition_actions
        transition.rewards = rewards.clone()
        transition.dones = dones
        transition.values = self.transition_values
        transition.actions_log_prob = self.transition_actions_log_prob
        transition.action_mean = self.transition_action_mean
        transition.action_sigma = self.transition_action_sigma
        if "time_outs" in infos:
            transition.rewards += self.gamma * torch.squeeze(
                transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(transition)
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def predict_region_routed_amp_reward(self, amp_obs, next_amp_obs, region_id, task_reward):
        """Rollout-time style reward, routed per-sample by region id. Mirrors
        him_on_policy_runner.py:161-178's masked predict_reward loop."""
        num_envs = amp_obs.shape[0]
        reward = torch.zeros(num_envs, device=amp_obs.device)
        for r, name in enumerate(REGION_NAMES):
            mask = region_id == r
            if not mask.any():
                continue
            r_out, _, _ = self.discriminators[name].predict_amp_reward(
                amp_obs[mask], next_amp_obs[mask], task_reward[mask], normalizer=self.amp_normalizer)
            reward[mask] = r_out
        return reward

    def update(self):
        mean_value_loss = mean_surrogate_loss = mean_amp_loss = mean_grad_pen_loss = 0.0
        mean_est_loss = mean_region_loss = mean_policy_pred = mean_expert_pred = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_expert_generators = {
            name: ds.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            for name, ds in self.amp_datasets.items()
        }

        for sample in generator:
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) = sample

            obs_current_batch = obs_batch[:, :self._obs_current_dim]
            obs_history_batch = obs_batch[:, self._obs_current_dim:]

            self.actor_critic.act(obs_current_batch.detach(), obs_history_batch.detach())
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch.detach())
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Auxiliary estimator losses (ported from him_ppo.py:226-228).
            gt_ball = critic_obs_batch[:, self.ball_gt_critic_obs_slice]
            gt_region = critic_obs_batch[:, self.region_id_critic_obs_index].long()
            est_loss = (self.actor_critic.estimate_ball - gt_ball).pow(2).mean()
            region_loss = nn.CrossEntropyLoss()(self.actor_critic.estimate_region, gt_region)

            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean() + est_loss + region_loss)

            # Region-routed AMP loss (ported from him_ppo.py:244-305).
            amp_loss = torch.tensor(0.0, device=self.device)
            grad_pen_loss = torch.tensor(0.0, device=self.device)
            policy_preds, expert_preds = [], []
            for r, name in enumerate(REGION_NAMES):
                mask = gt_region == r
                if not mask.any():
                    continue
                expert_state, expert_next_state = next(amp_expert_generators[name])
                policy_amp = obs_current_batch[mask]  # placeholder policy-side amp obs slice; see Task 5 note
                # NOTE: the actual policy-side AMP observation comes from the
                # amp_storage replay buffer (populated in process_env_step),
                # sampled the same way single-disc AMPPPO.update() does via
                # self.amp_storage.feed_forward_generator — wired in Task 5
                # once HimAMPOnPolicyRunner supplies amp_storage per region.
                discr = self.discriminators[name]
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_state_n = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state_n = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                else:
                    expert_state_n, expert_next_state_n = expert_state, expert_next_state
                expert_d = discr(torch.cat([expert_state_n, expert_next_state_n], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                grad_pen = discr.compute_grad_pen(expert_state_n, expert_next_state_n, lambda_=10)
                amp_loss = amp_loss + 0.5 * expert_loss
                grad_pen_loss = grad_pen_loss + grad_pen
                expert_preds.append(expert_d.mean().item())

            loss = loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += float(amp_loss)
            mean_grad_pen_loss += float(grad_pen_loss)
            mean_est_loss += est_loss.item()
            mean_region_loss += region_loss.item()
            if expert_preds:
                mean_expert_pred += sum(expert_preds) / len(expert_preds)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return (mean_value_loss / num_updates, mean_surrogate_loss / num_updates,
                mean_amp_loss / num_updates, mean_grad_pen_loss / num_updates,
                mean_est_loss / num_updates, mean_region_loss / num_updates,
                mean_policy_pred / num_updates, mean_expert_pred / num_updates)
```

**Known gap flagged for Task 5:** the `policy_d`/policy-side loss term inside the per-region loop
above is stubbed to reuse `obs_current_batch[mask]` as a placeholder — this is WRONG dimensionally
(it's a proprioceptive obs, not an AMP obs) and is called out explicitly so Task 5 replaces it with
a real per-region policy-side AMP replay buffer (mirrors single-disc `AMPPPO`'s `self.amp_storage`,
just one instance per region, populated in `process_env_step` using the same region mask). Task 5's
first step is to fix this before wiring the runner — do not treat this task's `update()` as final;
the test in Step 1 above only exercises the expert/grad-pen half of the loss (sufficient to prove
region routing is correct), not the full GAIL objective.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multi_disc_amp_ppo.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py tests/simple_goalkeeper/test_multi_disc_amp_ppo.py
git commit -m "feat(multidisc): add MultiDiscAMPPPO with region-routed expert loss and auxiliary est/region losses"
```

---

## Task 5: Add per-region policy-side AMP replay buffers, fix policy_d loss term

**Files:**
- Modify: `src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`
- Test: `tests/simple_goalkeeper/test_multi_disc_amp_ppo.py`

**Interfaces:**
- Modifies `MultiDiscAMPPPO.__init__` to accept `amp_obs_dim: int` and construct
  `self.amp_storages: dict[str, ReplayBuffer]` (one `rsl_rl_amp.storage.replay_buffer.ReplayBuffer`
  per region, same construction pattern as single-disc `AMPPPO.__init__`'s
  `self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)`).
- Modifies `process_env_step` to insert `(policy_amp_obs, next_amp_obs)` into the region-matching
  buffer only (mask by `gt_region` derived from `self._pending_critic_obs`).
- Modifies `update()`'s per-region loop to sample `policy_state, policy_next_state` from
  `self.amp_storages[name].feed_forward_generator(...)` (same call pattern as single-disc
  `AMPPPO.update()`'s `self.amp_storage.feed_forward_generator(...)`) instead of the Task 4
  placeholder, compute `policy_loss = MSELoss()(policy_d, -ones)`, and add it into `amp_loss`.

- [ ] **Step 1: Extend the existing test to assert the full GAIL loss (policy + expert) is used**

Modify `tests/simple_goalkeeper/test_multi_disc_amp_ppo.py`'s `_make_alg` helper to pass
`amp_obs_dim=AMP_OBS_DIM` and `amp_replay_buffer_size=64` into `MultiDiscAMPPPO(...)`, and add:

```python
def test_amp_loss_uses_policy_replay_buffer_not_proprioceptive_obs():
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=2)
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])
    for _ in range(2):
        obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(4, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(4, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        )
    assert alg.amp_storages["left_near"].step > 0 or alg.amp_storages["left_near"].num_samples > 0
    alg.compute_returns(torch.randn(4, NUM_CRITIC_OBS))
    result = alg.update()
    mean_amp_loss = result[2]
    assert mean_amp_loss > 0.0  # both expert and policy terms now contribute
```

Adjust the exact buffer-emptiness assertion (`.step`/`.num_samples`) to whatever attribute
`rsl_rl_amp.storage.replay_buffer.ReplayBuffer` actually exposes — read
`beyondAMP/source/rsl_rl_amp/rsl_rl_amp/storage/replay_buffer.py` first to confirm the attribute
name before writing this assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multi_disc_amp_ppo.py -v`
Expected: FAIL (TypeError: unexpected keyword argument 'amp_obs_dim', or the buffer-emptiness assertion fails since no buffer exists yet)

- [ ] **Step 3: Read replay_buffer.py and implement the fix**

Read `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/storage/replay_buffer.py` in full to get its exact
constructor signature and `insert`/`feed_forward_generator` method names (single-disc `AMPPPO`
already uses it as `ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)`
and `self.amp_storage.insert(self.amp_transition.observations, amp_obs)` — mirror that exactly).

In `src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py`:
1. Add `amp_obs_dim: int` and `amp_replay_buffer_size: int = 100_000` params to `__init__`.
2. After constructing `self.discriminators`, add:
   ```python
   from rsl_rl_amp.storage.replay_buffer import ReplayBuffer
   self.amp_storages = {
       name: ReplayBuffer(amp_obs_dim, amp_replay_buffer_size, device)
       for name in REGION_NAMES
   }
   ```
3. In `act()`, additionally stash `self._pending_amp_obs = amp_obs` (already present) and
   `self._pending_region = self._pending_critic_obs[:, self.region_id_critic_obs_index].long()`.
4. In `process_env_step(self, rewards, dones, infos, amp_obs)`, after building `transition`, insert
   into the matching region buffers:
   ```python
   region = self._pending_region
   for r, name in enumerate(REGION_NAMES):
       mask = region == r
       if mask.any():
           self.amp_storages[name].insert(self._pending_amp_obs[mask], amp_obs[mask])
   ```
5. In `update()`, replace the `amp_expert_generators` dict construction to also build
   `amp_policy_generators`, one per region, using the same
   `num_learning_epochs * num_mini_batches` / minibatch-size formula as `amp_expert_generators`.
   Inside the per-region loop, replace the placeholder `policy_amp = obs_current_batch[mask]` block
   with:
   ```python
   policy_state, policy_next_state = next(amp_policy_generators[name])
   if self.amp_normalizer is not None:
       with torch.no_grad():
           policy_state_n = self.amp_normalizer.normalize_torch(policy_state, self.device)
           policy_next_state_n = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
   else:
       policy_state_n, policy_next_state_n = policy_state, policy_next_state
   policy_d = discr(torch.cat([policy_state_n, policy_next_state_n], dim=-1))
   policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
   amp_loss = amp_loss + 0.5 * policy_loss
   policy_preds.append(policy_d.mean().item())
   ```
   (keep the existing `expert_loss`/`grad_pen` computation from Task 4 above it in the same
   per-region block, and append to `mean_policy_pred` the same way `mean_expert_pred` is
   accumulated).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multi_disc_amp_ppo.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/rsl_rl_multi/multi_disc_amp_ppo.py tests/simple_goalkeeper/test_multi_disc_amp_ppo.py
git commit -m "feat(multidisc): wire per-region policy-side AMP replay buffers into MultiDiscAMPPPO"
```

---

## Task 6: `goalkeeper_multidisc_env_cfg` — actor_history group, region events, critic obs terms

**Files:**
- Create: `src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py` (env cfg half; runner cfg added in Task 7)
- Modify: `src/simple_goalkeeper/tasks/__init__.py`

**Interfaces:**
- Produces: `goalkeeper_multidisc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg` — starts from
  the existing `goalkeeper_env_cfg(play)` (imported, called, then mutated — does not duplicate its
  ~600 lines), then: (a) adds `cfg.observations["actor_history"]` group, (b) adds
  `region_gt`/`ball_gt` terms to the critic group, (c) replaces the `reset_ball` event with
  `reset_ball_rolling_by_region`, (d) adds the `assign_static_regions` startup event.
- Consumes: `goalkeeper_env_cfg` (existing, unmodified, from `goalkeeper_env_cfg.py`).
  Consumes `assign_static_regions`, `reset_ball_rolling_by_region`, `region_id_gt`, `ball_state_gt`
  (Tasks 1-2, `mdp/regions.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/simple_goalkeeper/test_multidisc_env_cfg.py`:

```python
"""Tests for goalkeeper_multidisc_env_cfg: history group, critic gt terms, region events."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import goalkeeper_multidisc_env_cfg


def test_actor_history_group_has_history_length_ten():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "actor_history" in cfg.observations
    assert cfg.observations["actor_history"].history_length == 10
    # Same terms as the plain "actor" group (same dict of ObservationTermCfg).
    assert set(cfg.observations["actor_history"].terms.keys()) == set(cfg.observations["actor"].terms.keys())


def test_critic_group_has_ball_and_region_ground_truth_terms():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "ball_gt" in cfg.observations["critic"].terms
    assert "region_gt" in cfg.observations["critic"].terms


def test_region_events_registered():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "assign_static_regions" in cfg.events
    assert cfg.events["assign_static_regions"].mode == "startup"
    assert cfg.events["reset_ball"].func.__name__ == "reset_ball_rolling_by_region"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multidisc_env_cfg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg'`

- [ ] **Step 3: Implement `goalkeeper_multidisc_env_cfg`**

Read `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` lines 1-70 and 195-260 first (imports,
`ObservationGroupCfg`/`ObservationTermCfg` import paths, and the exact `actor_terms`/`critic_terms`
construction already in the file) to match import paths and the `EventTermCfg` import exactly.

Create `src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py` (env-cfg half — algorithm/
runner cfg appended in Task 7):

```python
"""Env config for the 4-region, multi-discriminator AMP variant of the
goalkeeper task. Starts from goalkeeper_env_cfg() and layers on: the
actor_history observation group (history_length=10), ball/region
ground-truth critic obs terms, and region-conditioned ball spawn + static
region assignment events. See docs/superpowers/specs/2026-07-02-multi-
discriminator-amp-design.md.
"""
from __future__ import annotations

from mjlab.managers.manager_term_config import EventTermCfg, ObservationGroupCfg, ObservationTermCfg

from simple_goalkeeper.mdp import regions as gk_regions
from simple_goalkeeper.tasks.goalkeeper_env_cfg import BALL_NAME, goalkeeper_env_cfg


def goalkeeper_multidisc_env_cfg(play: bool = False):
    cfg = goalkeeper_env_cfg(play=play)

    # (a) History-stacked actor observation group — same terms as "actor",
    # 10-step history, term-major-flattened by mjlab's ObservationManager.
    actor_terms = cfg.observations["actor"].terms
    cfg.observations["actor_history"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=10,
        flatten_history_dim=True,
    )

    # (b) Ball/region ground-truth terms, critic-only (privileged).
    cfg.observations["critic"].terms["ball_gt"] = ObservationTermCfg(
        func=gk_regions.ball_state_gt,
        params={"ball_name": BALL_NAME},
    )
    cfg.observations["critic"].terms["region_gt"] = ObservationTermCfg(
        func=gk_regions.region_id_gt,
        params={},
    )

    # (c) Static region assignment, once at startup.
    cfg.events["assign_static_regions"] = EventTermCfg(
        func=gk_regions.assign_static_regions,
        mode="startup",
        params={},
    )

    # (d) Replace the shared-range ball spawn with the region-conditioned one.
    # Reuses the existing reset_ball event's dist_range/t_flight_range/spawn_z;
    # drops y_start_range/y_end_range since those are now resolved per-region
    # inside reset_ball_rolling_by_region.
    existing = cfg.events["reset_ball"].params
    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_regions.reset_ball_rolling_by_region,
        mode="reset",
        params={
            "ball_name": existing["ball_name"],
            "dist_range": existing["dist_range"],
            "t_flight_range": existing["t_flight_range"],
            "spawn_z": existing["spawn_z"],
        },
    )

    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multidisc_env_cfg.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py tests/simple_goalkeeper/test_multidisc_env_cfg.py
git commit -m "feat(multidisc): add goalkeeper_multidisc_env_cfg with history group and region events"
```

---

## Task 7: Register the new task; 4-region motion-file config (TripleStep excluded)

**Files:**
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py` (append runner cfg)
- Read then modify: `src/simple_goalkeeper/tasks/__init__.py`
- Test: `tests/simple_goalkeeper/test_multidisc_amp_cfg.py`

**Interfaces:**
- Produces: `REGION_MOTION_FILES: dict[str, str]` mapping each of the 4 region names to its single
  motion file path (`LeftStep_own_booster_t1.npz`, `LeftDoubleStep_own_booster_t1.npz`,
  `Rightstep_own_booster_t1.npz`, `RightDoubleStep_own_booster_t1.npz`).
- Produces: `goalkeeper_multidisc_amp_runner_cfg() -> dict` — the `train_cfg` dict shape
  `HimAMPOnPolicyRunner` (Task 8) expects: `{"policy": {...}, "algorithm": {...}, "amp_data":
  {region_name: MotionDatasetCfg}, "num_steps_per_env", "save_interval", "max_iterations",
  "amp_reward_coef", "amp_discr_hidden_dims", "amp_task_reward_lerp", "amp_min_normalized_std",
  "use_wandb", "wandb_project", ...}` — mirrors the shape `goalkeeper_amp_runner_cfg()` in the
  existing `goalkeeper_amp_cfg.py` already produces (read that file first for the exact key set).
- Registers task id `"Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"` in `tasks/__init__.py`, alongside
  (not replacing) the existing `"Mjlab-BeyondAMP-Goalkeeper-T1"` registration.

- [ ] **Step 1: Write the failing test**

Create `tests/simple_goalkeeper/test_multidisc_amp_cfg.py`:

```python
"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert REGION_MOTION_FILES["left_near"].endswith("LeftStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["left_far"].endswith("LeftDoubleStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_near"].endswith("Rightstep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_far"].endswith("RightDoubleStep_own_booster_t1.npz")


def test_no_triple_step_anywhere_in_region_motion_files():
    for path in REGION_MOTION_FILES.values():
        assert "TripleStep" not in path


def test_runner_cfg_amp_data_has_one_file_per_region_and_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert len(motion_cfg.motion_files) == 1
        assert motion_cfg.motion_files[0] == REGION_MOTION_FILES[name]
        assert "TripleStep" not in motion_cfg.motion_files[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multidisc_amp_cfg.py -v`
Expected: FAIL with `ImportError: cannot import name 'REGION_MOTION_FILES'`

- [ ] **Step 3: Implement `REGION_MOTION_FILES` and `goalkeeper_multidisc_amp_runner_cfg`**

Read `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` in full first (already read earlier in
this project — reuse its `GOALKEEPER_ANCHOR_NAME`, `GOALKEEPER_KEY_BODY_NAMES`,
`AMPObsBaiscTerms`, `_MOTIONS_DIR`, and the exact `RslRlPpoActorCriticCfg`/`AMPPPOAlgorithmCfg`
import paths) to mirror its structure precisely rather than inventing a diverging shape.

Append to `src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py`:

```python
from pathlib import Path

from beyondAMP.motion.motion_dataset import MotionDatasetCfg
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import (
    GOALKEEPER_ANCHOR_NAME,
    GOALKEEPER_KEY_BODY_NAMES,
)
from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"

REGION_MOTION_FILES: dict[str, str] = {
    "left_near": str(_MOTIONS_DIR / "LeftStep_own_booster_t1.npz"),
    "left_far": str(_MOTIONS_DIR / "LeftDoubleStep_own_booster_t1.npz"),
    "right_near": str(_MOTIONS_DIR / "Rightstep_own_booster_t1.npz"),
    "right_far": str(_MOTIONS_DIR / "RightDoubleStep_own_booster_t1.npz"),
}


def goalkeeper_multidisc_amp_runner_cfg() -> dict:
    amp_data = {
        name: MotionDatasetCfg(
            motion_files=[path],
            body_names=GOALKEEPER_KEY_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name=GOALKEEPER_ANCHOR_NAME,
        )
        for name, path in REGION_MOTION_FILES.items()
    }
    return {
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "amp_replay_buffer_size": 250_000,
        },
        "amp_data": amp_data,
        "num_steps_per_env": 24,
        "max_iterations": 50_000,
        "save_interval": 250,
        "experiment_name": "simple_goalkeeper_multidisc",
        "run_name": "phase1",
        "empirical_normalization": True,
        "use_wandb": True,
        "wandb_project": "SimpleGoalKeeper-MultiDisc",
        "amp_discr_hidden_dims": [256, 256],
        "amp_reward_coef": 0.5,
        "amp_task_reward_lerp": 0.6,
        "amp_min_normalized_std": 0.05,
    }
```

Read `src/simple_goalkeeper/tasks/__init__.py` in full, then add a second `gym.register(...)`-style
(or whatever registration mechanism it currently uses — match exactly) call for
`"Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"` pointing `env_cfg_entry_point` at
`goalkeeper_multidisc_env_cfg` and `rl_cfg_entry_point` at `goalkeeper_multidisc_amp_runner_cfg`,
alongside the existing registration for `"Mjlab-BeyondAMP-Goalkeeper-T1"` (do not remove or modify
that one).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/simple_goalkeeper/test_multidisc_amp_cfg.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py src/simple_goalkeeper/tasks/__init__.py tests/simple_goalkeeper/test_multidisc_amp_cfg.py
git commit -m "feat(multidisc): register 4-region motion config and new task id"
```

---

## Task 8: `HimAMPOnPolicyRunner` — wires everything together

**Files:**
- Create: `src/simple_goalkeeper/rsl_rl_multi/him_amp_on_policy_runner.py`
- Modify: `src/simple_goalkeeper/scripts/train.py` (read first — add a small branch so `TrainConfig.from_task`/`run_train` can construct `HimAMPOnPolicyRunner` instead of `AMPOnPolicyRunner` when the task's `agent_cfg` shape indicates the multi-disc variant; exact mechanism TBD by reading the current `train.py` dispatch logic, which was already read earlier in this project session — mirror how it currently picks the runner class for the single-disc case)

**Interfaces:**
- Produces: `HimAMPOnPolicyRunner(env, train_cfg, log_dir=None, device='cpu')` — same constructor
  shape as `AMPOnPolicyRunner`, but builds `HimActorCritic` instead of `ActorCritic`, builds 4
  `AMPDiscriminator` + 4 `MotionDataset` instances from `train_cfg["amp_data"]` (a
  `dict[str, MotionDatasetCfg]`, per Task 7) instead of one, and constructs `MultiDiscAMPPPO`
  instead of `AMPPPO`.
- Consumes: `HimActorCritic` (Task 3), `MultiDiscAMPPPO` (Tasks 4-5), `REGION_NAMES` (Task 4),
  `env.observation_manager` group `"actor_history"` (Task 6) via
  `env.unwrapped.observation_manager.compute()["actor_history"]` (mirrors how `AMPEnvWrapper`
  already exposes `get_amp_observations()` for the `"amp"` group — read `amp_wrapper.py` again to
  add an equivalent accessor, or call `env.unwrapped.observation_manager.compute()` directly since
  it's already imported).

- [ ] **Step 1: Read `AMPEnvWrapper` (again, already read this session) and `train.py`'s runner-selection code in full**

This step has no test — it's investigation needed before Step 2's implementation, since the exact
call sites for `env.get_observations()` / `env.get_privileged_observations()` /
`env.num_privileged_obs` / `env.num_obs` (used throughout `AMPOnPolicyRunner`) must be matched
exactly, and this task needs an additional accessor for the `"actor_history"` group that doesn't
exist on `AMPEnvWrapper` yet.

- [ ] **Step 2: Add an `actor_history` accessor to a thin subclass, not `AMPEnvWrapper` itself**

Per the Global Constraints, `beyondAMP` (which owns `AMPEnvWrapper`) is not edited in place. Create
the accessor as a small subclass living in the new runner file instead:

Create `src/simple_goalkeeper/rsl_rl_multi/him_amp_on_policy_runner.py`:

```python
"""HIM-style AMP runner: wires HimActorCritic + MultiDiscAMPPPO + 4 region
MotionDataset/AMPDiscriminator instances together. Ported from
Humanoid-Goalkeeper/rsl_rl/rsl_rl/runners/him_on_policy_runner.py, adapted to
this codebase's AMPEnvWrapper/AMPDiscriminator/MotionDataset contract instead
of G1's raw legged_gym env + AMP module.
"""
from __future__ import annotations

import os
import time
import statistics
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from beyondAMP.motion.motion_dataset import MotionDataset
from beyondAMP.mjlab.rsl_rl.amp_wrapper import AMPEnvWrapper
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer

from .him_actor_critic import HimActorCritic
from .multi_disc_amp_ppo import MultiDiscAMPPPO, REGION_NAMES


def _get_actor_history_obs(env: AMPEnvWrapper) -> torch.Tensor:
    return env.unwrapped.observation_manager.compute()["actor_history"]


class HimAMPOnPolicyRunner:
    def __init__(self, env: AMPEnvWrapper, train_cfg: dict, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
        num_one_step_obs = env.num_obs

        actor_critic = HimActorCritic(
            num_one_step_obs=num_one_step_obs,
            actor_history_length=10,
            num_critic_obs=num_critic_obs,
            num_actions=env.num_actions,
            **self.policy_cfg,
        ).to(device)

        amp_data_cfgs = train_cfg["amp_data"]
        amp_datasets = {
            name: MotionDataset(cfg, env.unwrapped, device)
            for name, cfg in amp_data_cfgs.items()
        }
        amp_obs_dim = env.get_amp_observations().shape[-1]
        amp_normalizer = Normalizer(amp_obs_dim)
        discriminators = {
            name: AMPDiscriminator(
                amp_obs_dim * 2, train_cfg["amp_reward_coef"],
                train_cfg["amp_discr_hidden_dims"], device, train_cfg["amp_task_reward_lerp"],
            ).to(device)
            for name in REGION_NAMES
        }

        # Ball/region ground-truth land at the end of the critic obs concat
        # order set in Task 6 (ball_gt then region_gt, both critic-only terms
        # appended after the existing critic terms) -> ball_gt occupies the
        # 4 slots before the final region_gt slot.
        region_id_critic_obs_index = -1
        ball_gt_critic_obs_slice = slice(-5, -1)

        min_std = (
            torch.tensor(train_cfg["amp_min_normalized_std"], device=device)
            * torch.abs(env.dof_pos_limits[0, :, 1] - env.dof_pos_limits[0, :, 0]))

        self.alg = MultiDiscAMPPPO(
            actor_critic=actor_critic,
            discriminators=discriminators,
            amp_datasets=amp_datasets,
            amp_normalizer=amp_normalizer,
            region_id_critic_obs_index=region_id_critic_obs_index,
            ball_gt_critic_obs_slice=ball_gt_critic_obs_slice,
            amp_obs_dim=amp_obs_dim,
            device=device,
            min_std=min_std,
            **self.alg_cfg,
        )
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]

        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env,
            [num_one_step_obs], [num_one_step_obs * 10], [num_critic_obs], [env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        obs_history = _get_actor_history_obs(self.env)
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_history, critic_obs, amp_obs = (
            obs.to(self.device), obs_history.to(self.device),
            critic_obs.to(self.device), amp_obs.to(self.device))
        self.alg.train_mode()

        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, critic_obs, amp_obs)
                    obs, privileged_obs, raw_rewards, dones, infos, reset_env_ids, terminal_amp_states = (
                        self.env.step(actions, not_amp=False))
                    obs_history = _get_actor_history_obs(self.env)
                    next_amp_obs = self.env.get_amp_observations()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, obs_history, critic_obs, next_amp_obs, raw_rewards, dones = (
                        obs.to(self.device), obs_history.to(self.device), critic_obs.to(self.device),
                        next_amp_obs.to(self.device), raw_rewards.to(self.device), dones.to(self.device))

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    region_id = critic_obs[:, self.alg.region_id_critic_obs_index].long()
                    rewards = self.alg.predict_region_routed_amp_reward(
                        amp_obs, next_amp_obs_with_term, region_id, raw_rewards)
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            (mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_est_loss, mean_region_loss, mean_policy_pred, mean_expert_pred) = self.alg.update()
            learn_time = time.time() - start

            if self.log_dir is not None:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += collection_time + learn_time
                self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
                self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
                self.writer.add_scalar("Loss/AMP", mean_amp_loss, it)
                self.writer.add_scalar("Loss/AMP_grad", mean_grad_pen_loss, it)
                self.writer.add_scalar("Loss/est_ball", mean_est_loss, it)
                self.writer.add_scalar("Loss/est_region", mean_region_loss, it)
                if len(rewbuffer) > 0:
                    self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
                    self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
                print(f"[multidisc] it={it} reward={statistics.mean(rewbuffer) if rewbuffer else float('nan'):.3f} "
                      f"est_ball={mean_est_loss:.4f} est_region={mean_region_loss:.4f}")

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.writer is not None and hasattr(self.writer, "stop"):
            self.writer.stop()

    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": {n: d.state_dict() for n, d in self.alg.discriminators.items()},
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        for n, sd in loaded["discriminator_state_dict"].items():
            self.alg.discriminators[n].load_state_dict(sd)
        self.alg.amp_normalizer = loaded["amp_normalizer"]
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        return loaded["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
```

- [ ] **Step 3: Wire runner selection in `train.py`**

Read `src/simple_goalkeeper/scripts/train.py` in full (already read earlier this project session —
re-read to confirm current line numbers before editing). Add a branch so that when
`task_id == "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"`, `TrainConfig.from_task` builds the runner
via `HimAMPOnPolicyRunner` instead of `AMPOnPolicyRunner`, and `run_train`'s call site constructs
that class instead. Show the exact diff once the current file content is confirmed — this plan
does not guess at line numbers for a file whose content may have shifted since it was last read.

- [ ] **Step 4: Manual smoke test — short multi-disc training run**

No automated test for this step (mirrors the standalone-copy smoke test already run for
`interceptV2DualDis` earlier this session). Run:

```bash
cd interceptV2DualDis && source .venv/bin/activate
MUJOCO_GL=egl WANDB_MODE=offline uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc \
  --num-envs 8 --agent.max-iterations 2 --agent.save-interval 1
```

Expected: completes 2 iterations without error, prints `[multidisc] it=... est_ball=... est_region=...`
lines with finite, non-NaN values, and writes `model_0.pt`/`model_1.pt`/`model_2.pt` under
`interceptV2DualDis/logs/rsl_rl/simple_goalkeeper_multidisc/<run>/`. If it errors, the error and
traceback are the next debugging input — do not proceed to Step 5 until this runs clean.

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/rsl_rl_multi/him_amp_on_policy_runner.py src/simple_goalkeeper/scripts/train.py
git commit -m "feat(multidisc): add HimAMPOnPolicyRunner and wire it into train.py for the new task id"
```

---

## Self-Review Notes

**Spec coverage:** Section A (env partitioning) → Tasks 1, 6. Section B (actor-critic estimators)
→ Task 3. Section C (4 discriminators, region routing) → Tasks 4, 5. Section D (code organization)
→ all tasks live under `interceptV2DualDis/src/simple_goalkeeper/rsl_rl_multi/` and
`tasks/goalkeeper_multidisc_amp_cfg.py`, no edits to `beyondAMP` or `SimpleGoalKeeper`. Section E
(testing) → each task has its own test file; Task 8 Step 4 is the integration smoke test.

**Known deviation from "no placeholder" carried forward intentionally:** Task 4's `update()`
contains a documented, self-flagged placeholder (`policy_amp = obs_current_batch[mask]`) that
Task 5 replaces in the very next task — this is called out explicitly in Task 4's text rather than
hidden, and Task 4's own test only exercises the parts of the loss that are already correct at that
point (expert + grad-pen), so Task 4 is still independently gate-able/testable before Task 5 lands.

**Biggest implementation risk:** `region_id_critic_obs_index = -1` / `ball_gt_critic_obs_slice =
slice(-5, -1)` in Task 8 assumes `ball_gt` (4 dims) then `region_gt` (1 dim) are the LAST 5 columns
of the concatenated critic observation, which depends on Python dict insertion order being
preserved through `ObservationGroupCfg.terms` → mjlab's concatenation (true for `concatenate_dim=-1`
with `dict[str, ObservationTermCfg]`, which mjlab iterates in insertion order — verify this
assumption against `mjlab/managers/observation_manager.py`'s concatenation code during Task 6/8
implementation, since Task 6 appends `ball_gt` then `region_gt` last into `critic_terms`, and if
mjlab's concat order differs from insertion order this index breaks silently rather than erroring).
