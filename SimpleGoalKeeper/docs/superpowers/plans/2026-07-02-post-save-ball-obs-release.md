# Post-Save Ball-Observation Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zero the actor's ball observation once the ball has been saved or crossed the goal line, so the policy learns to disengage and return to the default pose post-save — while keeping the ball fully visible during the entire approach.

**Architecture:** Add a `hide_when_behind` flag to the actor's ball observation term (`ball_pos_xy_b` in `mdp/observations.py`) that multiplies the output by `~_ball_is_behind(...)` — the exact same latched flag that already activates the five `post*` recovery rewards and deactivates `footreach`. This is the "release" half of G1's post-save mechanism (G1 hides the ball via its `flying` mask once the ball stops approaching, `legged_robot.py:401,425-427`); the "pull" half (post-save rewards) is already ported. The full G1 visibility system (`always_visible=False`, commit 8803b6e) is NOT reused — it was reverted (3512989) after destabilizing run 2026-07-02_17-53-27 (25× action-rate penalty, 4× ball_exit terminations) because its warmup blackout and random vanish blind the actor during the approach.

**Tech Stack:** mjlab (MuJoCo-Warp), PyTorch, pytest, uv.

## Global Constraints

- All work in `SimpleGoalKeeper/` on `master` (repo root `/home/ibouwmeest/BEPImitationLearning`).
- Critic ball observations (`ball_pos_b` 3D, `ball_vel_b`) stay `always_visible=True` and ungated — matches G1's ungated privileged obs (`legged_robot.py:432`).
- The actor's term keeps `always_visible=True` (the reverted full-visibility system stays off). `hide_when_behind` is a new, orthogonal parameter.
- Training/play parity: the actor obs group is defined once in `goalkeeper_env_cfg()` and shared by train and play — a single edit satisfies the parity rule.
- Divergence from G1 must be documented in `SimpleGoalKeeper/CLAUDE.md` (G1 uses the `flying` mask as the release; SGK uses the `_ball_is_behind` flag — justification: run evidence above + requirement of full pre-save visibility).
- Run tests with: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/ -v`.
- Commits follow repo style: `feat(sgk): ...` / `docs(sgk): ...`, ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- A fresh training run is required after this lands; no existing checkpoint was trained under this gate.

---

## File Map

- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py` — add `hide_when_behind` param to `ball_pos_xy_b`.
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py:225-229` — actor term gains `"hide_when_behind": True`.
- Modify: `SimpleGoalKeeper/CLAUDE.md` — new divergence row.
- Create: `SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py` — unit tests for the gate.

---

## Task 1: `hide_when_behind` parameter on `ball_pos_xy_b`

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py` (function `ball_pos_xy_b`, currently lines 91-109)
- Test: `SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py` (create)

**Interfaces:**
- Consumes: `simple_goalkeeper.mdp.rewards._ball_is_behind(env, ball_name) -> torch.Tensor[bool] (N,)` — existing, unchanged. Fires when `ball_x_local < 0` OR `env._softstop_flag` OR `env._sb_flag`. `rewards.py` imports nothing from `observations.py`, so `from .rewards import _ball_is_behind` inside `observations.py` creates no cycle. Import it lazily inside the function body to keep module import order irrelevant.
- Produces: `ball_pos_xy_b(env, ball_name="ball", always_visible=False, hide_when_behind=False) -> torch.Tensor (N, 2)` — new keyword-only-in-practice param, default `False` so every existing call site is unaffected.

- [ ] **Step 1: Write the failing tests**

Create `SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py`:

```python
"""Post-save release gate: actor's ball obs zeroes once _ball_is_behind fires.

Uses fake env objects (same approach as test_live_rsi.py) — no mjlab env build.
Identity root quaternion (w,x,y,z)=(1,0,0,0) so body frame == world frame and
expected values are just (ball - robot) XY offsets.
"""
import torch

from simple_goalkeeper.mdp.observations import ball_pos_xy_b


class _FakeEntityData:
    def __init__(self, pos, quat):
        self.root_link_pos_w = pos
        self.root_link_quat_w = quat


class _FakeEntity:
    def __init__(self, pos, quat=None):
        n = pos.shape[0]
        if quat is None:
            quat = torch.zeros(n, 4)
            quat[:, 0] = 1.0  # identity (w, x, y, z)
        self.data = _FakeEntityData(pos, quat)


class _FakeScene(dict):
    def __init__(self, entities, env_origins):
        super().__init__(entities)
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, robot_pos, ball_pos, env_origins):
        n = robot_pos.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _FakeScene(
            {"robot": _FakeEntity(robot_pos), "ball": _FakeEntity(ball_pos)},
            env_origins,
        )


def _make_env(ball_x_local=1.5):
    """Robot at origin of each env; ball ball_x_local ahead of the goal line."""
    n = 4
    env_origins = torch.zeros(n, 3)
    robot_pos = env_origins.clone()
    ball_pos = env_origins.clone()
    ball_pos[:, 0] = ball_x_local
    ball_pos[:, 1] = 0.3
    return _FakeEnv(robot_pos, ball_pos, env_origins)


def test_visible_before_save_even_with_gate_enabled():
    env = _make_env(ball_x_local=1.5)  # in front, no flags set
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_zeroed_when_sb_flag_set():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.ones(4, dtype=torch.bool)
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_zeroed_when_softstop_flag_set():
    env = _make_env(ball_x_local=1.5)
    env._softstop_flag = torch.ones(4, dtype=torch.bool)
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_zeroed_when_ball_crossed_goal_line():
    env = _make_env(ball_x_local=-0.2)  # behind goal line, no flags
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_mixed_batch_only_saved_envs_zeroed():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.tensor([True, False, True, False])
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[2], torch.zeros(2))
    assert torch.allclose(out[1], torch.tensor([1.5, 0.3]), atol=1e-5)
    assert torch.allclose(out[3], torch.tensor([1.5, 0.3]), atol=1e-5)


def test_default_hide_when_behind_false_is_backward_compatible():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.ones(4, dtype=torch.bool)  # saved — but gate off
    out = ball_pos_xy_b(env, "ball", always_visible=True)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_post_save_ball_release.py -v`
Expected: FAIL — `TypeError: ball_pos_xy_b() got an unexpected keyword argument 'hide_when_behind'` (5 of 6 tests; the backward-compat test passes already).

- [ ] **Step 3: Implement the gate**

In `SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py`, replace the current `ball_pos_xy_b` (lines 91-109) with:

```python
def ball_pos_xy_b(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    always_visible: bool = False,
    hide_when_behind: bool = False,
) -> torch.Tensor:
    """Ball XY position in robot body frame (no Z). Shape (N, 2).

    Matches BoosterT1mjlab kick task observation space for deployment compatibility.

    hide_when_behind: post-save release gate. Zeroes the observation once
    _ball_is_behind fires (ball crossed the goal line OR a save was registered
    via _softstop_flag/_sb_flag — the same latched flag that activates the
    post* recovery rewards). This is the release half of G1's post-save
    mechanism (G1 hides the ball via its flying mask, legged_robot.py:401)
    without G1's warmup blackout / random vanish, which destabilized training
    when ported wholesale (run 2026-07-02_17-53-27, reverted 3512989).
    One-way within an episode: the flags are latched until reset.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_pos_w - robot.data.root_link_pos_w,
    )
    out = ball_pos_b_val[:, :2]
    if not always_visible:
        visible = _compute_ball_visibility(env, ball_name)
        out = out * visible.float().unsqueeze(-1)
    if hide_when_behind:
        from .rewards import _ball_is_behind
        out = out * (~_ball_is_behind(env, ball_name)).float().unsqueeze(-1)
    return out
```

Note: the import of `_ball_is_behind` is inside the function to avoid any module-level
import-order coupling between `observations.py` and `rewards.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_post_save_ball_release.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite to check nothing else broke**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/ -v`
Expected: all passed (27 pre-existing + 6 new)

- [ ] **Step 6: Commit**

```bash
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py
git commit -m "feat(sgk): add hide_when_behind post-save release gate to ball_pos_xy_b

Zeroes the actor's ball observation once _ball_is_behind fires (save
registered or ball crossed the goal line) — the release half of G1's
post-save disengage mechanism, without the warmup blackout / random
vanish that destabilized run 2026-07-02_17-53-27. Default False; not
yet wired into any obs term.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Wire the gate into the actor obs term + document the divergence

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py:225-229` (actor `ball_pos_b` term)
- Modify: `SimpleGoalKeeper/CLAUDE.md` (Divergences table + Ball visibility section)
- Test: `SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py` (append cfg-level tests)

**Interfaces:**
- Consumes: `ball_pos_xy_b(..., hide_when_behind: bool)` from Task 1.
- Produces: actor obs term params `{"ball_name": BALL_NAME, "always_visible": True, "hide_when_behind": True}`; critic terms unchanged.

- [ ] **Step 1: Write the failing cfg tests**

Append to `SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py`:

```python
def test_actor_ball_term_has_release_gate_in_train_and_play():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    for play in (False, True):
        cfg = goalkeeper_env_cfg(play=play)
        actor_term = cfg.observations["actor"].terms["ball_pos_b"]
        assert actor_term.params["always_visible"] is True
        assert actor_term.params["hide_when_behind"] is True


def test_critic_ball_terms_stay_ungated():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    cfg = goalkeeper_env_cfg(play=False)
    critic = cfg.observations["critic"].terms
    assert critic["ball_pos_b"].params["always_visible"] is True
    assert "hide_when_behind" not in critic["ball_pos_b"].params
    assert critic["ball_vel_b"].params["always_visible"] is True
    assert "hide_when_behind" not in critic["ball_vel_b"].params
```

Note for the implementer: if `cfg.observations` is not dict-subscriptable or the group/term
attribute names differ, inspect the object returned by `goalkeeper_env_cfg()` (e.g.
`cfg.observations.__dict__`) and adjust the two accessor lines — the assertions
themselves (params contain the right flags) are the contract, not the access syntax.

- [ ] **Step 2: Run tests to verify the actor-term test fails**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_post_save_ball_release.py -v`
Expected: `test_actor_ball_term_has_release_gate_in_train_and_play` FAILS (`KeyError: 'hide_when_behind'`); `test_critic_ball_terms_stay_ungated` PASSES.

- [ ] **Step 3: Wire the param into the actor term**

In `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, the actor term currently reads (lines 225-229):

```python
        "ball_pos_b": ObservationTermCfg(
            func=gk_mdp.ball_pos_xy_b,
            params={"ball_name": BALL_NAME, "always_visible": True},
```

Change the params line to:

```python
            params={"ball_name": BALL_NAME, "always_visible": True, "hide_when_behind": True},
```

And update the comment block above the actor obs group (around line 204) to say:

```python
    # Ball is fully visible during the approach (always_visible=True — the full
    # G1 visibility port was reverted, see CLAUDE.md). hide_when_behind adds the
    # post-save release: the obs zeroes once _ball_is_behind fires, so the policy
    # learns to disengage and recover to the default pose after a save.
```

Do NOT touch the critic terms (lines 239-247).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_post_save_ball_release.py -v`
Expected: 8 passed

- [ ] **Step 5: Add the CLAUDE.md divergence row**

In `SimpleGoalKeeper/CLAUDE.md`, add this row to the end of the Divergences table (before the `Effector type` row):

```markdown
| Actor ball-obs post-save release | `flying` mask hides ball once not approaching / catch window expired (`legged_robot.py:401,425-427`), trained under from iter 1; plus warmup blackout (3–10 steps, actions frozen to default via line 643) and noise-mode random vanish | **`hide_when_behind` gate: actor's `ball_pos_xy_b` zeroed once `_ball_is_behind` fires (save registered or ball crossed line); fully visible during the entire approach. Critic ungated.** | **FEAT 2026-07-02:** Solves the observed post-save ball-tracking (robot kept following the ball after saves in `sgk_play`). The full G1 gate port (8803b6e) was reverted (3512989) after destabilizing run 2026-07-02_17-53-27 (action_rate_l2 25× worse, ball_exit 4×, saves 65%→25%): the port made the warmup blackout ~43 steps vs G1's 3–10, dropped G1's action freeze during warmup, applied random vanish unconditionally vs noise-mode-only, and never resampled the vanish threshold. This gate keeps only the release semantics — same one-way `_ball_is_behind` flag that activates the `post*` recovery rewards — giving G1's release+pull structure without pre-save blindness. Requires a fresh training run. `observations.py:ball_pos_xy_b`, `goalkeeper_env_cfg.py` actor term. |
```

Also replace the `**Ball visibility:**` paragraph in the Reward Design section with:

```markdown
**Ball visibility:** actor's `ball_pos_b` uses `always_visible=True` (full pre-save
visibility) plus `hide_when_behind=True`: the observation zeroes once `_ball_is_behind`
fires and stays zero until reset — the post-save release. The full G1 visibility system
(warmup + flying cone + random vanish, `_compute_ball_visibility`) exists in
`observations.py` but is not active in training or play (see the post-save release
divergence row). Critic's `ball_pos_b`/`ball_vel_b` remain `always_visible=True`.
```

- [ ] **Step 6: Run the full suite one final time**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/ -v`
Expected: all passed

- [ ] **Step 7: Commit and push**

```bash
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py SimpleGoalKeeper/CLAUDE.md SimpleGoalKeeper/tests/simple_goalkeeper/test_post_save_ball_release.py
git commit -m "feat(sgk): enable post-save ball-obs release on the actor term

Actor's ball_pos_b now zeroes once _ball_is_behind fires, so the policy
learns to disengage after a save (release) while the existing post*
rewards pull it back to the default pose. Full visibility during the
approach is preserved. Critic stays ungated. Divergence documented.

Requires a fresh training run.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

## Post-implementation notes (not tasks)

- **Fresh run required.** The next run's deltas vs `2026-07-02_01-14-33` will be: (1) 4-motion AMP dataset (881e063), (2) 2mm foot-geom margin (714300f), (3) this release gate. If curves misbehave, these are the three suspects.
- **What to look for in `sgk_play`:** after a save, the robot should stop tracking the ball and settle toward the default pose (the `post*` rewards' target). Compare against model_9750's post-save chasing.
- **Not in scope (YAGNI):** gating `ball_vel_b` (actor doesn't observe it), latching beyond what `_softstop_flag`/`_sb_flag` already provide, any use of `_compute_ball_visibility`, sim2real vanish robustness.
