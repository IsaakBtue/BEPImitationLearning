# Blue-Ball Hard Gate + RSI Rebalance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the time-based fallback from the two-stage footreach schedule so the blue→green target switch depends only on a genuine foot landing; curriculum-ramp the `blue_ball_landed` bonus; rebalance RSI fractions to cut the "free landing" credit that was masking the real (near-zero) genuine landing rate; and add diagnostics that separate genuine landings from RSI-assisted ones.

**Architecture:** `mdp/rewards.py::_get_reach_target_y` loses its `elapsed >= t_flight/2` fallback (`phase1_active = wide & ~env._blue_landed`, no time term) and gains a new latch, `env._blue_airborne_at_reset`, marking episodes where the assigned foot was already airborne within 2 steps of reset (a proxy for RSI-seeded "free" landings). `scripts/play.py`'s sphere visualization is fixed to read the real `env._blue_landed` latch instead of re-deriving the now-removed time-based logic. `goalkeeper_env_cfg.py` gains a `blue_ball_landed_curriculum` entry (reusing the existing `reward_curriculum_ep_len`), rebalanced RSI fraction params, and a new `cfg.metrics` block. `mdp/events.py`'s `MotionResetManager.reset()` and `reset_from_motion_data()` default fractions change from tier=0.5/live=0.3 to tier=0.1/live=0.7. A new `mdp/metrics.py` exposes `blue_landed_genuine`/`blue_landed_rsi_assisted`, reading the new latch via `MetricsTermCfg` (no weight/dt scaling, unlike rewards).

**Tech Stack:** Python 3.11, PyTorch, mjlab (MuJoCo-Warp), pytest.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md`.
- No change to the 0.3 m landing radius or the airborne-then-contact detection logic itself.
- No change to `footreach`/`foot_proximity`'s `ball_close < 0.5 m` live-ball override — remains the true backstop against stalling.
- No change to the 20% standing-pose branch of the RSI three-way split; only tier (0.5→0.1) and live (0.3→0.7) move.
- `blue_ball_landed_curriculum` base weight stays 10.0 (unchanged seed value); ramps 10→25 across `_curriculumupdate` 0→3, identical formula/ceiling to `footreach_curriculum`.
- New `_blue_airborne_at_reset` latch: true if the assigned foot's first airborne transition happens while `episode_length_buf <= 2` (one step wider than this function's existing `just_reset = episode_length_buf <= 1` convention, to allow one physics settle step).
- `Episode_Metrics/*` values (this task's new diagnostics) are read directly as per-episode rates — no weight/dt conversion, unlike `Episode_Reward/*` values (see `SimpleGoalKeeper/CLAUDE.md`'s documented formula, which was misapplied once already in this investigation).
- Per `SimpleGoalKeeper/CLAUDE.md`: every divergence from G1 upstream must be documented in the "Divergences from G1 Upstream" table, and every fix/feature must get a dated entry in `SimpleGoalKeeper/docs/BugFixes.md`.
- This is a code-only change — it requires a fresh training run to evaluate; no task here launches or monitors one.

---

### Task 1: Remove the time-based fallback in `_get_reach_target_y`

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py:143-243` (`_get_reach_target_y`)
- Modify: `SimpleGoalKeeper/tests/simple_goalkeeper/test_reach_target_two_stage.py`
- Modify: `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py`

**Interfaces:**
- Consumes: `_get_ball_crossing_y(env, ball_name)`, `_get_correct_foot_idx(env, ball_name)`, `_DEFAULT_FEET_CFG`, `ContactSensor` (all already in `rewards.py`, unchanged signatures).
- Produces: `_get_reach_target_y(env, ball_name, wide_threshold=0.6, asset_cfg=_DEFAULT_FEET_CFG, landing_radius=0.3) -> torch.Tensor` — **signature unchanged**, but no longer reads `env._ball_t_flight`/`env.step_dt` at all. `env._blue_was_airborne`/`env._blue_landed` latches unchanged in meaning. Consumed by `footreach`/`foot_proximity` (unchanged call sites) and `blue_ball_landed` (unchanged).

- [ ] **Step 1: Update the existing tests that assert the old time-based fallback**

In `SimpleGoalKeeper/tests/simple_goalkeeper/test_reach_target_two_stage.py`, replace the test at lines 81-86:

```python
def test_wide_crossing_second_half_targets_full_point():
    # elapsed = 30 * 0.02 = 0.6s >= half (0.5s) -> second half -> full target.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=30)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)
```

with:

```python
def test_wide_crossing_stays_at_midpoint_past_former_half_flight_boundary():
    # Under the 2026-07-04 hard gate there is no more time-based fallback:
    # elapsed = 30*0.02 = 0.6s (past the OLD t_flight/2 = 0.5s boundary) must
    # STILL target the midpoint, since this fake env has no "robot"/
    # "feet_contact" in scene and can therefore never satisfy the landing gate.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=30)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)
```

Replace the test at lines 118-126:

```python
def test_flight_time_half_boundary_is_second_half_not_first():
    # elapsed == t_flight/2 exactly -> first_half uses strict '<', so this must
    # already be the full-target phase.
    # t_flight=1.0, step_dt=0.02 -> half=0.5s -> step 25 gives exactly 0.5s.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=25, step_dt=0.02)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)
```

with:

```python
def test_elapsed_time_no_longer_affects_phase_selection():
    # Under the hard gate, elapsed time plays no role at all -- exactly
    # t_flight/2 (the OLD boundary, step 25 at step_dt=0.02) must still be
    # midpoint, not full target.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=25, step_dt=0.02)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)
```

Delete the test at lines 128-135 entirely (its premise no longer applies — `_get_reach_target_y` no longer reads `_ball_t_flight` at all, so there is nothing to "fall back" from):

```python
def test_missing_t_flight_falls_back_to_full_target():
    # Safety fallback: if _ball_t_flight was never populated (shouldn't happen
    # post-reset in practice, but must not crash), always return the full target.
    env = _make_env(rel_cross_y=0.9, t_flight=1.0, episode_step=2)
    del env._ball_t_flight
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)
```

Replace the test at lines 150-172:

```python
def test_mixed_batch_wide_narrow_and_both_flight_halves_independently():
    # Four envs in one call: narrow / wide-first-half / wide-second-half / wide-negative-first-half.
    n = 4
    env_origins = torch.zeros(n, 3)
    env_origins[:, 1] = 5.0
    rel_cross_y = torch.tensor([0.3, 0.9, 0.9, -0.9])
    t_flight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    env = _FakeEnv(env_origins, rel_cross_y, t_flight, episode_step=2)
    env.episode_length_buf = torch.tensor([10, 10, 30, 10], dtype=torch.long)

    start_y = env.scene.env_origins[:, 1]
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")

    # env 0: narrow -> full target.
    assert torch.allclose(target[0], full_y[0])
    # env 1: wide, elapsed=0.2s < 0.5s -> midpoint.
    assert torch.allclose(target[1], start_y[1] + (full_y[1] - start_y[1]) / 2.0, atol=1e-5)
    # env 2: wide, elapsed=0.6s >= 0.5s -> full target.
    assert torch.allclose(target[2], full_y[2])
    # env 3: wide negative side, elapsed=0.2s < 0.5s -> midpoint, correct direction.
    assert torch.allclose(target[3], start_y[3] + (full_y[3] - start_y[3]) / 2.0, atol=1e-5)
    assert target[3] < start_y[3]
```

with:

```python
def test_mixed_batch_wide_narrow_independent_of_elapsed_time():
    # Four envs in one call: narrow / wide (early step) / wide (late step, past
    # the OLD t_flight/2 boundary) / wide-negative (early step). Under the hard
    # gate elapsed time is irrelevant -- both wide envs must behave identically
    # regardless of episode_step, since neither can ever land (no "robot"/
    # "feet_contact" in this fake env's scene).
    n = 4
    env_origins = torch.zeros(n, 3)
    env_origins[:, 1] = 5.0
    rel_cross_y = torch.tensor([0.3, 0.9, 0.9, -0.9])
    t_flight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    env = _FakeEnv(env_origins, rel_cross_y, t_flight, episode_step=2)
    env.episode_length_buf = torch.tensor([10, 10, 30, 10], dtype=torch.long)

    start_y = env.scene.env_origins[:, 1]
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")

    # env 0: narrow -> full target.
    assert torch.allclose(target[0], full_y[0])
    # env 1: wide, early step -> midpoint.
    assert torch.allclose(target[1], start_y[1] + (full_y[1] - start_y[1]) / 2.0, atol=1e-5)
    # env 2: wide, late step (past the OLD t_flight/2 boundary) -> STILL
    # midpoint, since elapsed time no longer matters and this env can never land.
    assert torch.allclose(target[2], start_y[2] + (full_y[2] - start_y[2]) / 2.0, atol=1e-5)
    # env 3: wide negative side, early step -> midpoint, correct direction.
    assert torch.allclose(target[3], start_y[3] + (full_y[3] - start_y[3]) / 2.0, atol=1e-5)
    assert target[3] < start_y[3]
```

In `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py`, replace the test at lines 151-159:

```python
def test_landing_gate_fallback_advances_at_half_flight_time_when_never_landed():
    # Foot stays airborne the whole time (never lands near the target) -- the
    # schedule must still fall back to the full target once elapsed >= t_flight/2,
    # exactly like the pre-landing-gate behavior.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=30, found_left=False, found_right=False)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, full_y)
```

with:

```python
def test_no_landing_means_target_stays_at_midpoint_indefinitely():
    # Foot stays airborne the whole time (never lands near the target) --
    # under the 2026-07-04 hard gate there is no more time-based fallback, so
    # the target must STAY at the midpoint even well past the OLD t_flight/2
    # boundary, unlike the pre-hard-gate behavior this replaces.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=30, found_left=False, found_right=False)
    full_y = _get_ball_crossing_y(env, "ball")
    start_y = env.scene.env_origins[:, 1]
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, expected_mid, atol=1e-5)
```

- [ ] **Step 2: Run the updated tests to verify they fail against the current implementation**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_reach_target_two_stage.py tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: FAIL — `test_wide_crossing_stays_at_midpoint_past_former_half_flight_boundary`, `test_elapsed_time_no_longer_affects_phase_selection`, `test_mixed_batch_wide_narrow_independent_of_elapsed_time`, and `test_no_landing_means_target_stays_at_midpoint_indefinitely` all fail (current code still switches to the full target once elapsed passes t_flight/2). All other tests in both files still pass.

- [ ] **Step 3: Remove the time-based fallback from `_get_reach_target_y`**

Replace the full body of `_get_reach_target_y` in `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` (currently lines 143-243) with:

```python
def _get_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    wide_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.3,
) -> torch.Tensor:
    """Two-stage reach target for wide crossings (2026-07-03, hard-gated 2026-07-04).

    For crossings where |crossing_y - start_y| > wide_threshold (same threshold
    _tier_npz_reset uses to seed a double/triple-step RSI pose, events.py), the
    target is the MIDPOINT between the robot's stance and the true crossing
    point until the assigned foot (_get_correct_foot_idx) physically lands
    there, then switches to the full crossing point. Narrow crossings always
    target the full point, unchanged from before this feature.

    2026-07-04 hard gate (user-directed design, REMOVES the prior timeout
    fallback): the switch to the full point now ONLY happens on a genuine
    landing -- the assigned foot must be airborne at some point after reset,
    then come into ground contact (feet_contact sensor) within landing_radius
    of the midpoint target. There is no elapsed-time escape hatch anymore: a
    robot that never lands stays targeting the midpoint for the whole flight.
    This is intentionally more aggressive than the prior soft/timeout gate --
    see docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md
    for the research behind this change (the old gate's ~35-40% apparent
    landing rate was mostly RSI free credit, not learned behavior; play-mode
    observation showed ~0% genuine landings). The episode still cannot stall:
    footreach/foot_proximity's separate ball_close (< 0.5 m) switch to the
    live ball position is unaffected by this function and remains the true
    backstop.

    Landing state (env._blue_was_airborne, env._blue_landed) is only tracked
    when both a "robot" entity and a "feet_contact" sensor are present in
    env.scene -- absent in the lightweight fake envs used by
    test_reach_target_two_stage.py, which exercise the two-stage switch in
    isolation (KeyError on either lookup silently skips the landing check, so
    those envs always see plain `wide` gating with landing never occurring --
    narrow crossings are unaffected either way).

    Root XY is pinned to env.scene.env_origins at every reset (reset_base,
    goalkeeper_env_cfg.py; _write_rsi_state, events.py, only ever overwrites Z),
    so "robot start Y" can always be read live with no new per-episode cache.

    Does not affect the separate live-ball switch already in footreach/
    foot_proximity (ball_x_local < 0.5 m), which still takes priority once the
    ball is genuinely close — this only changes the FROZEN target fed into
    that existing logic.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y

    rel = getattr(env, "_rsi_cross_y", None)
    lateral = rel if rel is not None else (full_y - start_y)
    wide = lateral.abs() > wide_threshold

    half_y = start_y + (full_y - start_y) / 2.0

    # --- landing gate (2026-07-04, hardened: no time fallback) ---
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

    phase1_active = wide & ~env._blue_landed
    return torch.where(phase1_active, half_y, full_y)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_reach_target_two_stage.py tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: PASS — 10 tests in `test_reach_target_two_stage.py` (11 minus the deleted `test_missing_t_flight_falls_back_to_full_target`), 7 tests in `test_blue_ball_landing_gate.py` (renamed test, same count).

- [ ] **Step 5: Run the full test suite to confirm no other regression**

Run: `cd SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: PASS — all tests pass (only the two files above touch this function).

- [ ] **Step 6: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py SimpleGoalKeeper/tests/simple_goalkeeper/test_reach_target_two_stage.py SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py
git commit -m "feat(sgk): hard-gate the blue-ball landing, remove the time-based fallback"
```

---

### Task 2: Fix `play.py`'s sphere visualization to reflect the hard gate

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py:194-296` (`_patch_viewer_intercept_vis`)

**Interfaces:**
- Consumes: `env._blue_landed` (Task 1, already set on the real env by `_get_reach_target_y` during the same step's reward computation, before this visualization code runs).
- Produces: no change to the function's external signature (`_patch_viewer_intercept_vis(native_viewer, env) -> None`); only the internal blue/green decision logic changes.

**Why this task exists:** `_patch_viewer_intercept_vis` independently re-derives the blue/green switch for rendering — it currently reads `env._ball_t_flight` and re-implements the exact `elapsed < t_flight/2` check Task 1 just deleted from the actual reward logic. Left unpatched, play mode would keep flipping the sphere green on a timer that no longer has anything to do with the real target (which now only flips on `env._blue_landed`), making it impossible to visually verify the hard gate is working — the entire point of this bundled change.

- [ ] **Step 1: Update the docstring and the phase-selection logic**

In `SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py`, replace the docstring (currently lines 195-209):

```python
    """Monkey-patch NativeMujocoViewer to draw the predicted interception sphere.

    Adds a sphere at the current reach target (env 0) and a vertical line from
    floor to sphere so it's visible from any camera angle. The sphere updates
    each render frame — it moves when a new episode starts and the crossing_y
    changes.

    Two-stage wide-crossing visualization (2026-07-03, mirrors
    mdp.rewards._get_reach_target_y exactly): when |crossing_y - start_y| > 0.6
    and the ball's flight is still in its first half, draws a BLUE sphere at the
    midpoint between the robot's stance and the true crossing point instead of
    the usual green one. Once flight time / 2 has elapsed (or the crossing is
    narrow), draws the usual GREEN sphere at the full crossing point — i.e. blue
    disappears and green appears (at the full, farther point) once the schedule
    flips. Lets a human watching sgk_play confirm the timing visually.
    """
```

with:

```python
    """Monkey-patch NativeMujocoViewer to draw the predicted interception sphere.

    Adds a sphere at the current reach target (env 0) and a vertical line from
    floor to sphere so it's visible from any camera angle. The sphere updates
    each render frame — it moves when a new episode starts and the crossing_y
    changes.

    Two-stage wide-crossing visualization (2026-07-03, mirrors
    mdp.rewards._get_reach_target_y exactly). 2026-07-04 hard gate: reads the
    real env._blue_landed latch instead of re-deriving a time-based
    approximation (the prior elapsed < t_flight/2 check this docstring used to
    describe no longer exists in _get_reach_target_y — see
    docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md).
    When |crossing_y - start_y| > 0.6 and the assigned foot has not yet landed
    at the midpoint, draws a BLUE sphere there instead of the usual green one.
    Once landing has occurred (or the crossing is narrow), draws the usual
    GREEN sphere at the full crossing point. Lets a human watching sgk_play
    confirm landing timing visually.
    """
```

Replace the phase-selection block (currently lines 264-291):

```python
        # Two-stage wide-crossing schedule (mirrors mdp.rewards._get_reach_target_y).
        start_y = float(origins[1])
        rel_t = getattr(raw_env, "_rsi_cross_y", None)
        rel = float(rel_t[0].item()) if rel_t is not None else (cross_y - start_y)
        wide = abs(rel) > 0.6

        t_flight_t = getattr(raw_env, "_ball_t_flight", None)
        first_half = False
        if wide and t_flight_t is not None:
            step_dt = float(getattr(raw_env, "step_dt", 0.02))
            elapsed = float(raw_env.episode_length_buf[0].item()) * step_dt
            first_half = elapsed < (float(t_flight_t[0].item()) / 2.0)

        if wide and first_half:
            # Phase 1: BLUE sphere at the midpoint — half the distance, half as far.
            mid_y = start_y + (cross_y - start_y) / 2.0
            _add_sphere(goal_x, mid_y, sphere_z, 0.08, [0.15, 0.4, 1.0, 0.75])
            _add_line(
                np.array([goal_x, mid_y, floor_z], dtype=np.float64),
                np.array([goal_x, mid_y, sphere_z], dtype=np.float64),
                0.008, [0.15, 0.4, 1.0, 0.6],
            )
        else:
            # Phase 2 (or narrow crossing): GREEN sphere at the full crossing point.
            _add_sphere(goal_x, cross_y, sphere_z, 0.08, [0.1, 1.0, 0.2, 0.75])
            _add_line(
                np.array([goal_x, cross_y, floor_z], dtype=np.float64),
                np.array([goal_x, cross_y, sphere_z], dtype=np.float64),
                0.008, [0.1, 1.0, 0.2, 0.6],
            )
```

with:

```python
        # Two-stage wide-crossing schedule (mirrors mdp.rewards._get_reach_target_y).
        start_y = float(origins[1])
        rel_t = getattr(raw_env, "_rsi_cross_y", None)
        rel = float(rel_t[0].item()) if rel_t is not None else (cross_y - start_y)
        wide = abs(rel) > 0.6

        # 2026-07-04 hard gate: phase 1 (blue) now lasts until a genuine
        # landing (env._blue_landed), not until elapsed time passes
        # t_flight/2 -- read the actual latch _get_reach_target_y maintains
        # rather than re-deriving the now-removed time-based approximation.
        landed_t = getattr(raw_env, "_blue_landed", None)
        landed = bool(landed_t[0].item()) if landed_t is not None else False

        if wide and not landed:
            # Phase 1: BLUE sphere at the midpoint — half the distance, half as far.
            mid_y = start_y + (cross_y - start_y) / 2.0
            _add_sphere(goal_x, mid_y, sphere_z, 0.08, [0.15, 0.4, 1.0, 0.75])
            _add_line(
                np.array([goal_x, mid_y, floor_z], dtype=np.float64),
                np.array([goal_x, mid_y, sphere_z], dtype=np.float64),
                0.008, [0.15, 0.4, 1.0, 0.6],
            )
        else:
            # Phase 2 (or narrow crossing): GREEN sphere at the full crossing point.
            _add_sphere(goal_x, cross_y, sphere_z, 0.08, [0.1, 1.0, 0.2, 0.75])
            _add_line(
                np.array([goal_x, cross_y, floor_z], dtype=np.float64),
                np.array([goal_x, cross_y, sphere_z], dtype=np.float64),
                0.008, [0.1, 1.0, 0.2, 0.6],
            )
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `cd SimpleGoalKeeper && uv run python -c "import simple_goalkeeper.scripts.play"`
Expected: no exceptions. (This function has no existing automated test coverage — it's viewer/rendering code exercised interactively. Manual verification via `uv run sgk_play` with a trained checkpoint is the post-plan follow-up, not a step here.)

- [ ] **Step 3: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/scripts/play.py
git commit -m "fix(sgk): play.py sphere vis reads the real landing latch, not elapsed time"
```

---

### Task 3: Curriculum-ramp the `blue_ball_landed` bonus weight

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py:162-172` (curriculum block, right after `footreach_curriculum`)

**Interfaces:**
- Consumes: `gk_mdp.reward_curriculum_ep_len` (existing, `mdp/events.py`), the existing `"blue_ball_landed"` `RewardTermCfg` entry (unchanged, `weight=10.0` remains just the seed value the curriculum overwrites).
- Produces: `cfg.curriculum["blue_ball_landed_curriculum"]`, active in both training and play configs.

- [ ] **Step 1: Add the curriculum entry**

In `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, immediately after the `footreach_curriculum` entry (currently ending at line 169):

```python
        cfg.curriculum["footreach_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "footreach",
                "base_weight": 10.0,     # G1 eereach_init=10 → max 25 at cu=3
                "update_interval": 500,
                "ep_len_divisor":  47,
            },
        )
        # blue_ball_landed is now load-bearing for the double-step choreography
        # (2026-07-04 hard gate removed its time-based fallback), not just an
        # auxiliary bonus -- ramp it like footreach, the subsystem it's coupled to.
        cfg.curriculum["blue_ball_landed_curriculum"] = CurriculumTermCfg(
            func=gk_mdp.reward_curriculum_ep_len,
            params={
                "reward_name": "blue_ball_landed",
                "base_weight": 10.0,     # unchanged seed value → max 25 at cu=3
                "update_interval": 500,
                "ep_len_divisor":  47,
            },
        )
```

- [ ] **Step 2: Verify the config builds without error**

Run: `cd SimpleGoalKeeper && uv run python -c "from simple_goalkeeper.tasks.goalkeeper_env_cfg import *" 2>&1 | tail -20`
Expected: no exceptions, no `KeyError`/`AttributeError` about `blue_ball_landed_curriculum` or the `blue_ball_landed` reward term it targets.

- [ ] **Step 3: Run the full test suite**

Run: `cd SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: PASS — no test currently asserts `blue_ball_landed`'s weight is flat, so nothing regresses.

- [ ] **Step 4: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat(sgk): curriculum-ramp blue_ball_landed's weight like footreach"
```

---

### Task 4: RSI fraction rebalance (tier 50%→10%, live 30%→70%)

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py:269-270` (`MotionResetManager.reset` defaults)
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py:367-368` (module-level `reset_from_motion_data` defaults — **this is what `sgk_play --rsi` actually uses**, per the existing 2026-07-03 note that it re-registers the event with no params dict)
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (training config's explicit params dict, currently ~line 563 — will have shifted by Task 3's earlier insertion; find by the exact `params={"tier_rsi_fraction": 0.5, "live_rsi_fraction": 0.3}` string below, not the line number)
- Modify: `SimpleGoalKeeper/tests/simple_goalkeeper/test_live_rsi.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change anywhere — `tier_rsi_fraction`/`live_rsi_fraction` keyword-arg names and positions are unchanged, only their default numeric values move (0.5→0.1, 0.3→0.7).

- [ ] **Step 1: Update the failing/affected tests first**

In `SimpleGoalKeeper/tests/simple_goalkeeper/test_live_rsi.py`, update the coin-flip boundary comment and the tier-forcing draw inside `test_reset_single_coin_flip_covers_the_whole_batch_at_once` (currently lines 228 and 244):

```python
    # Partition at defaults (tier=0.5, live=0.3): [0,0.5) tier, [0.5,0.8) live donor, [0.8,1) standing.
    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.6))  # forces donor branch
    mgr.reset(env, env_ids)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    # Donor branch copies live values (~100), unmistakably not the ~2.0 default-pose range.
    assert (robot.written_pos > 50).all()

    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.85))  # forces default branch
    mgr.reset(env, env_ids)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    # Default-pose branch: default_joint_pos is 0 here, scale*0==0, only the
    # small +-0.1 offset survives, so it must be nowhere near the donor range.
    assert (robot.written_pos.abs() < 0.2).all()

    tier_calls = []
    monkeypatch.setattr(mgr, "_tier_npz_reset", lambda env, ids, robot: tier_calls.append(ids.tolist()))
    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.1))  # forces tier NPZ branch
    mgr.reset(env, env_ids)
    assert tier_calls == [[0, 1, 2, 3]]
```

becomes:

```python
    # Partition at defaults (2026-07-04, tier=0.1, live=0.7): [0,0.1) tier,
    # [0.1,0.8) live donor, [0.8,1) standing.
    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.6))  # forces donor branch
    mgr.reset(env, env_ids)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    # Donor branch copies live values (~100), unmistakably not the ~2.0 default-pose range.
    assert (robot.written_pos > 50).all()

    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.85))  # forces default branch
    mgr.reset(env, env_ids)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    # Default-pose branch: default_joint_pos is 0 here, scale*0==0, only the
    # small +-0.1 offset survives, so it must be nowhere near the donor range.
    assert (robot.written_pos.abs() < 0.2).all()

    tier_calls = []
    monkeypatch.setattr(mgr, "_tier_npz_reset", lambda env, ids, robot: tier_calls.append(ids.tolist()))
    monkeypatch.setattr(torch, "rand", _forced_coin_flip(0.05))  # forces tier NPZ branch (< 0.1)
    mgr.reset(env, env_ids)
    assert tier_calls == [[0, 1, 2, 3]]
```

Replace `test_reset_partition_boundaries_at_default_fractions` (currently lines 249-264):

```python
def test_reset_partition_boundaries_at_default_fractions():
    """Draw partition (2026-07-03 three-way split, fractions raised same day
    30/50/20 -> 50/30/20): [0, 0.5) tier NPZ RSI, [0.5, 0.8) G1 continue_keep
    live donor, [0.8, 1.0) G1 randomized standing. The live-donor probability
    (0.3) plus tier (0.5) preserves an 80% chance of SOME RSI-style reset vs
    20% standing — same standing share as G1's literal
    `torch.rand(1).item() > 0.2` (legged_robot.py:669)."""
    tier, live = 0.5, 0.3
    for draw, branch in [
        (0.0, "tier"), (0.49, "tier"),
        (0.5, "live"), (0.6, "live"), (0.79, "live"),
        (0.8, "stand"), (0.99, "stand"),
    ]:
        got = "tier" if draw < tier else ("live" if draw < tier + live else "stand")
        assert got == branch, f"draw={draw}: got {got}, expected {branch}"
    assert abs((1.0 - (tier + live)) - 0.2) < 1e-9  # standing share matches G1
```

with:

```python
def test_reset_partition_boundaries_at_default_fractions():
    """Draw partition (2026-07-04 rebalance, tier 50%->10% / live 30%->70%,
    to cut the "free landing" RSI credit the blue-ball hard gate found — see
    docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md):
    [0, 0.1) tier NPZ RSI, [0.1, 0.8) G1 continue_keep live donor,
    [0.8, 1.0) G1 randomized standing. The live-donor probability (0.7) plus
    tier (0.1) preserves an 80% chance of SOME RSI-style reset vs 20%
    standing — same standing share as G1's literal
    `torch.rand(1).item() > 0.2` (legged_robot.py:669)."""
    tier, live = 0.1, 0.7
    for draw, branch in [
        (0.0, "tier"), (0.09, "tier"),
        (0.1, "live"), (0.5, "live"), (0.79, "live"),
        (0.8, "stand"), (0.99, "stand"),
    ]:
        got = "tier" if draw < tier else ("live" if draw < tier + live else "stand")
        assert got == branch, f"draw={draw}: got {got}, expected {branch}"
    assert abs((1.0 - (tier + live)) - 0.2) < 1e-9  # standing share matches G1
```

Replace `test_reset_three_way_split_statistics` (currently lines 267-301), just the docstring and the final expected list:

```python
def test_reset_three_way_split_statistics(monkeypatch):
    """Over many independent reset() calls the branch rates must converge to
    tier=0.5 / live=0.3 / standing=0.2 (one draw per call, not per env)."""
```

with:

```python
def test_reset_three_way_split_statistics(monkeypatch):
    """Over many independent reset() calls the branch rates must converge to
    tier=0.1 / live=0.7 / standing=0.2 (2026-07-04 rebalance; one draw per
    call, not per env)."""
```

and:

```python
    for branch, expected in [("tier", 0.5), ("live", 0.3), ("stand", 0.2)]:
```

with:

```python
    for branch, expected in [("tier", 0.1), ("live", 0.7), ("stand", 0.2)]:
```

Replace `test_reset_from_motion_data_passes_the_three_way_split_to_reset` (currently lines 376-398), the docstring and final assertions:

```python
def test_reset_from_motion_data_passes_the_three_way_split_to_reset(monkeypatch):
    """Regression guard (the 2026-07-01 audit found this wrapper silently
    overriding reset()'s fractions once before): pin the arguments the
    registered training event actually forwards — 0.5 tier / 0.3 live
    (raised same-day 2026-07-03 from 0.3/0.5)."""
```

with:

```python
def test_reset_from_motion_data_passes_the_three_way_split_to_reset(monkeypatch):
    """Regression guard (the 2026-07-01 audit found this wrapper silently
    overriding reset()'s fractions once before): pin the arguments the
    registered training event actually forwards — 0.1 tier / 0.7 live
    (rebalanced 2026-07-04 from 0.5/0.3, see
    docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md)."""
```

and:

```python
    assert captured["tier"] == 0.5
    assert captured["live"] == 0.3
```

with:

```python
    assert captured["tier"] == 0.1
    assert captured["live"] == 0.7
```

- [ ] **Step 2: Run the tests to verify they now fail against the current code**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: FAIL — `test_reset_single_coin_flip_covers_the_whole_batch_at_once`, `test_reset_partition_boundaries_at_default_fractions`, `test_reset_three_way_split_statistics`, `test_reset_from_motion_data_passes_the_three_way_split_to_reset` fail against the still-0.5/0.3 defaults.

- [ ] **Step 3: Change the default fractions in all three call sites**

In `SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py`, `MotionResetManager.reset` (currently lines 264-270):

```python
    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        tier_rsi_fraction: float = 0.5,
        live_rsi_fraction: float = 0.3,
```

becomes:

```python
    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        tier_rsi_fraction: float = 0.1,
        live_rsi_fraction: float = 0.7,
```

And the module-level `reset_from_motion_data` (currently lines 363-368):

```python
def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    tier_rsi_fraction: float = 0.5,
    live_rsi_fraction: float = 0.3,
) -> None:
```

becomes:

```python
def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    tier_rsi_fraction: float = 0.1,
    live_rsi_fraction: float = 0.7,
) -> None:
```

Also update this function's docstring (currently lines 372-379):

```python
    """Reset event: tier NPZ RSI vs continue_keep donor copy vs default pose.

    2026-07-03: three-way split, raised same day from 30/50/20 to 50/30/20
    (50% ball-conditioned NPZ / 30% G1 live donor / 20% G1 randomized standing)
    — see MotionResetManager.reset for the dilution rationale.

    FIX 2026-07-01: this wrapper hardcoded rsi_fraction=0.5, a silent 50/50
    split, diverging from both G1's actual 80/20 (`torch.rand(1).item() > 0.2`,
    legged_robot.py:669) and this project's own reset()/CLAUDE.md-documented
    80/20 intent. Caught by an independent fidelity audit — see
    docs/superpowers/plans/2026-07-01-live-env-rsi.md.
    """
```

to:

```python
    """Reset event: tier NPZ RSI vs continue_keep donor copy vs default pose.

    2026-07-03: three-way split, raised same day from 30/50/20 to 50/30/20
    (50% ball-conditioned NPZ / 30% G1 live donor / 20% G1 randomized standing)
    — see MotionResetManager.reset for the dilution rationale.

    2026-07-04: rebalanced 50/30/20 -> 10/70/20. The hard-gated blue-ball
    landing (mdp.rewards._get_reach_target_y) found the tier-RSI branch's
    random mid-clip DoubleStep/TripleStep frames can start with the assigned
    foot already lifted near the target, satisfying the landing latch "for
    free" — inflating the apparent landing rate without the policy learning
    anything. Cutting tier share to 10% forces most wide-crossing episodes to
    learn the landing from a normal starting state. See
    docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md.
    Standing share (20%) unchanged.

    FIX 2026-07-01: this wrapper hardcoded rsi_fraction=0.5, a silent 50/50
    split, diverging from both G1's actual 80/20 (`torch.rand(1).item() > 0.2`,
    legged_robot.py:669) and this project's own reset()/CLAUDE.md-documented
    80/20 intent. Caught by an independent fidelity audit — see
    docs/superpowers/plans/2026-07-01-live-env-rsi.md.
    """
```

In `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (this exact string is unique in the file, so find-and-replace by content, not line number — the line has shifted from 563 due to Task 3's earlier insertion):

```python
        params={"tier_rsi_fraction": 0.5, "live_rsi_fraction": 0.3},
```

becomes:

```python
        params={"tier_rsi_fraction": 0.1, "live_rsi_fraction": 0.7},
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: PASS — all tests in the file, including the four updated above.

- [ ] **Step 5: Run the full test suite**

Run: `cd SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py SimpleGoalKeeper/tests/simple_goalkeeper/test_live_rsi.py
git commit -m "feat(sgk): rebalance RSI fractions tier 50%->10%, live 30%->70%"
```

---

### Task 5: Diagnostic metrics — genuine vs. RSI-assisted landing

**Files:**
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py` (add `_blue_airborne_at_reset` latch inside `_get_reach_target_y`, second edit to the function Task 1 already changed)
- Create: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/metrics.py`
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py`
- Modify: `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (new `cfg.metrics` block + import)
- Modify: `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (append)

**Interfaces:**
- Consumes: `_get_reach_target_y(env, ball_name, asset_cfg=...)` (Task 1), `env._blue_landed`, new `env._blue_airborne_at_reset`.
- Produces: `env._blue_airborne_at_reset` (bool tensor, shape `(N,)`, episode-scoped latch). `blue_landed_genuine(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG) -> torch.Tensor` and `blue_landed_rsi_assisted(env, ball_name, asset_cfg=_DEFAULT_FEET_CFG) -> torch.Tensor`, both exported as `gk_mdp.blue_landed_genuine`/`gk_mdp.blue_landed_rsi_assisted` for the config's `cfg.metrics` wiring.

- [ ] **Step 1: Write the failing tests for the new latch**

Append to `SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py`:

```python
def test_airborne_at_reset_latches_when_lift_happens_within_two_steps():
    # Foot airborne on episode_step=1 (within the 2-step grace window) ->
    # _blue_airborne_at_reset must latch true, flagging this as a plausible
    # RSI artifact rather than something the policy did.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=1, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_airborne_at_reset[0].item()


def test_airborne_at_reset_does_not_latch_when_lift_happens_later():
    # Foot in contact at steps 1-2 (within the grace window), THEN airborne at
    # step 5 -- this is a policy-driven lift, not an RSI artifact, so
    # _blue_airborne_at_reset must stay false even though _blue_was_airborne
    # does become true.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=1, found_left=True, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_was_airborne[0].item()
    assert not env._blue_airborne_at_reset[0].item()

    env.episode_length_buf[:] = 5
    env.scene["feet_contact"].data.found[:, :4] = 0.0  # left foot now airborne
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_was_airborne[0].item()
    assert not env._blue_airborne_at_reset[0].item()


def test_airborne_at_reset_resets_on_new_episode():
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=1, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_airborne_at_reset[0].item()

    env.episode_length_buf[:] = 1  # new episode reset step
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # foot in contact again
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_airborne_at_reset[0].item()
```

Append also, at the top of the same file (near the other `from simple_goalkeeper.mdp.rewards import ...` lines), and then the metric-function tests:

```python
from simple_goalkeeper.mdp.metrics import blue_landed_genuine, blue_landed_rsi_assisted


def test_blue_landed_genuine_fires_when_airborne_transition_is_not_at_reset():
    # Foot in contact through step 2 (past the grace window already, but
    # still not airborne), THEN airborne at step 5, THEN lands at step 6 --
    # a policy-driven landing, so blue_landed_genuine must fire and
    # blue_landed_rsi_assisted must not.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=2, found_left=True, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())

    env.episode_length_buf[:] = 5
    env.scene["feet_contact"].data.found[:, :4] = 0.0  # left foot airborne
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())

    env.episode_length_buf[:] = 6
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot lands at the midpoint

    genuine = blue_landed_genuine(env, "ball", asset_cfg=_feet_cfg())
    rsi_assisted = blue_landed_rsi_assisted(env, "ball", asset_cfg=_feet_cfg())
    assert genuine.item() == 1.0
    assert rsi_assisted.item() == 0.0


def test_blue_landed_rsi_assisted_fires_when_airborne_transition_is_at_reset():
    # Foot airborne already on step 1 (within the grace window -- e.g. an RSI
    # reset pose with the foot mid-step), THEN lands at step 2. Landing must
    # be classified as RSI-assisted, not genuine.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=1, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())

    env.episode_length_buf[:] = 2
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot lands at the midpoint

    genuine = blue_landed_genuine(env, "ball", asset_cfg=_feet_cfg())
    rsi_assisted = blue_landed_rsi_assisted(env, "ball", asset_cfg=_feet_cfg())
    assert genuine.item() == 0.0
    assert rsi_assisted.item() == 1.0


def test_blue_landed_diagnostics_both_zero_without_a_landing():
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    genuine = blue_landed_genuine(env, "ball", asset_cfg=_feet_cfg())
    rsi_assisted = blue_landed_rsi_assisted(env, "ball", asset_cfg=_feet_cfg())
    assert genuine.item() == 0.0
    assert rsi_assisted.item() == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v -k "airborne_at_reset or blue_landed_genuine or blue_landed_rsi_assisted"`
Expected: FAIL — `AttributeError: '_FakeEnv'/env has no attribute '_blue_airborne_at_reset'` and `ModuleNotFoundError: No module named 'simple_goalkeeper.mdp.metrics'`.

- [ ] **Step 3: Add the `_blue_airborne_at_reset` latch to `_get_reach_target_y`**

In `SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py`, inside `_get_reach_target_y` (the version Task 1 just wrote), change:

```python
    # --- landing gate (2026-07-04, hardened: no time fallback) ---
    n = env.num_envs
    if not hasattr(env, "_blue_was_airborne"):
        env._blue_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_was_airborne[just_reset] = False
    env._blue_landed[just_reset] = False
```

to:

```python
    # --- landing gate (2026-07-04, hardened: no time fallback) ---
    n = env.num_envs
    if not hasattr(env, "_blue_was_airborne"):
        env._blue_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_airborne_at_reset = torch.zeros(n, dtype=torch.bool, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_was_airborne[just_reset] = False
    env._blue_landed[just_reset] = False
    env._blue_airborne_at_reset[just_reset] = False
```

and change:

```python
        env._blue_was_airborne |= ~foot_in_contact
```

to:

```python
        currently_airborne = ~foot_in_contact
        first_time_airborne = currently_airborne & ~env._blue_was_airborne
        near_reset = env.episode_length_buf <= 2
        env._blue_airborne_at_reset |= first_time_airborne & near_reset
        env._blue_was_airborne |= currently_airborne
```

Also update the function's docstring paragraph on landing state (currently "Landing state (env._blue_was_airborne, env._blue_landed) is only tracked when...") to read:

```
    Landing state (env._blue_was_airborne, env._blue_landed,
    env._blue_airborne_at_reset) is only tracked when both a "robot" entity
    and a "feet_contact" sensor are present in env.scene -- absent in the
    lightweight fake envs used by test_reach_target_two_stage.py, which
    exercise the two-stage switch in isolation (KeyError on either lookup
    silently skips the landing check, so those envs always see plain `wide`
    gating with landing never occurring -- narrow crossings are unaffected
    either way).

    env._blue_airborne_at_reset (2026-07-04, diagnostic only, does not affect
    the schedule): latched true if the assigned foot's FIRST transition to
    airborne happens within 2 steps of reset -- a proxy for "this episode's
    RSI pose plausibly started with the foot already lifted" rather than the
    policy causing the lift. See mdp.metrics.blue_landed_genuine /
    blue_landed_rsi_assisted, which classify blue_ball_landed's firing using
    this latch.
```

- [ ] **Step 4: Run the latch tests to verify they pass**

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v -k "airborne_at_reset"`
Expected: PASS (3 tests).

- [ ] **Step 5: Create `mdp/metrics.py`**

Create `SimpleGoalKeeper/src/simple_goalkeeper/mdp/metrics.py`:

```python
"""Diagnostic metrics for the blue-ball landing gate (2026-07-04).

Not rewards -- no weight, no dt scaling (mjlab.managers.metrics_manager:
"Unlike rewards, metrics have no weight, no dt scaling, and no normalization
by episode length"). Distinguishes a landing the policy caused from one the
RSI reset pose handed it for free, since the 2026-07-04 hard-gate + RSI-
rebalance design bundles three changes into one training run and needs a way
to tell their effects apart afterward. See
docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import _DEFAULT_FEET_CFG, _get_reach_target_y

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def blue_landed_genuine(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """1.0 while env._blue_landed is true and it was NOT an RSI-assisted
    landing (env._blue_airborne_at_reset is false), else 0.0.

    Wired with reduce="last" in MetricsTermCfg (goalkeeper_env_cfg.py), so the
    logged Episode_Metrics value is this exact per-episode 0/1 outcome, not a
    mean over the episode. See _get_reach_target_y for both latches.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure latches are fresh
    return (env._blue_landed & ~env._blue_airborne_at_reset).float()


def blue_landed_rsi_assisted(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """1.0 while env._blue_landed is true and the assigned foot's first
    airborne transition happened within 2 steps of reset (env.
    _blue_airborne_at_reset), else 0.0. Mutually exclusive with
    blue_landed_genuine given _blue_landed true; both zero when _blue_landed
    is false. See blue_landed_genuine.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    return (env._blue_landed & env._blue_airborne_at_reset).float()
```

- [ ] **Step 6: Export the new functions and run the metric-function tests**

In `SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py`, change:

```python
from . import observations, events, rewards, commands
```

to:

```python
from . import observations, events, rewards, commands, metrics
```

and add a new import line after the existing `from .rewards import (...)` block:

```python
from .metrics import blue_landed_genuine, blue_landed_rsi_assisted
```

Run: `cd SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_blue_ball_landing_gate.py -v`
Expected: PASS — all tests in the file, including the 3 new latch tests and 3 new metric-function tests (13 total: 7 from the original file + 3 latch + 3 metric).

- [ ] **Step 7: Wire the metrics into the config**

In `SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, add the import alongside the other manager imports (currently line 14-17):

```python
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
```

becomes:

```python
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
```

Then, immediately after the `cfg.rewards = {...}` dict closes (currently line 483, but will have shifted a few lines down from Task 3's earlier insertion — find it by the "Events — Domain Randomisation" section comment that immediately follows the closing `}`), add:

```python
    }

    # ------------------------------------------------------------------
    # Metrics — diagnostics only, no weight/dt scaling (mjlab.managers.
    # metrics_manager). Distinguishes a genuine (policy-driven) blue-ball
    # landing from an RSI-assisted one (2026-07-04, see mdp/metrics.py).
    # ------------------------------------------------------------------
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

(Note: the closing `}` of `cfg.rewards` already exists at line 483 — this step adds the new block immediately after it, it does not duplicate the brace.)

- [ ] **Step 8: Verify the config builds without error**

Run: `cd SimpleGoalKeeper && uv run python -c "from simple_goalkeeper.tasks.goalkeeper_env_cfg import *" 2>&1 | tail -20`
Expected: no exceptions, no `AttributeError`/`KeyError` about `cfg.metrics`, `blue_landed_genuine`, or `blue_landed_rsi_assisted`.

- [ ] **Step 9: Run the full test suite**

Run: `cd SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: PASS — all tests, including everything from Tasks 1-4.

- [ ] **Step 10: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py SimpleGoalKeeper/src/simple_goalkeeper/mdp/metrics.py SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py SimpleGoalKeeper/tests/simple_goalkeeper/test_blue_ball_landing_gate.py
git commit -m "feat(sgk): add genuine-vs-RSI-assisted blue-ball landing diagnostics"
```

---

### Task 6: Documentation

**Files:**
- Modify: `SimpleGoalKeeper/CLAUDE.md` (Divergences from G1 Upstream table + Reward Design table + a new Diagnostics note)
- Modify: `SimpleGoalKeeper/docs/BugFixes.md`

**Interfaces:** None (docs only).

- [ ] **Step 1: Update the "landing gate" divergence-table row and add an RSI-fraction row**

In `SimpleGoalKeeper/CLAUDE.md`, find the existing table row starting `| \`footreach\`/\`foot_proximity\` reach target: landing gate on the two-stage schedule |` and replace its third column (the "SimpleGoalKeeper value" cell, currently starting "**Landing gate (2026-07-04, user-directed design):** the blue→green switch on wide crossings now ALSO fires early...") with:

```markdown
**Landing gate (2026-07-04, hardened same day):** the blue→green switch on wide crossings now ONLY fires on a genuine foot landing — the assigned foot must be airborne at some point then land in ground contact within 0.3 m of the blue midpoint target (`feet_contact` sensor). The original `elapsed >= t_flight/2` time-based fallback was REMOVED (not just supplemented) after research showed the soft-gated version's ~35-40% apparent landing rate (correctly converted via the `Episode_Reward` formula below) was mostly explained by RSI-seeded "free" landings, not learned behavior — confirmed by ~0% genuine landings visible in `sgk_play`. `blue_ball_landed` (one-shot bonus, now curriculum-ramped 10→25 like `footreach` rather than flat +10) pays out the first time the landing latch fires. New diagnostic metrics `blue_landed_genuine`/`blue_landed_rsi_assisted` (`mdp/metrics.py`, `Episode_Metrics/*`, no weight/dt scaling) classify each landing using a new latch, `env._blue_airborne_at_reset` (true if the assigned foot's first airborne transition happens within 2 steps of reset). `footreach`/`foot_proximity`'s separate `ball_close < 0.5 m` live-ball override is unaffected and remains the true backstop against stalling.
```

And update its fourth column (the "Justification" cell) by appending a new paragraph after the existing text:

```markdown
 **2026-07-04 hardening:** research (`Episode_Reward/blue_ball_landed` converted via this file's own documented formula — see "Reading TensorBoard" below — gives ~35-40% of all episodes, ~73-83% of wide crossings) showed the soft gate looked healthy on paper, but a Monte Carlo of `_tier_npz_reset`'s sampling plus user observation in `sgk_play` (0% visible genuine landings) showed this was almost entirely RSI free credit: tier-RSI resets draw a random frame from the DoubleStep/TripleStep motion clips, which can already have the assigned foot lifted near the blue target. Removing the fallback outright, rebalancing RSI (see the row below), and adding the genuine/RSI-assisted split were shipped together in one run — user's explicit choice over testing the RSI rebalance in isolation first — with the new diagnostic metrics as the agreed mitigation for that risk. Full rationale: `docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md`. Covered by `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (13 tests) and the rewritten tests in `tests/simple_goalkeeper/test_reach_target_two_stage.py`. Not yet validated against a training run.
```

Then add a new table row directly after that row (same table):

```markdown
| RSI three-way split fractions | N/A — G1 has no NPZ-tier RSI mechanism at all (see the RSI mechanism row above; this only changes the SGK-specific tier/live/standing mix). | **Rebalanced 2026-07-04: tier 50%→10%, live 30%→70%, standing unchanged 20%.** `MotionResetManager.reset()` defaults, module-level `reset_from_motion_data()` defaults (used by `sgk_play --rsi`), and `goalkeeper_env_cfg.py`'s training params dict all updated together (same three-call-site pattern as the 2026-07-03 30/50/20→50/30/20 change). | **Deliberate divergence, in tension with the 2026-07-03 change this reverses part of:** tier-RSI was raised to 50% specifically because 30% wasn't enough exposure to produce double-stepping at all. Cutting it to 10% risks reintroducing that failure mode, just observed through a different symptom this time (no double-stepping, vs. the illusory landing metric this row's parent describes) — documented as an accepted risk, not resolved, since the user chose to bundle this with the hard gate in one run rather than test it in isolation first. `mdp/events.py:MotionResetManager.reset,reset_from_motion_data`, `tasks/goalkeeper_env_cfg.py`. |
```

- [ ] **Step 2: Update the Reward Design table and add a Diagnostics note**

In `SimpleGoalKeeper/CLAUDE.md`, find the row `| \`blue_ball_landed\` | +10.0 | One-shot bonus when the assigned foot lands...` and replace it with:

```markdown
| `blue_ball_landed` | +10→25 (curriculum, 2026-07-04) | One-shot bonus when the assigned foot lands (airborne-then-contact) within 0.3 m of the blue-ball midpoint target on a wide crossing. Since 2026-07-04 this is the ONLY way the two-stage schedule advances to the green target on wide crossings — the prior time-based fallback was removed. |
```

Then, immediately after the Reward Design table (before the "Terminations:" line), add:

```markdown
**Diagnostics (not rewards, no weight/dt scaling — `cfg.metrics`, 2026-07-04):** `blue_landed_genuine` and `blue_landed_rsi_assisted` (`Episode_Metrics/*`) classify each `blue_ball_landed` firing as policy-driven or RSI-seeded, using `env._blue_airborne_at_reset` (true if the assigned foot's first airborne transition happens within 2 steps of reset). Read directly as per-episode rates — unlike `Episode_Reward/*` one-shot values, no conversion formula needed.
```

- [ ] **Step 3: Add a BugFixes.md entry**

Append to `SimpleGoalKeeper/docs/BugFixes.md`:

```markdown

---

## 2026-07-04 — blue-ball landing gate hardened to a pure hard gate; RSI rebalanced; landing diagnostics added

**What changed:** three coupled changes, shipped together in one training run (user's explicit choice):

1. `mdp.rewards._get_reach_target_y` loses its `elapsed >= t_flight/2` time-based fallback entirely. `phase1_active = wide & ~env._blue_landed` — no time term at all. A robot that never lands at the blue midpoint on a wide crossing now stays targeting it for the whole flight, until `footreach`/`foot_proximity`'s separate `ball_close < 0.5 m` live-ball override takes over (unaffected by this change, remains the true backstop against stalling). `scripts/play.py`'s sphere visualization, which had been independently re-deriving the now-removed time-based switch for rendering, was fixed to read the real `env._blue_landed` latch instead.
2. `blue_ball_landed`'s reward weight changed from flat +10.0 to curriculum-ramped 10→25 (reusing `reward_curriculum_ep_len`, identical formula/ceiling to `footreach_curriculum`), since it's no longer just an auxiliary bonus — it now gates whether the double-step choreography is reachable at all.
3. RSI three-way split rebalanced: `tier_rsi_fraction` 0.5→0.1, `live_rsi_fraction` 0.3→0.7 (standing unchanged at 0.2), across all three call sites (`MotionResetManager.reset()` defaults, module-level `reset_from_motion_data()` defaults — used by `sgk_play --rsi` — and `goalkeeper_env_cfg.py`'s training params dict).

Plus a new diagnostic: `env._blue_airborne_at_reset` (latched true if the assigned foot's first airborne transition happens within 2 steps of reset) and two new `cfg.metrics` entries, `blue_landed_genuine`/`blue_landed_rsi_assisted` (`mdp/metrics.py`, `Episode_Metrics/*`, no weight/dt scaling), classifying each landing as policy-driven or RSI-seeded.

**Why it was wrong:** the 2026-07-04 soft/timeout-gated landing mechanism (previous entry above) looked healthy in `Episode_Reward/blue_ball_landed` (~35-40% of all episodes, ~73-83% of wide crossings once correctly converted via this file's own documented one-shot-reward formula) but the user observed ~0% genuine landings in `sgk_play`. Root cause: `_tier_npz_reset` routes 50% of wide-crossing resets to a random frame from the DoubleStep/TripleStep motion clips, which can already have the assigned foot lifted and positioned near the blue target — satisfying the landing latch within a step or two of reset with zero credit due to the policy. The metric was measuring RSI seeding, not learned behavior.

**Why this design over alternatives:** considered testing the RSI rebalance alone first (keeping today's soft gate unchanged) to isolate whether it produces genuine learning before also hard-gating the schedule — user chose to bundle all three changes into one run instead, accepting the interpretability cost in exchange for speed; the new genuine/RSI-assisted diagnostic split is the agreed mitigation for that risk. Also notable: `tier_rsi_fraction` was raised 30%→50% on 2026-07-03 specifically because 30% wasn't enough exposure to produce double-stepping at all — dropping it back to 10% risks reintroducing that failure mode under a different symptom, documented as an accepted risk rather than resolved.

**Evidence:** `tests/simple_goalkeeper/test_blue_ball_landing_gate.py` (13 tests), rewritten tests in `tests/simple_goalkeeper/test_reach_target_two_stage.py`, updated fraction tests in `tests/simple_goalkeeper/test_live_rsi.py`. Full research and design: `docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md`. Not yet validated against a training run — the next run should be checked via `Episode_Metrics/blue_landed_genuine` vs. `blue_landed_rsi_assisted` (if genuine stays near zero while RSI-assisted accounts for most landings, the RSI rebalance risk above has materialized and `tier_rsi_fraction` may need to be walked back up) and visually in `sgk_play` for the blue→green marker now switching only on a genuine landing.
```

- [ ] **Step 4: Commit**

```bash
cd /home/ibouwmeest/BEPImitationLearning
git add SimpleGoalKeeper/CLAUDE.md SimpleGoalKeeper/docs/BugFixes.md
git commit -m "docs(sgk): document the blue-ball hard gate, RSI rebalance, and landing diagnostics"
```

---

## Post-plan note (not a task)

This is a code-only change — it requires a fresh training run to evaluate, and does not touch the currently-running (already-diverging) run `2026-07-04_03-47-17`. Once a fresh run has enough iterations to produce wide-crossing episodes past the difficulty ramp-up (~iter 3000, per the current run's `ball_difficulty` curve), review `Episode_Metrics/blue_landed_genuine` vs. `blue_landed_rsi_assisted` first, then confirm visually in `sgk_play` that the blue→green marker only switches on a genuine landing. If `blue_landed_genuine` stays near zero, `tier_rsi_fraction` may need to be walked back up from 10% — see the accepted-risk note in Task 4 and the CLAUDE.md divergence-table row this plan adds.
