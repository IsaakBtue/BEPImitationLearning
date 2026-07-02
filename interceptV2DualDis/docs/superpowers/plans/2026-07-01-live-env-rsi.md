# Live-Environment RSI Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SUPERSEDED 2026-07-01 (same day, after all 6 tasks below shipped):** the tier-scoped design in this plan (side/tier-matched donor pools, maturity/warmup gating, NPZ fallback, `live_rsi_usage_curriculum`) was fully implemented and committed, then explicitly reverted per user request in favor of a **literal, unscoped port of G1's `continue_keep`** — no side/tier matching, no exclusion of the current reset batch, no maturity check, no fallback (none of these exist in G1 either). Rationale for the tier-scoping (below) is kept for context, but `MotionResetManager.reset()` no longer implements it. The actual current behavior: one `torch.rand(1)` draw per `reset()` call decides the whole batch; the RSI branch copies `dof_pos` from `torch.randint(0, env.num_envs, ...)` — any live env, unconditionally; `joint_vel` is always zeroed; root state is left to the separate `reset_base` event. The old side/tier NPZ-pool infrastructure (`self.pools`, `_write_rsi_state`, `_STEM_TO_POOL`) is untouched and still used by the `sgk_play_rsi` diagnostic script, just no longer called from `reset()`. See the `CLAUDE.md` divergence table entry for the up-to-date description.

> **FOLLOW-UP AUDIT 2026-07-01 (later same day):** per user request, an independent subagent audit compared the literal-port implementation above against G1's actual source line-by-line and found it was still not fully literal. Three real divergences, all now fixed:
> 1. **Clamp on the wrong branch.** G1 never clips in the `continue_keep` donor-copy branch — only inside the *other* branch's `randomize_initial_joint_pos` sub-case (`legged_robot.py:673-675`). The first literal-port attempt clamped the donor branch (to `soft_joint_pos_limits`, which doesn't even exist as a concept in G1) and didn't clamp the else branch at all. Fixed: donor branch now copies raw with no clamp; else branch clips to `robot.data.joint_pos_limits` (hard, matching G1's `dof_pos_limits`).
> 2. **Else branch was missing G1's actual active randomization.** G1's real config (`g1/g1_29_config.py:279-282`) sets `randomize_initial_joint_pos=True`, `initial_joint_pos_scale=[0.5,1.5]`, `initial_joint_pos_offset=[-0.1,0.1]` — the else branch is `standpos * U(0.5,1.5) + U(-0.1,0.1)`, clipped, not a flat default pose. The first literal-port attempt used a flat `default_joint_pos` copy, which only corresponds to G1's *inactive* (`randomize_initial_joint_pos=False`) base-class default, not what G1 actually runs. Fixed: else branch now applies the same scale/offset via `sample_uniform`.
> 3. **`rsi_fraction` was hardcoded to 0.5 in production, not 0.8.** `reset_from_motion_data` (the function actually registered as the training event) called `mgr.reset(..., rsi_fraction=0.5)`, silently overriding `reset()`'s own default of 0.8 — a 50/50 split had been running since at least 2026-06-20, contradicting both G1's 80/20 and this project's own CLAUDE.md documentation the whole time. Fixed: hardcoded to `0.8`.
>
> Also confirmed independently (not fixed, flagged as separate/out-of-scope): G1's `_reset_root_states` randomizes root linear+angular velocity ±0.3 unconditionally on every reset; SimpleGoalKeeper's `reset_base` event has always used `velocity_range={}`, which resolves to a hard zero. This is a real divergence but touches every reset in the environment (not just RSI), so it wasn't changed without a separate explicit decision.
>
> Lesson for next time: "literal port" claims need verification against the actual active config values, not just the function's structure — the first attempt matched G1's control flow shape but missed that G1's own *default* config differs from what G1's G1-specific config file actually turns on.

**Goal:** Replace SimpleGoalKeeper's static-NPZ-frame RSI (for the double/triple/wide step pools only) with sampling from other *currently running* training environments' live joint state, so mid-motion reset poses reflect the policy's own evolving stepping technique instead of a fixed mocap clip — while keeping every safety property that made the two prior attempts at this fail or get abandoned.

**Architecture:** `MotionResetManager` (in `src/simple_goalkeeper/mdp/events.py`) gains a per-env "which tier is this episode" tag and reuses mjlab's existing `episode_length_buf`/`common_step_counter`. When a double/triple/wide-tier reset fires, it first looks for other envs currently mid-episode in the *same* tier that are past a minimum maturity age, and if enough exist, copies **joint positions only** from a random one (root pose/velocity is left untouched — already set by the existing `reset_base` event). If too few eligible donors exist (early training, small eval runs, or a warmup window after startup), it transparently falls back to the existing static-NPZ path, so nothing can regress to worse-than-today behavior.

**Tech Stack:** mjlab (MuJoCo-Warp), beyondAMP, PyTorch, uv, pytest (new dev dependency).

## Global Constraints

- Do not modify `Humanoid-Goalkeeper/` (frozen upstream reference) or `Imitationlearningbooster/` — read-only references for this work.
- Per `SimpleGoalKeeper/CLAUDE.md`'s Design Rule, every reward/reset/spawn divergence from G1 must be justified and logged in the "Divergences from G1 Upstream" table — Task 6 does this.
- Per `SimpleGoalKeeper/CLAUDE.md`'s Change Approval Workflow, this plan itself is the change list — do not start Task 1 until the user has approved this plan.
- Root pose/velocity must **never** be copied from another live env — this is the exact bug that caused yaw-drift propagation in a prior attempt (see Context below). Only `joint_pos` may be copied from a live donor.
- Every new reset path must have a safe fallback to the existing static-NPZ pool so behavior can never silently become "no RSI happened."

## Context: why this design, not a naive port

**What Humanoid-Goalkeeper (G1) actually does** (confirmed by reading `legged_gym/legged_gym/envs/base/legged_robot.py:657-681`): there is no dedicated "shadow env that just replays mocap." G1's `continue_keep` mechanism instead treats **all** `num_envs` training environments as each other's live-state pool:

```python
# legged_robot.py:669 (G1, read-only reference)
if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
    self.dof_pos[env_ids] = self.dof_pos[torch.randint(0, self.num_envs, (len(env_ids),), device=self.dof_pos.device)]
```

Only `dof_pos` is copied — root pose always resets to `base_init_state` + env origin, velocities are randomized independently, never copied. Diversity comes from the natural asynchrony of thousands of independently-progressing episodes, not from any explicit staggering logic. G1 has no per-env "which motion category" concept — any other env is a valid donor, because G1's hand-catch task doesn't need gait-specific tier matching the way SimpleGoalKeeper's foot-stepping does.

**Why we can't copy this 1:1:** SimpleGoalKeeper's whole RSI design is distance-conditioned (side × {double, triple, wide} — see `_STEM_TO_POOL` in `events.py`) because single/double/triple-step gaits are structurally different motions, unlike G1's generic reach. A random *any-other-env* donor would frequently hand a wide-target episode a standing or single-step donor pose, which defeats the purpose. This plan's design keeps G1's "copy dof_pos only, leave root alone" rule but scopes the donor pool to the same (side, tier) bucket the resetting env was just assigned.

**What went wrong last time (from `SimpleGoalKeeperObsHis/DIVERGENCE_FROM_UPSTREAM.md` and `.sessionmd/2026-06-18.md`, plus git history):**
1. Commit `92e6b39` ported live-env RSI into SimpleGoalKeeper; commit `cd9811e` (6 minutes later) flipped the reset coin-flip from batch-level back to per-env "for visual variety"; commit `6832e64` (11 minutes after *that*) deleted the whole mechanism, believing (incorrectly, per ILB's own log) that ILB didn't use it. **Lesson: the approach was abandoned by a documentation misunderstanding, not because live-env RSI itself was proven broken.** This plan is not repeating that mistake — the mechanism is being re-added deliberately, not by accident, and this file is the paper trail so it doesn't get deleted on a second misunderstanding.
2. A real, diagnosed bug (`SimpleGoalKeeperObsHis/.sessionmd/2026-06-18.md:20`, "Critical" issue #1): copying **full state** (root quat + velocity, not just joints) from a live donor propagated that donor's accumulated yaw drift into the new episode, pointing the robot the wrong way. **Fixed by copying `joint_pos` only** — this plan's `_write_live_donor_state` (Task 2) does exactly that and never touches root state.
3. The user's specific concern — "sampling from environments that were already reset" / losing the butterfly-effect diversity — was not found as a previously diagnosed-and-fixed bug, but it's a real, structurally plausible failure mode: if the donor pool includes envs that are *themselves* being reset in the same batched call, you'd hand out fresh, undiverged reset noise instead of a matured live state. **This plan closes that gap explicitly**: donors must (a) not be in the current reset batch, and (b) have been running for at least `_MIN_MATURITY_STEPS` steps since their own last reset (Task 1's `_select_live_donors`).

**What this does *not* fix:** the standing-start double/triple-step initiation gap identified in the previous debugging session (default `sgk_play` starts 100% standing; only ~20% of *wide*-tier training episodes start standing, ~7.5% of all episodes) is untouched by this change — it targets the quality/naturalness of the 80%-RSI branch, not the standing-start branch. If a future check shows the initiation gap persists after this lands, that is a separate, follow-up piece of work, not a sign this plan failed.

---

### Task 1: Pool-ID mapping + pure live-donor selection logic

**Files:**
- Modify: `pyproject.toml` (add pytest dev dependency)
- Create: `tests/__init__.py`
- Create: `tests/simple_goalkeeper/__init__.py`
- Create: `tests/simple_goalkeeper/test_live_rsi.py`
- Modify: `src/simple_goalkeeper/mdp/events.py:65-90` (add pool-ID constants near existing `_SINGLE_THRESH`/`_STEM_TO_POOL`)

**Interfaces:**
- Produces: `_POOL_KEYS: list[tuple[str, str]]` (the 6 (side, tier) keys in a fixed order), `_POOL_ID: dict[tuple[str, str], int]` (maps each key to 0-5), `_select_live_donors(pool_id: torch.Tensor, episode_steps: torch.Tensor, exclude_ids: torch.Tensor, target_pool: int, min_maturity_steps: int) -> torch.Tensor` (module-level function in `events.py`) — Task 2/3 call this directly.

- [ ] **Step 1: Add pytest as a dev dependency**

Edit `pyproject.toml`, add after the `[tool.uv]` block (after line 22):

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Sync the dev group**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv sync --group dev`
Expected: completes without error, `pytest` importable via `uv run python -c "import pytest"`.

- [ ] **Step 3: Create the test package skeleton**

Create `tests/__init__.py` (empty file).
Create `tests/simple_goalkeeper/__init__.py` (empty file).

- [ ] **Step 4: Write the failing test for pool-ID mapping and donor selection**

Create `tests/simple_goalkeeper/test_live_rsi.py`:

```python
import torch

from simple_goalkeeper.mdp.events import _POOL_KEYS, _POOL_ID, _select_live_donors


def test_pool_id_covers_six_side_tier_combinations():
    assert len(_POOL_KEYS) == 6
    assert set(_POOL_KEYS) == {
        ("left", "double"), ("left", "triple"), ("left", "wide"),
        ("right", "double"), ("right", "triple"), ("right", "wide"),
    }
    assert set(_POOL_ID.values()) == {0, 1, 2, 3, 4, 5}
    assert set(_POOL_ID.keys()) == set(_POOL_KEYS)


def test_select_live_donors_matches_pool_and_maturity():
    # 6 envs: pool assignment per env, episode age per env.
    pool_id = torch.tensor([0, 0, 1, 0, -1, 0])
    episode_steps = torch.tensor([50, 3, 50, 50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    # env 0: pool matches, mature -> eligible
    # env 1: pool matches, but only 3 steps old -> excluded (immature)
    # env 2: wrong pool -> excluded
    # env 3: pool matches, mature -> eligible
    # env 4: pool -1 (standing) -> excluded
    # env 5: pool matches, mature -> eligible
    assert sorted(donors.tolist()) == [0, 3, 5]


def test_select_live_donors_excludes_current_reset_batch():
    pool_id = torch.tensor([0, 0, 0])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([1], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert sorted(donors.tolist()) == [0, 2]


def test_select_live_donors_returns_empty_when_no_match():
    pool_id = torch.tensor([1, 2, 3])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert donors.numel() == 0
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: FAIL with `ImportError: cannot import name '_POOL_KEYS'` (or similar — the names don't exist yet).

- [ ] **Step 6: Add the pool-ID constants and `_select_live_donors` to events.py**

In `src/simple_goalkeeper/mdp/events.py`, immediately after the existing threshold block (after line 67, i.e. after `_TRIPLE_THRESH = 0.60`), add:

```python
# Live-donor RSI: fixed ordering of the 6 (side, tier) pools, used to tag
# each env with an integer "which tier is my current episode" id so other
# envs can be checked for eligibility as live-state donors. -1 = no tier
# (standing / single-range / not yet classified).
_POOL_KEYS: list[tuple[str, str]] = [
    ("left", "double"), ("left", "triple"), ("left", "wide"),
    ("right", "double"), ("right", "triple"), ("right", "wide"),
]
_POOL_ID: dict[tuple[str, str], int] = {key: i for i, key in enumerate(_POOL_KEYS)}
_POOL_ID_NONE = -1

# Live-donor RSI tuning. See docs/superpowers/plans/2026-07-01-live-env-rsi.md
# for why these exist — they close the exact gap a prior attempt lost:
# a donor must not be in the current reset batch (never simultaneously
# resetting) and must be at least this many steps past its OWN last reset
# (never "freshly reset noise" masquerading as a matured live state).
_MIN_MATURITY_STEPS = 10
_MIN_DONOR_POOL_SIZE = 16
_LIVE_RSI_WARMUP_STEPS = 500  # env.common_step_counter threshold; matches the
                              # 500-step cadence already used by reward curricula.


def _select_live_donors(
    pool_id: torch.Tensor,
    episode_steps: torch.Tensor,
    exclude_ids: torch.Tensor,
    target_pool: int,
    min_maturity_steps: int,
) -> torch.Tensor:
    """Indices of envs eligible to donate live dof_pos state for `target_pool`.

    Eligible = currently tagged with target_pool, not in the batch being
    reset right now, and has been running at least min_maturity_steps since
    its own last reset.
    """
    num_envs = pool_id.shape[0]
    exclude_mask = torch.zeros(num_envs, dtype=torch.bool, device=pool_id.device)
    if exclude_ids.numel() > 0:
        exclude_mask[exclude_ids] = True
    eligible = (
        (pool_id == target_pool)
        & (episode_steps >= min_maturity_steps)
        & (~exclude_mask)
    )
    return eligible.nonzero(as_tuple=True)[0]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tests/ src/simple_goalkeeper/mdp/events.py
git commit -m "feat(sgk): add pool-ID mapping and live-donor selection logic"
```

---

### Task 2: Live-donor state writer + per-env pool tag

**Files:**
- Modify: `src/simple_goalkeeper/mdp/events.py:120-125` (`MotionResetManager.__init__` — add running counters)
- Modify: `src/simple_goalkeeper/mdp/events.py` (add `_write_live_donor_state` method to `MotionResetManager`, after `_write_rsi_state` at line 200)
- Modify: `tests/simple_goalkeeper/test_live_rsi.py` (add fake-env test double + tests)

**Interfaces:**
- Consumes: nothing new from Task 1 beyond what's already imported.
- Produces: `MotionResetManager._write_live_donor_state(self, env, ids, donor_idx, robot) -> None`, `MotionResetManager.live_rsi_hits: int`, `MotionResetManager.live_rsi_total: int` — Task 3 calls the method and increments the counters; Task 4 reads the counters.

- [ ] **Step 1: Write the failing test using a minimal fake env/robot**

Append to `tests/simple_goalkeeper/test_live_rsi.py`:

```python
class _FakeRobotData:
    def __init__(self, joint_pos, soft_limits):
        self.joint_pos = joint_pos
        self.soft_joint_pos_limits = soft_limits


class _FakeRobot:
    def __init__(self, joint_pos, soft_limits):
        self.data = _FakeRobotData(joint_pos, soft_limits)
        self.written_pos = None
        self.written_vel = None
        self.written_ids = None

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.written_pos = joint_pos
        self.written_vel = joint_vel
        self.written_ids = env_ids


class _FakeEnv:
    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device


def test_write_live_donor_state_copies_joint_pos_only_and_clamps():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_dof = 4
    joint_pos = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],   # env 0 (will be reset — irrelevant source)
        [1.0, 1.0, 1.0, 1.0],   # env 1 (donor)
        [2.0, 2.0, 2.0, 2.0],   # env 2 (donor)
        [5.0, 5.0, 5.0, 5.0],   # env 3 (donor, out of joint limits — must clamp)
    ])
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(4, 1, 1)
    robot = _FakeRobot(joint_pos, soft_limits)
    env = _FakeEnv(num_envs=4)

    mgr = MotionResetManager()
    ids = torch.tensor([0], dtype=torch.long)
    donor_idx = torch.tensor([3], dtype=torch.long)  # force-pick the out-of-range donor

    torch.manual_seed(0)
    mgr._write_live_donor_state(env, ids, donor_idx, robot)

    assert robot.written_ids.tolist() == [0]
    # Clamped to [-3, 3] even though donor env 3 had joint_pos == 5.0.
    assert torch.allclose(robot.written_pos, torch.tensor([[3.0, 3.0, 3.0, 3.0]]))
    # Velocities are always zero for live-donor resets (never copied from the
    # donor) — this is the exact fix for the yaw/velocity-drift bug from the
    # prior attempt (SimpleGoalKeeperObsHis, 2026-06-18).
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py::test_write_live_donor_state_copies_joint_pos_only_and_clamps -v`
Expected: FAIL with `AttributeError: 'MotionResetManager' object has no attribute '_write_live_donor_state'`.

- [ ] **Step 3: Add running counters to `__init__`**

In `src/simple_goalkeeper/mdp/events.py`, modify `MotionResetManager.__init__` (currently lines 120-124):

```python
    def __init__(self) -> None:
        # Combined pool (fallback / AMP reference compatibility)
        self.frames: dict[str, torch.Tensor] = {}
        # Per-type pools keyed by (side, steps): side ∈ {left, right}, steps ∈ {single, double, triple, wide}
        self.pools: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        # Live-donor usage bookkeeping, read by live_rsi_usage_curriculum (Task 4)
        # and reset to 0 there each time it reports — purely for observability.
        self.live_rsi_hits = 0
        self.live_rsi_total = 0
```

- [ ] **Step 4: Add `_write_live_donor_state` method**

In `src/simple_goalkeeper/mdp/events.py`, immediately after `_write_rsi_state` (after line 200, before `def reset(`), add:

```python
    def _write_live_donor_state(
        self,
        env: "ManagerBasedRlEnv",
        ids: torch.Tensor,
        donor_idx: torch.Tensor,
        robot: "Entity",
    ) -> None:
        """Copy joint positions only from randomly sampled live donor envs.

        Root pose/velocity are intentionally left untouched (already set by
        the reset_base event this same reset cycle) — copying root state
        from another live env caused yaw-drift propagation in a prior
        attempt (SimpleGoalKeeperObsHis, 2026-06-18 session notes). Joint
        velocities are zeroed rather than copied for the same reason: a
        donor's current velocity reflects whatever it's mid-doing right now,
        which may not be physically consistent with the NEW episode's ball
        trajectory.
        """
        n = len(ids)
        picks = donor_idx[torch.randint(0, donor_idx.numel(), (n,), device=env.device)]
        joint_pos = robot.data.joint_pos[picks].clone()
        joint_vel = torch.zeros_like(joint_pos)
        limits = robot.data.soft_joint_pos_limits[ids]
        joint_pos.clamp_(limits[..., 0], limits[..., 1])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/simple_goalkeeper/mdp/events.py tests/simple_goalkeeper/test_live_rsi.py
git commit -m "feat(sgk): add live-donor joint-state writer (dof_pos only, no root/vel copy)"
```

---

### Task 3: Wire live-donor sampling into `MotionResetManager.reset()`

**Files:**
- Modify: `src/simple_goalkeeper/mdp/events.py:202-291` (the `reset()` method)

**Interfaces:**
- Consumes: `_POOL_ID`, `_select_live_donors`, `_MIN_MATURITY_STEPS`, `_MIN_DONOR_POOL_SIZE`, `_LIVE_RSI_WARMUP_STEPS` (Task 1), `self._write_live_donor_state` (Task 2), `env.episode_length_buf` and `env.common_step_counter` (native mjlab `ManagerBasedRlEnv` attributes — confirmed present at `mjlab/envs/manager_based_rl_env.py:231-232,430-431,590`, no new plumbing needed).
- Produces: `env._rsi_pool_id: torch.Tensor` (per-env tier tag, lazily created) — nothing outside this method reads it, but Task 4's curriculum term reads `self.live_rsi_hits`/`self.live_rsi_total` which this task increments.

This task has no new pure-function unit tests of its own (the logic being added is already covered by Task 1/2's unit tests) — it is wiring, verified by an integration test using the fake-env harness from Task 2, extended to a fuller fake `Entity`/scene.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/simple_goalkeeper/test_live_rsi.py`:

```python
def test_reset_prefers_live_donor_when_pool_is_large_and_warm(monkeypatch):
    from simple_goalkeeper.mdp import events as events_mod
    from simple_goalkeeper.mdp.events import MotionResetManager

    monkeypatch.setattr(events_mod, "_MIN_DONOR_POOL_SIZE", 2)
    monkeypatch.setattr(events_mod, "_LIVE_RSI_WARMUP_STEPS", 0)
    monkeypatch.setattr(events_mod, "_MIN_MATURITY_STEPS", 0)

    num_envs = 4
    num_dof = 2
    joint_pos = torch.zeros(num_envs, num_dof)
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    robot = _FakeRobot(joint_pos, soft_limits)
    robot.data.default_joint_pos = torch.zeros(num_envs, num_dof)
    robot.write_root_link_pose_to_sim = lambda *a, **k: None
    robot.write_root_link_velocity_to_sim = lambda *a, **k: None

    class _Scene:
        def __init__(self, robot):
            self._robot = robot
            self.env_origins = torch.zeros(num_envs, 3)

        def __getitem__(self, name):
            return self._robot

    env = _FakeEnv(num_envs=num_envs)
    env.scene = _Scene(robot)
    env.common_step_counter = 999
    env.episode_length_buf = torch.tensor([0, 100, 100, 100])  # env 0 is resetting now
    env._rsi_cross_y = torch.tensor([0.9, 0.0, 0.0, 0.0])  # env 0 wants a "wide" target
    env._rsi_pool_id = torch.tensor([-1, 2, 2, -1])  # envs 1,2 already tagged "left wide" (pool 2)

    mgr = MotionResetManager()
    mgr.frames = {"joint_pos": torch.zeros(1, num_dof), "joint_vel": torch.zeros(1, num_dof),
                  "root_pos": torch.zeros(1, 3), "root_quat": torch.zeros(1, 4),
                  "root_lin_vel": torch.zeros(1, 3), "root_ang_vel": torch.zeros(1, 3)}

    env_ids = torch.tensor([0], dtype=torch.int32)
    torch.manual_seed(0)
    mgr.reset(env, env_ids, rsi_fraction=1.0)

    assert robot.written_ids is not None
    assert robot.written_ids.tolist() == [0]
    assert mgr.live_rsi_hits == 1
    assert mgr.live_rsi_total == 1
    # env 0 gets tagged into the pool it was just assigned (left, wide) = 2.
    assert env._rsi_pool_id[0].item() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py::test_reset_prefers_live_donor_when_pool_is_large_and_warm -v`
Expected: FAIL (either `AttributeError` on `env._rsi_pool_id` not being read/written yet, or the assertion on `mgr.live_rsi_hits` failing since it's never incremented).

- [ ] **Step 3: Rewrite `reset()` to wire in live-donor sampling**

Replace the full `reset()` method in `src/simple_goalkeeper/mdp/events.py` (lines 202-291) with:

```python
    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        rsi_fraction: float = 0.8,
    ) -> None:
        """80% distance-conditioned RSI + 20% HOME_KEYFRAME standing.

        reset_ball fires first, so ball pos/vel reflect the NEW episode trajectory.
        We predict the ball's goal-line crossing Y and pick:
          left/right — from sign of crossing_y
          tier       — from |crossing_y| vs thresholds:
            < 0.20 m  → standing pose (no RSI)
            0.20–0.40 → double  (MediumStep, SafeMedium)
            0.40–0.60 → triple  (FarStep, SafeFar)
            ≥ 0.60    → wide    (DoubleStep, TripleStep)

        For double/triple/wide tiers, each reset first tries to copy joint
        positions from another env currently mid-episode in the SAME tier
        (live-donor RSI — see docs/superpowers/plans/2026-07-01-live-env-rsi.md).
        Falls back to the static NPZ pool when too few live donors exist yet
        (early training / small eval runs / the startup warmup window).

        20% standing resets prevent the AMP dive→RSI transition artefact and
        ensure the policy learns to save from a standing start (deployment scenario).
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        if not hasattr(env, "_rsi_pool_id"):
            env._rsi_pool_id = torch.full((env.num_envs,), _POOL_ID_NONE, dtype=torch.long, device=env.device)

        robot: Entity = env.scene[asset_cfg.name]
        n = len(env_ids)

        use_rsi     = torch.rand(n, device=env.device) < rsi_fraction
        rsi_local   = use_rsi.nonzero(as_tuple=True)[0]
        stand_local = (~use_rsi).nonzero(as_tuple=True)[0]

        # Collect extra env_ids that are redirected from RSI → standing.
        extra_stand_ids: list[torch.Tensor] = []

        # ── 80 %: distance-conditioned RSI ────────────────────────────────
        if len(rsi_local) > 0:
            ids_rsi = env_ids[rsi_local]

            # Read the crossing Y stored by reset_ball_rolling (fires before this event).
            # Using the stored value avoids the entity data-buffer lag: ball.data.*
            # is only refreshed after the next physics step, so reading it here would
            # give the PREVIOUS episode's ball state.
            cross_y = getattr(env, "_rsi_cross_y", None)
            if cross_y is None:
                # Fallback for the very first reset before reset_ball has ever fired.
                self._write_rsi_state(env, ids_rsi, self.frames, robot)
                env._rsi_pool_id[ids_rsi.long()] = _POOL_ID_NONE
            else:
                cy     = cross_y[ids_rsi]
                is_left = cy > 0
                abs_cy  = cy.abs()

                is_single = abs_cy < _SINGLE_THRESH
                is_double = (abs_cy >= _SINGLE_THRESH) & (abs_cy < _DOUBLE_THRESH)
                is_triple = (abs_cy >= _DOUBLE_THRESH) & (abs_cy < _TRIPLE_THRESH)
                is_wide   = abs_cy >= _TRIPLE_THRESH

                # Single-range: ball arrives near centre — start standing so the
                # policy reacts freely rather than committing to a lateral step.
                single_local = is_single.nonzero(as_tuple=True)[0]
                if len(single_local) > 0:
                    extra_stand_ids.append(ids_rsi[single_local])
                    env._rsi_pool_id[ids_rsi[single_local].long()] = _POOL_ID_NONE

                for (side, steps), mask in [
                    (("left",  "double"), is_left  & is_double),
                    (("left",  "triple"), is_left  & is_triple),
                    (("left",  "wide"),   is_left  & is_wide),
                    (("right", "double"), ~is_left & is_double),
                    (("right", "triple"), ~is_left & is_triple),
                    (("right", "wide"),   ~is_left & is_wide),
                ]:
                    local_ids = mask.nonzero(as_tuple=True)[0]
                    if len(local_ids) == 0:
                        continue
                    target_ids = ids_rsi[local_ids]
                    target_pool = _POOL_ID[(side, steps)]

                    used_live = False
                    if env.common_step_counter >= _LIVE_RSI_WARMUP_STEPS:
                        donor_idx = _select_live_donors(
                            env._rsi_pool_id, env.episode_length_buf, ids_rsi.long(),
                            target_pool=target_pool, min_maturity_steps=_MIN_MATURITY_STEPS,
                        )
                        if donor_idx.numel() >= _MIN_DONOR_POOL_SIZE:
                            self._write_live_donor_state(env, target_ids.long(), donor_idx, robot)
                            used_live = True

                    if not used_live:
                        pool = self.pools.get((side, steps), self.frames)
                        self._write_rsi_state(env, target_ids, pool, robot)

                    env._rsi_pool_id[target_ids.long()] = target_pool
                    self.live_rsi_total += len(target_ids)
                    if used_live:
                        self.live_rsi_hits += len(target_ids)

        # ── HOME_KEYFRAME standing pose (20% random + all single-range) ───
        ids_stand_parts: list[torch.Tensor] = []
        if len(stand_local) > 0:
            ids_stand_parts.append(env_ids[stand_local])
            env._rsi_pool_id[env_ids[stand_local].long()] = _POOL_ID_NONE
        ids_stand_parts.extend(extra_stand_ids)
        if ids_stand_parts:
            ids_stand = torch.cat(ids_stand_parts)
            ns = len(ids_stand)
            home_pos = robot.data.default_joint_pos[ids_stand]
            home_vel = torch.zeros(ns, home_pos.shape[-1], device=env.device)
            robot.write_joint_state_to_sim(home_pos, home_vel, env_ids=ids_stand)
            # Root state already correct from reset_base.
```

Note the `exclude_ids=ids_rsi.long()` passed to `_select_live_donors`: this excludes the **entire current reset batch**, not just `target_ids` — the exact guard against "sampling from environments that were already reset" the user was worried about, since none of the envs being reset in this call can be a valid donor for each other.

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simple_goalkeeper/mdp/events.py
git commit -m "feat(sgk): wire live-donor RSI into MotionResetManager.reset() with NPZ fallback"
```

---

### Task 4: Observability — log the live-donor hit rate

**Files:**
- Modify: `src/simple_goalkeeper/mdp/events.py` (add `live_rsi_usage_curriculum` class, after `correct_foot_save_curriculum`)
- Modify: `src/simple_goalkeeper/mdp/__init__.py:3` (export it)
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` (register it in the curriculum dict, near the other curriculum registrations around line 192)

**Interfaces:**
- Consumes: `MotionResetManager.get().live_rsi_hits` / `.live_rsi_total` (Task 2/3).
- Produces: `Episode/Curriculum/live_rsi_usage/live_rsi_fraction` scalar in TensorBoard/wandb.

- [ ] **Step 1: Write the failing test**

Append to `tests/simple_goalkeeper/test_live_rsi.py`:

```python
def test_live_rsi_usage_curriculum_reports_and_resets_counters():
    from simple_goalkeeper.mdp.events import MotionResetManager, live_rsi_usage_curriculum

    mgr = MotionResetManager.get()
    mgr.live_rsi_hits = 3
    mgr.live_rsi_total = 4

    result = live_rsi_usage_curriculum(env=None, env_ids=None)

    assert torch.isclose(result["live_rsi_fraction"], torch.tensor(0.75))
    assert mgr.live_rsi_hits == 0
    assert mgr.live_rsi_total == 0


def test_live_rsi_usage_curriculum_handles_zero_total():
    from simple_goalkeeper.mdp.events import MotionResetManager, live_rsi_usage_curriculum

    mgr = MotionResetManager.get()
    mgr.live_rsi_hits = 0
    mgr.live_rsi_total = 0

    result = live_rsi_usage_curriculum(env=None, env_ids=None)

    assert torch.isclose(result["live_rsi_fraction"], torch.tensor(0.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -k live_rsi_usage -v`
Expected: FAIL with `ImportError: cannot import name 'live_rsi_usage_curriculum'`.

- [ ] **Step 3: Add the function**

In `src/simple_goalkeeper/mdp/events.py`, after the `correct_foot_save_curriculum` class (after its closing `return {"weight": torch.tensor(float(new_weight))}` line), add:

```python
def live_rsi_usage_curriculum(env: "ManagerBasedRlEnv", env_ids: torch.Tensor, **kwargs) -> dict:
    """Logging-only curriculum term: reports the live-donor RSI hit rate.

    Not a real curriculum — doesn't change any reward weight. Piggybacks on
    the curriculum manager's reset-time logging so the live-vs-NPZ-fallback
    split for double/triple/wide-tier resets shows up in TensorBoard/wandb
    under Episode/Curriculum/live_rsi_usage/live_rsi_fraction, instead of
    being invisible the way per-pool RSI branching was before this change.
    """
    mgr = MotionResetManager.get()
    total = max(mgr.live_rsi_total, 1)
    fraction = mgr.live_rsi_hits / total
    mgr.live_rsi_hits = 0
    mgr.live_rsi_total = 0
    return {"live_rsi_fraction": torch.tensor(float(fraction))}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/simple_goalkeeper/test_live_rsi.py -v`
Expected: 8 passed.

- [ ] **Step 5: Export it from `mdp/__init__.py`**

Modify `src/simple_goalkeeper/mdp/__init__.py:3`, change:

```python
from .events import init_motion_loader, reset_from_motion_data, reset_ball_local_frame, reset_ball_global_frame, reset_ball_rolling, tick_catchstep, ball_difficulty_curriculum, reward_curriculum_ep_len, correct_foot_save_curriculum, ball_exit_termination, sharpforce_termination, shank_height_termination
```

to:

```python
from .events import init_motion_loader, reset_from_motion_data, reset_ball_local_frame, reset_ball_global_frame, reset_ball_rolling, tick_catchstep, ball_difficulty_curriculum, reward_curriculum_ep_len, correct_foot_save_curriculum, live_rsi_usage_curriculum, ball_exit_termination, sharpforce_termination, shank_height_termination
```

- [ ] **Step 6: Register it as a curriculum term**

In `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`, after the `for _name, _base in (...)` loop that registers `correct_foot_save_curriculum` terms (after line 192, still inside the `if not play:` block), add:

```python
        cfg.curriculum["live_rsi_usage"] = CurriculumTermCfg(
            func=gk_mdp.live_rsi_usage_curriculum,
            params={},
        )
```

- [ ] **Step 7: Verify the whole test suite still passes**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run pytest tests/ -v`
Expected: 8 passed.

- [ ] **Step 8: Commit**

```bash
git add src/simple_goalkeeper/mdp/events.py src/simple_goalkeeper/mdp/__init__.py src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py tests/simple_goalkeeper/test_live_rsi.py
git commit -m "feat(sgk): log live-donor RSI hit rate to TensorBoard/wandb"
```

---

### Task 5: Smoke-test with a real (small) training + play run

**Files:** none modified — verification only.

- [ ] **Step 1: Short dry-run training to confirm no crashes and that live RSI engages**

Run: `cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && timeout 600 uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 2048 2>&1 | tee /tmp/sgk_live_rsi_smoketest.log`

Let it run past iteration ~25 (with `num_steps_per_env=24`, `common_step_counter` crosses the `_LIVE_RSI_WARMUP_STEPS=500` threshold at iteration ~21). Then Ctrl-C or let the timeout stop it.

Expected: no exceptions/tracebacks. `grep "live_rsi_fraction" /tmp/sgk_live_rsi_smoketest.log` — if wandb console mirroring is on you'll see it in the periodic log dump; otherwise open the run in wandb/TensorBoard and confirm `Episode/Curriculum/live_rsi_usage/live_rsi_fraction` is present and > 0 for iterations after ~21 (it will be 0 for the warmup iterations before that — that's correct, not a bug).

- [ ] **Step 2: Confirm behavior is unchanged for standing-start play (regression check)**

Run: `uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --checkpoint-file logs/rsl_rl/simple_goalkeeper/2026-07-01_12-24-06_phase1/model_5000.pt --difficulty 1`

Expected: same behavior as before this change (default play never touches live-donor code — `common_step_counter` stays near 0 for a short play session, so it's always below `_LIVE_RSI_WARMUP_STEPS` and falls back to the static NPZ path exactly like today).

- [ ] **Step 3: Confirm RSI-forced play still works and pool tagging doesn't crash with few envs**

Run: `uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --checkpoint-file logs/rsl_rl/simple_goalkeeper/2026-07-01_12-24-06_phase1/model_5000.pt --difficulty 1 --rsi True --num-envs 8`

Expected: runs without error. With only 8 envs, `_MIN_DONOR_POOL_SIZE=16` can never be satisfied, so this always uses the static NPZ fallback too — this is correct, not a bug, and worth explicitly noting if asked "why didn't it use live donors here."

- [ ] **Step 4: No commit for this task** — it's a verification checkpoint. If any step fails, stop and return to the relevant earlier task rather than patching around the smoke test.

---

### Task 6: Documentation

**Files:**
- Modify: `SimpleGoalKeeper/CLAUDE.md` (Divergences from G1 Upstream table)

- [ ] **Step 1: Add a divergence table row**

In `SimpleGoalKeeper/CLAUDE.md`, add a new row to the "Divergences from G1 Upstream" table:

```markdown
| RSI donor source (double/triple/wide tiers) | `continue_keep`: copy `dof_pos` from a random *other* live training env (any env, no tier matching) | **Tier-scoped live-donor RSI**: copy `dof_pos` from a random live env currently in the SAME (side, tier) bucket; falls back to the static NPZ pool when fewer than 16 eligible donors exist or within the first 500 global steps | **2026-07-01:** G1 doesn't need tier matching (generic hand reach); SimpleGoalKeeper's single/double/triple-step gaits are structurally different motions, so an untiered donor would frequently hand a wide-target episode a standing/single-step pose. Root pose/velocity are never copied from a donor (G1 doesn't either) — a prior attempt (SimpleGoalKeeperObsHis, 2026-06-18) copied full root state and caused yaw-drift propagation; fixed here by design, not as an afterthought. See `docs/superpowers/plans/2026-07-01-live-env-rsi.md`. |
```

- [ ] **Step 2: Commit**

```bash
git add SimpleGoalKeeper/CLAUDE.md
git commit -m "docs(sgk): log live-donor RSI divergence from G1's continue_keep"
```

---

## Self-Review

**Spec coverage:** user asked for (1) a live-env RSI method like Humanoid-Goalkeeper's, ported with awareness of the two documented prior-attempt failure modes (yaw-drift from copying full state, and losing diversity by sampling already-reset envs) — covered by Tasks 1-3, with the design explicitly closing both gaps (Task 2 never copies root/velocity; Task 1/3's `exclude_ids` + maturity gate stops same-batch/freshly-reset donors). (2) Plan it out, not implement yet — this document is the plan; Task 5/6 round it out with verification and the mandatory CLAUDE.md documentation the project's own rules require.

**Placeholder scan:** no TBD/TODO markers; every step has runnable code or an exact command with expected output.

**Type consistency:** `_select_live_donors` signature (`pool_id, episode_steps, exclude_ids, target_pool, min_maturity_steps`) is identical between its Task 1 definition and Task 3's call site. `_write_live_donor_state(self, env, ids, donor_idx, robot)` matches between Task 2's definition and Task 3's call. `live_rsi_usage_curriculum(env, env_ids, **kwargs)` matches the existing `correct_foot_save_curriculum.__call__` signature convention used elsewhere in the file.
