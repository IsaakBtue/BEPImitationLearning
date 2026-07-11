import pytest
import torch


class _FakeRobotData:
    def __init__(self, joint_pos, default_joint_pos, soft_limits, hard_limits):
        self.joint_pos = joint_pos
        self.default_joint_pos = default_joint_pos
        self.soft_joint_pos_limits = soft_limits
        self.joint_pos_limits = hard_limits


class _PoisonedSoftLimitsData(_FakeRobotData):
    """soft_joint_pos_limits raises if ever read — G1's continue_keep has no
    soft-limit concept at all; reset() must never touch it (only the hard
    joint_pos_limits, and only in the else/default-pose branch)."""

    def __init__(self, joint_pos, default_joint_pos, hard_limits):
        self.joint_pos = joint_pos
        self.default_joint_pos = default_joint_pos
        self.joint_pos_limits = hard_limits
        self._soft = None

    @property
    def soft_joint_pos_limits(self):
        raise AssertionError(
            "reset() read soft_joint_pos_limits — G1's continue_keep has no "
            "soft-limit concept; this attribute must never be touched."
        )

    @soft_joint_pos_limits.setter
    def soft_joint_pos_limits(self, value):
        self._soft = value


class _FakeRobot:
    def __init__(self, data):
        self.data = data
        self.written_pos = None
        self.written_vel = None
        self.written_ids = None
        self.write_joint_state_calls = 0
        self.root_pose_calls = 0
        self.root_velocity_calls = 0

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.written_pos = joint_pos
        self.written_vel = joint_vel
        self.written_ids = env_ids
        self.write_joint_state_calls += 1

    def write_root_link_pose_to_sim(self, *a, **k):
        self.root_pose_calls += 1

    def write_root_link_velocity_to_sim(self, *a, **k):
        self.root_velocity_calls += 1


class _Scene:
    def __init__(self, robot):
        self._robot = robot

    def __getitem__(self, name):
        return self._robot


class _FakeEnv:
    def __init__(self, num_envs, robot, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = _Scene(robot)


def _make_env(num_envs, num_dof, joint_values, hard_bound=10.0):
    """joint_values: (num_envs, num_dof) tensor of each env's CURRENT live dof_pos."""
    default_joint_pos = torch.zeros(num_envs, num_dof)
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    hard_limits = torch.tensor([[-hard_bound, hard_bound]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    data = _FakeRobotData(joint_values, default_joint_pos, soft_limits, hard_limits)
    robot = _FakeRobot(data)
    env = _FakeEnv(num_envs=num_envs, robot=robot)
    return env, robot


def _make_env_poisoned_soft_limits(num_envs, num_dof, joint_values, hard_bound=10.0):
    """Same as _make_env, but reading soft_joint_pos_limits raises immediately."""
    default_joint_pos = torch.zeros(num_envs, num_dof)
    hard_limits = torch.tensor([[-hard_bound, hard_bound]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    data = _PoisonedSoftLimitsData(joint_values, default_joint_pos, hard_limits)
    robot = _FakeRobot(data)
    env = _FakeEnv(num_envs=num_envs, robot=robot)
    return env, robot


def test_reset_root_state_is_never_touched():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.arange(num_envs * num_dof, dtype=torch.float32).reshape(num_envs, num_dof)
    env, robot = _make_env(num_envs, num_dof, joint_values)

    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)
    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)

    # Root pose/velocity are handled by the separate reset_base event, matching
    # G1's continue_keep, which only ever touches dof_pos/dof_vel.
    assert robot.root_pose_calls == 0
    assert robot.root_velocity_calls == 0


def test_reset_dof_vel_always_zero_on_both_branches():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.ones(num_envs, num_dof) * 2.5
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)   # forces the donor-copy branch
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=0.0)   # forces the default-pose branch
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))


def test_reset_rsi_fraction_1_copies_dof_pos_from_another_live_env_without_clamping(monkeypatch):
    """G1's torch.clip only appears in the OTHER (else) branch — the
    continue_keep donor-copy branch never clips, since a live env's dof_pos
    is already physically valid. Use a donor value outside even the FAKE
    hard limit to prove no clamping of any kind happens here."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 3
    # Env 1 (the donor torch.randint is forced to pick) is outside both the
    # soft [-3, 3] AND the fake hard [-3, 3] bound used by _make_env's default.
    joint_values = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    env, robot = _make_env(num_envs, num_dof, joint_values, hard_bound=3.0)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.ones(1, dtype=torch.long))
    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)  # tier_rsi_fraction=0.0, live_rsi_fraction=1.0 -> always the donor branch

    assert robot.written_ids.tolist() == [0]
    # Donor (env 1) had joint_pos == 5.0 — copied verbatim, no clamp applied.
    assert torch.allclose(robot.written_pos, torch.tensor([[5.0, 5.0, 5.0]]))


def test_reset_rsi_fraction_0_randomizes_around_default_pose_and_clips_to_hard_limits(monkeypatch):
    """G1's active config (g1_29_config.py: randomize_initial_joint_pos=True,
    initial_joint_pos_scale=[0.5, 1.5], initial_joint_pos_offset=[-0.1, 0.1])
    scales+offsets the default pose per-joint, then clips to the HARD dof
    limits — not a flat copy of default_joint_pos, and not the soft limits."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.ones(num_envs, num_dof) * 99.0  # would be obviously wrong if ever copied
    env, robot = _make_env(num_envs, num_dof, joint_values, hard_bound=1.0)
    robot.data.default_joint_pos = torch.ones(num_envs, num_dof) * 2.0
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    # scale=1.5, offset=0.1 -> 2.0*1.5+0.1 = 3.1, clipped to hard bound [-1, 1] -> 1.0
    monkeypatch.setattr(
        "simple_goalkeeper.mdp.events.sample_uniform",
        lambda lo, hi, shape, device: torch.full(shape, hi),
    )
    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=0.0)  # rsi_fraction=0.0 -> always the default-pose branch

    assert robot.written_ids.tolist() == [0]
    assert torch.allclose(robot.written_pos, torch.ones(1, num_dof))


def test_reset_donor_pool_is_not_tier_restricted_and_not_batch_excluded(monkeypatch):
    """No side/tier matching (G1 has none) and no exclusion of the current
    reset batch (G1 samples torch.randint(0, num_envs, ...) unconditionally,
    so a resetting env CAN be sampled as its own or another resetting env's
    donor — this test locks in that this project matches that, deliberately,
    rather than silently reintroducing tier/exclusion logic)."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 1
    joint_values = torch.tensor([[1.0], [2.0], [3.0]])
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0, 1, 2], dtype=torch.int32)  # the WHOLE population resets together

    torch.manual_seed(0)
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.zeros(len(env_ids), dtype=torch.long))
    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)

    # torch.randint patched to always return index 0 -> every env, including
    # env 0 reading from itself, copies joint_values[0] == 1.0. No exclusion,
    # no tier filter — the donor pool is genuinely env.num_envs wide.
    assert torch.allclose(robot.written_pos, torch.full((3, 1), 1.0))


def test_reset_single_coin_flip_covers_the_whole_batch_at_once(monkeypatch):
    """G1 draws ONE torch.rand(1) per reset() call, not one per env — so
    every env in this call must land on the SAME branch. Only the coin-flip
    call (shape (1,)) is forced; other torch.rand uses (e.g. inside
    sample_uniform's scale/offset randomization) fall through to the real
    implementation so they aren't broken by this patch."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 5, 1
    joint_values = torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1) + 100.0
    env, robot = _make_env(num_envs, num_dof, joint_values, hard_bound=1000.0)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    real_rand = torch.rand

    def _forced_coin_flip(value):
        def _fake(*a, **k):
            if a and a[0] == 1:
                return torch.tensor([value])
            return real_rand(*a, **k)
        return _fake

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


def test_reset_three_way_split_statistics(monkeypatch):
    """Over many independent reset() calls the branch rates must converge to
    tier=0.1 / live=0.7 / standing=0.2 (2026-07-04 rebalance; one draw per
    call, not per env)."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 1
    joint_values = torch.ones(num_envs, num_dof) * 7.0  # donor value, distinguishable
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    counts = {"tier": 0, "live": 0, "stand": 0}
    monkeypatch.setattr(
        mgr, "_tier_npz_reset",
        lambda env, ids, robot: counts.__setitem__("tier", counts["tier"] + 1),
    )

    torch.manual_seed(42)
    n_trials = 3000
    for _ in range(n_trials):
        tier_before = counts["tier"]
        mgr.reset(env, env_ids)
        if counts["tier"] > tier_before:
            continue
        # Donor value is 7.0; the default-pose branch is 0.0 * scale +- 0.1.
        if abs(robot.written_pos.item()) > 1.0:
            counts["live"] += 1
        else:
            counts["stand"] += 1

    for branch, expected in [("tier", 0.1), ("live", 0.7), ("stand", 0.2)]:
        rate = counts[branch] / n_trials
        assert abs(rate - expected) < 0.05, (
            f"{branch} rate {rate} far from expected {expected}"
        )


def test_reset_donor_sampling_is_roughly_uniform_over_full_population():
    """G1 samples torch.randint(0, num_envs, ...) uniformly — no env should be
    systematically favored or excluded as a donor over many draws."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 5, 1
    joint_values = torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1)
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([2], dtype=torch.int32)  # always reset the same single env

    torch.manual_seed(7)
    n_trials = 2000
    counts = torch.zeros(num_envs)
    for _ in range(n_trials):
        mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)  # always the donor branch
        donor_value = int(robot.written_pos.item())
        counts[donor_value] += 1

    expected = n_trials / num_envs
    for i in range(num_envs):
        assert abs(counts[i].item() - expected) < expected * 0.35, (
            f"env {i} sampled {counts[i].item()} times, expected ~{expected}"
        )


def test_reset_env_ids_none_resets_the_full_population():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 2
    joint_values = torch.ones(num_envs, num_dof) * 4.0
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()

    mgr.reset(env, None, tier_rsi_fraction=0.0, live_rsi_fraction=0.0)  # default-pose branch, full population

    assert robot.written_ids.tolist() == [0, 1, 2]
    # default_joint_pos is 0 here, so only the +-0.1 offset survives —
    # unmistakably not the donor value of 4.0.
    assert (robot.written_pos.abs() < 0.2).all()


def test_reset_empty_env_ids_is_a_no_op():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 2
    joint_values = torch.ones(num_envs, num_dof) * 4.0
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()

    mgr.reset(env, torch.tensor([], dtype=torch.int32), tier_rsi_fraction=0.0, live_rsi_fraction=1.0)

    assert robot.written_ids is None


def test_reset_single_env_population_donor_pool_is_itself():
    """num_envs=1 is the degenerate case of G1's torch.randint(0, num_envs, ...):
    the only possible donor is the env itself. Must not crash or raise."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 1, 2
    joint_values = torch.tensor([[1.5, -1.5]])
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)

    assert robot.written_ids.tolist() == [0]
    assert torch.allclose(robot.written_pos, torch.tensor([[1.5, -1.5]]))


def test_reset_from_motion_data_passes_the_three_way_split_to_reset(monkeypatch):
    """Regression guard (the 2026-07-01 audit found this wrapper silently
    overriding reset()'s fractions once before): pin the arguments the
    registered training event actually forwards — 0.1 tier / 0.7 live
    (rebalanced 2026-07-04 from 0.5/0.3, see
    docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md)."""
    import simple_goalkeeper.mdp.events as events_mod

    captured = {}

    class _StubMgr:
        def init(self, env):
            pass

        def reset(self, env, env_ids, asset_cfg, tier_rsi_fraction, live_rsi_fraction):
            captured["tier"] = tier_rsi_fraction
            captured["live"] = live_rsi_fraction

    monkeypatch.setattr(events_mod.MotionResetManager, "get", staticmethod(lambda: _StubMgr()))

    events_mod.reset_from_motion_data(env=object(), env_ids=None)

    assert captured["tier"] == 0.1
    assert captured["live"] == 0.7


@pytest.mark.parametrize("live_fraction", [0.2, 0.5, 0.8, 0.95])
def test_reset_matches_live_fraction_statistically_at_multiple_values(live_fraction):
    """With tier_rsi_fraction=0.0 the donor rate must track live_rsi_fraction
    generally, not just at the default — pins the partition formula itself."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 1
    joint_values = torch.ones(num_envs, num_dof) * 50.0
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    torch.manual_seed(123)
    n_trials = 2000
    donor_hits = 0
    for _ in range(n_trials):
        mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=live_fraction)
        if abs(robot.written_pos.item()) > 1.0:
            donor_hits += 1

    rate = donor_hits / n_trials
    assert abs(rate - live_fraction) < 0.05, (
        f"donor-branch rate {rate} far from expected {live_fraction}"
    )


def test_reset_write_joint_state_called_exactly_once_per_reset_call():
    """Whichever branch fires, write_joint_state_to_sim must be called
    exactly once — no double-write, no branch calling it twice."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 1
    joint_values = torch.ones(num_envs, num_dof) * 5.0
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)
    assert robot.write_joint_state_calls == 1

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=0.0)
    assert robot.write_joint_state_calls == 2


def test_reset_donor_branch_never_reads_soft_joint_pos_limits():
    """G1's continue_keep branch has no soft-limit concept at all — confirm
    reset() genuinely never touches robot.data.soft_joint_pos_limits here,
    not just that the end result happens to look unclamped."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 1
    joint_values = torch.ones(num_envs, num_dof) * 5.0
    env, robot = _make_env_poisoned_soft_limits(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=1.0)  # donor branch — must not raise

    assert robot.written_ids.tolist() == [0]


def test_reset_default_pose_branch_never_reads_soft_joint_pos_limits():
    """G1's randomize_initial_joint_pos branch clips to the HARD dof limits,
    never the soft ones — confirm the else branch doesn't read them either."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 1
    joint_values = torch.ones(num_envs, num_dof) * 5.0
    env, robot = _make_env_poisoned_soft_limits(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=0.0, live_rsi_fraction=0.0)  # default-pose branch — must not raise

    assert robot.written_ids.tolist() == [0]


# ── Tier-routed NPZ RSI branch (reinstated 2026-07-03) ──────────────────────


def _make_tier_env(cross_y_values):
    """Fake env with a stored crossing-Y buffer and sentinel pools."""
    num_envs = len(cross_y_values)
    num_dof = 2
    joint_values = torch.zeros(num_envs, num_dof)
    env, robot = _make_env(num_envs, num_dof, joint_values)
    env._rsi_cross_y = torch.tensor(cross_y_values, dtype=torch.float32)
    return env, robot


def _mgr_with_recording_pools():
    from simple_goalkeeper.mdp.events import MotionResetManager

    mgr = MotionResetManager()
    for side in ("left", "right"):
        for steps in ("double", "triple", "wide"):
            mgr.pools[(side, steps)] = {"name": f"{side}_{steps}"}
    mgr.frames = {"name": "combined"}

    calls = []
    mgr._write_rsi_state = lambda env, ids, pool, robot: calls.append(
        (pool["name"], ids.tolist())
    )
    return mgr, calls


def test_tier_branch_routes_by_crossing_y_side_and_magnitude():
    """left = cy > 0; |cy| tiers: <0.20 standing, 0.20-0.40 double,
    0.40-0.50 triple, >=0.50 wide (the DoubleStep/TripleStep pool).
    2026-07-05: wide boundary lowered 0.60 -> 0.50, in sync with
    mdp.rewards._get_reach_target_y's wide_threshold."""
    env, robot = _make_tier_env([0.7, -0.7, 0.3, -0.45, 0.1, 1.05])
    mgr, calls = _mgr_with_recording_pools()
    env_ids = torch.arange(6, dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=1.0, live_rsi_fraction=0.0)

    routed = {pool: ids for pool, ids in calls}
    assert routed["left_wide"] == [0, 5]    # 0.7 and 1.05 -> wide, left
    assert routed["right_wide"] == [1]      # -0.7 -> wide, right
    assert routed["left_double"] == [2]     # 0.3 -> double, left
    assert routed["right_triple"] == [3]    # -0.45 -> triple, right
    # env 4 (|cy| = 0.1 < 0.20): standing HOME pose, written directly.
    assert robot.written_ids.tolist() == [4]
    assert torch.allclose(robot.written_pos, torch.zeros(1, 2))
    assert torch.allclose(robot.written_vel, torch.zeros(1, 2))


def test_tier_branch_wide_pool_serves_all_far_crossings_up_to_y_end_max():
    """Every crossing the widened +-1.1 y_end can produce beyond 0.50 lands in
    the wide pool — the DoubleStep/TripleStep frames the tier split exists for."""
    env, robot = _make_tier_env([0.61, 0.9, 1.1, -0.61, -0.9, -1.1])
    mgr, calls = _mgr_with_recording_pools()
    env_ids = torch.arange(6, dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=1.0, live_rsi_fraction=0.0)

    routed = {pool: ids for pool, ids in calls}
    assert routed["left_wide"] == [0, 1, 2]
    assert routed["right_wide"] == [3, 4, 5]


def test_tier_branch_falls_back_to_combined_pool_without_cross_y():
    """Before reset_ball has ever fired there is no crossing buffer — the
    whole batch resets from the combined pool instead of crashing."""
    num_envs, num_dof = 3, 2
    env, robot = _make_env(num_envs, num_dof, torch.zeros(num_envs, num_dof))
    assert not hasattr(env, "_rsi_cross_y")
    mgr, calls = _mgr_with_recording_pools()
    env_ids = torch.arange(3, dtype=torch.int32)

    mgr.reset(env, env_ids, tier_rsi_fraction=1.0, live_rsi_fraction=0.0)

    assert calls == [("combined", [0, 1, 2])]


def test_reset_ball_rolling_stores_predicted_crossing_y():
    """reset_ball_rolling must store the goal-line crossing Y (x=0 lerp of the
    spawn->target segment) into env._rsi_cross_y for the tier branch. Verified
    geometrically: cross_y = y_start + (y_end - y_start) * x_start / (x_start + 0.3).
    Uses the real event on a fake ball entity."""
    from simple_goalkeeper.mdp import events as events_mod

    class _BallData:
        pass

    class _Ball:
        def __init__(self):
            self.data = _BallData()

        def write_root_link_pose_to_sim(self, *a, **k):
            pass

        def write_root_link_velocity_to_sim(self, *a, **k):
            pass

    class _TierScene(dict):
        env_origins = torch.zeros(4, 3)

    class _Env:
        num_envs = 4
        device = "cpu"
        scene = _TierScene({"ball": _Ball()})
        _ball_difficulty = 1.0

    env = _Env()
    torch.manual_seed(0)
    events_mod.reset_ball_rolling(
        env, torch.arange(4, dtype=torch.int32), "ball",
        dist_range=(2.0, 2.0),          # x_start fixed at 2.0
        y_start_range=(0.0, 0.0),       # y_start fixed at 0.0
        y_end_range=(-1.1, 1.1),
        t_flight_range=(1.0, 1.0),
    )

    assert hasattr(env, "_rsi_cross_y")
    cy = env._rsi_cross_y
    # With y_start=0: cross_y = y_end * 2.0/2.3 — strictly inside the target,
    # same sign, and bounded by the widened range.
    assert (cy.abs() <= 1.1 * 2.0 / 2.3 + 1e-5).all()
    assert (cy.abs() > 0.0).all()  # dead-zone sampling never returns exactly 0
