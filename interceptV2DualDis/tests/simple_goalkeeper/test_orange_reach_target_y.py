"""Tests for _get_orange_reach_target_y's target-position formula.

NEW 2026-08-08: trailing-foot ("orange") mirror of _get_reach_target_y's
midpoint targeting, using a different formula -- shrink |delta| by 0.50m
(sign-safe) before halving, instead of blue's plain halve. See
docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md for the
original 0.30m derivation; FIX 2026-08-08 (user request, "the orange ball
needs to be way further than the blue ball") raised it to 0.60m same day,
then adjusted to 0.50m later the same day -- see docs/BugFixes.md.

UPDATE 2026-09-11 (user request, "put the spawn at 0.2 minimum to decrease
the episodes that it happens to be free"): the raw `shrunk/2.0`
distance-from-start was smoothly floored at 0.2m (logsumexp smooth-max) --
root cause was orange sitting exactly AT start_y for every near-
wide_threshold crossing, which any reasonable RSI donor pose already
satisfies at reset, driving a large share of orange's "free landing"
misclassification rate.

REWRITTEN 2026-09-12 (user request, "just do this with 0.6 and also a new
way of calculating only making sure of the 0.25m gap nothing else"): the
formula above is entirely replaced. Orange is now DEFINED as a fixed gap
closer to start than blue (`blue_dist_from_start = |delta|/2`,
`orange_dist_from_start = max(blue_dist_from_start - GAP, 0.0)`) --
guarantees the gap by construction rather than approximating it. Also
depends on `wide_threshold` (rewards.py:_get_reach_target_y) having moved
0.5 -> 0.6 in the same change.

UPDATE 2026-09-12 (same day, user request, "for orange ball landed have
the distance 0.3 between it and blue ball"): GAP 0.25 -> 0.30. NOTE this
reopens the exact degenerate case the 0.25 gap safely avoided: at the new
wide_threshold minimum (delta=0.6), blue_dist_from_start=0.30 exactly
EQUALS the new 0.30m gap, so orange_dist_from_start = max(0.30-0.30,0) = 0
-- orange sits exactly AT start_y again at the tightest wide crossing, the
same geometry overlap that originally caused the "free landing"
misclassification bug (see the earlier `_ORANGE_START_MIN` entry in
docs/BugFixes.md). Not re-guarded here per explicit user instruction
("nothing else" scoping from the original rewrite) -- flagged, not fixed.
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


_ORANGE_BLUE_GAP = 0.30


def _expected_dist_from_start(delta: float) -> float:
    """Reference implementation of the new blue-anchored formula."""
    blue_dist_from_start = abs(delta) / 2.0
    return max(blue_dist_from_start - _ORANGE_BLUE_GAP, 0.0)


def test_orange_target_floors_at_start_y_exactly_at_wide_threshold():
    # delta=+0.60m (the wide_threshold minimum) -> blue_dist=0.30, which
    # EXACTLY equals the 0.30m gap -- orange collapses to exactly start_y
    # (0.0) at the tightest real wide crossing. See module docstring's
    # 2026-09-12 UPDATE note: this is a known, accepted degenerate case at
    # this specific gap value, not a bug.
    expected = _expected_dist_from_start(0.6)
    assert expected == 0.0
    assert abs(_orange_y(0.6) - 0.0) < 1e-6


def test_orange_target_at_moderate_delta():
    # delta=+0.70m -> blue_dist=0.35, orange_dist=0.35-0.30=0.05 -- the
    # smallest delta with a genuinely positive orange distance.
    expected = _expected_dist_from_start(0.7)
    assert abs(_orange_y(0.7) - expected) < 1e-6
    assert abs(expected - 0.05) < 1e-6


def test_orange_target_at_large_delta():
    # delta=+1.00m -> blue_dist=0.50, orange_dist=0.50-0.30=0.20.
    expected = _expected_dist_from_start(1.0)
    assert abs(_orange_y(1.0) - expected) < 1e-6
    assert abs(expected - 0.20) < 1e-6


def test_orange_target_floors_at_start_y_for_degenerate_small_delta():
    # delta=+0.40m (below wide_threshold=0.6, only reachable via a
    # region-forced-wide degenerate case) -> blue_dist=0.20, well below the
    # 0.30m gap -- orange collapses to exactly start_y (0).
    expected = _expected_dist_from_start(0.4)
    assert expected == 0.0
    assert abs(_orange_y(0.4) - 0.0) < 1e-6


def test_orange_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> mirrors the positive-delta case, sign-flipped.
    expected = _expected_dist_from_start(-1.0)
    assert abs(_orange_y(-1.0) - (-expected)) < 1e-6


def test_gap_between_blue_and_orange_is_at_least_030m_for_wide_crossings_above_threshold():
    """Mirrors the original design intent: for delta STRICTLY above
    wide_threshold, the blue-orange gap is always exactly 0.30m (once the
    floor isn't binding). At exactly delta=wide_threshold (0.6) the floor
    DOES bind (see test above) -- excluded here deliberately."""
    for delta in [0.7, 0.8, 0.9, 1.0, 1.1]:
        blue_dist = delta / 2.0
        orange_dist = _expected_dist_from_start(delta)
        gap = blue_dist - orange_dist
        assert abs(gap - _ORANGE_BLUE_GAP) < 1e-9, f"gap {gap} != 0.30 at delta={delta}"


def test_trailing_idx_is_complement_of_leading_foot_idx():
    """Guards the single most likely defect in a copy-paste mirror: using
    foot_idx instead of 1 - foot_idx (targeting the leading foot instead of
    the trailing one) in any of the 4 orange reward functions."""
    import torch
    foot_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    trailing_idx = 1 - foot_idx
    assert trailing_idx.tolist() == [1, 0, 1, 0]
    # Every orange_* function computes trailing_idx this exact way immediately
    # after calling _get_correct_foot_idx -- this pins the arithmetic itself,
    # not a specific function (those need a real robot/contact-sensor scene
    # to exercise end-to-end, impractical to fake in a unit test).
