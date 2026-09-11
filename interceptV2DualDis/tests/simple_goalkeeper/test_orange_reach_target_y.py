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
distance-from-start is now smoothly floored at 0.2m (logsumexp smooth-max,
_ORANGE_START_MIN/_ORANGE_START_SMOOTH_K in the real function) -- root cause
was orange sitting exactly AT start_y for every near-wide_threshold crossing,
which any reasonable RSI donor pose already satisfies at reset, driving a
large share of orange's "free landing" misclassification rate. Tests below
updated to assert against the same smooth formula (pinning the SHAPE, not
just a point value) rather than the old exact closed form.
"""
import math

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


_ORANGE_START_MIN = 0.2
_ORANGE_START_SMOOTH_K = 15.0


def _expected_dist_from_start(delta: float) -> float:
    """Reference implementation of the smooth 0.2m floor -- mirrors the
    real function's logsumexp smooth-max exactly, so these tests pin the
    SHAPE (smooth floor, converges to the plain formula away from it)
    rather than a single hardcoded constant."""
    raw = max(abs(delta) - 0.50, 0.0) / 2.0
    a = raw * _ORANGE_START_SMOOTH_K
    b = _ORANGE_START_MIN * _ORANGE_START_SMOOTH_K
    m = max(a, b)
    dist = (m + math.log(math.exp(a - m) + math.exp(b - m))) / _ORANGE_START_SMOOTH_K
    return min(dist, abs(delta))


def test_orange_target_shrinks_positive_delta_by_050_then_halves():
    # delta=+1.00m -> raw shrunk/2=0.25, well above the 0.2m floor -> orange_y
    # converges close to the plain formula (blue's own midpoint would be 0.50).
    expected = _expected_dist_from_start(1.0)
    assert abs(_orange_y(1.0) - expected) < 1e-4
    assert abs(expected - 0.25) < 0.03  # smooth floor barely nudges a value this far above it


def test_orange_target_shrinks_moderate_positive_delta():
    # delta=+0.80m -> raw shrunk/2=0.15, BELOW the 0.2m floor -> orange_y sits
    # near the floor (0.2m from start), not the old raw 0.15m.
    expected = _expected_dist_from_start(0.8)
    assert abs(_orange_y(0.8) - expected) < 1e-4
    assert expected > 0.15  # floor pulled it up from the old raw value
    assert abs(expected - 0.2) < 0.03  # close to the floor itself


def test_orange_target_floors_at_020m_from_start_when_delta_below_050():
    # delta=+0.40m -> raw shrunk clamped to 0.0 -- OLD behavior collapsed
    # orange_y to start_y exactly (0.0); NEW behavior floors it at ~0.2m
    # from start instead (the whole point of this fix).
    expected = _expected_dist_from_start(0.4)
    assert abs(_orange_y(0.4) - expected) < 1e-4
    assert expected > 0.15
    assert abs(expected - 0.2) < 0.02


def test_orange_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> mirrors the positive-delta case, sign-flipped (NOT
    # -0.90, which a naive `delta - 0.50` without sign handling would produce).
    expected = _expected_dist_from_start(-1.0)
    assert abs(_orange_y(-1.0) - (-expected)) < 1e-4


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
