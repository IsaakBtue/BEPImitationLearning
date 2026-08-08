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
