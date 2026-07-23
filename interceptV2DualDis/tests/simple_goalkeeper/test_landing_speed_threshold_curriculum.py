"""Tests for _get_reach_target_y's landing_speed_threshold curriculum.

FIX 2026-07-23 (twice same day): the threshold was first made flat/curriculum-
independent (0.15 m/s at every difficulty) after the old eased range (2.0 m/s
at d=0 -> 1.0 m/s at d=1) let a foot merely striding through the eased
landing_radius zone register as "landed" without stopping. Later the same
day, per user request, the curriculum was restored -- but with a narrower,
deliberately conservative easy end (0.30 m/s, not the old leaky 1.0/2.0 m/s)
so an early policy still gets a genuinely easier bar without reintroducing
the pass-through bug. See docs/BugFixes.md.
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_reach_target_y


class _Scene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeEnv:
    """Deliberately omits 'robot'/'feet_contact' scene entries so
    _get_reach_target_y's try/except falls into the robot=None branch --
    the landing_speed_threshold lerp is computed and cached unconditionally
    BEFORE that branch, so no robot/contact-sensor mocking is needed to
    test it in isolation."""

    def __init__(self, num_envs: int = 4, ball_difficulty: float = 1.0):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(num_envs)
        self._ball_difficulty = ball_difficulty
        self._rsi_cross_y = torch.full((num_envs,), 1.0)  # wide crossing, > 0.65
        self.scene = _Scene(num_envs)


def _resolved_threshold(ball_difficulty: float) -> float:
    env = _FakeEnv(ball_difficulty=ball_difficulty)
    _get_reach_target_y(env, "ball")
    return env._blue_landing_speed_threshold_current


def test_landing_speed_threshold_strict_at_full_difficulty():
    assert abs(_resolved_threshold(1.0) - 0.15) < 1e-9


def test_landing_speed_threshold_eases_at_zero_difficulty():
    """FIX 2026-07-23 (restored curriculum): easy end is 0.30 m/s, a real
    curriculum step but deliberately NOT the old 1.0/2.0 m/s that let a
    foot merely striding through the eased radius zone register as
    landed."""
    assert abs(_resolved_threshold(0.0) - 0.30) < 1e-9


def test_landing_speed_threshold_lerps_monotonically_between_ends():
    thresholds = [_resolved_threshold(d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for a, b in zip(thresholds, thresholds[1:]):
        assert a >= b  # strictly non-increasing as difficulty rises
    assert thresholds[0] > thresholds[-1]  # genuinely eases, not flat


def test_landing_speed_threshold_never_reaches_old_leaky_values():
    """Regression guard for the original bug: at NO difficulty should the
    resolved threshold reach the old 1.0/2.0 m/s values that were confirmed
    to let a foot pass through the eased radius zone without stopping."""
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert _resolved_threshold(d) <= 0.30 + 1e-9
