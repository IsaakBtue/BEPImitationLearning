"""Tests for _get_reach_target_y's landing_speed_threshold curriculum.

FIX 2026-07-23 (twice same day): the threshold was first made flat/curriculum-
independent (0.15 m/s at every difficulty) after the old eased range (2.0 m/s
at d=0 -> 1.0 m/s at d=1) let a foot merely striding through the eased
landing_radius zone register as "landed" without stopping. Later the same
day, per user request, the curriculum was restored -- but with a narrower,
deliberately conservative easy end (0.30 m/s, not the old leaky 1.0/2.0 m/s)
so an early policy still gets a genuinely easier bar without reintroducing
the pass-through bug.

FIX 2026-07-24: reverted back to the original eased range (2.0 m/s at d=0 ->
1.0 m/s at d=1), per user request, after the narrower 0.30->0.15 m/s band
coincided with blue_ball_landed declining over iterations 0-6700 of
blue_v2_landinggatefix_2026-07-23 (0.041 peak -> 0.007) with no recovery --
the strict end looked hard enough to be suppressing the landing gradient
outright rather than just filtering false-positive "strides through" landings.
This deliberately reintroduces the documented pass-through leak risk as a
trade against the widened landing_radius (0.08m -> 0.15m, same date, same
request). See docs/BugFixes.md.
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
    assert abs(_resolved_threshold(1.0) - 1.0) < 1e-9


def test_landing_speed_threshold_eases_at_zero_difficulty():
    """FIX 2026-07-24 (reverted): easy end is 2.0 m/s, the original
    pre-2026-07-23 value -- reinstated after the narrower 0.30 m/s easy end
    coincided with blue_ball_landed declining rather than recovering."""
    assert abs(_resolved_threshold(0.0) - 2.0) < 1e-9


def test_landing_speed_threshold_lerps_monotonically_between_ends():
    thresholds = [_resolved_threshold(d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for a, b in zip(thresholds, thresholds[1:]):
        assert a >= b  # strictly non-increasing as difficulty rises
    assert thresholds[0] > thresholds[-1]  # genuinely eases, not flat


def test_landing_speed_threshold_within_reverted_band():
    """FIX 2026-07-24: the never-reaches-old-leaky-values regression guard
    from 2026-07-23 is intentionally gone -- 1.0/2.0 m/s IS the current
    band again, by deliberate user request. This just pins the resolved
    range to what's actually configured, so an accidental further change
    is still caught."""
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        threshold = _resolved_threshold(d)
        assert 1.0 - 1e-9 <= threshold <= 2.0 + 1e-9
