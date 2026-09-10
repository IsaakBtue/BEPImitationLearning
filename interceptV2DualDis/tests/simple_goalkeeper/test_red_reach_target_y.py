"""Tests for _get_red_reach_target_y's target-position formula and gate.

REWRITTEN 2026-09-11 (user request, "i dont want yellow ball/gold ball
anymore, i just want red ball to be in the 60% along the green ball full
length it only appears once blue ball is landed"): the orange waypoint is
gone -- red is now the trailing foot's ONLY waypoint, gated on
env._blue_landed_genuine alone (was blue AND orange), positioned at 60% of
the way from start_y to full_y (green) -- was a flat/clamped offset near
green. See docs/BugFixes.md, 2026-09-11.
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_red_reach_target_y


class _Scene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeEnv:
    """Deliberately omits 'robot'/'feet_contact' scene entries so
    _get_red_reach_target_y's try/except falls into the robot=None branch;
    the target-Y formula and the blue gate are both computed unconditionally
    before that branch."""

    def __init__(self, num_envs: int, crossing_delta: float, blue_landed=None):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(num_envs)
        self._rsi_cross_y = torch.full((num_envs,), crossing_delta)
        self.scene = _Scene(num_envs)
        if blue_landed is not None:
            self._blue_landed_genuine = torch.full((num_envs,), blue_landed, dtype=torch.bool)


def _red_y(crossing_delta: float) -> float:
    env = _FakeEnv(num_envs=4, crossing_delta=crossing_delta)
    result = _get_red_reach_target_y(env, "ball")
    return result[0].item()


def test_red_target_is_60_percent_of_start_to_green_positive_side():
    # delta=+1.00m -> full_y=1.0, start_y=0.0, red_y=0.6*1.0=0.6.
    assert abs(_red_y(1.0) - 0.6) < 1e-6


def test_red_target_is_60_percent_of_start_to_green_smaller_delta():
    # delta=+0.60m -> red_y=0.6*0.60=0.36.
    assert abs(_red_y(0.6) - 0.36) < 1e-6


def test_red_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> full_y=-1.0, red_y=0.6*(-1.0)=-0.6.
    assert abs(_red_y(-1.0) - (-0.6)) < 1e-6


def test_red_active_false_when_blue_attr_missing():
    """Defensive fallback: real term order registers red after blue,
    so this should never trigger live, but must not crash if it does."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_false_when_blue_not_landed():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=False)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_true_when_blue_landed():
    """Red now activates on blue alone -- orange no longer exists."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True)
    _get_red_reach_target_y(env, "ball")
    assert bool(env._red_active[0].item())


def test_red_wide_reuses_blue_wide_directly():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0)
    env._blue_wide = torch.tensor([True, False, True, False])
    _get_red_reach_target_y(env, "ball")
    assert env._red_wide.tolist() == [True, False, True, False]
