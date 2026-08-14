"""Tests for _get_red_reach_target_y's target-position formula and gate.

NEW 2026-08-15: trailing-foot second-stage ("red") mirror of
_get_orange_reach_target_y, active only once both blue and orange are
genuinely landed (env._blue_landed_genuine & env._orange_landed_genuine).
Unlike orange (anchored at start_y, shrinks toward it), red is anchored at
full_y (green) and shrinks backward -- see that function's docstring for why
a literal start_y-anchored copy of orange's formula could never place a
point past blue toward green. See docs/BugFixes.md, 2026-08-15.
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_red_reach_target_y


class _Scene(dict):
    def __init__(self, num_envs: int):
        super().__init__()
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeEnv:
    """Mirrors tests/simple_goalkeeper/test_orange_reach_target_y.py's _FakeEnv --
    deliberately omits 'robot'/'feet_contact' scene entries so
    _get_red_reach_target_y's try/except falls into the robot=None branch;
    the target-Y formula and the blue/orange gate are both computed
    unconditionally before that branch."""

    def __init__(self, num_envs: int, crossing_delta: float, blue_landed=None, orange_landed=None):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(num_envs)
        self._rsi_cross_y = torch.full((num_envs,), crossing_delta)
        self.scene = _Scene(num_envs)
        if blue_landed is not None:
            self._blue_landed_genuine = torch.full((num_envs,), blue_landed, dtype=torch.bool)
        if orange_landed is not None:
            self._orange_landed_genuine = torch.full((num_envs,), orange_landed, dtype=torch.bool)


def _red_y(crossing_delta: float) -> float:
    env = _FakeEnv(num_envs=4, crossing_delta=crossing_delta)
    result = _get_red_reach_target_y(env, "ball")
    return result[0].item()


def test_red_target_shrinks_positive_delta_by_050_then_halves_from_green():
    # delta=+1.00m -> full_y=1.0, shrunk=0.50 -> red_y=1.0-0.25=0.75
    # (blue's own midpoint would be 0.50, orange's would be 0.25 -- red sits
    # symmetrically opposite orange around blue.)
    assert abs(_red_y(1.0) - 0.75) < 1e-6


def test_red_target_shrinks_moderate_positive_delta():
    # delta=+0.80m -> shrunk=0.30 -> red_y=0.80-0.15=0.65
    assert abs(_red_y(0.8) - 0.65) < 1e-6


def test_red_target_collapses_to_green_when_delta_below_050():
    # delta=+0.40m -> shrunk clamped to 0.0 -> red_y collapses to full_y (0.40),
    # NOT start_y like orange's degenerate case (0.0) -- the anchor is flipped.
    assert abs(_red_y(0.4) - 0.4) < 1e-6


def test_red_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> full_y=-1.0, shrunk=-0.50 -> red_y=-1.0-(-0.25)=-0.75
    assert abs(_red_y(-1.0) - (-0.75)) < 1e-6


def test_red_active_false_when_blue_and_orange_attrs_missing():
    """Defensive fallback: real term order registers red after blue/orange,
    so this should never trigger live, but must not crash if it does."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_false_when_only_blue_landed():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True, orange_landed=False)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_false_when_only_orange_landed():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=False, orange_landed=True)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_true_when_both_blue_and_orange_landed():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True, orange_landed=True)
    _get_red_reach_target_y(env, "ball")
    assert bool(env._red_active[0].item())


def test_red_wide_reuses_blue_wide_directly():
    """Mirrors orange's own env._orange_wide = env._blue_wide reuse --
    red must read the SAME cached flag, not recompute its own."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0)
    env._blue_wide = torch.tensor([True, False, True, False])
    _get_red_reach_target_y(env, "ball")
    assert env._red_wide.tolist() == [True, False, True, False]
