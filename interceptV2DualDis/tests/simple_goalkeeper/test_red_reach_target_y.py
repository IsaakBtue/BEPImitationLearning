"""Tests for _get_red_reach_target_y's target-position formula and gate.

NEW 2026-08-15: trailing-foot second-stage ("red") mirror of
_get_orange_reach_target_y, active only once both blue and orange are
genuinely landed (env._blue_landed_genuine & env._orange_landed_genuine).
Anchored at full_y (green), a flat 0.4m offset toward start (sign-safe),
CLAMPED to never exceed blue's own distance from green (|delta|/2) -- NOT a
shrink-then-halve formula like orange's. FIX 2026-08-15 (user request): the
original shrink-based formula's distance from green scaled with the total
crossing distance and collapsed to as little as 0.029m from green on a real
checkpoint rollout for crossings just over the 0.5m wide threshold. Then the
first flat-0.4m-offset fix was found (also live) to place red BEFORE blue
for crossings under ~0.8m total distance -- clamped so red instead collapses
onto blue's own position in that range rather than overshooting past it.
See docs/BugFixes.md, 2026-08-15.
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


def test_red_target_is_flat_040_offset_from_green_positive_side():
    # delta=+1.00m -> full_y=1.0, red_y=1.0-0.4=0.6 -- fixed offset,
    # independent of the crossing distance (unlike orange's shrink formula).
    assert abs(_red_y(1.0) - 0.6) < 1e-6


def test_red_target_at_exactly_080_gets_the_full_offset():
    # delta=+0.80m -> blue's own distance from green is |delta|/2=0.40,
    # exactly equal to the 0.4m cap -- boundary case, clamp is a no-op here.
    assert abs(_red_y(0.8) - 0.4) < 1e-6


def test_red_target_clamps_to_blue_position_below_080():
    # delta=+0.55m (just over the 0.5m wide threshold, well under the 0.8m
    # clamp boundary): blue's own distance from green is |delta|/2=0.275,
    # LESS than the 0.4m offset -- red_y must clamp to blue_y (0.275) rather
    # than overshoot past it toward start (which an uncapped 0.4m offset
    # would give: 0.55-0.4=0.15, confirmed live to invert start<orange<
    # blue<red<green into start<orange<red<blue<green -- the bug this
    # clamp fixes).
    blue_y = 0.0 + 0.55 / 2.0  # start_y=0 in this test's _FakeEnv
    assert abs(_red_y(0.55) - blue_y) < 1e-6


def test_red_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> full_y=-1.0, red_y=-1.0-(-0.4)=-0.6
    assert abs(_red_y(-1.0) - (-0.6)) < 1e-6


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
