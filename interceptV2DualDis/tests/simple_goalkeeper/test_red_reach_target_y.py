"""Tests for _get_red_reach_target_y's target-position formula and gate.

REWRITTEN 2026-09-11 (user request, "i dont want yellow ball/gold ball
anymore, i just want red ball to be in the 60% along the green ball full
length it only appears once blue ball is landed"): red's POSITION formula
changed to 60% of the way from start_y to full_y (green) -- was a
flat/clamped offset near green.

RESTORED 2026-09-11 (same day, user request, "revert the yellow ball i
want it back... do this by means of git"): red's GATE reverted back to
requiring BOTH blue AND orange genuinely landed (was blue-only for one
same-session interval while orange was deleted). The 60%-of-start-to-green
POSITION formula above is unaffected -- only the gate tests changed back.

DELAYED 2026-09-11 (same day, user request, "delay when going from yellow
to red"): the gate additionally requires 25 steps to have passed since
orange FIRST genuinely landed (`_RED_ACTIVATION_DELAY_STEPS`,
`env._orange_landed_genuine_step`). `test_red_active_true_when_both_blue_
and_orange_landed` updated to advance past the delay explicitly; new tests
added for the delay boundary itself.

See docs/BugFixes.md, 2026-09-11 (all three entries).
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


def test_red_target_is_60_percent_of_start_to_green_positive_side():
    # delta=+1.00m -> full_y=1.0, start_y=0.0, red_y=0.6*1.0=0.6.
    assert abs(_red_y(1.0) - 0.6) < 1e-6


def test_red_target_is_60_percent_of_start_to_green_smaller_delta():
    # delta=+0.60m -> red_y=0.6*0.60=0.36.
    assert abs(_red_y(0.6) - 0.36) < 1e-6


def test_red_target_sign_safe_for_right_side_crossings():
    # delta=-1.00m -> full_y=-1.0, red_y=0.6*(-1.0)=-0.6.
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


def test_red_active_false_immediately_after_both_landed_delay_not_elapsed():
    """Same tick orange genuinely lands: delay hasn't elapsed yet."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True, orange_landed=True)
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_active_true_once_delay_elapses_after_both_landed():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True, orange_landed=True)
    _get_red_reach_target_y(env, "ball")  # tick 0: orange_landed_genuine_step latches to 0
    assert not bool(env._red_active[0].item())
    env.episode_length_buf += 25  # advance exactly _RED_ACTIVATION_DELAY_STEPS
    _get_red_reach_target_y(env, "ball")
    assert bool(env._red_active[0].item())


def test_red_active_false_just_before_delay_elapses():
    env = _FakeEnv(num_envs=4, crossing_delta=1.0, blue_landed=True, orange_landed=True)
    _get_red_reach_target_y(env, "ball")
    env.episode_length_buf += 24  # one step short of the delay
    _get_red_reach_target_y(env, "ball")
    assert not bool(env._red_active[0].item())


def test_red_wide_reuses_blue_wide_directly():
    """Mirrors orange's own env._orange_wide = env._blue_wide reuse --
    red must read the SAME cached flag, not recompute its own."""
    env = _FakeEnv(num_envs=4, crossing_delta=1.0)
    env._blue_wide = torch.tensor([True, False, True, False])
    _get_red_reach_target_y(env, "ball")
    assert env._red_wide.tolist() == [True, False, True, False]
