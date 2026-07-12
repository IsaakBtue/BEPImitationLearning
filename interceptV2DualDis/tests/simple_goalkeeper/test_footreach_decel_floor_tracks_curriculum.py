"""2026-07-12: regression test for the footreach decel-zone fix.

Before this fix, footreach's vel_sigma decay floor (_BLUE_DECEL_FLOOR) was a
hardcoded 0.08 -- the STRICT, full-difficulty landing_radius -- even though
_get_reach_target_y's real landing_radius is curriculum-eased up to 0.20 at
ball_difficulty=0. That meant the "slow down near blue" incentive stayed
inert (never fully decayed to neutral) exactly in the loose-tolerance regime
where landing should be easiest to first discover. Fix: _get_reach_target_y
now caches its own eased radius on env._blue_landing_radius_current, and
footreach reads that instead of a hardcoded constant.

This test only exercises the caching in _get_reach_target_y (the part that
runs before the robot/feet_contact-dependent landing-detection block), since
that's the mechanism under test and it avoids needing a full scene mock.
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_reach_target_y


class _FakeScene:
    def __init__(self, num_envs: int):
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, key):
        raise KeyError(key)  # no robot/feet_contact/ball -- landing-detection block is skipped


class _FakeEnv:
    def __init__(self, num_envs: int, difficulty: float):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _FakeScene(num_envs)
        self.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.int64)
        self._ball_difficulty = difficulty
        # Bypass _get_ball_crossing_y's ball-entity lookup entirely.
        self._rsi_cross_y = torch.full((num_envs,), 0.9)  # a wide crossing


def test_landing_radius_cache_eases_from_loose_to_strict_with_difficulty():
    easy = _FakeEnv(num_envs=2, difficulty=0.0)
    _get_reach_target_y(easy, "ball")
    assert easy._blue_landing_radius_current == 0.20  # fully eased, loose

    hard = _FakeEnv(num_envs=2, difficulty=1.0)
    _get_reach_target_y(hard, "ball")
    assert abs(hard._blue_landing_radius_current - 0.08) < 1e-6  # fully saturated, strict

    mid = _FakeEnv(num_envs=2, difficulty=0.5)
    _get_reach_target_y(mid, "ball")
    expected_mid = 0.20 + (0.08 - 0.20) * 0.5
    assert abs(mid._blue_landing_radius_current - expected_mid) < 1e-6


def test_footreach_decel_floor_would_have_been_wrong_under_the_old_hardcoded_value():
    """Confirms the specific bug this fix closes: at low difficulty, the real
    landing_radius (0.20) is looser than the old hardcoded decel floor (0.08)
    -- footreach's vel_sigma bonus would NOT have reached neutral by the time
    a foot actually satisfied the real (eased) landing check."""
    easy = _FakeEnv(num_envs=1, difficulty=0.0)
    _get_reach_target_y(easy, "ball")
    real_landing_radius = easy._blue_landing_radius_current
    old_hardcoded_decel_floor = 0.08
    assert real_landing_radius > old_hardcoded_decel_floor
