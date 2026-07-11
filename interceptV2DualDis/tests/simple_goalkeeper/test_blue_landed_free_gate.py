"""2026-07-12: regression coverage for the reward-leak fix (docs/BugFixes.md,
"_BLUE_LANDED_SEED_FRACTION RSI teleport leaked free reward..."). Both
blue_ball_landed and track_blue_landing_success must ignore landings where
env._blue_landed_was_free is set (RSI-seeded, not policy-caused) -- previously
neither excluded it, only stopball/softstop did.

blue_ball_landed calls _get_reach_target_y first (real scene/sensor
dependencies); monkeypatched to a no-op here since these tests target only the
free-landing gate, not the reach-target computation itself.
"""
import torch

from simple_goalkeeper.mdp import rewards
from simple_goalkeeper.mdp.events import track_blue_landing_success


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.int64)


def _patch_no_op_reach_target(monkeypatch):
    monkeypatch.setattr(rewards, "_get_reach_target_y", lambda *a, **k: None)


def test_blue_ball_landed_does_not_fire_for_rsi_seeded_free_landing(monkeypatch):
    _patch_no_op_reach_target(monkeypatch)
    env = _FakeEnv(2)
    env._blue_landed = torch.tensor([True, True])
    env._blue_landed_was_free = torch.tensor([True, False])  # env0 seeded, env1 genuine

    fired = rewards.blue_ball_landed(env, ball_name="ball")

    assert fired.tolist() == [0.0, 1.0]


def test_blue_ball_landed_fires_again_for_genuine_landing_after_episode_reset(monkeypatch):
    _patch_no_op_reach_target(monkeypatch)
    env = _FakeEnv(1)
    env._blue_landed = torch.tensor([True])
    env._blue_landed_was_free = torch.tensor([False])

    first = rewards.blue_ball_landed(env, ball_name="ball")
    second = rewards.blue_ball_landed(env, ball_name="ball")  # same step: one-shot latch
    assert first.tolist() == [1.0]
    assert second.tolist() == [0.0]

    # New episode: bonus flag clears, a fresh genuine landing can fire again.
    env.episode_length_buf = torch.tensor([1])
    third = rewards.blue_ball_landed(env, ball_name="ball")
    assert third.tolist() == [1.0]


def test_track_blue_landing_success_excludes_free_landings_from_rolling_rate():
    env = _FakeEnv(4)
    env.common_step_counter = 0
    # Pre-set the accumulator state (rather than relying on the function's own
    # lazy-init, which seeds _blue_success_last_update=-500 -- with
    # common_step_counter=0 that would trigger the periodic update on the very
    # first call) so the first call below is guaranteed to only accumulate.
    env._blue_landing_success_rate = 0.0
    env._blue_success_window_count = 0
    env._blue_wide_window_count = 0
    env._blue_success_last_update = 0
    env._blue_wide = torch.tensor([True, True, True, False])
    # env0: genuine landing; env1: seeded/free landing (must not count);
    # env2: wide, never landed; env3: narrow (excluded via _blue_wide).
    env._blue_landed = torch.tensor([True, True, False, False])
    env._blue_landed_was_free = torch.tensor([False, True, False, False])
    env_ids = torch.arange(4)

    track_blue_landing_success(env, env_ids)
    assert env._blue_wide_window_count == 3
    assert env._blue_success_window_count == 1
    assert env._blue_landing_success_rate == 0.0  # not yet updated

    # Second call, 500 steps later: triggers the periodic update, which reads
    # the accumulated window counts (doubled: this call re-accumulates the
    # same env_ids again before the threshold check) and resets them.
    env.common_step_counter = 500
    track_blue_landing_success(env, env_ids)
    assert env._blue_landing_success_rate == 2 / 6  # (1+1) genuine / (3+3) wide
    assert env._blue_wide_window_count == 0
    assert env._blue_success_window_count == 0


def test_track_blue_landing_success_all_free_landings_yields_zero_rate():
    env = _FakeEnv(2)
    env.common_step_counter = 500
    env._blue_wide = torch.tensor([True, True])
    env._blue_landed = torch.tensor([True, True])
    env._blue_landed_was_free = torch.tensor([True, True])  # both seeded, not genuine

    track_blue_landing_success(env, torch.arange(2))

    assert env._blue_landing_success_rate == 0.0
