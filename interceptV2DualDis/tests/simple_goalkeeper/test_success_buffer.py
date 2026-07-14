"""Tests for SuccessReplayBuffer -- the self-imitation replay buffer (see
success_buffer.py docstring for the full rationale: genuine blue-ball
landings are rare and get washed out by denser competing reward gradients
over continued training, so the transitions leading up to a genuine landing
are stashed here and replayed as an auxiliary behavior-cloning loss).
"""
import torch

from simple_goalkeeper.rsl_rl_multi.success_buffer import SuccessReplayBuffer


def _make_buffer(num_envs=4, capacity=100, lookback=5, obs_current_dim=3, obs_history_dim=6, action_dim=2):
    return SuccessReplayBuffer(
        obs_current_dim=obs_current_dim,
        obs_history_dim=obs_history_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        device="cpu",
        capacity=capacity,
        lookback=lookback,
    )


def test_empty_buffer_has_zero_length():
    buf = _make_buffer()
    assert len(buf) == 0


def test_commit_success_flushes_lookback_window_into_permanent_buffer():
    buf = _make_buffer(num_envs=2, capacity=100, lookback=5)
    reset_mask = torch.zeros(2, dtype=torch.bool)

    # Record 3 distinct steps for both envs (env 0 will "land", env 1 won't).
    for step in range(3):
        obs_current = torch.full((2, 3), float(step))
        obs_history = torch.full((2, 6), float(step))
        actions = torch.full((2, 2), float(step))
        buf.record_step(obs_current, obs_history, actions, reset_mask)

    assert len(buf) == 0  # nothing committed yet

    buf.commit_success(torch.tensor([0]))  # only env 0 landed
    assert len(buf) == 3  # all 3 recorded steps for env 0 flushed

    sampled_obs_current, _, sampled_actions = buf.sample(3)
    # All 3 committed transitions came from env 0's recorded steps 0, 1, 2.
    values = set(sampled_obs_current[:, 0].round().int().tolist())
    assert values <= {0, 1, 2}


def test_reset_mask_clears_rolling_window_for_that_env():
    buf = _make_buffer(num_envs=2, capacity=100, lookback=5)
    reset_mask = torch.zeros(2, dtype=torch.bool)

    # Env 0 accumulates 2 steps of an "old episode".
    for step in range(2):
        buf.record_step(
            torch.full((2, 3), float(step)), torch.full((2, 6), float(step)),
            torch.full((2, 2), float(step)), reset_mask,
        )

    # Env 0 resets (new episode starts) -- its rolling window must be wiped
    # before the new episode's first transition is recorded.
    reset_mask = torch.tensor([True, False])
    buf.record_step(
        torch.full((2, 3), 99.0), torch.full((2, 6), 99.0),
        torch.full((2, 2), 99.0), reset_mask,
    )

    buf.commit_success(torch.tensor([0]))
    # Only the single post-reset transition (value 99) should have survived
    # for env 0 -- the pre-reset steps 0/1 must not leak into the new episode.
    assert len(buf) == 1
    sampled_obs_current, _, _ = buf.sample(1)
    assert sampled_obs_current[0, 0].item() == 99.0


def test_capacity_wraparound_keeps_buffer_size_bounded():
    buf = _make_buffer(num_envs=1, capacity=5, lookback=10)
    reset_mask = torch.zeros(1, dtype=torch.bool)

    for step in range(8):
        buf.record_step(
            torch.full((1, 3), float(step)), torch.full((1, 6), float(step)),
            torch.full((1, 2), float(step)), reset_mask,
        )
    buf.commit_success(torch.tensor([0]))  # 8 valid steps committed into a 5-slot buffer

    assert len(buf) == 5  # capped at capacity, not 8


def test_sample_raises_or_is_empty_before_any_commit():
    buf = _make_buffer()
    assert len(buf) == 0
    # Sampling from an empty buffer would divide by zero envs -- callers
    # (MultiDiscAMPPPO.update) are responsible for checking len() first, but
    # confirm the buffer itself doesn't silently return nonsense.
    import pytest
    with pytest.raises(RuntimeError):
        buf.sample(4)
