"""Self-imitation replay buffer for genuine blue-ball landings.

FEAT 2026-07-14: across every fix attempted for the blue-ball landing
problem (AMP dilution, reward-shape, retiming, contact-gating, the leaky
settle-window counter), the same shape kept recurring once a run finally
discovered genuine landings: the rate rose, then decayed back toward the
historical ~0-5% floor over the following iterations (e.g.
blue_leaky_settle_count_2026-07-13: 44.8% -> 14.6% -> 8.3% -> 2.5% across
four checkpoints). The landing reward is sparse and rare relative to the
much denser, constantly-firing competing signals (footreach's speed bonus,
AMP imitation reward, the curriculum-scaled blue_overshoot_penalty) --
ordinary on-policy PPO has no mechanism to specifically protect a
rediscovered rare success from being washed out by that denser gradient.

This is a simplified self-imitation approach (Oh et al. 2018, "Self-
Imitation Learning", arXiv:1806.05635) adapted to fit this project's
existing on-policy RolloutStorage-based PPO loop without a large
return/value-bootstrapping rewrite: rather than the full advantage-weighted
SIL loss (which needs episode-return bookkeeping this codebase's plain
RolloutStorage doesn't retain past its own update), this stores the
(obs_current, obs_history, action) transitions from the steps leading up
to and including every genuine blue-ball landing into a persistent replay
buffer, and PPO's update() adds a small behavior-cloning-style loss
(negative log-likelihood of the recorded actions under the CURRENT policy)
sampled from that buffer alongside the normal surrogate loss. This directly
reinforces "do what you did when you succeeded" every update, independent
of how rare successes are in the current on-policy rollout batch.
"""
from __future__ import annotations

import torch


class SuccessReplayBuffer:
    """FIFO replay buffer of (obs_current, obs_history, action) transitions
    drawn from the lookback window preceding each genuine landing.

    Two pieces of state:
    - A small per-env ring buffer (`lookback` steps) of the most recent
      transitions, continuously overwritten every step.
    - A larger permanent FIFO buffer that `commit_success` flushes an env's
      current lookback window into, the moment that env achieves a genuine
      landing this step.
    """

    def __init__(
        self,
        obs_current_dim: int,
        obs_history_dim: int,
        action_dim: int,
        num_envs: int,
        device: str,
        capacity: int = 8192,
        lookback: int = 40,
    ) -> None:
        self.capacity = capacity
        self.lookback = lookback
        self.num_envs = num_envs
        self.device = device

        self._buf_obs_current = torch.zeros(capacity, obs_current_dim, device=device)
        self._buf_obs_history = torch.zeros(capacity, obs_history_dim, device=device)
        self._buf_actions = torch.zeros(capacity, action_dim, device=device)
        self._size = 0
        self._ptr = 0

        self._roll_obs_current = torch.zeros(lookback, num_envs, obs_current_dim, device=device)
        self._roll_obs_history = torch.zeros(lookback, num_envs, obs_history_dim, device=device)
        self._roll_actions = torch.zeros(lookback, num_envs, action_dim, device=device)
        self._roll_valid = torch.zeros(lookback, num_envs, dtype=torch.bool, device=device)
        self._roll_ptr = 0

    def record_step(
        self,
        obs_current: torch.Tensor,
        obs_history: torch.Tensor,
        actions: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> None:
        """Call once per real env step with the (obs, action) the policy
        actually acted on, BEFORE overwriting obs with the post-step values.

        reset_mask: envs whose episode just ended on this exact step (so
        their rolling window must not leak into whatever unrelated episode
        starts running in that env slot next).
        """
        if reset_mask.any():
            self._roll_valid[:, reset_mask] = False

        p = self._roll_ptr
        self._roll_obs_current[p] = obs_current.detach()
        self._roll_obs_history[p] = obs_history.detach()
        self._roll_actions[p] = actions.detach()
        self._roll_valid[p] = True
        self._roll_ptr = (p + 1) % self.lookback

    def commit_success(self, env_ids: torch.Tensor) -> None:
        """Flush the full valid lookback window for each of env_ids into the
        permanent buffer. Call with the set of envs that newly achieved a
        genuine (non-free) blue landing on the current step."""
        if env_ids.numel() == 0:
            return
        order = (torch.arange(self.lookback, device=self.device) + self._roll_ptr) % self.lookback
        for eid in env_ids.tolist():
            valid = self._roll_valid[order, eid]
            idxs = order[valid]
            n = idxs.numel()
            if n == 0:
                continue
            obs_current = self._roll_obs_current[idxs, eid]
            obs_history = self._roll_obs_history[idxs, eid]
            actions = self._roll_actions[idxs, eid]
            self._push_batch(obs_current, obs_history, actions)

    def _push_batch(self, obs_current: torch.Tensor, obs_history: torch.Tensor, actions: torch.Tensor) -> None:
        n = obs_current.shape[0]
        if n >= self.capacity:
            obs_current, obs_history, actions = obs_current[-self.capacity:], obs_history[-self.capacity:], actions[-self.capacity:]
            n = self.capacity
        end = self._ptr + n
        if end <= self.capacity:
            self._buf_obs_current[self._ptr:end] = obs_current
            self._buf_obs_history[self._ptr:end] = obs_history
            self._buf_actions[self._ptr:end] = actions
        else:
            first = self.capacity - self._ptr
            self._buf_obs_current[self._ptr:] = obs_current[:first]
            self._buf_obs_history[self._ptr:] = obs_history[:first]
            self._buf_actions[self._ptr:] = actions[:first]
            self._buf_obs_current[:end - self.capacity] = obs_current[first:]
            self._buf_obs_history[:end - self.capacity] = obs_history[first:]
            self._buf_actions[:end - self.capacity] = actions[first:]
        self._ptr = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return self._buf_obs_current[idx], self._buf_obs_history[idx], self._buf_actions[idx]
