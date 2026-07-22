import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, states, next_states):
        """Add new states to memory."""
        
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def clear(self):
        """Discard all stored transitions. FIX 2026-07-08 (interceptV2DualDis):
        G1's AMP discriminator (Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/
        him_ppo.py) has no persistent replay buffer at all -- its "policy"
        sample for the discriminator loss is drawn directly from the same
        on-policy HIMRolloutStorage minibatch as everything else, and that
        storage is cleared every update() call. A port using this
        ReplayBuffer as a large (e.g. 250k-transition) FIFO queue that
        persists across many updates trains the discriminator against a
        stale mix of past-policy behavior instead of the current policy --
        this method lets a caller reproduce G1's on-policy-only behavior by
        clearing the buffer at the end of every update(), same as the main
        rollout storage.
        """
        self.step = 0
        self.num_samples = 0

    def feed_forward_generator(self, num_mini_batches, num_learning_epochs):
        """FIX 2026-07-21 (AMP-saturation investigation): previously drew
        `num_learning_epochs * num_mini_batches` independent WITH-REPLACEMENT
        samples of size `mini_batch_size` (the cross-region total,
        num_envs*num_transitions_per_env // num_mini_batches -- not this
        region's own share) from this region's pool every single call. In
        the current 4-region/4-minibatch config that size coincidentally
        equals the region's entire pool, so each of the 20 draws
        independently resampled ~all ~98k unique on-policy transitions with
        replacement -- ~20x average reuse per transition, high per-draw
        variance, no relation to the deterministic 5x (num_learning_epochs)
        reuse G1's HIMRolloutStorage.mini_batch_generator gives its own
        mask-filtered discriminator batches (permutation generated ONCE per
        update(), same partition replayed unchanged across every epoch --
        confirmed via him_rollout_storage.py:137 and the identical,
        already-verified pattern in this file's own
        MultiDiscAMPPPO._mini_batch_generator_with_next). A discriminator
        retrained ~20x per update against a narrow, independently-resampled
        slice of the CURRENT policy's own rollout is a textbook setup for
        exactly the fast, permanent saturation this investigation is
        chasing (AMP paper section 6.3: replay/reuse scheme is the named
        mitigation for a discriminator overfitting to the current policy's
        narrow distribution).

        Fix: match G1's structure instead of an ad hoc size. Permute this
        region's pool ONCE per call (i.e. once per update(), matching the
        main PPO generator's `torch.randperm` timing and the caller's own
        `_mini_batch_generator_with_next`), split into `num_mini_batches`
        equal chunks (no replacement within a chunk), and replay the SAME
        chunks unchanged across all `num_learning_epochs` passes -- each
        unique on-policy transition is now seen by the discriminator exactly
        `num_learning_epochs` times (5, matching G1), deterministically, not
        an average of ~20 with unbounded variance. Batch size per draw is
        now this region's own fair share (`num_samples // num_mini_batches`)
        rather than the cross-region total -- confirmed safe: the LSGAN loss
        (`multi_disc_amp_ppo.py`'s `expert_loss`/`policy_loss`) reduces each
        side independently via `torch.nn.MSELoss` against a same-shaped
        target, exactly like G1's own variable-sized per-motion masked
        batches -- there is no requirement that policy and expert draw sizes
        match each other.
        """
        chunk_size = max(1, self.num_samples // num_mini_batches)
        permutation = np.random.permutation(self.num_samples)
        chunks = [permutation[i * chunk_size:(i + 1) * chunk_size] for i in range(num_mini_batches)]
        for _ in range(num_learning_epochs):
            for sample_idxs in chunks:
                yield (self.states[sample_idxs].to(self.device),
                       self.next_states[sample_idxs].to(self.device))
