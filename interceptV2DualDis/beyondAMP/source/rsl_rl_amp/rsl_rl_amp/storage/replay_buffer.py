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

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.next_states[sample_idxs].to(self.device))
