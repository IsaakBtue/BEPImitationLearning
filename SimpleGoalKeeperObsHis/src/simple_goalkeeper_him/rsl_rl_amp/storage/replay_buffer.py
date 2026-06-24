# src/simple_goalkeeper_him/rsl_rl_amp/storage/replay_buffer.py
"""Replay buffer for AMP expert transitions."""
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, buffer_size, device="cpu"):
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.device = device
        self.buffer_s = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_s_next = torch.zeros(buffer_size, obs_dim, device=device)
        self.head = 0
        self.count = 0

    def insert(self, s, s_next):
        """Insert batch of transitions into ring buffer."""
        batch_size = s.shape[0]
        if self.head + batch_size <= self.buffer_size:
            self.buffer_s[self.head:self.head+batch_size] = s
            self.buffer_s_next[self.head:self.head+batch_size] = s_next
        else:
            part1 = self.buffer_size - self.head
            self.buffer_s[self.head:] = s[:part1]
            self.buffer_s_next[self.head:] = s_next[:part1]
            self.buffer_s[:batch_size-part1] = s[part1:]
            self.buffer_s_next[:batch_size-part1] = s_next[part1:]
        self.head = (self.head + batch_size) % self.buffer_size
        self.count = min(self.count + batch_size, self.buffer_size)

    def feed_forward_generator(self, mini_batch_size):
        """Yield mini-batches from buffer."""
        while True:
            indices = torch.randint(0, self.count, (mini_batch_size,), device=self.device)
            yield self.buffer_s[indices], self.buffer_s_next[indices]

    def ready(self, mini_batch_size):
        """Check if buffer has enough samples."""
        return self.count >= mini_batch_size
