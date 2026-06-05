# src/imitationlearningbooster/rsl_rl_amp/utils/motion_loader.py
"""Expert AMP obs sampler for T1 goalkeeper motions.

Adapted from ccrpRepo/AMP_mjlab rsl_rl/utils/motion_loader.py.
Key differences: joint_pos only (46-dim) instead of full body kinematics (195-dim).
One instance per motion file; runner creates 6 and routes by motion_type_id.
"""
from pathlib import Path
import numpy as np
import torch


MOTION_NAMES = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]


class GoalkeeperMotionLoader:
    """Loads one npz motion file and yields expert (s, s_next) pairs of joint_pos."""

    def __init__(self, npz_path: str, device: str = "cpu"):
        self.device = device
        d = np.load(npz_path, allow_pickle=True)
        # joint_pos: (150, 23) float32
        jp = torch.tensor(d["joint_pos"], dtype=torch.float32, device=device)
        self.frames: torch.Tensor = jp          # (T, 23)
        self.T = jp.shape[0]
        self.observation_dim = jp.shape[1] * 2  # 46
        self.joint_names: list = list(d["joint_names"]) if "joint_names" in d else None

    def feed_forward_generator(self, mini_batch_size: int):
        """Yield (s, s_next) of shape (mini_batch_size, 23) each, repeated.

        Fix 3.1: Apply temporal jitter when sampling the next frame, broadening the
        temporal distribution of expert transitions to match G1 upstream MotionLib
        behaviour (ratio = fps/env_fps * U(0.25, 1.25) bilinear interpolation).
        Here we use a simplified discrete version: offset ~ U(0.75, 1.25) steps,
        rounded to nearest integer and clamped to valid range.
        """
        max_valid_frame = self.T - 1
        while True:
            frame_idx = torch.randint(0, max_valid_frame, (mini_batch_size,), device=self.device)
            # Temporal jitter: sample next frame offset in [0.75, 1.25] steps
            jitter = (0.75 + 0.5 * torch.rand(mini_batch_size, device=self.device))  # U(0.75, 1.25)
            offset = torch.clamp(jitter.round().long(), min=1)
            next_frame_idx = torch.clamp(frame_idx + offset, max=max_valid_frame)
            yield self.frames[frame_idx], self.frames[next_frame_idx]

    def get_obs(self, num_samples: int) -> torch.Tensor:
        """Return (N, 46) concat of two consecutive random frames."""
        idx = torch.randint(0, self.T - 1, (num_samples,), device=self.device)
        return torch.cat([self.frames[idx], self.frames[idx + 1]], dim=-1)
