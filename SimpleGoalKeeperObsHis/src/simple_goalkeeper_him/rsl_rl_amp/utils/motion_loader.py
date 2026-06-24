# src/simple_goalkeeper_him/rsl_rl_amp/utils/motion_loader.py
"""Expert AMP obs sampler for T1 goalkeeper motions (2-discriminator, 21-DOF).

Two motion groups — left and right — each backed by multiple NPZ files that are
concatenated into a single frame pool at load time.  One GoalkeeperMotionLoader
instance per group; the runner routes AMP reward/loss by env motion_type_id.
"""
from pathlib import Path
import numpy as np
import torch


MOTION_NAMES = ["left", "right"]

LEFT_FILES = [
    "LeftDoubleStep_own_booster_t1.npz",
    "LeftQuickStep_own_booster_t1.npz",
    "LeftStep_own_booster_t1.npz",
    "LeftTripleStep_own_booster_t1.npz",
]
RIGHT_FILES = [
    "RightDoubleStep_own_booster_t1.npz",
    "Rightstep_own_booster_t1.npz",
    "RightTripleStep_own_booster_t1.npz",
]
MOTION_FILE_GROUPS: dict[str, list[str]] = {
    "left":  LEFT_FILES,
    "right": RIGHT_FILES,
}


class GoalkeeperMotionLoader:
    """Loads one or more NPZ motion files and yields expert (s, s_next) pairs of joint_pos.

    When given a list of paths the joint_pos arrays are concatenated along the
    time axis so sampling is uniform across all files in the group.
    """

    def __init__(self, npz_paths: "list[str] | str", device: str = "cpu"):
        self.device = device
        if isinstance(npz_paths, (str, Path)):
            npz_paths = [npz_paths]

        frames_list: list[torch.Tensor] = []
        self.joint_names: list | None = None

        for path in npz_paths:
            d = np.load(str(path), allow_pickle=True)
            # joint_pos: (T, 21) float32 — absolute DOF angles at 50 Hz
            jp = torch.tensor(d["joint_pos"], dtype=torch.float32, device=device)
            frames_list.append(jp)
            if self.joint_names is None and "joint_names" in d:
                self.joint_names = list(d["joint_names"])

        self.frames: torch.Tensor = torch.cat(frames_list, dim=0)   # (T_total, 21)
        self.T = self.frames.shape[0]
        self.observation_dim = self.frames.shape[1] * 2  # 42

    def feed_forward_generator(self, mini_batch_size: int):
        """Yield (s, s_next) of shape (mini_batch_size, 21) each, repeated.

        Temporal jitter: offset ~ U(0.75, 1.25) steps, rounded and clamped.
        Mirrors G1 upstream MotionLib bilinear interpolation behaviour.
        """
        max_valid_frame = self.T - 1
        while True:
            frame_idx = torch.randint(0, max_valid_frame, (mini_batch_size,), device=self.device)
            jitter = (0.75 + 0.5 * torch.rand(mini_batch_size, device=self.device))
            offset = torch.clamp(jitter.round().long(), min=1)
            next_frame_idx = torch.clamp(frame_idx + offset, max=max_valid_frame)
            yield self.frames[frame_idx], self.frames[next_frame_idx]

    def get_obs(self, num_samples: int) -> torch.Tensor:
        """Return (N, 42) concat of two consecutive random frames."""
        idx = torch.randint(0, self.T - 1, (num_samples,), device=self.device)
        return torch.cat([self.frames[idx], self.frames[idx + 1]], dim=-1)
