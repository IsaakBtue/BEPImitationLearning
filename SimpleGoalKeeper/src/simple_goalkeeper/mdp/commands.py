"""Ghost motion command for overlay visualization."""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg, MotionLoader


class GhostMotionCommand(MotionCommand):
    """Shows the reference motion ghost in world space without RSI teleportation.

    Identical to MotionCommand except _resample_command() only resets the motion
    clock — it never teleports the simulated robot. The ghost plays the reference
    motion at its NPZ world-space position, independent of the robot.
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = 0


class GhostMotionCommandCfg(MotionCommandCfg):
    def build(self, env) -> GhostMotionCommand:
        return GhostMotionCommand(self, env)


class CyclingGhostMotionCommand(GhostMotionCommand):
    """Ghost overlay that cycles through a list of NPZ motion files in order.

    Each time the current clip ends (time_steps reaches the end), the next file
    in the list is loaded and the ghost restarts from frame 0.
    """

    def __init__(self, cfg: "CyclingGhostMotionCommandCfg", env) -> None:
        self._motion_files: List[str] = list(cfg.motion_files)
        self._motion_index: int = 0
        cfg.motion_file = self._motion_files[0]
        super().__init__(cfg, env)
        print(f"[GhostMotion] Playing: {Path(self._motion_files[0]).name}  (1/{len(self._motion_files)})")

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._motion_index = (self._motion_index + 1) % len(self._motion_files)
        self.motion = MotionLoader(
            self._motion_files[self._motion_index],
            self.body_indexes,
            device=self.device,
        )
        self.time_steps[env_ids] = 0
        idx = self._motion_index
        name = Path(self._motion_files[idx]).name
        print(f"[GhostMotion] Playing: {name}  ({idx + 1}/{len(self._motion_files)})")


class CyclingGhostMotionCommandCfg(MotionCommandCfg):
    motion_files: List[str] = None  # set at build time

    def build(self, env) -> CyclingGhostMotionCommand:
        return CyclingGhostMotionCommand(self, env)
