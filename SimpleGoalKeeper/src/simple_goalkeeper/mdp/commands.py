"""Ghost motion command for overlay visualization."""
from __future__ import annotations

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class GhostMotionCommand(MotionCommand):
    """Shows a ghost robot overlay without RSI teleportation.

    Identical to MotionCommand but _resample_command() only resets the
    motion clock — it never teleports the simulated robot. The ghost
    restarts from frame 0 at every episode reset.
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = 0


class GhostMotionCommandCfg(MotionCommandCfg):
    def build(self, env) -> GhostMotionCommand:
        return GhostMotionCommand(self, env)
