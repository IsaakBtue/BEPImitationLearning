"""Ghost motion command for overlay visualization."""
from __future__ import annotations

import copy

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer


class GhostMotionCommand(MotionCommand):
    """Shows a ghost robot overlay without RSI teleportation.

    Identical to MotionCommand but _resample_command() only resets the
    motion clock — it never teleports the simulated robot. The ghost
    restarts from frame 0 at every episode reset.
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = 0

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Ghost visualization with correct absolute joint angles.

        NPZ stores joint_pos relative to default (for AMP discriminator
        matching joint_pos_rel). MuJoCo qpos expects absolute angles, so
        we restore them by adding default_joint_pos back.
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode != "ghost":
            super()._debug_vis_impl(visualizer)
            return

        if self._ghost_model is None:
            self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(self._ghost_model.ngeom):
                if (
                    self._ghost_model.geom_contype[gi] != 0
                    or self._ghost_model.geom_conaffinity[gi] != 0
                ):
                    self._ghost_model.geom_rgba[gi, 3] = 0
                else:
                    self._ghost_model.geom_rgba[gi] = self._ghost_color

        entity = self._env.scene[self.cfg.entity_name]
        indexing = entity.indexing
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_q_adr = indexing.joint_q_adr.cpu().numpy()
        default_joint_pos = self.robot.data.default_joint_pos[0].cpu().numpy()

        for batch in env_indices:
            qpos = np.zeros(self._env.sim.mj_model.nq)
            qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
            qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
            qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy() + default_joint_pos

            visualizer.add_ghost_mesh(
                qpos,
                model=self._ghost_model,
                label=f"ghost_{batch}",
            )


class GhostMotionCommandCfg(MotionCommandCfg):
    def build(self, env) -> GhostMotionCommand:
        return GhostMotionCommand(self, env)
