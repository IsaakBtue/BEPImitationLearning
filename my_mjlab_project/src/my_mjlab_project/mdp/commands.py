"""Multi-motion command: randomly assigns one of N motion clips per env at reset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg, MotionLoader
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Ball end-target ranges per motion type (y_min, y_max, z_min, z_max) in env-local frame.
# Matches g1_29_config.py ranges_0 … ranges_5 "height" and "width" fields.
_BALL_END_RANGES = [
    ( 0.2,  1.2, 0.4, 1.2),  # 0 lefthand
    (-1.2, -0.2, 0.4, 1.2),  # 1 righthand
    ( 0.0,  1.0, 1.2, 1.6),  # 2 leftjump
    (-1.0,  0.0, 1.2, 1.6),  # 3 rightjump
    ( 0.2,  1.2, 0.1, 0.3),  # 4 leftstep
    (-1.2, -0.2, 0.1, 0.3),  # 5 rightstep
]


class MultiMotionCommand(MotionCommand):
    """MotionCommand that loads multiple clips and assigns one per env at reset."""

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        # self.motion is loaders[0], already loaded by parent via cfg.motion_file
        self.loaders: list[MotionLoader] = [self.motion]
        for mf in cfg.motion_files[1:]:
            self.loaders.append(MotionLoader(mf, self.body_indexes, device=self.device))

        self.motion_type_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # (num_clips,) lookup table for clip lengths — used in _update_command
        self._time_step_totals = torch.tensor(
            [loader.time_step_total for loader in self.loaders],
            dtype=torch.long,
            device=self.device,
        )

        # Recompute bin bookkeeping using the max clip length
        max_total = int(self._time_step_totals.max().item())
        self.bin_count = int(max_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = kernel / kernel.sum()

    # ------------------------------------------------------------------
    # Internal gather helpers
    # ------------------------------------------------------------------

    def _gather(self, attr: str) -> torch.Tensor:
        """Return (num_envs, ...) tensor gathered per-env from the correct loader."""
        ref = getattr(self.loaders[0], attr)
        out = torch.zeros((self.num_envs,) + ref.shape[1:], dtype=ref.dtype, device=self.device)
        for m_idx, loader in enumerate(self.loaders):
            mask = self.motion_type_ids == m_idx
            if not mask.any():
                continue
            env_ids = mask.nonzero(as_tuple=True)[0]
            t = self.time_steps[env_ids].clamp(0, loader.time_step_total - 1)
            out[env_ids] = getattr(loader, attr)[t]
        return out

    def _gather_anchor(self, attr: str) -> torch.Tensor:
        """Return (num_envs, feat_dim) tensor for the anchor body only."""
        ref = getattr(self.loaders[0], attr)
        out = torch.zeros((self.num_envs,) + ref.shape[2:], dtype=ref.dtype, device=self.device)
        for m_idx, loader in enumerate(self.loaders):
            mask = self.motion_type_ids == m_idx
            if not mask.any():
                continue
            env_ids = mask.nonzero(as_tuple=True)[0]
            t = self.time_steps[env_ids].clamp(0, loader.time_step_total - 1)
            out[env_ids] = getattr(loader, attr)[t, self.motion_anchor_body_index]
        return out

    # ------------------------------------------------------------------
    # Motion data properties — override all MotionCommand properties
    # ------------------------------------------------------------------

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._gather("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather("body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._gather_anchor("body_pos_w") + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._gather_anchor("body_quat_w")

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._gather_anchor("body_lin_vel_w")

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._gather_anchor("body_ang_vel_w")

    # ------------------------------------------------------------------
    # Command update / resample — override to handle per-env clip lengths
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Randomly assign a motion clip to each resetting env
        self.motion_type_ids[env_ids] = torch.randint(
            0, len(self.loaders), (len(env_ids),), device=self.device
        )

        # Sample start frame uniformly within the assigned clip
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        else:
            # adaptive falls back to per-clip uniform for multi-motion
            for m_idx, loader in enumerate(self.loaders):
                mask = self.motion_type_ids[env_ids] == m_idx
                if not mask.any():
                    continue
                count = int(mask.sum().item())
                self.time_steps[env_ids[mask]] = torch.randint(
                    0, loader.time_step_total, (count,), device=self.device
                )

        # Reference-state initialization (RSI)
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        range_list = [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, :3]
        root_ori = quat_mul(
            quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]),
            root_ori,
        )

        range_list = [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,
        )

        self._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )

        # Spawn ball trajectory matched to the assigned motion type
        if self.cfg.ball_name:
            try:
                ball: Entity = self._env.scene[self.cfg.ball_name]
                self._reset_ball(env_ids, ball)
            except (KeyError, AttributeError):
                pass

    def _reset_ball(self, env_ids: torch.Tensor, ball: Entity) -> None:
        """Place ball at a trajectory that matches the assigned motion type."""
        g = 9.81
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]  # (n, 3)
        motion_types = self.motion_type_ids[env_ids]    # (n,)

        # Ball start: x=3-5m ahead of goal, y/z broadly randomized
        x_start = sample_uniform(3.0, 5.0, (n,), device=self.device)

        # Build per-env end-target ranges from the motion type table
        end_ranges = torch.tensor(_BALL_END_RANGES, device=self.device)  # (6, 4)
        per_env_ranges = end_ranges[motion_types]  # (n, 4)

        y_end = sample_uniform(
            per_env_ranges[:, 0], per_env_ranges[:, 1], (n,), device=self.device
        )
        z_end = sample_uniform(
            per_env_ranges[:, 2], per_env_ranges[:, 3], (n,), device=self.device
        )
        y_start = sample_uniform(-1.8, 1.8, (n,), device=self.device)
        z_start = sample_uniform(0.3, 1.8, (n,), device=self.device)

        t_flight = sample_uniform(0.5, 1.0, (n,), device=self.device)

        # Compute velocity to reach end target from start in t_flight seconds
        dx = -x_start - 0.3  # end x ≈ -0.3 (slightly behind goal line)
        dy = y_end - y_start
        dz = z_end - z_start

        vx = dx / t_flight
        vy = dy / t_flight
        vz = (dz + 0.5 * g * t_flight**2) / t_flight

        # Positions in world frame
        ball_pos_w = torch.stack([
            origins[:, 0] + x_start,
            origins[:, 1] + y_start,
            z_start,
        ], dim=1)
        ball_quat_w = torch.zeros((n, 4), device=self.device)
        ball_quat_w[:, 0] = 1.0  # identity wxyz
        ball_pose = torch.cat([ball_pos_w, ball_quat_w], dim=-1)  # (n, 7)

        ball_vel = torch.stack([vx, vy, vz], dim=1)  # (n, 3)
        ball_ang_vel = torch.zeros((n, 3), device=self.device)
        ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=-1)  # (n, 6)

        ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
        ball.write_root_link_velocity_to_sim(ball_velocity, env_ids=env_ids)

    def _update_command(self) -> None:
        self.time_steps += 1
        per_env_totals = self._time_step_totals[self.motion_type_ids]
        env_ids = torch.where(self.time_steps >= per_env_totals)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)
        self.update_relative_body_poses()

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    """Config for MultiMotionCommand. Set ``motion_files``; ``motion_file`` is derived automatically."""

    motion_files: tuple[str, ...] = field(default_factory=tuple)
    motion_file: str = ""  # set from motion_files[0] in __post_init__
    ball_name: str = "ball"  # scene entity name for the ball; set to "" to skip ball reset

    def __post_init__(self) -> None:
        if self.motion_files:
            self.motion_file = self.motion_files[0]

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
