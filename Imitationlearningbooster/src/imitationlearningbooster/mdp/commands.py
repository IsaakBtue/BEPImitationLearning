"""Multi-motion command for Booster T1 goalkeeper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg, MotionLoader
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Ball end-target ranges per motion type (x_min, x_max, z_min, z_max) in env-local frame.
# Ball approaches from +Y; robot faces +Y so left hand is at -X, right hand at +X.
# Order matches MOTION_NAMES in rsl_rl_amp/utils/motion_loader.py:
#   ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
# These are the FULL (difficulty=1.0) ranges — easy ranges are interpolated at runtime.
_BALL_END_RANGES = [
    (-0.84, -0.2, 0.40, 1.20),  # 0 lefthand  — left side (-X), mid-height arm catch
    ( 0.20,  0.84, 0.40, 1.20),  # 1 righthand — right side (+X), mid-height arm catch
    (-0.84, -0.2, 0.80, 1.50),  # 2 leftjump  — left side (-X), high (diving jump)
    ( 0.20,  0.84, 0.80, 1.50),  # 3 rightjump — right side (+X), high (diving jump)
    (-0.84, -0.2, 0.20, 0.70),  # 4 leftstep  — left side (-X), low (lateral step)
    ( 0.20,  0.84, 0.20, 0.70),  # 5 rightstep — right side (+X), low (lateral step)
]

# Easy (difficulty=0.0) end-target ranges, used as the lerp starting point.
# difficulty=0: centre-ish shots easy to intercept.
# difficulty=1: full range matching _BALL_END_RANGES.
_BALL_END_RANGES_EASY = [
    (-0.40, -0.2, 0.55, 1.05),  # 0 lefthand  — narrower, easier
    ( 0.20,  0.40, 0.55, 1.05),  # 1 righthand — narrower, easier
    (-0.40, -0.2, 0.90, 1.30),  # 2 leftjump  — high, narrower
    ( 0.20,  0.40, 0.90, 1.30),  # 3 rightjump — high, narrower
    (-0.40, -0.2, 0.30, 0.60),  # 4 leftstep  — low, narrower
    ( 0.20,  0.40, 0.30, 0.60),  # 5 rightstep — low, narrower
]


class MultiMotionCommand(MotionCommand):
    """MotionCommand that loads multiple clips and assigns one per env at reset."""

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.loaders: list[MotionLoader] = [self.motion]
        for mf in cfg.motion_files[1:]:
            self.loaders.append(MotionLoader(mf, self.body_indexes, device=self.device))

        self.motion_type_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._time_step_totals = torch.tensor(
            [loader.time_step_total for loader in self.loaders],
            dtype=torch.long,
            device=self.device,
        )

        max_total = int(self._time_step_totals.max().item())
        self.bin_count = int(max_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = kernel / kernel.sum()

    def _gather(self, attr: str) -> torch.Tensor:
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

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        # npz already has corrected foot heights (see fix_motion_foot.py): no extra z shift here.
        pos = self._gather("body_pos_w")
        return pos + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        # convert_booster.py already applied +90° yaw rotation — no correction needed here
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

    def _resample_command(self, env_ids: torch.Tensor, reset_ball: bool = True) -> None:
        self.motion_type_ids[env_ids] = torch.randint(
            0, len(self.loaders), (len(env_ids),), device=self.device
        )

        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        else:
            for m_idx, loader in enumerate(self.loaders):
                mask = self.motion_type_ids[env_ids] == m_idx
                if not mask.any():
                    continue
                count = int(mask.sum().item())
                self.time_steps[env_ids[mask]] = torch.randint(
                    0, loader.time_step_total, (count,), device=self.device
                )

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

        # Only reset the ball at true episode resets, not when the motion loops mid-episode.
        # Matches original: one ball per episode. Caller passes reset_ball=False for loops.
        if reset_ball and self.cfg.ball_name:
            try:
                ball: Entity = self._env.scene[self.cfg.ball_name]
                self._reset_ball(env_ids, ball)
            except (KeyError, AttributeError):
                pass

    def _reset_ball(self, env_ids: torch.Tensor, ball: Entity) -> None:
        # Ensure env_ids is 1D and of correct type
        if env_ids.dim() > 1:
            env_ids = env_ids.squeeze()
        if env_ids.dtype not in (torch.long, torch.int64):
            env_ids = env_ids.long()

        g = 9.81
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        motion_types = self.motion_type_ids[env_ids]

        # Ball approaches from +Y (rotated 90° clockwise vs original +X approach).
        y_start = sample_uniform(3.0, 5.0, (n,), device=self.device)

        # Difficulty curriculum: linearly interpolate between easy and full ranges.
        # difficulty=0.0 (easy): narrow centre zone; difficulty=1.0: full range.
        difficulty = float(getattr(self._env, "_ball_difficulty", 0.0))
        difficulty = max(0.0, min(1.0, difficulty))

        end_ranges_full = torch.tensor(_BALL_END_RANGES, device=self.device)       # [M, 4]
        end_ranges_easy = torch.tensor(_BALL_END_RANGES_EASY, device=self.device)  # [M, 4]

        # Interpolate: full_range * difficulty + easy_range * (1 - difficulty)
        end_ranges = end_ranges_easy + difficulty * (end_ranges_full - end_ranges_easy)

        per_env_ranges = end_ranges[motion_types]  # [n, 4]

        x_end = sample_uniform(per_env_ranges[:, 0], per_env_ranges[:, 1], (n,), device=self.device)
        z_end = sample_uniform(per_env_ranges[:, 2], per_env_ranges[:, 3], (n,), device=self.device)
        x_start = sample_uniform(-1.8, 1.8, (n,), device=self.device)
        z_start = sample_uniform(0.3, 1.8, (n,), device=self.device)

        t_flight = sample_uniform(0.4, 1.0, (n,), device=self.device)

        dx = x_end - x_start
        dy = -y_start - 0.3          # target Y ≈ -0.3 (just behind goal line)
        dz = z_end - z_start

        vx = dx / t_flight
        vy = dy / t_flight
        vz = (dz + 0.5 * g * t_flight**2) / t_flight

        ball_pos_w = torch.stack([
            origins[:, 0] + x_start,
            origins[:, 1] + y_start,
            origins[:, 2] + z_start,
        ], dim=1)
        ball_quat_w = torch.zeros((n, 4), device=self.device)
        ball_quat_w[:, 0] = 1.0
        ball_pose = torch.cat([ball_pos_w, ball_quat_w], dim=-1)

        ball_vel = torch.stack([vx, vy, vz], dim=1)
        ball_ang_vel = torch.zeros((n, 3), device=self.device)
        ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=-1)

        # ----------------------------------------------------------------
        # Compute predicted intercept point (Feature P3A: eereach target).
        # Mirrors upstream assign_ball_states catch_prop formula:
        #     catch_prop = (0.1 - x_start) / (x_end_local - x_start)  [original X axis]
        # Port uses Y axis: ball goes from +y_start → −0.3 (goal).
        # catch_prop = fraction of Y travel when ball is at y_local = 0.1 (arm-reach plane).
        # y_start_local > 0; total dy = -(y_start + 0.3).
        # At y_local=0.1: fraction = (y_start - 0.1) / (y_start + 0.3)
        # end_target_w = ball_start_w + delta_w * catch_prop
        # ----------------------------------------------------------------
        catch_prop = (y_start - 0.1) / (y_start + 0.3)      # [n]
        catch_prop = catch_prop.clamp(0.0, 1.0)

        delta_w = torch.stack([
            (x_end - x_start),          # vx * t_flight
            dy,                          # vy * t_flight  (world y)
            (z_end - z_start),          # vz*t - 0.5g*t² = dz → dz is just delta, not full arc
        ], dim=1)                        # [n, 3]

        # ball_start in world frame
        ball_start_world = ball_pos_w    # [n, 3]

        # end_target in world frame = start + delta * catch_prop
        end_target_w = ball_start_world + delta_w * catch_prop.unsqueeze(-1)   # [n, 3]
        # Clip X so it stays within a reasonable arm-reach zone (≈ original clip).
        end_target_w[:, 0] = end_target_w[:, 0].clamp(
            origins[:, 0] - 1.0, origins[:, 0] + 1.0
        )

        # Store on env for use by eereach reward.
        if not hasattr(self._env, "_ball_end_target"):
            self._env._ball_end_target = torch.zeros(
                self._env.num_envs, 3, dtype=torch.float, device=self.device
            )
        self._env._ball_end_target[env_ids] = end_target_w

        # ----------------------------------------------------------------
        # Catchstep warmup: during the first N steps after reset the ball
        # observations are masked so the policy holds a standing pose while
        # the ball is launched and reaches a stable trajectory.
        # Mirrors upstream: catchstep = 50 (fixed), decremented each step.
        # startstep = 50 − randint(3,10): the mask lifts when catchstep < startstep.
        # ----------------------------------------------------------------
        if not hasattr(self._env, "_catchstep"):
            self._env._catchstep = torch.zeros(
                self._env.num_envs, dtype=torch.int, device=self.device
            )
        self._env._catchstep[env_ids] = 50

        ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
        ball.write_root_link_velocity_to_sim(ball_velocity, env_ids=env_ids)

    def _update_command(self) -> None:
        self.time_steps += 1
        per_env_totals = self._time_step_totals[self.motion_type_ids]
        env_ids = torch.where(self.time_steps >= per_env_totals)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids, reset_ball=False)
        self.update_relative_body_poses()

        # Decrement catchstep warmup counter each step (mirrors upstream post_physics_step).
        if hasattr(self._env, "_catchstep"):
            self._env._catchstep = (self._env._catchstep - 1).clamp(min=0)

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    """Config for MultiMotionCommand."""

    motion_files: tuple[str, ...] = field(default_factory=tuple)
    motion_file: str = ""
    ball_name: str = "ball"

    def __post_init__(self) -> None:
        if self.motion_files:
            self.motion_file = self.motion_files[0]

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
