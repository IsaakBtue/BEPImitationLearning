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

# Ball end-target ranges per motion type (y_min, y_max, z_min, z_max) in env-local frame.
# Ball approaches from +X (forwards); robot faces +X so left hand is at -Y, right hand at +Y.
# Order matches MOTION_NAMES in rsl_rl_amp/utils/motion_loader.py:
#   ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
# These are the FULL (difficulty=1.0) ranges — easy ranges are interpolated at runtime.
# Lateral Y max reduced 0.84→0.65: at 0.84m the robot needs a lateral step + full arm
# extension simultaneously — ~50% of shots were unreachable. 0.65m is within T1's reach.
_BALL_END_RANGES = [
    (-0.65, -0.15, 0.40, 1.15),  # 0 lefthand  — left side (-Y), mid-height arm catch
    ( 0.15,  0.65, 0.40, 1.15),  # 1 righthand — right side (+Y), mid-height arm catch
    (-0.65, -0.15, 0.85, 1.40),  # 2 leftjump  — left side (-Y), high (diving jump)
    ( 0.15,  0.65, 0.85, 1.40),  # 3 rightjump — right side (+Y), high (diving jump)
    (-0.65, -0.15, 0.20, 0.65),  # 4 leftstep  — left side (-Y), low (lateral step)
    ( 0.15,  0.65, 0.20, 0.65),  # 5 rightstep — right side (+Y), low (lateral step)
]

# Easy (difficulty=0.0) end-target ranges, used as the lerp starting point.
# difficulty=0: centre-ish shots easy to intercept.
# difficulty=1: full range matching _BALL_END_RANGES.
_BALL_END_RANGES_EASY = [
    (-0.35, -0.10, 0.55, 1.05),  # 0 lefthand  — narrower, easier
    ( 0.10,  0.35, 0.55, 1.05),  # 1 righthand — narrower, easier
    (-0.35, -0.10, 0.90, 1.25),  # 2 leftjump  — high, narrower
    ( 0.10,  0.35, 0.90, 1.25),  # 3 rightjump — high, narrower
    (-0.35, -0.10, 0.30, 0.55),  # 4 leftstep  — low, narrower
    ( 0.10,  0.35, 0.30, 0.55),  # 5 rightstep — low, narrower
]


class MultiMotionCommand(MotionCommand):
    """MotionCommand that loads multiple clips and assigns one per env at reset."""

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        # If --motion-file was passed via CLI, the play script overrides cfg.motion_file
        # before the env is built. In that case use only that single file.
        if cfg.motion_file and cfg.motion_file not in cfg.motion_files:
            self.loaders = [MotionLoader(cfg.motion_file, self.body_indexes, device=self.device)]
        else:
            self.loaders = [self.motion]
            for mf in cfg.motion_files[1:]:
                self.loaders.append(MotionLoader(mf, self.body_indexes, device=self.device))

        self.motion_type_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cycle_counter: int = 0

        # Static permanent partitioning (mirrors G1 end_regions).
        # If enabled, each env is permanently assigned to one motion type at init
        # and never reassigned at episode reset. This matches G1's _init_buffers:
        #   six = num_envs // 6; end_regions = cat([zeros(six), ones(six), ...])
        # Benefit: each env specialises in one motion style → AMP discriminator for
        # that style gets clean, consistent training signal from the same env group
        # every rollout, rather than mixed signal from all envs cycling through styles.
        if cfg.static_partition:
            n = len(self.loaders)
            six = self.num_envs // n
            parts = []
            for m_idx in range(n):
                # Last group absorbs any remainder (num_envs % n extra envs).
                count = six if m_idx < n - 1 else self.num_envs - six * (n - 1)
                parts.append(torch.full((count,), m_idx, dtype=torch.long, device=self.device))
            self.motion_type_ids = torch.cat(parts)
            self._static_motion_type_ids = self.motion_type_ids.clone()

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
        # npz natively faces +X — no rotation needed.
        return self._gather("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        # npz natively faces +X — no rotation needed.
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
        if self.cfg.static_partition:
            # Restore permanent assignment — never re-randomise, even on mid-episode clip loops.
            self.motion_type_ids[env_ids] = self._static_motion_type_ids[env_ids]
        elif self.cfg.cycle_motions:
            n = len(self.loaders)
            assigned = torch.tensor(
                [(self._cycle_counter + i) % n for i in range(len(env_ids))],
                dtype=torch.long,
                device=self.device,
            )
            self.motion_type_ids[env_ids] = assigned
            self._cycle_counter = (self._cycle_counter + len(env_ids)) % n
        else:
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
        # Booster motion data ground reference is ~8 cm below simulation floor.
        # Lifting root by 0.05 m puts feet at ~0.05 m clearance, preventing underground spawn.
        root_pos[:, 2] += 0.05
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

        # Ball approaches from +X (forwards from robot).
        x_start = sample_uniform(3.0, 4.5, (n,), device=self.device)

        # Difficulty curriculum: linearly interpolate between easy and full ranges.
        # difficulty=0.0 (easy): narrow centre zone; difficulty=1.0: full range.
        difficulty = float(getattr(self._env, "_ball_difficulty", 0.0))
        difficulty = max(0.0, min(1.0, difficulty))

        end_ranges_full = torch.tensor(_BALL_END_RANGES, device=self.device)       # [M, 4]
        end_ranges_easy = torch.tensor(_BALL_END_RANGES_EASY, device=self.device)  # [M, 4]

        # Interpolate: full_range * difficulty + easy_range * (1 - difficulty)
        end_ranges = end_ranges_easy + difficulty * (end_ranges_full - end_ranges_easy)

        per_env_ranges = end_ranges[motion_types]  # [n, 4]

        y_end = sample_uniform(per_env_ranges[:, 0], per_env_ranges[:, 1], (n,), device=self.device)
        z_end = sample_uniform(per_env_ranges[:, 2], per_env_ranges[:, 3], (n,), device=self.device)
        # Narrowed from ±1.8 m: wide y_start caused extreme curved trajectories that
        # were physically impossible to intercept regardless of policy quality.
        y_start = sample_uniform(-0.8, 0.8, (n,), device=self.device)
        # Narrowed from (0.3, 1.8): extreme start heights produced very high vz,
        # making ball timing unpredictable and the apex unreachable.
        z_start = sample_uniform(0.5, 1.4, (n,), device=self.device)

        # Minimum t_flight raised 0.4→0.5 s: at t=0.4 and x_start=4.5 m, vx=-9.6 m/s —
        # too fast for a 50 Hz controller to react. 0.5 s keeps vx ≤ -9.6 m/s at worst.
        t_flight = sample_uniform(0.5, 1.0, (n,), device=self.device)

        dx = -x_start - 0.3          # target X ≈ -0.3 (just behind goal line)
        dy = y_end - y_start
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
        # New system uses X axis: ball goes from +x_start → −0.3 (goal).
        # catch_prop = fraction of X travel when ball is at x_local = 0.1 (arm-reach plane).
        # x_start_local > 0; total dx = -(x_start + 0.3).
        # At x_local=0.1: fraction = (x_start - 0.1) / (x_start + 0.3)
        # end_target_w = ball_start_w + delta_w * catch_prop
        # ----------------------------------------------------------------
        catch_prop = (x_start - 0.1) / (x_start + 0.3)      # [n]
        catch_prop = catch_prop.clamp(0.0, 1.0)

        delta_w = torch.stack([
            dx,                          # vx * t_flight  (world x)
            (y_end - y_start),          # vy * t_flight
            (z_end - z_start),          # vz*t - 0.5g*t² = dz → dz is just delta, not full arc
        ], dim=1)                        # [n, 3]

        # ball_start in world frame
        ball_start_world = ball_pos_w    # [n, 3]

        # end_target in world frame = start + delta * catch_prop
        end_target_w = ball_start_world + delta_w * catch_prop.unsqueeze(-1)   # [n, 3]
        # Clip Y so it stays within a reasonable arm-reach zone (≈ original clip).
        end_target_w[:, 1] = end_target_w[:, 1].clamp(
            origins[:, 1] - 1.0, origins[:, 1] + 1.0
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
                self._env.num_envs, dtype=torch.long, device=self.device
            )
        self._env._catchstep[env_ids] = 50

        # startstep: per-env random threshold; ball obs reveal when catchstep < startstep.
        # Upstream uses a single scalar updated once per N steps; port randomises per-env
        # per-episode for better diversity. Range: 50 - randint(3,10) → [40, 47].
        if not hasattr(self._env, "_startstep"):
            self._env._startstep = torch.zeros(
                self._env.num_envs, dtype=torch.long, device=self.device
            )
        self._env._startstep[env_ids] = 50 - torch.randint(
            3, 11, (len(env_ids),), device=self.device
        )

        # vanish_step: random per-episode step at which ball obs disappear mid-flight.
        # Mirrors upstream vanish_step = randint(0, 30), re-sampled at each episode reset.
        # Upstream uses catchstep > vanish_step; the port uses _ball_visible_step > _vanish_step
        # (counts consecutive flying steps), which is semantically equivalent but more robust.
        if not hasattr(self._env, "_vanish_step"):
            self._env._vanish_step = torch.zeros(
                self._env.num_envs, dtype=torch.long, device=self.device
            )
        self._env._vanish_step[env_ids] = torch.randint(
            0, 30, (len(env_ids),), device=self.device
        )

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
    cycle_motions: bool = False
    """If True, assign motion types round-robin across episodes instead of randomly."""
    static_partition: bool = False
    """If True, permanently assign each env to one motion type at init (like G1 end_regions).
    Envs are split into equal groups: group 0 → motion 0 (lefthand), group 1 → motion 1, etc.
    The assignment never changes at episode reset, giving each AMP discriminator a dedicated,
    consistent env group that always trains the same motion style."""

    def __post_init__(self) -> None:
        if self.motion_files:
            self.motion_file = self.motion_files[0]

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
