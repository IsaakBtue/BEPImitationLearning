"""Goalkeeper environment — Isaac Lab port of legged_robot.py.

API changes vs Isaac Gym:
  - Quaternions are wxyz (Isaac Lab) not xyzw (Isaac Gym)
  - Joint order is breadth-first (Isaac Lab) not depth-first (Isaac Gym)
  - No gym.refresh_*; state is auto-updated via scene.update()
  - Ball is a RigidObject (SphereCfg), not a URDF actor
  - Contact forces come from ContactSensor
  - PD control is manual: stiffness=0/damping=0, set_joint_effort_target()
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse, convert_quat

from .goalkeeper_env_cfg import GoalkeeperEnvCfg
from .goalkeeper_utils import load_imitation_dataset, MotionLib, torch_rand_float


class GoalkeeperEnv(DirectRLEnv):
    """Humanoid goalkeeper environment for Isaac Lab 2.2.1."""

    cfg: GoalkeeperEnvCfg

    def __init__(self, cfg: GoalkeeperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # dt for this policy step (seconds)
        self.dt = self.cfg.decimation * self.cfg.sim.dt

        # Build joint index maps from articulation (breadth-first order)
        self._build_joint_maps()
        self._build_body_maps()

        # Allocate buffers
        self._init_buffers()

        # Load AMP motion dataset
        self._load_motions()

        self.play = self.cfg.play
        self.add_noise = self.cfg.add_noise

        # Curriculum state
        self.curriculumupdate = 0.0
        self.curriculumsigma = self.cfg.catch_sigma
        self.common_step_counter = 0
        self.last_step_counter = 0

        # Push/ball randomization intervals (steps)
        self.push_interval = int(math.ceil(self.cfg.push_interval_s / self.dt))
        self.ball_interval = int(math.ceil(self.cfg.ball_interval_s / self.dt))

        # Reward scales (multiply by dt to get per-step)
        self._build_reward_scales()

        # Noise vector
        self.noise_scale_vec = self._get_noise_scale_vec()

    # ------------------------------------------------------------------
    # DirectRLEnv interface
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Create simulation scene: robot, ball, contact sensor, terrain."""
        self._robot = Articulation(self.cfg.robot)
        self._ball = RigidObject(self.cfg.ball)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = TerrainImporter(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Register with scene manager
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["ball"] = self._ball
        self.scene.sensors["contact_sensor"] = self._contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Clip actions and prepare delayed action buffer."""
        self.actions = torch.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions)

        # Build per-decimation delayed actions
        self.delayed_actions = self.actions.unsqueeze(0).repeat(self.cfg.decimation, 1, 1)
        if self.cfg.delay:
            delay_steps = torch.randint(0, self.cfg.decimation, (self.num_envs, 1), device=self.device)
            for i in range(self.cfg.decimation):
                self.delayed_actions[i] = (
                    self.last_actions
                    + (self.actions - self.last_actions) * (i >= delay_steps).float()
                )
        self._decimation_step = 0

    def _apply_action(self) -> None:
        """Compute PD torques and apply to robot joints."""
        action = self.delayed_actions[self._decimation_step]
        self._decimation_step = min(self._decimation_step + 1, self.cfg.decimation - 1)

        # Randomize joint injections each decimation step
        if self.cfg.randomize_joint_injection:
            self.joint_injection = torch_rand_float(
                self.cfg.joint_injection_range[0], self.cfg.joint_injection_range[1],
                (self.num_envs, self.num_dof), self.device,
            ) * self.torque_limits.unsqueeze(0)
            self.joint_injection[:, self.curriculum_dof_indices] = 0.0

        torques = self._compute_torques(action)
        self._robot.set_joint_effort_target(torques)

        # Apply drag force on ball
        self._apply_ball_drag()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (terminated, timed_out) tensors."""
        self._update_derived_quantities()
        self.common_step_counter += 1

        # Knee too low
        knee_z = self._robot.data.body_pos_w[:, self.knee_indices, 2]
        terminated = torch.min(knee_z, dim=-1).values < 0.10

        # Gravity tilt (upper body)
        gravity_tilt = torch.any(
            torch.norm(self.projected_gravity[:, :2], dim=-1, keepdim=True) > 0.8, dim=1
        )
        terminated = terminated | gravity_tilt

        # Sharp contact force at feet
        feet_forces = self._contact_sensor.data.net_forces_w[:, self.contact_feet_indices, :]
        sharp_contact = (
            torch.mean(torch.norm(feet_forces, dim=-1), dim=-1) > 1.5 * self.cfg.max_contact_force
        )
        terminated = terminated | sharp_contact

        # Timeout
        timed_out = self.episode_length_buf >= self.max_episode_length

        return terminated, timed_out

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return total reward."""
        # Update curriculum reward scales
        if "eereach" in self._reward_scales:
            self._reward_scales["eereach"] = self._eereach_init * (1 + 0.5 * self.curriculumupdate)
        if "success" in self._reward_scales:
            self._reward_scales["success"] = self._success_init * (1 + 0.5 * self.curriculumupdate)
        if "stopball" in self._reward_scales:
            self._reward_scales["stopball"] = self._stop_init * (1 + 0.5 * self.curriculumupdate)
        if self.curriculumupdate > 1.0 and "dof_pos_limits" in self._reward_scales:
            self._reward_scales["dof_pos_limits"] = self._dof_pos_init * 2.0
            self._reward_scales["torque_limits"] = self._torque_init * 2.0
        if self.curriculumupdate > 2.0 and "dof_pos_limits" in self._reward_scales:
            self._reward_scales["dof_pos_limits"] = self._dof_pos_init * 3.0
            self._reward_scales["torque_limits"] = self._torque_init * 3.0

        rew_buf = torch.zeros(self.num_envs, device=self.device)
        for name, scale in self._reward_scales.items():
            fn = getattr(self, f"_reward_{name}", None)
            if fn is not None:
                rew = fn() * scale
                rew_buf += rew
                self.episode_sums[name] += rew

        return rew_buf

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments."""
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Track success rate
        self.success_rate[env_ids, 0] += self.success_flag[env_ids]
        self.success_rate[env_ids, 1] += 1.0
        reach_count = self.success_rate[:, 1] > 9
        self.success_rate[reach_count, 2] = (
            self.success_rate[reach_count, 0] / self.success_rate[reach_count, 1]
        )
        self.success_rate[reach_count, :2] *= 0.0

        # Reset joint state
        self._reset_dofs(env_ids)
        # Reset root state (robot + ball)
        self._reset_root_states(env_ids)

        # Reset randomized gains
        if self.cfg.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.kp_range[0], self.cfg.kp_range[1], (len(env_ids), self.num_dof), self.device
            )
        if self.cfg.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.kd_range[0], self.cfg.kd_range[1], (len(env_ids), self.num_dof), self.device
            )
        if self.cfg.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(
                self.cfg.actuation_offset_range[0], self.cfg.actuation_offset_range[1],
                (len(env_ids), self.num_dof), self.device,
            ) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[env_ids[:, None], self.curriculum_dof_indices] = 0.0

        # Reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.joint_powers[env_ids] = 0.0
        self.actor_history_buf[env_ids] = 0.0
        self.reach_goal_timer[env_ids] = 0.0

        # Log episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums:
            ep_len = torch.clip(self.episode_length_buf[env_ids], min=1).float()
            self.extras["episode"][f"rew_{key}"] = torch.mean(
                self.episode_sums[key][env_ids] / ep_len / self.dt
            )
            self.episode_sums[key][env_ids] = 0.0

        self.extras["time_outs"] = self.reset_time_outs

        # Curriculum update
        if (self.common_step_counter - self.last_step_counter) > 500:
            self.startstep = 50 - random.randint(3, 10)
            avg_ep_len = torch.mean(self.episode_length_buf[env_ids].float()) / 50.0
            self.curriculumupdate = float(avg_ep_len.item())
            self.command_ranges[:, 0] = torch.clip(
                self.command_ranges[:, 0] - 0.3 * self.curriculumupdate,
                self.command_bound[:, 0], self.command_bound[:, 1],
            )
            self.command_ranges[:, 1] = torch.clip(
                self.command_ranges[:, 1] + 0.3 * self.curriculumupdate,
                self.command_bound[:, 0], self.command_bound[:, 1],
            )
            self.command_ranges[:, 2] = torch.clip(
                self.command_ranges[:, 2] - 0.3 * self.curriculumupdate,
                self.command_bound[:, 2], self.command_bound[:, 3],
            )
            self.command_ranges[:, 3] = torch.clip(
                self.command_ranges[:, 3] + 0.3 * self.curriculumupdate,
                self.command_bound[:, 2], self.command_bound[:, 3],
            )
            self.last_step_counter = self.common_step_counter

    def _get_observations(self) -> dict:
        """Build observation dict with 'policy' (history) and 'critic' (privileged)."""
        # Update ball target tracking
        self._update_ball_target()
        # Domain rand on ball
        self._randomize_balls()
        if self.cfg.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()

        # Read current state
        joint_pos = self._robot.data.joint_pos   # (N, num_dof)
        joint_vel = self._robot.data.joint_vel   # (N, num_dof)

        upper_body_quat = self._robot.data.body_quat_w[:, self.upper_body_index, :]  # wxyz
        hand_pos_w = self._robot.data.body_pos_w[:, self.hand_indices, :3]
        hand_pos_l = quat_apply_inverse(
            self._robot.data.body_quat_w[:, self.upper_body_index, :],
            hand_pos_w[:, 0, :] - self.torso_pos,
        )
        hand_pos_r = quat_apply_inverse(
            self._robot.data.body_quat_w[:, self.upper_body_index, :],
            hand_pos_w[:, 1, :] - self.torso_pos,
        )

        # Ball position in robot-local frame
        initial_vanish = (self.catchstep < self.startstep).view(-1, 1)
        ball_local = quat_apply_inverse(
            self._robot.data.root_quat_w,
            self._ball.data.root_pos_w - self.torso_pos,
        ) * initial_vanish

        # Ball in air check
        ball_local_x = ball_local[:, 0]
        ball_vel_w = self._ball.data.root_lin_vel_w
        ball_speed_x = ball_vel_w[:, 0]
        flying = (
            (ball_local_x > 0.05) & (ball_local_x < 3.4)
            & (ball_local[:, 1] > -2.0) & (ball_local[:, 1] < 2.0)
            & (ball_local[:, 2] < 1.8)
            & (self.catchstep > 0)
            & ((ball_local_x < self.ball_last[:, 0]) | (self.ball_last[:, 0] == 0.0))
        ).view(-1, 1)
        random_vanish = (self.catchstep > self.vanish_step).view(-1, 1)
        self.ball_last = ball_local.clone()

        # Current one-step observation (96-dim)
        current_actor_obs = torch.cat([
            ball_local,                                           # 3
            self.base_ang_vel * self.cfg.obs_scales_ang_vel,      # 3
            self.projected_gravity,                               # 3
            (joint_pos - self.default_dof_pos) * self.cfg.obs_scales_dof_pos,  # 29
            joint_vel * self.cfg.obs_scales_dof_vel,              # 29
            self.actions,                                         # 29
        ], dim=-1)  # total = 96

        # Apply noise
        if self.add_noise:
            current_actor_obs = current_actor_obs + (
                2 * torch.rand_like(current_actor_obs) - 1
            ) * self.noise_scale_vec
            current_actor_obs[:, :self.cfg.num_ballobs] *= flying * random_vanish
        else:
            current_actor_obs[:, :self.cfg.num_ballobs] *= flying

        # Roll history buffer: drop oldest, append new
        self.actor_history_buf = torch.cat(
            [self.actor_history_buf[:, self.cfg.num_one_step_observations:], current_actor_obs], dim=1
        )

        # Privileged observation (113-dim) — no noise
        end_target_local = quat_apply_inverse(
            self._robot.data.root_quat_w, self.end_target - self.torso_pos
        )
        privileged_obs = torch.cat([
            ball_local,                                           # 3
            self.base_ang_vel * self.cfg.obs_scales_ang_vel,      # 3
            self.projected_gravity,                               # 3
            (joint_pos - self.default_dof_pos) * self.cfg.obs_scales_dof_pos,  # 29
            joint_vel * self.cfg.obs_scales_dof_vel,              # 29
            self.actions,                                         # 29
            self.base_lin_vel * self.cfg.obs_scales_lin_vel,      # 3
            self.end_regions.unsqueeze(-1).float() / 3.0,         # 1
            end_target_local,                                     # 3
            ball_vel_w * self.cfg.obs_scales_ball_vel,            # 3
            hand_pos_r,                                           # 3
            hand_pos_l,                                           # 3
            self.dist.unsqueeze(-1),                              # 1
        ], dim=-1)  # total = 113

        # Clip
        self.actor_history_buf = torch.clip(self.actor_history_buf, -self.cfg.clip_observations, self.cfg.clip_observations)
        privileged_obs = torch.clip(privileged_obs, -self.cfg.clip_observations, self.cfg.clip_observations)

        # Update trackers after obs
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = joint_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = torch.cat([
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
        ], dim=-1)
        self.catchstep -= 1

        return {"policy": self.actor_history_buf, "critic": privileged_obs}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_joint_maps(self):
        """Build DOF index maps from articulation joint names (breadth-first)."""
        joint_names = self._robot.data.joint_names
        self.num_dof = len(joint_names)
        self.dof_names = list(joint_names)

        def _find_idx(names_list):
            idx = []
            for n in names_list:
                if n in self.dof_names:
                    idx.append(self.dof_names.index(n))
                else:
                    print(f"[GoalkeeperEnv] Warning: joint '{n}' not found in articulation.")
            return torch.tensor(idx, dtype=torch.long, device=self.device)

        self.curriculum_dof_indices = _find_idx(self.cfg.curriculum_joints)
        self.left_leg_joint_indices = _find_idx(self.cfg.left_leg_joints)
        self.right_leg_joint_indices = _find_idx(self.cfg.right_leg_joints)
        self.leg_joint_indices = torch.cat([self.left_leg_joint_indices, self.right_leg_joint_indices])
        self.left_hip_joint_indices = _find_idx(self.cfg.left_hip_joints)
        self.right_hip_joint_indices = _find_idx(self.cfg.right_hip_joints)
        self.hip_joint_indices = torch.cat([self.left_hip_joint_indices, self.right_hip_joint_indices])
        self.left_arm_joint_indices = _find_idx(self.cfg.left_arm_joints)
        self.right_arm_joint_indices = _find_idx(self.cfg.right_arm_joints)
        self.arm_joint_indices = torch.cat([self.left_arm_joint_indices, self.right_arm_joint_indices])
        self.elbow_joint_indices = _find_idx(self.cfg.elbow_joints)
        self.wrist_joint_indices = _find_idx(self.cfg.wrist_joints)
        self.upper_body_joint_indices = torch.cat([self.elbow_joint_indices, self.wrist_joint_indices])
        self.waist_joint_indices = _find_idx(self.cfg.waist_joints)
        self.ankle_joint_indices = _find_idx(self.cfg.ankle_joints)

    def _build_body_maps(self):
        """Build rigid body index maps."""
        body_names = list(self._robot.data.body_names)
        self.num_bodies = len(body_names)

        def _find_body_idx(substr: str):
            return [i for i, n in enumerate(body_names) if substr in n]

        def _find_body_exact(name: str):
            return body_names.index(name) if name in body_names else 0

        feet_names = [n for n in body_names if self.cfg.foot_name in n]
        contact_foot_names = [n for n in body_names if self.cfg.contact_foot_names in n]
        hand_names = [n for n in body_names if self.cfg.hand_name in n]
        knee_names_list = [n for n in body_names if any(k in n for k in self.cfg.knee_names)]
        penalized_names = []
        for p in self.cfg.penalize_contacts_on:
            penalized_names.extend([n for n in body_names if p in n])

        self.contact_feet_indices = torch.tensor(
            [body_names.index(n) for n in contact_foot_names], dtype=torch.long, device=self.device
        )
        self.hand_indices = torch.tensor(
            [body_names.index(n) for n in hand_names[:2]], dtype=torch.long, device=self.device
        )
        self.knee_indices = torch.tensor(
            [body_names.index(n) for n in knee_names_list], dtype=torch.long, device=self.device
        )
        self.penalised_contact_indices = torch.tensor(
            [body_names.index(n) for n in set(penalized_names)], dtype=torch.long, device=self.device
        )
        self.upper_body_index = _find_body_exact(self.cfg.upper_body_link)
        self.torso_body_index = _find_body_exact(self.cfg.torso_link)

        keyframe_names_list = [n for n in body_names if self.cfg.keyframe_name in n]
        self.keyframe_indices = torch.tensor(
            [body_names.index(n) for n in keyframe_names_list], dtype=torch.long, device=self.device
        )
        self.keyframe_names = keyframe_names_list

        print(f"[GoalkeeperEnv] Bodies={self.num_bodies}, DOFs={self.num_dof}")
        print(f"[GoalkeeperEnv] Contact feet={contact_foot_names}")
        print(f"[GoalkeeperEnv] Hands={hand_names[:2]}")
        print(f"[GoalkeeperEnv] Knees={knee_names_list}")

    def _init_buffers(self):
        """Allocate all state buffers."""
        N = self.num_envs
        nd = self.num_dof

        # Actor history obs buffer (N, 960)
        self.actor_history_buf = torch.zeros(N, self.cfg.observation_space, device=self.device)

        # Action buffers
        self.actions = torch.zeros(N, nd, device=self.device)
        self.last_actions = torch.zeros(N, nd, device=self.device)
        self.last_last_actions = torch.zeros(N, nd, device=self.device)
        self.delayed_actions = torch.zeros(self.cfg.decimation, N, nd, device=self.device)
        self._decimation_step = 0

        # Torque buffer
        self.torques = torch.zeros(N, nd, device=self.device)
        self.last_torques = torch.zeros_like(self.torques)

        # Velocity buffers
        self.last_dof_vel = torch.zeros(N, nd, device=self.device)
        self.last_root_vel = torch.zeros(N, 6, device=self.device)
        self.base_lin_vel = torch.zeros(N, 3, device=self.device)
        self.base_ang_vel = torch.zeros(N, 3, device=self.device)
        self.projected_gravity = torch.zeros(N, 3, device=self.device)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N, 1)

        # PD gains
        self.p_gains = torch.zeros(nd, device=self.device)
        self.d_gains = torch.zeros(nd, device=self.device)
        self._setup_pd_gains()
        self.Kp_factors = torch.ones(N, nd, device=self.device)
        self.Kd_factors = torch.ones(N, nd, device=self.device)
        if self.cfg.randomize_kp:
            self.Kp_factors = torch_rand_float(
                self.cfg.kp_range[0], self.cfg.kp_range[1], (N, nd), self.device
            )
        if self.cfg.randomize_kd:
            self.Kd_factors = torch_rand_float(
                self.cfg.kd_range[0], self.cfg.kd_range[1], (N, nd), self.device
            )

        # DOF limits (needed for actuation offset and torque clamping)
        self._setup_dof_limits()

        # Actuation noise
        self.joint_injection = torch.zeros(N, nd, device=self.device)
        self.actuation_offset = torch.zeros(N, nd, device=self.device)
        if self.cfg.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(
                self.cfg.actuation_offset_range[0], self.cfg.actuation_offset_range[1],
                (N, nd), self.device,
            ) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[:, self.curriculum_dof_indices] = 0.0

        # Default DOF positions
        self.default_dof_pos = torch.zeros(nd, device=self.device)
        for i, name in enumerate(self.dof_names):
            angle = self.cfg.robot.init_state.joint_pos.get(name, 0.0)
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_poses = self.default_dof_pos.repeat(N, 1)

        # cfg.init_pos is in IsaacGym DFS order; reorder to IsaacLab BFS order by name lookup
        _G1_DFS_ORDER = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        _init_pos_dfs = self.cfg.init_pos
        _standpos_arr = torch.zeros(nd, device=self.device)
        for bfs_idx, name in enumerate(self.dof_names):
            if name in _G1_DFS_ORDER:
                dfs_idx = _G1_DFS_ORDER.index(name)
                if dfs_idx < len(_init_pos_dfs):
                    _standpos_arr[bfs_idx] = _init_pos_dfs[dfs_idx]
        self.standpos = _standpos_arr.unsqueeze(0)

        if self.standpos.shape[1] != nd:
            print(f"[GoalkeeperEnv] init_pos length mismatch ({self.standpos.shape[1]} vs {nd}), using zeros")
            self.standpos = self.default_dof_pos.clone()
        self.init_dof_pos = self.default_dof_poses.clone()

        # Ball / task buffers
        self.end_target = torch.zeros(N, 3, device=self.device)
        self.dist = 5.0 * torch.ones(N, device=self.device)
        self.ball_vel_scalar = torch.zeros(N, device=self.device)
        self.has_in_air = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.stop_flag = torch.zeros(N, device=self.device)
        self.success_flag = torch.zeros(N, device=self.device)
        self.success_rate = torch.zeros(N, 3, device=self.device)
        self.ball_start = torch.zeros(N, 3, device=self.device)
        self.ball_end = torch.zeros(N, 3, device=self.device)
        self.ball_last = torch.zeros(N, 3, device=self.device)
        self.vanish_step = torch.randint(0, 30, (N,), device=self.device)
        self.catchstep = 50 * torch.ones(N, dtype=torch.int, device=self.device)
        self.startstep = 50 - random.randint(3, 10)
        self.reach_goal_timer = torch.zeros(N, device=self.device)
        self.joint_powers = torch.zeros(N, 100, nd, device=self.device)
        self.reset_time_outs = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Domain rand
        self.payload = torch.zeros(N, 1, device=self.device)
        self.com_displacement = torch.zeros(N, 3, device=self.device)
        self.friction_coeffs = torch.ones(N, 1, device=self.device)
        self.restitution_coeffs = torch.zeros(N, 1, device=self.device)

        # Command ranges (6 regions)
        six = N // 6
        self.end_regions = torch.cat([
            torch.full((six,), 0, dtype=torch.long, device=self.device),
            torch.full((six,), 1, dtype=torch.long, device=self.device),
            torch.full((six,), 2, dtype=torch.long, device=self.device),
            torch.full((six,), 3, dtype=torch.long, device=self.device),
            torch.full((six,), 4, dtype=torch.long, device=self.device),
            torch.full((N - 5 * six,), 5, dtype=torch.long, device=self.device),
        ])
        self.command_ranges = torch.zeros(N, 4, device=self.device)
        self.command_bound = torch.zeros(N, 4, device=self.device)
        self.init_ranges = torch.zeros(4, device=self.device)
        cr = self.cfg.commands_ranges
        for env_idx in range(N):
            region = self.end_regions[env_idx].item()
            rk = f"ranges_{region}"
            rr = cr[rk]
            self.command_ranges[env_idx, 0] = rr["width"][0]
            self.command_ranges[env_idx, 1] = rr["width"][1]
            self.command_ranges[env_idx, 2] = rr["height"][0]
            self.command_ranges[env_idx, 3] = rr["height"][1]
            self.command_bound[env_idx, 0] = rr["maxw"][0]
            self.command_bound[env_idx, 1] = rr["maxw"][1]
            self.command_bound[env_idx, 2] = rr["maxh"][0]
            self.command_bound[env_idx, 3] = rr["maxh"][1]
        self.init_ranges[0] = cr["ranges_1"]["maxw"][0]
        self.init_ranges[1] = cr["ranges_0"]["maxw"][1]
        self.init_ranges[2] = cr["ranges_4"]["maxh"][0]
        self.init_ranges[3] = cr["ranges_2"]["maxh"][1]

        # Episode sums
        self.episode_sums = {
            name: torch.zeros(N, device=self.device)
            for name in self._get_all_reward_names()
        }

    def _setup_pd_gains(self):
        """Set p_gains and d_gains from config."""
        for i, name in enumerate(self.dof_names):
            found = False
            for key, val in self.cfg.stiffness.items():
                if key in name:
                    self.p_gains[i] = val
                    found = True
                    break
            for key, val in self.cfg.damping.items():
                if key in name:
                    self.d_gains[i] = val
                    break
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0

    def _setup_dof_limits(self):
        """Read DOF limits from articulation data."""
        soft_limit = self.cfg.soft_dof_pos_limit
        limits = self._robot.data.joint_pos_limits  # (num_envs, num_dof, 2) or (num_dof, 2)
        if limits.dim() == 3:
            limits = limits[0]  # (num_dof, 2)
        self.hard_dof_pos_limits = limits.clone().to(self.device)
        m = (limits[:, 0] + limits[:, 1]) / 2
        r = limits[:, 1] - limits[:, 0]
        self.dof_pos_limits = torch.zeros_like(limits)
        self.dof_pos_limits[:, 0] = m - 0.5 * r * soft_limit
        self.dof_pos_limits[:, 1] = m + 0.5 * r * soft_limit

        vel_limits = self._robot.data.joint_vel_limits
        if vel_limits.dim() == 2:
            vel_limits = vel_limits[0]
        self.dof_vel_limits = vel_limits.to(self.device)

        effort_limits = self._robot.data.joint_effort_limits
        if effort_limits.dim() == 2:
            effort_limits = effort_limits[0]
        self.torque_limits = effort_limits.to(self.device)
        # Replace zeros with a high value (some URDFs don't specify effort limits)
        self.torque_limits = torch.where(
            self.torque_limits == 0, torch.full_like(self.torque_limits, 400.0), self.torque_limits
        )

    def _load_motions(self):
        """Load AMP motion dataset."""
        try:
            multidataset, mapping = load_imitation_dataset(
                self.cfg.dataset_folder, self.cfg.dataset_joint_mapping
            )
            self.motion_lib = MotionLib(
                multidataset, mapping, self.dof_names, self.keyframe_names,
                self.cfg.dataset_frame_rate, self.cfg.dataset_min_time,
                self.device, self.cfg.amp_obs_type, self.cfg.amp_num_steps,
            )
        except Exception as e:
            print(f"[GoalkeeperEnv] Warning: motion dataset load failed: {e}")
            self.motion_lib = None

    def _build_reward_scales(self):
        """Build reward scale dict from config fields, multiplied by dt."""
        self._reward_scales = {}
        reward_field_map = {
            "eereach": self.cfg.rew_eereach,
            "success": self.cfg.rew_success,
            "stopball": self.cfg.rew_stopball,
            "stayonline": self.cfg.rew_stayonline,
            "noretreat": self.cfg.rew_noretreat,
            "successland": self.cfg.rew_successland,
            "feetorientaion": self.cfg.rew_feetorientaion,
            "penalize_sharpcontact": self.cfg.rew_penalize_sharpcontact,
            "penalize_kneeheight": self.cfg.rew_penalize_kneeheight,
            "feet_slippage": self.cfg.rew_feet_slippage,
            "postorientation": self.cfg.rew_postorientation,
            "postangvel": self.cfg.rew_postangvel,
            "postupperdofpos": self.cfg.rew_postupperdofpos,
            "postwaistdofpos": self.cfg.rew_postwaistdofpos,
            "postlinvel": self.cfg.rew_postlinvel,
            "ang_vel_xy": self.cfg.rew_ang_vel_xy,
            "dof_acc": self.cfg.rew_dof_acc,
            "smoothness": self.cfg.rew_smoothness,
            "torques": self.cfg.rew_torques,
            "dof_vel": self.cfg.rew_dof_vel,
            "dof_pos_limits": self.cfg.rew_dof_pos_limits,
            "dof_vel_limits": self.cfg.rew_dof_vel_limits,
            "torque_limits": self.cfg.rew_torque_limits,
            "deviation_waist_pitch_joint": self.cfg.rew_deviation_waist_pitch_joint,
        }
        for name, scale in reward_field_map.items():
            if scale != 0.0:
                self._reward_scales[name] = scale * self.dt

        self._eereach_init = self._reward_scales.get("eereach", 0.0)
        self._success_init = self._reward_scales.get("success", 0.0)
        self._stop_init = self._reward_scales.get("stopball", 0.0)
        self._dof_pos_init = self._reward_scales.get("dof_pos_limits", 0.0)
        self._torque_init = self._reward_scales.get("torque_limits", 0.0)

    def _get_all_reward_names(self):
        return [
            "eereach", "success", "stopball", "stayonline", "noretreat",
            "successland", "feetorientaion", "penalize_sharpcontact", "penalize_kneeheight",
            "feet_slippage", "postorientation", "postangvel", "postupperdofpos",
            "postwaistdofpos", "postlinvel", "ang_vel_xy", "dof_acc", "smoothness",
            "torques", "dof_vel", "dof_pos_limits", "dof_vel_limits",
            "torque_limits", "deviation_waist_pitch_joint",
        ]

    def _get_noise_scale_vec(self) -> torch.Tensor:
        """Build per-element noise scale vector for actor obs."""
        nd = self.num_dof
        nb = self.cfg.num_ballobs
        ns = self.cfg.noise_level
        vec = torch.zeros(nb + 6 + 2 * nd + nd, device=self.device)
        vec[:nb] = self.cfg.noise_ball * ns
        vec[nb: nb + 3] = self.cfg.noise_ang_vel * ns * self.cfg.obs_scales_ang_vel
        vec[nb + 3: nb + 6] = self.cfg.noise_gravity * ns
        vec[nb + 6: nb + 6 + nd] = self.cfg.noise_dof_pos * ns * self.cfg.obs_scales_dof_pos
        vec[nb + 6 + nd: nb + 6 + 2 * nd] = self.cfg.noise_dof_vel * ns * self.cfg.obs_scales_dof_vel
        vec[nb + 6 + 2 * nd:] = 0.0  # previous actions
        return vec

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _update_derived_quantities(self):
        """Update base velocity, projected gravity, torso position."""
        upper_quat = self._robot.data.body_quat_w[:, self.upper_body_index, :]  # wxyz
        self.base_lin_vel = quat_apply_inverse(
            upper_quat, self._robot.data.body_lin_vel_w[:, self.upper_body_index, :]
        )
        self.base_ang_vel = quat_apply_inverse(
            upper_quat, self._robot.data.body_ang_vel_w[:, self.upper_body_index, :]
        )
        self.projected_gravity = quat_apply_inverse(upper_quat, self.gravity_vec)

        joint_vel = self._robot.data.joint_vel
        self.torques = self._compute_torques(self.actions)
        joint_powers = torch.abs(self.torques * joint_vel).unsqueeze(1)
        self.joint_powers = torch.cat([joint_powers, self.joint_powers[:, :-1]], dim=1)

        self.reset_time_outs = self.episode_length_buf >= self.max_episode_length

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Manual PD torque computation."""
        actions_scaled = actions * self.cfg.action_scale
        joint_pos_target = self.default_dof_poses + actions_scaled
        joint_pos_target[self.catchstep > self.startstep] = self.init_dof_pos[self.catchstep > self.startstep]

        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        torques = (
            self.p_gains * self.Kp_factors * (joint_pos_target - joint_pos)
            - self.d_gains * self.Kd_factors * joint_vel
        )
        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _apply_ball_drag(self):
        """Apply aerodynamic drag force to ball."""
        ball_vel = self._ball.data.root_lin_vel_w  # (N, 3)
        rho, Cd, r = 1.225, 0.47, 0.1
        A = math.pi * r * r
        speed = torch.norm(ball_vel, dim=1, keepdim=True)
        drag = -0.5 * rho * Cd * A * speed * ball_vel
        drag += torch.empty_like(drag).uniform_(-0.5, 0.5)
        forces = drag.unsqueeze(1)  # (N, 1, 3)
        torques_zero = torch.zeros_like(forces)
        # Isaac Lab 0.54+ renamed this API
        self._ball.permanent_wrench_composer.set_forces_and_torques(forces, torques_zero, is_global=True)

    def _update_ball_target(self):
        """Update end_target based on ball approach."""
        ball_pos_w = self._ball.data.root_pos_w
        env_origins = self.scene.env_origins
        ball_local_x = ball_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]

        approach_idx = (
            (ball_local_x < 0.5) & (ball_local_x > 0.1)
            & (ball_vel_x - self.ball_vel_scalar < 2.0)
        ).nonzero(as_tuple=False).flatten()

        self.end_target[approach_idx] = ball_pos_w[approach_idx, :3].clone()
        self.end_target[:, 0] = torch.clip(
            self.end_target[:, 0],
            min=env_origins[:, 0] + 0.1,
            max=env_origins[:, 0] + 1.0,
        )

        hand_pos_w = self._robot.data.body_pos_w[:, self.hand_indices, :3]
        hand_l, hand_r = hand_pos_w[:, 0, :], hand_pos_w[:, 1, :]

        for region_id, hand in [(0, hand_l), (1, hand_r), (2, hand_l),
                                 (3, hand_r), (4, hand_l), (5, hand_r)]:
            mask = self.end_regions == region_id
            if mask.any():
                self.dist[mask] = torch.norm(self.end_target[mask] - hand[mask], dim=1)

    def _reset_dofs(self, env_ids: torch.Tensor):
        """Reset joint positions and velocities for selected environments."""
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)

        if self.cfg.continue_keep and torch.rand(1).item() > 0.2 and not self.play:
            random_env_ids = torch.randint(0, self.num_envs, (len(env_ids),), device=self.device)
            joint_pos = self._robot.data.joint_pos[random_env_ids].clone()
        elif self.cfg.randomize_initial_joint_pos:
            scale = torch_rand_float(
                self.cfg.initial_joint_pos_scale[0], self.cfg.initial_joint_pos_scale[1],
                (len(env_ids), self.num_dof), self.device,
            )
            offset = torch_rand_float(
                self.cfg.initial_joint_pos_offset[0], self.cfg.initial_joint_pos_offset[1],
                (len(env_ids), self.num_dof), self.device,
            )
            joint_pos = torch.clip(self.standpos * scale + offset, dof_lower, dof_upper)
        else:
            joint_pos = self.standpos * torch.ones(len(env_ids), self.num_dof, device=self.device)

        joint_vel = torch.zeros_like(joint_pos)
        self.init_dof_pos[env_ids] = joint_pos.clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_root_states(self, env_ids: torch.Tensor):
        """Reset robot root and ball states."""
        env_origins = self.scene.env_origins

        # Robot pose: position + identity quaternion wxyz
        robot_pos = env_origins[env_ids].clone()
        robot_pos[:, 2] += 0.8
        if not self.play:
            robot_pos[:, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), self.device)
        robot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        if self.play:
            robot_lin_vel = torch.zeros(len(env_ids), 3, device=self.device)
            robot_ang_vel = torch.zeros(len(env_ids), 3, device=self.device)
        else:
            robot_lin_vel = torch_rand_float(-0.3, 0.3, (len(env_ids), 3), self.device)
            robot_ang_vel = torch_rand_float(-0.3, 0.3, (len(env_ids), 3), self.device)
        robot_state = torch.cat([robot_pos, robot_quat, robot_lin_vel, robot_ang_vel], dim=-1)
        self._robot.write_root_state_to_sim(robot_state, env_ids)

        # Ball
        ball_vel = self._assign_ball_states(env_ids)
        ball_pos = self.ball_start[env_ids]
        ball_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        ball_ang_vel_zero = torch.zeros(len(env_ids), 3, device=self.device)
        ball_state = torch.cat([ball_pos, ball_quat, ball_vel, ball_ang_vel_zero], dim=-1)
        self._ball.write_root_state_to_sim(ball_state, env_ids)

        # Ball trackers
        self.ball_vel_scalar[env_ids] = ball_vel[:, 0]
        self.has_in_air[env_ids] = False
        self.stop_flag[env_ids] = 0.0
        self.success_flag[env_ids] = 0.0
        self.dist[env_ids] = 5.0
        self.ball_last[env_ids] = 0.0
        self.vanish_step[env_ids] = torch.randint(0, 30, (len(env_ids),), device=self.device)

    def _assign_ball_states(self, ball_ids: torch.Tensor, g: float = 9.81) -> torch.Tensor:
        """Compute ball start position and velocity for given env ids."""
        dtype = torch.float
        device = self.device
        env_origins = self.scene.env_origins
        nb = len(ball_ids)

        self.catchstep[ball_ids] = 50 * torch.ones(nb, dtype=torch.int, device=device)

        # Random ball start (3-5m ahead of robot)
        ball_start_local = torch.stack([
            2.0 * torch.rand(nb, dtype=dtype, device=device) + 3.0,
            torch.rand(nb, dtype=dtype, device=device) * (
                self.init_ranges[1] - self.init_ranges[0]) + self.init_ranges[0],
            torch.rand(nb, dtype=dtype, device=device) * (
                self.init_ranges[3] - self.init_ranges[2]) + self.init_ranges[2],
        ], dim=1)

        # Ball target (behind robot)
        ball_end_local = torch.stack([
            -0.5 * torch.rand(nb, dtype=dtype, device=device) - 0.1,
            (torch.rand(nb, dtype=dtype, device=device)
             * (self.command_ranges[ball_ids, 1] - self.command_ranges[ball_ids, 0])
             + self.command_ranges[ball_ids, 0]),
            (torch.rand(nb, dtype=dtype, device=device)
             * (self.command_ranges[ball_ids, 3] - self.command_ranges[ball_ids, 2])
             + self.command_ranges[ball_ids, 2]),
        ], dim=1)

        # World frame
        self.ball_start[ball_ids] = ball_start_local + env_origins[ball_ids]
        self.ball_start[ball_ids, 2] = ball_start_local[:, 2]
        self.ball_end[ball_ids] = ball_end_local + env_origins[ball_ids]
        self.ball_end[ball_ids, 2] = ball_end_local[:, 2]

        delta = ball_end_local - ball_start_local
        catch_prop = (0.1 - ball_start_local[:, 0:1]) / (ball_end_local[:, 0:1] - ball_start_local[:, 0:1] + 1e-6)
        self.end_target[ball_ids] = self.ball_start[ball_ids] + (self.ball_end[ball_ids] - self.ball_start[ball_ids]) * catch_prop

        t_flight = 0.4 + 0.6 * torch.rand(1, dtype=dtype, device=device)
        ball_vel = torch.empty_like(delta)
        ball_vel[:, 0:2] = delta[:, 0:2] / t_flight
        ball_vel[:, 2] = (delta[:, 2] + 0.5 * g * t_flight ** 2) / t_flight

        return ball_vel

    def _push_robots(self):
        """Apply random velocity impulse to all robots."""
        max_vel = self.cfg.max_push_vel_xy
        robot_state = self._robot.data.root_state_w.clone()
        robot_state[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), self.device)
        self._robot.write_root_state_to_sim(robot_state)

    def _randomize_balls(self):
        """Periodic random velocity perturbation on ball."""
        if self.common_step_counter % self.ball_interval == 0:
            max_vel = self.cfg.max_ball_vel
            ball_state = self._ball.data.root_state_w.clone()
            ball_state[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), self.device)
            self._ball.write_root_state_to_sim(ball_state)

    # ------------------------------------------------------------------
    # Reward functions (ported from legged_robot.py)
    # ------------------------------------------------------------------

    @property
    def torso_pos(self):
        return self._robot.data.body_pos_w[:, self.torso_body_index, :3]

    def _reward_eereach(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_pos = self._ball.data.root_pos_w
        ball_local_x = ball_pos[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        phase1 = ((ball_local_x > 1.5) & (ball_vel_x - self.ball_vel_scalar < 2.0))

        end_target_local = self.end_target - self.torso_pos
        asidegoal = torch.clip(end_target_local[:, 1], -1.0, 1.0)
        asidegoal[torch.abs(asidegoal) < 0.3] = 0.0
        verticalgoal = torch.clip(self.torso_pos[:, 2] - torch.clip(self.end_target[:, 2], 0.3, 1.2), 0.0, 1.0)
        phase1_rew = 1.0 - (verticalgoal + torch.abs(asidegoal)) / 2.0

        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        jump_scale = 3.0 + 3.0 * self.curriculumupdate

        body_vel = self._robot.data.body_lin_vel_w[:, self.upper_body_index, :]
        vel_sigma = torch.zeros(self.num_envs, device=self.device)
        for region, axis, sign in [(0, 1, 1), (1, 1, -1), (4, 1, 1), (5, 1, -1)]:
            mask = self.end_regions == region
            vel_sigma[mask] = 1 + 3.0 * torch.clip(sign * body_vel[mask, axis], 0.0, 3.0)
        for region in [2, 3]:
            mask = self.end_regions == region
            vel_sigma[mask] = 1 + jump_scale * torch.clip(body_vel[mask, 2], 0.0, 3.0)
        vel_sigma[behind] = 2.0

        taskrew = 1.0 - 1.0 / (1.0 + torch.exp(-self.curriculumsigma * (self.dist - self.cfg.reach_th)))
        taskrew *= vel_sigma
        taskrew[phase1] = phase1_rew[phase1]
        gravity_pen = torch.clip(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0.0, 1.0)
        return taskrew * (1 - gravity_pen)

    def _reward_success(self) -> torch.Tensor:
        return (self.success_flag + 1.0) * (self.dist < self.cfg.strict_th)

    def _reward_stopball(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_pos = self._ball.data.root_pos_w
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        ball_local_x = ball_pos[:, 0] - env_origins[:, 0]
        changevel = (ball_vel_x - self.ball_vel_scalar > 2.0) & (ball_local_x > 0.0)
        stopped_ids = (changevel | (ball_local_x < 0.0)).nonzero(as_tuple=False).flatten()
        success_ids = ((self.stop_flag == 0) & changevel).nonzero(as_tuple=False).flatten()
        rew = 1.0 * (self.stop_flag == 0) * changevel
        self.success_flag[success_ids] = 1.0
        self.stop_flag[stopped_ids] = 1.0
        return rew

    def _reward_feetorientaion(self) -> torch.Tensor:
        if len(self.contact_feet_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)
        feet_quat = self._robot.data.body_quat_w[:, self.contact_feet_indices[:2], :]
        grav = self.gravity_vec
        left_g = quat_apply_inverse(feet_quat[:, 0, :], grav)
        right_g = quat_apply_inverse(feet_quat[:, 1, :], grav)
        orient = torch.sum(torch.square(left_g[:, :2]), dim=1) + torch.sum(torch.square(right_g[:, :2]), dim=1)
        return torch.exp(orient * -5)

    def _reward_successland(self) -> torch.Tensor:
        if len(self.contact_feet_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)
        feet_forces = self._contact_sensor.data.net_forces_w[:, self.contact_feet_indices[:2], 2]
        robot_z = self._robot.data.root_pos_w[:, 2]
        jump = robot_z > 1.0
        self.has_in_air = torch.logical_or(self.has_in_air, jump)
        has_contact = (feet_forces[:, 0] > 1.0) & (feet_forces[:, 1] > 1.0)
        one_feet = (
            ((feet_forces[:, 0] > 1.0) & (feet_forces[:, 1] < 1.0))
            | ((feet_forces[:, 0] < 1.0) & (feet_forces[:, 1] > 1.0))
        ) & self.has_in_air
        successful = torch.logical_and(has_contact, self.has_in_air)
        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)
        rew = self.has_in_air.float() + successful.float() * 5.0 + one_feet.float() * -1.0
        return rew * jump_ids

    def _reward_feet_slippage(self) -> torch.Tensor:
        if len(self.contact_feet_indices) < 1:
            return torch.zeros(self.num_envs, device=self.device)
        foot_vel = self._robot.data.body_lin_vel_w[:, self.contact_feet_indices, :]
        foot_forces = self._contact_sensor.data.net_forces_w[:, self.contact_feet_indices, :]
        contact_vel = torch.sum(
            torch.norm(foot_vel, dim=-1) * (torch.norm(foot_forces, dim=-1) > 1.0), dim=1
        )
        return torch.exp(contact_vel * -10)

    def _reward_penalize_sharpcontact(self) -> torch.Tensor:
        if len(self.contact_feet_indices) < 1:
            return torch.zeros(self.num_envs, device=self.device)
        feet_forces = self._contact_sensor.data.net_forces_w[:, self.contact_feet_indices, :]
        return (torch.mean(torch.norm(feet_forces, dim=-1), dim=-1) > self.cfg.max_contact_force).float()

    def _reward_penalize_kneeheight(self) -> torch.Tensor:
        knee_z = self._robot.data.body_pos_w[:, self.knee_indices, 2]
        return (torch.min(knee_z, dim=-1).values < 0.15).float()

    def _reward_postorientation(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_local_x = self._ball.data.root_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -3) * behind

    def _reward_postangvel(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_local_x = self._ball.data.root_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3) * behind

    def _reward_postlinvel(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_local_x = self._ball.data.root_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1) * -3) * behind

    def _reward_postupperdofpos(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_local_x = self._ball.data.root_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        joint_pos = self._robot.data.joint_pos
        mse = torch.sum(torch.square(
            joint_pos[:, self.upper_body_joint_indices]
            - self.default_dof_pos[:, self.upper_body_joint_indices]
        ), dim=-1)
        return torch.exp(mse * -1) * behind

    def _reward_postwaistdofpos(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        ball_local_x = self._ball.data.root_pos_w[:, 0] - env_origins[:, 0]
        ball_vel_x = self._ball.data.root_lin_vel_w[:, 0]
        behind = (ball_local_x < 0.0) | (ball_vel_x - self.ball_vel_scalar > 2.0)
        joint_pos = self._robot.data.joint_pos
        mse = torch.sum(torch.square(
            joint_pos[:, self.waist_joint_indices]
            - self.default_dof_pos[:, self.waist_joint_indices]
        ), dim=-1)
        return torch.exp(-3 * mse) * behind

    def _reward_stayonline(self) -> torch.Tensor:
        env_origins = self.scene.env_origins
        dist = torch.clip(torch.abs(self.torso_pos[:, 0] - env_origins[:, 0]), 0.2, 1.2) - 0.2
        return dist

    def _reward_noretreat(self) -> torch.Tensor:
        return -1.0 * torch.clip(self.base_lin_vel[:, 0], -1.0, 0.0)

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        joint_vel = self._robot.data.joint_vel
        return torch.sum(torch.square((self.last_dof_vel - joint_vel) / self.dt), dim=1)

    def _reward_smoothness(self) -> torch.Tensor:
        return torch.sum(torch.square(
            self.actions - self.last_actions - self.last_actions + self.last_last_actions
        ), dim=1)

    def _reward_torques(self) -> torch.Tensor:
        p_safe = torch.where(self.p_gains.unsqueeze(0) == 0, torch.ones_like(self.p_gains.unsqueeze(0)), self.p_gains.unsqueeze(0))
        return torch.sum(torch.square(self.torques / p_safe), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        return torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos
        out = -(joint_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out += (joint_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out, dim=1)

    def _reward_dof_vel_limits(self) -> torch.Tensor:
        return torch.sum(
            (torch.abs(self._robot.data.joint_vel) - self.dof_vel_limits * self.cfg.soft_dof_vel_limit).clip(min=0.0),
            dim=1,
        )

    def _reward_torque_limits(self) -> torch.Tensor:
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.soft_torque_limit).clip(min=0.0), dim=1
        )

    def _reward_deviation_waist_pitch_joint(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos
        if len(self.waist_joint_indices) < 3:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(joint_pos - self.default_dof_pos)[:, self.waist_joint_indices[2]], dim=-1)

    # AMP observation accessor (for external use)
    def get_amp_observations(self) -> torch.Tensor:
        return self._robot.data.joint_pos.clone()
