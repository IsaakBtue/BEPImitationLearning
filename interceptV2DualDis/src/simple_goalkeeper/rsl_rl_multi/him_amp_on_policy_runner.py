"""HIM-style AMP runner: wires HimActorCritic + MultiDiscAMPPPO + 4 region
MotionDataset/AMPDiscriminator instances together. Ported from
Humanoid-Goalkeeper/rsl_rl/rsl_rl/runners/him_on_policy_runner.py, adapted to
this codebase's AMPEnvWrapper/AMPDiscriminator/MotionDataset contract instead
of G1's raw legged_gym env + AMP module.

Observation sourcing (verified against Task 6's approved env cfg and mjlab's
ObservationManager):

* ``env.get_observations()`` / ``env.num_obs`` / ``step()``'s returned ``obs``
  all refer to the ``"actor"`` group, which Task 6 configured with
  ``history_length=10`` -- i.e. the *history-stacked* observation flattened to
  ``(num_envs, num_one_step_obs * 10)``. They are NOT the single-step obs.
* ``HimActorCritic.act(obs_current, obs_history)`` needs the single-step obs as
  ``obs_current``. Task 6 added a dedicated ``"actor_current"`` group
  (``history_length=0`` on every term) for exactly this. We fetch it directly
  from the observation manager after every reset and every step.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from beyondAMP.mjlab.rsl_rl.amp_wrapper import AMPEnvWrapper
from beyondAMP.motion.motion_dataset import MotionDataset
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer

from .him_actor_critic import HimActorCritic
from .multi_disc_amp_ppo import REGION_NAMES, MultiDiscAMPPPO


def _get_actor_current_obs(env: AMPEnvWrapper) -> torch.Tensor:
    """Single-step actor observation (Task 6's ``actor_current`` group).

    Divergence from G1: G1's HimOnPolicyRunner slices obs_current directly out
    of the history tensor's newest frame, so obs_current and obs_history's
    current slot share one noise/delay realization. Here obs_current comes
    from a separate observation group with its own independent
    enable_corruption sampling -- mjlab flattens history per-term rather than
    as a single trailing block, so recovering G1's exact slice-based approach
    would require a per-term gather/reassemble. Documented and accepted as a
    justified divergence in CLAUDE.md's "Multi-disc obs_current sourcing" row
    (2026-07-02) rather than fixed -- assessed as functionally safe for a
    learned MLP by two independent reviews.
    """
    return env.unwrapped.observation_manager.compute()["actor_current"]


def _get_actor_history_obs(env: AMPEnvWrapper) -> torch.Tensor:
    """History-stacked actor observation (Task 6's ``actor`` group, hist=10)."""
    return env.unwrapped.observation_manager.compute()["actor"]


class HimAMPOnPolicyRunner:
    def __init__(self, env: AMPEnvWrapper, train_cfg: dict, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        num_critic_obs = (
            env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
        )
        # NB: env.num_obs is the "actor" group dim, which is history-flattened
        # (num_one_step_obs * 10). The genuine single-step width comes from the
        # dedicated "actor_current" group instead.
        num_one_step_obs = int(
            env.unwrapped.observation_manager.group_obs_dim["actor_current"][0]
        )

        actor_critic = HimActorCritic(
            num_one_step_obs=num_one_step_obs,
            actor_history_length=10,
            num_critic_obs=num_critic_obs,
            num_actions=env.num_actions,
            **self.policy_cfg,
        ).to(device)

        amp_data_cfgs = train_cfg["amp_data"]
        amp_datasets = {
            name: MotionDataset(cfg, env.unwrapped, device)
            for name, cfg in amp_data_cfgs.items()
        }
        amp_obs_dim = env.get_amp_observations().shape[-1]
        amp_normalizer = Normalizer(amp_obs_dim)
        discriminators = {
            name: AMPDiscriminator(
                amp_obs_dim * 2,
                train_cfg["amp_reward_coef"],
                train_cfg["amp_discr_hidden_dims"],
                device,
                train_cfg["amp_task_reward_lerp"],
            ).to(device)
            for name in REGION_NAMES
        }

        # Ball/region ground-truth land at the end of the critic obs concat
        # order set in Task 6 (ball_gt then region_gt, both critic-only terms
        # appended after the existing critic terms) -> ball_gt occupies the
        # 4 slots before the final region_gt slot.
        region_id_critic_obs_index = -1
        ball_gt_critic_obs_slice = slice(-5, -1)

        min_std = (
            torch.tensor(train_cfg["amp_min_normalized_std"], device=device)
            * torch.abs(env.dof_pos_limits[0, :, 1] - env.dof_pos_limits[0, :, 0])
        )

        self.alg = MultiDiscAMPPPO(
            actor_critic=actor_critic,
            discriminators=discriminators,
            amp_datasets=amp_datasets,
            amp_normalizer=amp_normalizer,
            region_id_critic_obs_index=region_id_critic_obs_index,
            ball_gt_critic_obs_slice=ball_gt_critic_obs_slice,
            amp_obs_dim=amp_obs_dim,
            device=device,
            min_std=min_std,
            **self.alg_cfg,
        )
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]

        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [num_one_step_obs],
            [num_one_step_obs * 10],
            [num_critic_obs],
            [env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Single-step obs (actor_current) and history obs (actor) are both
        # fetched directly from the observation manager -- see module docstring.
        obs = _get_actor_current_obs(self.env)
        obs_history = _get_actor_history_obs(self.env)
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_history, critic_obs, amp_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
            amp_obs.to(self.device),
        )
        self.alg.train_mode()

        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, critic_obs, amp_obs)
                    # step()'s first return value is the "actor" (history-stacked)
                    # group -- NOT obs_current. Discard it and re-fetch both the
                    # single-step and history observations independently.
                    _step_obs, privileged_obs, raw_rewards, dones, infos, reset_env_ids, terminal_amp_states = (
                        self.env.step(actions, not_amp=False)
                    )
                    obs = _get_actor_current_obs(self.env)
                    obs_history = _get_actor_history_obs(self.env)
                    next_amp_obs = self.env.get_amp_observations()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, obs_history, critic_obs, next_amp_obs, raw_rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
                        next_amp_obs.to(self.device),
                        raw_rewards.to(self.device),
                        dones.to(self.device),
                    )

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    region_id = critic_obs[:, self.alg.region_id_critic_obs_index].long()
                    rewards = self.alg.predict_region_routed_amp_reward(
                        amp_obs, next_amp_obs_with_term, region_id, raw_rewards
                    )
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            (mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_est_loss, mean_region_loss, mean_policy_pred, mean_expert_pred) = self.alg.update()
            learn_time = time.time() - start

            if self.log_dir is not None:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += collection_time + learn_time
                self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
                self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
                self.writer.add_scalar("Loss/AMP", mean_amp_loss, it)
                self.writer.add_scalar("Loss/AMP_grad", mean_grad_pen_loss, it)
                self.writer.add_scalar("Loss/est_ball", mean_est_loss, it)
                self.writer.add_scalar("Loss/est_region", mean_region_loss, it)
                if len(rewbuffer) > 0:
                    self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
                    self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
                print(f"[multidisc] it={it} reward={statistics.mean(rewbuffer) if rewbuffer else float('nan'):.3f} "
                      f"est_ball={mean_est_loss:.4f} est_region={mean_region_loss:.4f}")

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.writer is not None and hasattr(self.writer, "stop"):
            self.writer.stop()

    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": {n: d.state_dict() for n, d in self.alg.discriminators.items()},
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        for n, sd in loaded["discriminator_state_dict"].items():
            self.alg.discriminators[n].load_state_dict(sd)
        self.alg.amp_normalizer = loaded["amp_normalizer"]
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        return loaded["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
