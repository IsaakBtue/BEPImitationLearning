# src/imitationlearningbooster/rsl_rl_amp/runners/him_amp_runner.py
"""GoalkeeperAmpRunner — faithful port of G1 HIMOnPolicyRunner + HIMPPO to mjlab/rsl_rl-v5.

Architecture mirrors G1 exactly (him_on_policy_runner.py + him_ppo.py):
  - One shared Adam: actor + critic + all 6 discriminator trunks/heads.
  - Each mini-batch step: loss = surrogate + value + smooth + amp_disc → single backward.
  - 20 gradient steps per iteration (num_learning_epochs × num_mini_batches, default 5×4).
  - Smooth loss uses consecutive (t, t+1) obs pairs from rollout storage + cont mask.
  - Normalizer updated with already-normalised data (matches G1 HIMPPO L304-305).
  - AMP reward kernel: noise-sampling (σ=0.3, N=20) × 0.5 → G1 runner L175-178.

Only T1-specific adaptations documented in DIVERGENCE_FROM_UPSTREAM.md:
  - 6 separate Discriminator objects (one per motion class) vs G1's one shared AMP module.
  - AMP obs from mjlab infos["amp"] group vs G1's env.get_amp_observations().
  - Terminal-state amp_obs fix for reset envs (necessary because mjlab resets envs
    at episode boundary before the runner sees the done flag).
  - Curriculum and ball difficulty logic lives in resets.py (T1-only).
"""
from __future__ import annotations

import os
import time
from collections import deque
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from imitationlearningbooster.rsl_rl_amp.modules.discriminator import Discriminator
from imitationlearningbooster.rsl_rl_amp.utils.normalizer import EmpiricalNormalization
from imitationlearningbooster.rsl_rl_amp.utils.motion_loader import GoalkeeperMotionLoader, MOTION_NAMES

AMP_OBS_DIM = 46   # 23 DOF × 2 consecutive frames (same as G1's num_obs=58 for 29-DOF G1)
AMP_COEF    = 0.4  # 40% AMP + 60% task — G1 amp_coef


class GoalkeeperAmpRunner(MotionTrackingOnPolicyRunner):
    """AMP-augmented PPO runner for Booster T1 goalkeeper.

    Inherits checkpoint save/load, ONNX export, and W&B logging from base.
    The learn() loop and update step are replaced to inject AMP in the G1 style.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
        super().__init__(env, train_cfg, log_dir, device, registry_name)

        data_dir   = Path(__file__).parent.parent.parent / "motions" / "data"
        amp_coef   = train_cfg.get("amp_coef", AMP_COEF)
        disc_hidden = train_cfg.get("amp_discr_hidden_dims", [512, 256])
        self._amp_coef = amp_coef

        # ── 6 discriminators (one per motion class) ─────────────────────────
        # G1 HIMPPO L90-100: 6 deepcopies of one AMP module, all managed together.
        self.discriminators = {
            name: Discriminator(
                input_dim=AMP_OBS_DIM,
                amp_reward_coef=0.1,
                hidden_layer_sizes=disc_hidden,
                device=device,
            )
            for name in MOTION_NAMES
        }

        # ── 6 expert motion loaders ──────────────────────────────────────────
        self.motion_loaders = {
            name: GoalkeeperMotionLoader(
                str(data_dir / f"{name}_t1.npz"), device=device
            )
            for name in MOTION_NAMES
        }

        # ── Shared AMP obs normalizer ────────────────────────────────────────
        self.amp_normalizer = EmpiricalNormalization(shape=(AMP_OBS_DIM,), device=device)

        # ── Add disc params to the SAME optimizer as actor-critic ────────────
        # G1 HIMPPO L101-116: one Adam for actor_critic + all 6 amp modules.
        # weight_decay mirrors G1: trunk = 10e-4 = 1e-3, head = 10e-2 = 1e-1.
        for name, disc in self.discriminators.items():
            self.alg.optimizer.add_param_group({
                "params": list(disc.trunk.parameters()),
                "weight_decay": 1e-3,
                "name": f"amp_trunk_{name}",
            })
            self.alg.optimizer.add_param_group({
                "params": list(disc.amp_linear.parameters()),
                "weight_decay": 1e-1,
                "name": f"amp_head_{name}",
            })

        # ── Rollout buffers (T × N) filled during env steps ─────────────────
        num_steps = train_cfg.get("num_steps_per_env", 100)
        num_envs  = env.num_envs
        self._rollout_amp_obs    = torch.zeros(num_steps, num_envs, AMP_OBS_DIM,    device=device)
        self._rollout_motion_ids = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)

        # ── Smoothness regularisation coefficients (G1 HIMPPO L55-57) ───────
        self._smooth_lower = train_cfg.get("smoothness_lower_bound", 0.1)
        self._smooth_upper = train_cfg.get("smoothness_upper_bound", 1.0)

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _get_motion_type_ids(self) -> torch.Tensor:
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        return cmd.motion_type_ids.clone()  # (N,) long, values 0-5

    def _extract_amp_obs(self, obs_dict) -> torch.Tensor:
        """Pull single-frame AMP joint-pos obs from mjlab obs_dict."""
        amp_raw = obs_dict.get("amp", None)
        if amp_raw is None:
            return torch.zeros(self.env.num_envs, AMP_OBS_DIM // 2, device=self.device)
        if hasattr(amp_raw, "get"):
            amp_raw = amp_raw.get("amp_obs", amp_raw)
        if hasattr(amp_raw, "to"):
            return amp_raw.to(self.device)
        return torch.zeros(self.env.num_envs, AMP_OBS_DIM // 2, device=self.device)

    def _verify_joint_order(self):
        """Assert robot joint_names matches npz joint order. Runs once at learn() start."""
        try:
            base_env   = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            robot_names = list(base_env.scene["robot"].joint_names)
        except Exception:
            print("[AMP] WARNING: could not read robot.joint_names — joint order unverified.")
            return
        for motion_name, loader in self.motion_loaders.items():
            if loader.joint_names is None:
                print(f"[AMP] WARNING: {motion_name}_t1.npz has no joint_names key.")
                continue
            npz_names = [n if isinstance(n, str) else n.decode() for n in loader.joint_names]
            if npz_names != robot_names:
                mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(npz_names, robot_names)) if a != b]
                raise RuntimeError(
                    f"[AMP] FATAL: joint order mismatch in {motion_name}_t1.npz vs robot.\n"
                    f"  npz ({len(npz_names)}): {npz_names}\n"
                    f"  robot ({len(robot_names)}): {robot_names}\n"
                    f"  first mismatches: {mismatch[:5]}"
                )
        print(f"[AMP] Joint order verified OK: {robot_names}")

    # ────────────────────────────────────────────────────────────────────────
    # Main training loop  (mirrors G1 HIMOnPolicyRunner.learn)
    # ────────────────────────────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self._verify_joint_order()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.logger.init_logging_writer()

        obs_dict  = self.env.get_observations()
        amp_obs   = self._extract_amp_obs(obs_dict)  # (N, 23) — single frame

        self.alg.train_mode()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        num_steps = self.cfg["num_steps_per_env"]

        rewbuffer    = deque(maxlen=100)
        lenbuffer    = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        cur_rew = torch.zeros(self.env.num_envs, device=self.device)
        cur_amp = torch.zeros(self.env.num_envs, device=self.device)
        cur_len = torch.zeros(self.env.num_envs, device=self.device)

        for it in range(start_it, total_it):
            t0 = time.time()

            # ── Rollout (G1 HIMOnPolicyRunner.learn L150-207) ───────────────
            with torch.inference_mode():
                for step in range(num_steps):
                    actions = self.alg.act(obs_dict)
                    obs_dict_next, task_rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )

                    # AMP obs from next step
                    next_amp_obs_raw = self._extract_amp_obs(obs_dict_next)

                    # Terminal-state fix: mjlab resets env before runner sees done,
                    # so next_amp_obs for reset envs is already the NEW episode's
                    # initial state. Use pre-step amp_obs instead (G1 L162: old_amp_state).
                    next_amp_obs = next_amp_obs_raw.clone()
                    reset_ids = dones.bool().nonzero(as_tuple=False).flatten()
                    if len(reset_ids) > 0:
                        next_amp_obs[reset_ids] = amp_obs[reset_ids]

                    # 2-frame obs: [prev_joint_pos, curr_joint_pos]  (G1 L162)
                    amp_obs_2frame = torch.cat([amp_obs, next_amp_obs], dim=-1)  # (N, 46)

                    # AMP reward per env — G1 L165-178 (noise-sampling kernel × 0.5)
                    task_r  = task_rewards.to(self.device)
                    motion_ids = self._get_motion_type_ids()
                    amp_r_all  = torch.zeros_like(task_r)
                    for disc_idx, name in enumerate(MOTION_NAMES):
                        mask = (motion_ids == disc_idx)
                        if mask.any():
                            with torch.no_grad():
                                self.discriminators[name].eval()
                                obs_n = self.amp_normalizer.normalize_torch(
                                    amp_obs_2frame[mask], self.device
                                )
                                M, D = obs_n.shape
                                noise    = torch.randn(M, 20, D, device=self.device) * 0.3
                                perturbed = obs_n.unsqueeze(1) + noise
                                d_all    = self.discriminators[name](
                                    perturbed.view(M * 20, D)
                                ).view(M, 20)
                                sq_err   = (d_all - 1.0).pow(2)
                                amp_r_all[mask] = torch.clamp(
                                    1.0 - 0.25 * sq_err.min(dim=-1).values, min=0.0
                                ) * 0.5  # G1 runner L177
                                self.discriminators[name].train()

                    # Blend: G1 L185  rewards = amp_r*amp_coef + raw_r*(1-amp_coef)
                    blended_r = self._amp_coef * amp_r_all + (1.0 - self._amp_coef) * task_r

                    # Store rollout buffers (flushed each _update_with_amp call)
                    self._rollout_amp_obs[step]    = amp_obs_2frame
                    self._rollout_motion_ids[step] = motion_ids

                    self.alg.process_env_step(obs_dict_next, blended_r, dones, infos)
                    self.logger.process_env_step(blended_r, dones, infos, None)

                    cur_rew += blended_r
                    cur_amp += amp_r_all
                    cur_len += 1
                    new_ids = dones.bool()
                    rewbuffer.extend(cur_rew[new_ids].cpu().tolist())
                    amp_rewbuffer.extend(cur_amp[new_ids].cpu().tolist())
                    lenbuffer.extend(cur_len[new_ids].cpu().tolist())
                    cur_rew[new_ids] = 0
                    cur_amp[new_ids] = 0
                    cur_len[new_ids] = 0

                    obs_dict = obs_dict_next
                    amp_obs  = next_amp_obs_raw  # advance without terminal fix

                collect_time = time.time() - t0
                t1 = time.time()
                self.alg.compute_returns(obs_dict)

            # ── Joint PPO + AMP + smoothness update ─────────────────────────
            # G1: HIMPPO.update() — one optimizer, one backward per mini-batch step.
            loss_dict = self._update_with_amp()

            learn_time = time.time() - t1
            self.current_learning_iteration = it

            loss_dict["mean_amp_reward"] = (
                sum(amp_rewbuffer) / len(amp_rewbuffer) if amp_rewbuffer else 0.0
            )
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it,
                collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.log_dir and it % self.cfg.get("save_interval", 200) == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.log_dir:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.logger.writer is not None:
            self.logger.stop_logging_writer()

    # ────────────────────────────────────────────────────────────────────────
    # Update — mirrors G1 HIMPPO.update() faithfully
    # ────────────────────────────────────────────────────────────────────────

    def _update_with_amp(self) -> dict[str, float]:
        """G1-faithful joint update: one Adam, one backward per mini-batch step.

        G1 HIMPPO.update() structure replicated here:
          for epoch in num_learning_epochs:           # default 5
            for mb in num_mini_batches:               # default 4
              loss = surrogate + value + smooth + amp_disc
              optimizer.zero_grad()
              loss.backward()
              clip_grad_norm_(actor + critic)         # disc grads unclipped (G1 L310)
              optimizer.step()
        Total: 20 gradient steps per iteration.
        """
        storage = self.alg.storage
        T = storage.num_transitions_per_env
        N = storage.num_envs
        num_epochs      = self.alg.num_learning_epochs
        num_mini_batches = self.alg.num_mini_batches
        batch_size       = T * N
        mini_batch_size  = batch_size // num_mini_batches

        actor  = self.alg.actor
        critic = self.alg.critic

        # ── Flatten storage tensors (T*N,…) ─────────────────────────────────
        # Matches G1 HIMRolloutStorage.mini_batch_generator flatten ordering.
        obs_flat      = storage.observations.flatten(0, 1)       # TensorDict (T*N,…)
        act_flat      = storage.actions.flatten(0, 1)            # (T*N, act)
        val_flat      = storage.values.flatten(0, 1)             # (T*N, 1)
        ret_flat      = storage.returns.flatten(0, 1)            # (T*N, 1)
        adv_flat      = storage.advantages.flatten(0, 1)         # (T*N, 1)
        old_lp_flat   = storage.actions_log_prob.flatten(0, 1)   # (T*N, 1)
        old_dp_flat   = tuple(p.flatten(0, 1) for p in storage.distribution_params)
        dones_flat    = storage.dones.flatten(0, 1).float()      # (T*N, 1)
        cont_flat     = 1.0 - dones_flat                         # (T*N, 1) — 0 at done steps

        # ── Next-obs for smooth loss (G1 HIMPPO L235-237) ───────────────────
        # obs[t+1] at step t (clamped to last step at t = T-1).
        # Flat layout: index i = step*(N) + env, so step = i//N, env = i%N.
        # next-step index = min(step+1, T-1)*N + env.
        raw        = torch.arange(batch_size, device=self.device)
        shifted    = raw + N
        same_last  = (T - 1) * N + raw % N   # last-step entry for same env
        next_idx   = torch.where(shifted < batch_size, shifted, same_last)
        obs_next_flat = obs_flat[next_idx]    # TensorDict (T*N,…), next-step obs

        # ── Flatten AMP rollout buffers (same T*N ordering) ─────────────────
        amp_obs_flat = self._rollout_amp_obs.flatten(0, 1)      # (T*N, 46)
        mids_flat    = self._rollout_motion_ids.flatten(0, 1)   # (T*N,)

        # ── Smoothness coefficients (G1 HIMPPO L232-233) ────────────────────
        eps                = self._smooth_lower / (self._smooth_upper - self._smooth_lower)
        policy_smooth_coef = self._smooth_upper * eps           # ≈ 0.111
        value_smooth_coef  = 0.1 * policy_smooth_coef          # G1 value_smoothness_coef=0.1

        # ── Running loss accumulators ────────────────────────────────────────
        mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
        mean_amp_loss   = mean_smooth_loss    = 0.0
        num_updates = num_epochs * num_mini_batches

        # One random permutation, shared across all epochs (matches rsl_rl mini_batch_generator)
        indices = torch.randperm(batch_size, device=self.device)

        for _epoch in range(num_epochs):
            for mb in range(num_mini_batches):
                start = mb * mini_batch_size
                stop  = (mb + 1) * mini_batch_size
                bidx  = indices[start:stop]   # (B,) random indices

                obs_b      = obs_flat[bidx]       # TensorDict (B,…)
                act_b      = act_flat[bidx]        # (B, act)
                val_b      = val_flat[bidx]        # (B, 1)
                ret_b      = ret_flat[bidx]        # (B, 1)
                adv_b      = adv_flat[bidx]        # (B, 1)
                old_lp_b   = old_lp_flat[bidx]    # (B, 1)
                old_dp_b   = tuple(p[bidx] for p in old_dp_flat)
                cont_b     = cont_flat[bidx]       # (B, 1)
                obs_next_b = obs_next_flat[bidx]   # TensorDict (B,…)
                amp_obs_b  = amp_obs_flat[bidx]    # (B, 46)
                mids_b     = mids_flat[bidx]       # (B,)

                # ── Actor forward (stochastic) — G1 HIMPPO L184 ─────────────
                actor(obs_b, stochastic_output=True)
                log_prob = actor.get_output_log_prob(act_b)
                entropy  = actor.output_entropy
                dist_params = tuple(actor.output_distribution_params)
                mu_b        = dist_params[0]   # action mean — live in graph for smooth loss

                # ── KL adaptive LR — G1 HIMPPO L197-209 ────────────────────
                if self.alg.desired_kl is not None and self.alg.schedule == "adaptive":
                    with torch.inference_mode():
                        kl      = actor.get_kl_divergence(old_dp_b, dist_params)
                        kl_mean = kl.mean()
                        if kl_mean > self.alg.desired_kl * 2.0:
                            self.alg.learning_rate = max(1e-5, self.alg.learning_rate / 1.5)
                        elif kl_mean < self.alg.desired_kl / 2.0 and kl_mean > 0.0:
                            self.alg.learning_rate = min(1e-2, self.alg.learning_rate * 1.5)
                        for pg in self.alg.optimizer.param_groups:
                            pg["lr"] = self.alg.learning_rate

                # ── Surrogate loss — G1 HIMPPO L212-215 ─────────────────────
                ratio     = torch.exp(log_prob - old_lp_b.squeeze(-1))
                adv       = adv_b.squeeze(-1)
                surrogate = -adv * ratio
                surrogate_clipped = -adv * ratio.clamp(
                    1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # ── Value loss — G1 HIMPPO L218-224 ─────────────────────────
                values_pred = critic(obs_b)
                if self.alg.use_clipped_value_loss:
                    val_clipped = val_b + (values_pred - val_b).clamp(
                        -self.alg.clip_param, self.alg.clip_param
                    )
                    value_loss = torch.max(
                        (values_pred - ret_b).pow(2),
                        (val_clipped - ret_b).pow(2),
                    ).mean()
                else:
                    value_loss = (ret_b - values_pred).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.alg.value_loss_coef * value_loss
                    - self.alg.entropy_coef * entropy.mean()
                )

                # ── Smooth loss — G1 HIMPPO L231-242 ────────────────────────
                # mix_w = cont * U(-1,1) → 0 at terminal transitions.
                mix_w    = cont_b * (torch.rand_like(cont_b) - 0.5) * 2.0    # (B, 1)
                mix_obs_b = obs_b.apply(
                    lambda x, y: x + mix_w * (y - x), obs_next_b
                )                                                               # TensorDict (B,…)
                mu_mix   = actor(mix_obs_b, stochastic_output=False)           # (B, act)
                val_mix  = critic(mix_obs_b)                                   # (B, 1)
                policy_smooth_loss = torch.square(
                    torch.norm(mu_b - mu_mix, dim=-1)
                ).mean()
                value_smooth_loss = torch.square(
                    torch.norm(values_pred - val_mix, dim=-1)
                ).mean()
                smooth_loss = (
                    policy_smooth_coef * policy_smooth_loss
                    + value_smooth_coef * value_smooth_loss
                )
                loss = loss + smooth_loss

                # ── AMP disc loss — G1 HIMPPO L244-303 ──────────────────────
                # Expert obs: fresh sample from motion loader each mini-batch (G1 L251-284).
                amp_expert_b = torch.zeros_like(amp_obs_b)
                for disc_idx, name in enumerate(MOTION_NAMES):
                    mask = (mids_b == disc_idx)
                    if mask.any():
                        n = mask.sum().item()
                        exp_s, exp_s_next = next(
                            self.motion_loaders[name].feed_forward_generator(n)
                        )
                        amp_expert_b[mask] = torch.cat(
                            [exp_s, exp_s_next], dim=-1
                        ).to(self.device)

                # Normalise (G1 HIMPPO L287-288)
                amp_expert_norm = self.amp_normalizer.normalize_torch(amp_expert_b, self.device)
                amp_policy_norm = self.amp_normalizer.normalize_torch(amp_obs_b,   self.device)

                # Per-motion disc loss (G1 HIMPPO L290-303 via amp.compute_loss)
                amp_loss_b = torch.tensor(0.0, device=self.device)
                for disc_idx, name in enumerate(MOTION_NAMES):
                    mask = (mids_b == disc_idx)
                    if mask.any():
                        disc     = self.discriminators[name]
                        expert_d = disc(amp_expert_norm[mask])
                        policy_d = disc(amp_policy_norm[mask])
                        # G1 amp.py L164-169: MSE expert→+1, policy→-1, grad_pen×0.1
                        e_loss   = (expert_d - 1).pow(2).mean()
                        p_loss   = (policy_d  + 1).pow(2).mean()
                        gp       = disc.compute_grad_pen(
                            amp_expert_norm[mask], lambda_=5.0
                        ) * 0.1  # effective λ=0.5, matches G1 amp.py L169
                        amp_loss_b = amp_loss_b + e_loss + p_loss + gp

                # Update normaliser with already-normalised data (G1 HIMPPO L304-305)
                self.amp_normalizer.update(amp_policy_norm.cpu().detach())
                self.amp_normalizer.update(amp_expert_norm.cpu().detach())

                loss = loss + amp_loss_b

                # ── Single backward + step — G1 HIMPPO L307-311 ─────────────
                self.alg.optimizer.zero_grad()
                loss.backward()
                # G1 L310: clip_grad_norm_ on actor_critic only; disc grads unclipped.
                nn.utils.clip_grad_norm_(
                    chain(actor.parameters(), critic.parameters()),
                    self.alg.max_grad_norm,
                )
                self.alg.optimizer.step()

                mean_value_loss    += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy       += entropy.mean().item()
                mean_amp_loss      += amp_loss_b.item()
                mean_smooth_loss   += smooth_loss.item()

        storage.clear()

        return {
            "value":        mean_value_loss    / num_updates,
            "surrogate":    mean_surrogate_loss / num_updates,
            "entropy":      mean_entropy        / num_updates,
            "amp_disc_loss": mean_amp_loss      / num_updates,
            "smooth_loss":  mean_smooth_loss    / num_updates,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Checkpoint save / load
    # ────────────────────────────────────────────────────────────────────────

    def save(self, path: str, infos=None):
        amp_state = {
            "discriminators":  {n: d.state_dict() for n, d in self.discriminators.items()},
            "amp_normalizer":  self.amp_normalizer.state_dict(),
        }
        super().save(path, infos={**(infos or {}), "amp": amp_state})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "amp" in infos:
            amp = infos["amp"]
            for name, state in amp.get("discriminators", {}).items():
                if name in self.discriminators:
                    self.discriminators[name].load_state_dict(state)
            if "amp_normalizer" in amp:
                self.amp_normalizer.load_state_dict(amp["amp_normalizer"])
        return infos
