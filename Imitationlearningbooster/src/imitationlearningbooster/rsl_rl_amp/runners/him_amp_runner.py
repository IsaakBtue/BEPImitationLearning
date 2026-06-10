# src/imitationlearningbooster/rsl_rl_amp/runners/him_amp_runner.py
"""GoalkeeperAmpRunner — AMP training loop for T1 goalkeeper.

Pattern from ccrpRepo/AMP_mjlab rsl_rl/runners/amp_on_policy_runner.py.
Key adaptations:
  - 6 discriminators (one per motion class) routed by motion_type_ids
  - AMP obs from infos["observations"]["amp"] (mjlab ObservationGroup)
  - Terminal state: copy amp_obs for reset envs before they are overwritten
  - Uses mjlab v5 runner API (MotionTrackingOnPolicyRunner as base)
"""
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.optim as optim

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from imitationlearningbooster.rsl_rl_amp.modules.discriminator import Discriminator
from imitationlearningbooster.rsl_rl_amp.utils.normalizer import EmpiricalNormalization
from imitationlearningbooster.rsl_rl_amp.storage.replay_buffer import ReplayBuffer
from imitationlearningbooster.rsl_rl_amp.utils.motion_loader import GoalkeeperMotionLoader, MOTION_NAMES

AMP_OBS_DIM = 46       # 23 DOF × 2 frames
AMP_COEF = 0.4         # 40% AMP + 60% task (matches original Goalkeeper)
AMP_BUFFER_SIZE = 100_000


class GoalkeeperAmpRunner(MotionTrackingOnPolicyRunner):
    """AMP-augmented PPO runner for Booster T1 goalkeeper.

    Inherits checkpoint save/load, ONNX export, and logging from base.
    Adds 6 discriminators + AMP reward blending to the PPO training loop.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
        super().__init__(env, train_cfg, log_dir, device, registry_name)

        data_dir = Path(__file__).parent.parent.parent / "motions" / "data"
        amp_coef = train_cfg.get("amp_coef", 0.4)  # 40% AMP + 60% task (matches Goalkeeper)
        amp_disc_hidden = train_cfg.get("amp_discr_hidden_dims", [512, 256])

        # 6 discriminators — one per motion class
        self.discriminators = {
            name: Discriminator(
                input_dim=AMP_OBS_DIM,
                amp_reward_coef=0.1,  # raw AMP reward scale (blending happens in runner)
                hidden_layer_sizes=amp_disc_hidden,
                device=device,
            )
            for name in MOTION_NAMES
        }

        # 6 expert motion loaders
        self.motion_loaders = {
            name: GoalkeeperMotionLoader(
                str(data_dir / f"{name}_t1.npz"), device=device
            )
            for name in MOTION_NAMES
        }

        # 6 replay buffers (one per motion class)
        self.replay_buffers = {
            name: ReplayBuffer(
                obs_dim=AMP_OBS_DIM // 2,  # single-frame dim = 23
                buffer_size=AMP_BUFFER_SIZE,
                device=device,
            )
            for name in MOTION_NAMES
        }

        # AMP obs normalizer (shared, updated with all policy + expert transitions)
        self.amp_normalizer = EmpiricalNormalization(shape=(AMP_OBS_DIM,), device=device)

        # Single Adam optimizer for all 6 discriminators
        disc_params = []
        for disc in self.discriminators.values():
            disc_params.append({"params": disc.trunk.parameters(), "weight_decay": 1e-3})
            disc_params.append({"params": disc.amp_linear.parameters(), "weight_decay": 1e-1})
        # lr lowered 1e-4→1e-5: at 1e-4 the discriminator overtook the policy by iter ~700
        # (disc_loss stuck at 4.2, AMP reward collapsed). G1 uses a shared optimizer with
        # the same adaptive LR as PPO; a separate disc optimizer must be much slower.
        self.disc_optimizer = optim.Adam(disc_params, lr=1e-5)

        # Buffer to store amp_obs and motion_ids during rollout
        num_steps = train_cfg.get("num_steps_per_env", 100)
        num_envs = env.num_envs
        self._rollout_amp_obs = torch.zeros(num_steps, num_envs, AMP_OBS_DIM, device=device)
        self._rollout_motion_ids = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self._amp_coef = amp_coef

    def _get_motion_type_ids(self) -> torch.Tensor:
        """Read which motion class each env is currently tracking (0–5)."""
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        return cmd.motion_type_ids.clone()  # (N,) long

    def _verify_joint_order(self):
        """Assert that robot.joint_names matches the joint order in the npz files.

        Catches silent AMP failure caused by mismatched joint ordering between
        mjlab (robot.data.joint_pos) and the converted motion npz files.
        Runs once at the start of training; skipped if npz predates joint_names addition.
        """
        try:
            base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            robot = base_env.scene["robot"]
            robot_names = list(robot.joint_names)
        except Exception:
            print("[AMP] WARNING: could not read robot.joint_names — joint order unverified.")
            return

        for motion_name, loader in self.motion_loaders.items():
            if loader.joint_names is None:
                print(
                    f"[AMP] WARNING: {motion_name}_t1.npz has no joint_names key. "
                    "Re-run convert_all.py to regenerate npz files with joint order metadata."
                )
                continue
            npz_names = [n if isinstance(n, str) else n.decode() for n in loader.joint_names]
            if npz_names != robot_names:
                mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(npz_names, robot_names)) if a != b]
                raise RuntimeError(
                    f"[AMP] FATAL: joint order mismatch in {motion_name}_t1.npz vs robot.\n"
                    f"  npz ({len(npz_names)} joints): {npz_names}\n"
                    f"  robot ({len(robot_names)} joints): {robot_names}\n"
                    f"  first mismatches: {mismatch[:5]}\n"
                    "Re-run convert_all.py to regenerate npz files."
                )
        print(f"[AMP] Joint order verified OK: {robot_names}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self._verify_joint_order()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.logger.init_logging_writer()

        obs_dict = self.env.get_observations()
        # Initial AMP obs (pre-step) from env's "amp" group
        amp_obs = self._extract_amp_obs(obs_dict)

        self.alg.train_mode()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        num_steps = self.cfg["num_steps_per_env"]

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        cur_rew = torch.zeros(self.env.num_envs, device=self.device)
        cur_amp = torch.zeros(self.env.num_envs, device=self.device)
        cur_len = torch.zeros(self.env.num_envs, device=self.device)

        for it in range(start_it, total_it):
            t0 = time.time()

            with torch.inference_mode():
                for step in range(num_steps):
                    actions = self.alg.act(obs_dict)
                    obs_dict_next, task_rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )

                    # AMP obs from next obs_dict "amp" group
                    next_amp_obs_raw = self._extract_amp_obs(obs_dict_next)

                    # Terminal state fix: for reset envs the amp obs is already the NEW
                    # episode's initial state. Use pre-step amp_obs instead.
                    next_amp_obs = next_amp_obs_raw.clone()
                    reset_ids = dones.bool().nonzero(as_tuple=False).flatten()
                    if len(reset_ids) > 0:
                        next_amp_obs[reset_ids] = amp_obs[reset_ids]

                    # Build 2-frame AMP obs: [prev_joint_pos, curr_joint_pos]
                    amp_obs_2frame = torch.cat([amp_obs, next_amp_obs], dim=-1)  # (N, 46)

                    # Discriminator reward per env (routing by motion_type_id)
                    task_r = task_rewards.to(self.device)
                    motion_ids = self._get_motion_type_ids()  # (N,)
                    amp_r_all = torch.zeros_like(task_r)

                    # Compute AMP reward per discriminator
                    for disc_idx, name in enumerate(MOTION_NAMES):
                        mask = (motion_ids == disc_idx)
                        if mask.any():
                            with torch.no_grad():
                                obs_norm = self.amp_normalizer.normalize_torch(
                                    amp_obs_2frame[mask], self.device
                                )
                                d = self.discriminators[name](obs_norm)
                                amp_r_all[mask] = torch.clamp(
                                    1.0 - 0.25 * (d - 1.0).pow(2), min=0.0
                                ).squeeze(-1) * 0.5  # matches G1 Goalkeeper

                    # Blend: amp_coef * AMP + (1 - amp_coef) * task = 0.4 * AMP + 0.6 * task
                    blended_r = self._amp_coef * amp_r_all + (1.0 - self._amp_coef) * task_r

                    # Store for discriminator update
                    self._rollout_amp_obs[step] = amp_obs_2frame
                    self._rollout_motion_ids[step] = motion_ids

                    # Store in replay buffers per motion class
                    for disc_idx, name in enumerate(MOTION_NAMES):
                        mask = (motion_ids == disc_idx)
                        if mask.any():
                            self.replay_buffers[name].insert(amp_obs[mask], next_amp_obs[mask])

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
                    amp_obs = next_amp_obs_raw  # advance (use un-fixed version for next step)

                collect_time = time.time() - t0
                t1 = time.time()
                self.alg.compute_returns(obs_dict)

            # PPO update
            loss_dict = self.alg.update()

            # Discriminator update — 4 steps per iteration (not 20).
            # G1's "20 steps" are joint actor-critic+discriminator steps inside the same
            # optimizer. Our discriminator has a separate optimizer that runs AFTER the
            # policy update, so 20 pure disc steps per iteration lets the discriminator
            # overtake the policy before it can respond. 4 steps (one epoch) keeps the
            # discriminator from getting too far ahead of the policy each iteration.
            num_mini_batches = self.cfg.get("algorithm", {}).get("num_mini_batches", 4)
            num_disc_updates = num_mini_batches  # 4 steps, not 5×4=20
            # Cap per-discriminator batch to avoid OOM in compute_grad_pen (create_graph=True
            # on 25k samples exhausts 31 GB VRAM; G1 upstream gets ~17k on an 8 GB GPU).
            # Default 4096 is close to G1's effective batch without the OOM risk.
            amp_disc_batch_cap = self.cfg.get("amp_disc_mini_batch_size", 4096)
            disc_per_motion = min(
                (self.env.num_envs * num_steps) // num_mini_batches // len(MOTION_NAMES),
                amp_disc_batch_cap,
            )

            disc_loss_total = 0.0
            for _ in range(num_disc_updates):
                # One gradient step per disc update (matches G1's 20 separate SGD steps).
                disc_loss_iter = torch.tensor(0.0, device=self.device)
                for name in MOTION_NAMES:
                    if not self.replay_buffers[name].ready(disc_per_motion):
                        continue
                    pol_s, pol_s_next = next(self.replay_buffers[name].feed_forward_generator(
                        disc_per_motion
                    ))
                    exp_s, exp_s_next = next(self.motion_loaders[name].feed_forward_generator(
                        disc_per_motion
                    ))

                    pol_raw = torch.cat([pol_s, pol_s_next], dim=-1)
                    exp_raw = torch.cat([exp_s, exp_s_next], dim=-1)

                    # Update normalizer with raw (unnormalized) observations.
                    self.amp_normalizer.update(pol_raw.cpu())
                    self.amp_normalizer.update(exp_raw.cpu())

                    pol_obs = self.amp_normalizer.normalize_torch(pol_raw, self.device)
                    exp_obs = self.amp_normalizer.normalize_torch(exp_raw, self.device)

                    disc = self.discriminators[name]
                    expert_d = disc(exp_obs)
                    policy_d = disc(pol_obs)
                    expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
                    policy_loss = torch.nn.MSELoss()(policy_d, -torch.ones_like(policy_d))
                    grad_pen = disc.compute_grad_pen(exp_obs, lambda_=5.0) * 0.1  # matches G1 effective lambda=0.5
                    disc_loss_iter = disc_loss_iter + (expert_loss + policy_loss) + grad_pen

                self.disc_optimizer.zero_grad()
                if disc_loss_iter.requires_grad:
                    disc_loss_iter.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.disc_optimizer.param_groups for p in g["params"]],
                        max_norm=1.0,
                    )
                    self.disc_optimizer.step()
                disc_loss_total += disc_loss_iter.item()

            learn_time = time.time() - t1
            self.current_learning_iteration = it

            loss_dict["amp_disc_loss"] = disc_loss_total
            loss_dict["mean_amp_reward"] = sum(amp_rewbuffer) / len(amp_rewbuffer) if amp_rewbuffer else 0.0
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

    def _extract_amp_obs(self, obs_dict) -> torch.Tensor:
        """Extract AMP joint-pos observation from an obs_dict (TensorDict or plain dict)."""
        amp_raw = obs_dict.get("amp", None)
        if amp_raw is None:
            return torch.zeros(self.env.num_envs, AMP_OBS_DIM // 2, device=self.device)
        if hasattr(amp_raw, "get"):
            amp_raw = amp_raw.get("amp_obs", amp_raw)
        if hasattr(amp_raw, "to"):
            return amp_raw.to(self.device)
        return torch.zeros(self.env.num_envs, AMP_OBS_DIM // 2, device=self.device)

    def save(self, path: str, infos=None):
        amp_state = {
            "discriminators": {n: d.state_dict() for n, d in self.discriminators.items()},
            "amp_normalizer": self.amp_normalizer.state_dict(),
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
