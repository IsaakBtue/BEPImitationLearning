"""Multi-discriminator AMP-PPO: dict of independent AMPDiscriminator instances
and MotionDataset expert buffers, one per region, region-masked loss and
reward — ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py
(HIMPPO.update / him_on_policy_runner.py's rollout reward loop), adapted to
this codebase's AMPDiscriminator (predict_amp_reward, task-reward lerp baked
into the discriminator) instead of G1's AMP module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl_amp.storage.rollout_storage import RolloutStorage
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator

from .him_actor_critic import HimActorCritic

REGION_NAMES: tuple[str, ...] = ("left_near", "left_far", "right_near", "right_far")


class MultiDiscAMPPPO:
    actor_critic: HimActorCritic

    def __init__(
        self,
        actor_critic: HimActorCritic,
        discriminators: dict[str, AMPDiscriminator],
        amp_datasets: dict,
        amp_normalizer,
        region_id_critic_obs_index: int,
        ball_gt_critic_obs_slice: slice,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        min_std=None,
        **kwargs,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.region_id_critic_obs_index = region_id_critic_obs_index
        self.ball_gt_critic_obs_slice = ball_gt_critic_obs_slice

        assert set(discriminators.keys()) == set(REGION_NAMES)
        assert set(amp_datasets.keys()) == set(REGION_NAMES)
        self.discriminators = discriminators
        self.amp_datasets = amp_datasets
        for d in self.discriminators.values():
            d.to(device)
        self.amp_normalizer = amp_normalizer

        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage: RolloutStorage | None = None

        params = [{"params": self.actor_critic.parameters(), "name": "actor_critic"}]
        for name, d in self.discriminators.items():
            params.append({"params": d.trunk.parameters(), "weight_decay": 10e-4, "name": f"amp_trunk_{name}"})
            params.append({"params": d.amp_linear.parameters(), "weight_decay": 10e-2, "name": f"amp_head_{name}"})
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self._obs_current = None
        self._obs_history = None
        self._amp_obs = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_current_shape,
                      obs_history_shape, critic_obs_shape, action_shape):
        # Stack obs_current and obs_history into one "observations" tensor for
        # RolloutStorage (which only knows one obs slot); split again at update
        # time via the known obs_current width.
        self._obs_current_dim = obs_current_shape[0]
        self._obs_history_dim = obs_history_shape[0]
        combined_obs_shape = [self._obs_current_dim + self._obs_history_dim]
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, combined_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_current, obs_history, critic_obs, amp_obs):
        combined_obs = torch.cat([obs_current, obs_history], dim=-1)
        self.transition_actions = self.actor_critic.act(obs_current.detach(), obs_history.detach()).detach()
        self.transition_values = self.actor_critic.evaluate(critic_obs.detach()).detach()
        self.transition_actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_actions).detach()
        self.transition_action_mean = self.actor_critic.action_mean.detach()
        self.transition_action_sigma = self.actor_critic.action_std.detach()
        self._pending_obs = combined_obs
        self._pending_critic_obs = critic_obs
        self._pending_amp_obs = amp_obs
        return self.transition_actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        transition = RolloutStorage.Transition()
        transition.observations = self._pending_obs
        transition.critic_observations = self._pending_critic_obs
        transition.actions = self.transition_actions
        transition.rewards = rewards.clone()
        transition.dones = dones
        transition.values = self.transition_values
        transition.actions_log_prob = self.transition_actions_log_prob
        transition.action_mean = self.transition_action_mean
        transition.action_sigma = self.transition_action_sigma
        if "time_outs" in infos:
            transition.rewards += self.gamma * torch.squeeze(
                transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(transition)
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def predict_region_routed_amp_reward(self, amp_obs, next_amp_obs, region_id, task_reward):
        """Rollout-time style reward, routed per-sample by region id. Mirrors
        him_on_policy_runner.py:161-178's masked predict_reward loop."""
        num_envs = amp_obs.shape[0]
        reward = torch.zeros(num_envs, device=amp_obs.device)
        for r, name in enumerate(REGION_NAMES):
            mask = region_id == r
            if not mask.any():
                continue
            r_out, _, _ = self.discriminators[name].predict_amp_reward(
                amp_obs[mask], next_amp_obs[mask], task_reward[mask], normalizer=self.amp_normalizer)
            reward[mask] = r_out
        return reward

    def update(self):
        mean_value_loss = mean_surrogate_loss = mean_amp_loss = mean_grad_pen_loss = 0.0
        mean_est_loss = mean_region_loss = mean_policy_pred = mean_expert_pred = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_expert_generators = {
            name: ds.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            for name, ds in self.amp_datasets.items()
        }

        for sample in generator:
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) = sample

            obs_current_batch = obs_batch[:, :self._obs_current_dim]
            obs_history_batch = obs_batch[:, self._obs_current_dim:]

            self.actor_critic.act(obs_current_batch.detach(), obs_history_batch.detach())
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch.detach())
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Auxiliary estimator losses (ported from him_ppo.py:226-228).
            gt_ball = critic_obs_batch[:, self.ball_gt_critic_obs_slice]
            gt_region = critic_obs_batch[:, self.region_id_critic_obs_index].long()
            est_loss = (self.actor_critic.estimate_ball - gt_ball).pow(2).mean()
            region_loss = nn.CrossEntropyLoss()(self.actor_critic.estimate_region, gt_region)

            loss = (surrogate_loss + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean() + est_loss + region_loss)

            # Region-routed AMP loss (ported from him_ppo.py:244-305).
            amp_loss = torch.tensor(0.0, device=self.device)
            grad_pen_loss = torch.tensor(0.0, device=self.device)
            policy_preds, expert_preds = [], []
            for r, name in enumerate(REGION_NAMES):
                mask = gt_region == r
                if not mask.any():
                    continue
                expert_state, expert_next_state = next(amp_expert_generators[name])
                policy_amp = obs_current_batch[mask]  # placeholder policy-side amp obs slice; see Task 5 note
                # NOTE: the actual policy-side AMP observation comes from the
                # amp_storage replay buffer (populated in process_env_step),
                # sampled the same way single-disc AMPPPO.update() does via
                # self.amp_storage.feed_forward_generator — wired in Task 5
                # once HimAMPOnPolicyRunner supplies amp_storage per region.
                discr = self.discriminators[name]
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_state_n = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state_n = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                else:
                    expert_state_n, expert_next_state_n = expert_state, expert_next_state
                expert_d = discr(torch.cat([expert_state_n, expert_next_state_n], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                grad_pen = discr.compute_grad_pen(expert_state_n, expert_next_state_n, lambda_=10)
                amp_loss = amp_loss + 0.5 * expert_loss
                grad_pen_loss = grad_pen_loss + grad_pen
                expert_preds.append(expert_d.mean().item())

            loss = loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_est_loss += est_loss.item()
            mean_region_loss += region_loss.item()
            if expert_preds:
                mean_expert_pred += sum(expert_preds) / len(expert_preds)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return (mean_value_loss / num_updates, mean_surrogate_loss / num_updates,
                mean_amp_loss / num_updates, mean_grad_pen_loss / num_updates,
                mean_est_loss / num_updates, mean_region_loss / num_updates,
                mean_policy_pred / num_updates, mean_expert_pred / num_updates)
