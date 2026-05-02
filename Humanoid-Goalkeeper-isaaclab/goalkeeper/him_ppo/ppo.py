# Ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py
# Changed imports to use local package modules.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .actor_critic import ActorCritic
from .storage import HIMRolloutStorage


class HIMPPO:
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 value_smoothness_coef=0.1,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.1,
                 amp=None,
                 amp_normalizer=None,
                 motion_buffer=None,
                 ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound

        # 6 independent discriminators (one per motion type)
        self.amp = {
            "lefthand":  deepcopy(amp),
            "righthand": deepcopy(amp),
            "leftjump":  deepcopy(amp),
            "rightjump": deepcopy(amp),
            "leftstep":  deepcopy(amp),
            "rightstep": deepcopy(amp),
        }
        for model in self.amp.values():
            model.to(self.device)

        params = [{'params': self.actor_critic.parameters(), 'name': 'actor_critic'}]
        for key in self.amp:
            params.append({'params': self.amp[key].trunk.parameters(),
                           'weight_decay': 10e-4, 'name': f'amp_trunk_{key}'})
            params.append({'params': self.amp[key].amp_linear.parameters(),
                           'weight_decay': 10e-2, 'name': f'amp_head_{key}'})
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.amp_normalizer = amp_normalizer
        self.motion_buffer = motion_buffer

        self.transition = HIMRolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape, amp_obs_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,
                                         critic_obs_shape, action_shape, amp_obs_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if obs.isnan().any():
            obs = torch.zeros_like(obs)
            critic_obs = torch.zeros_like(critic_obs)

        self.transition.actions = self.actor_critic.act(obs)[0].detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def process_amp_state(self, amp_state):
        self.transition.amp_observations = amp_state

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_est_loss = 0
        mean_region_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (obs_batch, next_obs_batch, critic_obs_batch, actions_batch,
             next_critic_obs_batch, cont_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             amp_obs_batch) in generator:

            _, estball_batch, estregion_batch = self.actor_critic.act(obs_batch)

            # Ground-truth ball estimator target: end_target(3) + ball_vel(3) = 6 dims
            gtball_batch = critic_obs_batch[:, -13:-7]
            # Ground-truth region: recover int region_id from normalized value region/3
            gtregion_batch = (3 * critic_obs_batch[:, -14]).long()

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive KL learning-rate schedule
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate (PPO clip) loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Auxiliary losses: ball estimator (MSE) + region estimator (CrossEntropy)
            est_loss = (estball_batch - gtball_batch).pow(2).mean()
            region_loss = nn.CrossEntropyLoss()(estregion_batch, gtregion_batch)

            loss = (surrogate_loss + est_loss + region_loss +
                    self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean())

            # Smoothness loss (continuity regularisation across episode boundary)
            epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
            policy_smooth_coef = self.smoothness_upper_bound * epsilon
            value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

            mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
            mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
            mix_critic_obs_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)
            policy_smooth_loss = torch.square(
                torch.norm(mu_batch - self.actor_critic.act_inference(mix_obs_batch), dim=-1)
            ).mean()
            value_smooth_loss = torch.square(
                torch.norm(value_batch - self.actor_critic.evaluate(mix_critic_obs_batch), dim=-1)
            ).mean()
            smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss
            loss += smooth_loss

            # AMP discriminator losses
            amp_loss, expert_loss, policy_loss = 0.0, 0.0, 0.0
            if self.amp is not None:
                # motion_id = 3 * (region_id / 3) = region_id
                motion_ids = 3 * critic_obs_batch[:, self.actor_critic.num_one_step_obs + 3]

                amp_expert_obs_batch_mask = torch.zeros_like(amp_obs_batch)
                motion_masks = []
                for motion_val in range(6):
                    mask = motion_ids == motion_val
                    motion_masks.append(mask)
                    if mask.any():
                        n = obs_batch[mask].shape[0]
                        motion_key = ["lefthand", "righthand", "leftjump",
                                      "rightjump", "leftstep", "rightstep"][motion_val]
                        amp_expert_obs_batch_mask[mask] = self.motion_buffer[motion_key].get_expert_obs(
                            batch_size=n
                        ).to(self.device)

                amp_expert_obs_batch = self.amp_normalizer.normalize_torch(amp_expert_obs_batch_mask, self.device)
                amp_obs_batch_norm = self.amp_normalizer.normalize_torch(amp_obs_batch, self.device)

                for motion_val, motion_key in enumerate(
                    ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
                ):
                    mask = motion_masks[motion_val]
                    if mask.any():
                        loss_part, expert_loss_part, policy_loss_part = self.amp[motion_key].compute_loss(
                            amp_obs_batch_norm[mask], amp_expert_obs_batch[mask]
                        )
                        amp_loss = amp_loss + loss_part
                        expert_loss = expert_loss + expert_loss_part
                        policy_loss = policy_loss + policy_loss_part

                loss = loss + amp_loss
                self.amp_normalizer.update(amp_obs_batch_norm.cpu().detach().numpy())
                self.amp_normalizer.update(amp_expert_obs_batch.cpu().detach().numpy())

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_est_loss += est_loss.item()
            mean_region_loss += region_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_est_loss /= num_updates
        mean_region_loss /= num_updates

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_est_loss, mean_region_loss, amp_loss, expert_loss, policy_loss
