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
from rsl_rl_amp.storage.replay_buffer import ReplayBuffer
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
        amp_obs_dim: int,
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
        amp_replay_buffer_size: int = 100_000,
        min_std=None,
        success_buffer=None,
        sil_coef: float = 0.1,
        sil_batch_size: int = 256,
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
        self.amp_storages: dict[str, ReplayBuffer] = {
            name: ReplayBuffer(amp_obs_dim, amp_replay_buffer_size, device)
            for name in REGION_NAMES
        }

        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage: RolloutStorage | None = None

        # region_estimator shares the single actor_critic param group/LR with
        # actor, critic, history_encoder, and ball_estimator -- matching G1
        # exactly (Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py:
        # 101-116: one 'actor_critic' group covers the whole ActorCritic,
        # region_estimator included, no separate LR).
        #
        # 2026-07-05 this fork briefly split region_estimator into its own
        # undamped, always-high-LR param group (region_estimator_learning_rate,
        # exempted from the adaptive-KL throttle) to fix a real regression
        # (region_estimator accuracy collapsing under schedule="adaptive").
        # That rationale rested on a factual error: it assumed G1 uses
        # schedule="fixed" as its proven config. It doesn't -- G1's actual
        # config chain (g1_29_config.py -> legged_robot_config.py:326)
        # inherits schedule="adaptive" unmodified, with region_estimator
        # fully exposed to the same KL-based throttle as everything else, and
        # G1 has no documented region_estimator collapse. The split was a net
        # new divergence from G1, not a fix, and every training run since it
        # landed has diverged (to NaN) far earlier than the pre-split
        # baseline. Reverted 2026-07-06; see docs/BugFixes.md.
        self.main_params = list(actor_critic.parameters())
        params = [
            {"params": self.main_params, "name": "actor_critic"},
        ]
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

        # FEAT 2026-07-14: self-imitation buffer, see success_buffer.py for
        # the full rationale. Optional -- if None, update() behaves exactly
        # as before (no SIL term added to the loss).
        self.success_buffer = success_buffer
        self.sil_coef = sil_coef
        self.sil_batch_size = sil_batch_size

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
        # Ported from Humanoid-Goalkeeper/rsl_rl/algorithms/him_ppo.py:138-140 --
        # G1 zeros out a NaN'd observation batch before it reaches the network,
        # rather than letting it propagate into the actor/critic and corrupt
        # their weights on the next backward pass.
        if obs_current.isnan().any() or obs_history.isnan().any():
            obs_current = torch.zeros_like(obs_current)
            obs_history = torch.zeros_like(obs_history)
            critic_obs = torch.zeros_like(critic_obs)
        combined_obs = torch.cat([obs_current, obs_history], dim=-1)
        self.transition_actions = self.actor_critic.act(obs_current.detach(), obs_history.detach()).detach()
        self.transition_values = self.actor_critic.evaluate(critic_obs.detach()).detach()
        self.transition_actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_actions).detach()
        self.transition_action_mean = self.actor_critic.action_mean.detach()
        self.transition_action_sigma = self.actor_critic.action_std.detach()
        self._pending_obs = combined_obs
        self._pending_critic_obs = critic_obs
        self._pending_amp_obs = amp_obs
        self._pending_region = critic_obs[:, self.region_id_critic_obs_index].long()
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

        region = self._pending_region
        for r, name in enumerate(REGION_NAMES):
            mask = region == r
            if mask.any():
                self.amp_storages[name].insert(self._pending_amp_obs[mask], amp_obs[mask])

        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs.detach()).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def predict_region_routed_amp_reward(self, amp_obs, next_amp_obs, region_id, task_reward):
        """Rollout-time style reward, routed per-sample by region id. Mirrors
        him_on_policy_runner.py:161-178's masked predict_reward loop.

        2026-07-05: also returns per-sample d_logits/amp_reward (previously
        discarded via `_, _`) so the runner can log Train/mean_amp_reward and
        Train/mean_discri_logits like the stock single-discriminator runner.
        """
        num_envs = amp_obs.shape[0]
        reward = torch.zeros(num_envs, device=amp_obs.device)
        d_logits = torch.zeros(num_envs, device=amp_obs.device)
        amp_reward = torch.zeros(num_envs, device=amp_obs.device)
        for r, name in enumerate(REGION_NAMES):
            mask = region_id == r
            if not mask.any():
                continue
            r_out, d_out, a_out = self.discriminators[name].predict_amp_reward(
                amp_obs[mask], next_amp_obs[mask], task_reward[mask], normalizer=self.amp_normalizer)
            reward[mask] = r_out
            d_logits[mask] = d_out
            amp_reward[mask] = a_out
        return reward, d_logits, amp_reward

    def update(self):
        mean_value_loss = mean_surrogate_loss = mean_amp_loss = mean_grad_pen_loss = 0.0
        mean_est_loss = mean_region_loss = mean_policy_pred = mean_expert_pred = 0.0
        mean_sil_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        minibatch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_expert_generators = {
            name: ds.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches, minibatch_size,
            )
            for name, ds in self.amp_datasets.items()
        }
        amp_policy_generators = {
            name: rb.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches, minibatch_size,
            )
            for name, rb in self.amp_storages.items()
        }

        for sample in generator:
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch) = sample

            # 2026-07-06: run 2026-07-06_01-14-09_intercept_phase1 showed
            # action_rate_l2/action_acc_l2 (unbounded quadratic penalties on
            # raw policy output, mjlab_mdp default) occasionally exploding to
            # -1e8/-1e9 for a subset of envs. Bootstrapped through GAE this
            # produced value_function loss up to 1e26 and never recovered.
            # G1 (Humanoid-Goalkeeper/rsl_rl/algorithms/him_ppo.py) has no
            # equivalent clamp, but it also has no unbounded-magnitude reward
            # source like this and additionally runs a value/policy
            # smooth_loss regularizer this fork's plain RolloutStorage can't
            # support (no next_obs/cont tracking) -- porting that is a larger
            # storage change, deferred. This clamp is the minimal targeted
            # fix: bound the value regression target so one anomalous step
            # can never blow up value_loss's gradient magnitude by 20 orders
            # of magnitude. 1000 is >> any plausible genuine return here
            # (softstop alone maxes out at curriculum weight 250 one-shot).
            returns_batch = returns_batch.clamp(-1000.0, 1000.0)
            target_values_batch = target_values_batch.clamp(-1000.0, 1000.0)

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
                    # Matches G1 (him_ppo.py:208-209): every param group,
                    # region_estimator included, gets rescaled -- no exemption.
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

            # FEAT 2026-07-14: self-imitation loss, see success_buffer.py.
            # Behavior-cloning-style NLL of recorded successful actions under
            # the CURRENT policy, sampled fresh every minibatch so it stays
            # part of the same gradient step as the main PPO loss rather than
            # a separate update pass. No-op (sil_loss stays 0) until the
            # buffer has accumulated at least one full sample batch's worth
            # of genuine-landing transitions.
            sil_loss = torch.tensor(0.0, device=self.device)
            if self.success_buffer is not None and len(self.success_buffer) >= self.sil_batch_size:
                sil_obs_current, sil_obs_history, sil_actions = self.success_buffer.sample(self.sil_batch_size)
                self.actor_critic.act(sil_obs_current.detach(), sil_obs_history.detach())
                sil_log_prob = self.actor_critic.get_actions_log_prob(sil_actions)
                sil_loss = -sil_log_prob.mean()
                loss = loss + self.sil_coef * sil_loss

            # Region-routed AMP loss (ported from him_ppo.py:244-305).
            amp_loss = torch.tensor(0.0, device=self.device)
            grad_pen_loss = torch.tensor(0.0, device=self.device)
            policy_preds, expert_preds = [], []
            normalizer_states = []
            for r, name in enumerate(REGION_NAMES):
                mask = gt_region == r
                if not mask.any():
                    continue
                expert_state, expert_next_state = next(amp_expert_generators[name])
                policy_state, policy_next_state = next(amp_policy_generators[name])
                discr = self.discriminators[name]
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_state_n = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state_n = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                        policy_state_n = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state_n = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                else:
                    expert_state_n, expert_next_state_n = expert_state, expert_next_state
                    policy_state_n, policy_next_state_n = policy_state, policy_next_state
                expert_d = discr(torch.cat([expert_state_n, expert_next_state_n], dim=-1))
                policy_d = discr(torch.cat([policy_state_n, policy_next_state_n], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                # FIX 2026-07-08: match G1 exactly (Humanoid-Goalkeeper/rsl_rl/
                # rsl_rl/modules/amp.py:124-173). G1's compute_grad_pen defaults
                # to lambda_=5, then compute_loss applies an additional *0.1
                # scaling on top (grad_pen = self.compute_grad_pen(...) * 0.1),
                # net effective coefficient 0.5. This port previously called
                # lambda_=10 with no further scaling -- effective coefficient
                # 10, ~20x stronger regularization than G1, which over-flattens
                # the discriminator's logit landscape. Also: G1's gail_loss is
                # the unweighted sum (expert_loss + policy_loss); this port
                # previously averaged (0.5 * each), halving amp_loss's relative
                # magnitude against the rest of the total loss. Both now match
                # G1's actual formula, not just its outcome. See docs/BugFixes.md.
                grad_pen = discr.compute_grad_pen(expert_state_n, expert_next_state_n, lambda_=5) * 0.1
                amp_loss = amp_loss + expert_loss + policy_loss
                grad_pen_loss = grad_pen_loss + grad_pen
                expert_preds.append(expert_d.mean().item())
                policy_preds.append(policy_d.mean().item())
                normalizer_states.append((policy_state, expert_state))

            loss = loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            # Single joint clip over the whole actor_critic, matching G1
            # exactly (him_ppo.py:310).
            nn.utils.clip_grad_norm_(self.main_params, self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            # Refresh the (shared) normalizer's running mean/std from every
            # region processed this minibatch — ported from amp_ppo.py:227-229,
            # which does this once per minibatch for its single region's worth
            # of data. Here each region draws its own policy/expert pair, so
            # each pair feeds the update, keeping the normalizer's statistics
            # from staying frozen at init.
            if self.amp_normalizer is not None:
                for policy_state, expert_state in normalizer_states:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_est_loss += est_loss.item()
            mean_region_loss += region_loss.item()
            mean_sil_loss += sil_loss.item()
            if expert_preds:
                mean_expert_pred += sum(expert_preds) / len(expert_preds)
            if policy_preds:
                mean_policy_pred += sum(policy_preds) / len(policy_preds)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        # FIX 2026-07-08: match G1's on-policy-only AMP discriminator training
        # (Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py -- its
        # "policy" sample for the discriminator loss comes directly from the
        # same HIMRolloutStorage minibatch as everything else, cleared every
        # update). This port previously left amp_storages as a persistent
        # 250k-transition FIFO buffer, training the discriminator against a
        # stale mix of past-policy behavior across many updates instead of
        # strictly the current on-policy rollout. Clearing here makes each
        # amp_storages[name] hold only the transitions collected since the
        # last update, matching G1's semantics exactly. See docs/BugFixes.md.
        for rb in self.amp_storages.values():
            rb.clear()
        return (mean_value_loss / num_updates, mean_surrogate_loss / num_updates,
                mean_amp_loss / num_updates, mean_grad_pen_loss / num_updates,
                mean_est_loss / num_updates, mean_region_loss / num_updates,
                mean_policy_pred / num_updates, mean_expert_pred / num_updates,
                mean_sil_loss / num_updates)
