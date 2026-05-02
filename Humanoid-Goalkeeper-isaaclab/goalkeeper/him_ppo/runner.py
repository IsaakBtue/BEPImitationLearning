# Ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/runners/him_on_policy_runner.py
# Changes: local imports; tensorboard-only logger; git dependency removed.

import time
import os
import statistics
from collections import deque

from torch.utils.tensorboard import SummaryWriter

import torch

from .ppo import HIMPPO
from .actor_critic import ActorCritic
from .amp import AMP
from .normalizer import Normalizer


class HIMOnPolicyRunner:

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        num_critic_obs = (self.env.num_privileged_obs
                          if self.env.num_privileged_obs is not None
                          else self.env.num_one_step_obs)

        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self.actor_history_length = self.env.actor_history_length

        actor_critic: ActorCritic = ActorCritic(
            self.num_actor_obs,
            self.num_critic_obs,
            self.env.num_one_step_obs,
            self.actor_history_length,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self.amp_cfg = train_cfg["amp"]
        self.amp_coef = self.amp_cfg['amp_coef']
        amp = AMP(self.amp_cfg['num_obs'], self.amp_cfg['amp_coef'], device=self.device).to(self.device)
        amp_normalizer = Normalizer(self.amp_cfg['num_obs'])

        self.alg: HIMPPO = HIMPPO(
            actor_critic,
            amp=amp,
            amp_normalizer=amp_normalizer,
            motion_buffer=self.env.motions,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            [self.env.num_amp_obs],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        amp_state = self.env.get_amp_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        raw_rewbuffer = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_raw_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    old_amp_state = amp_state

                    obs, privileged_obs, raw_rewards, dones, infos, termination_ids, termination_privileged_obs = \
                        self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    raw_rewards = raw_rewards.to(self.device)
                    dones = dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    amp_state = self.env.get_amp_observations().to(self.device)
                    amp_state_ = torch.cat([old_amp_state, amp_state], dim=1).to(self.device)
                    self.alg.process_amp_state(amp_state_)

                    # Compute AMP reward per motion type
                    num_envs = obs.shape[0]
                    amp_reward = torch.zeros(num_envs, device=self.device)
                    motion_ids = 3 * critic_obs[:, self.alg.actor_critic.num_one_step_obs + 3]
                    for motion_key, motion_val in zip(
                        ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"],
                        [0, 1, 2, 3, 4, 5],
                    ):
                        mask = motion_ids == motion_val
                        if mask.any():
                            rew = self.alg.amp[motion_key].predict_reward(
                                amp_state_[mask], normalizer=self.alg.amp_normalizer
                            ).squeeze(1) * 0.5
                            amp_reward[mask] = rew

                    # Override next_critic_obs for terminated envs (pre-reset obs)
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    rewards = amp_reward * self.amp_coef + raw_rewards * (1 - self.amp_coef)
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_raw_reward_sum += raw_rewards
                        cur_amp_reward_sum += amp_reward
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        raw_rewbuffer.extend(cur_raw_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        amp_rewbuffer.extend(cur_amp_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_raw_reward_sum[new_ids] = 0
                        cur_amp_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_est_loss, mean_region_loss, \
                amp_loss, expert_loss, policy_loss = self.alg.update()

            learn_time = time.time() - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()
            self.current_learning_iteration = it

        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std[0:10].mean()
        fps = int(self.num_steps_per_env * self.env.num_envs /
                  (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/estball', locs['mean_est_loss'], locs['it'])
        self.writer.add_scalar('Loss/region', locs['mean_region_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/amp_loss', locs['amp_loss'] if isinstance(locs['amp_loss'], float) else locs['amp_loss'].item(), locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_raw_reward', statistics.mean(locs['raw_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_amp_reward', statistics.mean(locs['amp_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        it_str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{it_str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s """
                          f"""(collection: {locs['collection_time']:.3f}s, """
                          f"""learning: {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimate ball loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                          f"""{'Region loss:':>{pad}} {locs['mean_region_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean raw reward:':>{pad}} {statistics.mean(locs['raw_rewbuffer']):.2f}\n"""
                          f"""{'Mean AMP reward:':>{pad}} {statistics.mean(locs['amp_rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{it_str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s """
                          f"""(collection: {locs['collection_time']:.3f}s, """
                          f"""learning: {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimate ball loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                          f"""{'Region loss:':>{pad}} {locs['mean_region_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        eta = self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it'])
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {eta:.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration + 1,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
