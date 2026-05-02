"""Wrapper that translates the Isaac Lab gymnasium env interface to the HIM-PPO env interface.

HIMOnPolicyRunner expects:
  env.step(actions)  → (obs, priv_obs, raw_rewards, dones, infos, term_ids, term_priv_obs)
  env.get_observations()           → (N, num_obs)         actor history
  env.get_privileged_observations()→ (N, num_privileged_obs) critic obs
  env.get_amp_observations()       → (N, num_amp_joints)  current per-frame amp obs
  env.reset()
  env.num_obs, env.num_privileged_obs, env.num_one_step_obs, env.actor_history_length
  env.num_actions, env.num_amp_obs, env.motions
  env.num_envs, env.episode_length_buf, env.max_episode_length
"""
from __future__ import annotations

import torch


class GoalkeeperHimWrapper:
    """Wraps gymnasium GoalkeeperEnv for use with HIMOnPolicyRunner."""

    def __init__(self, gym_env, device: str = "cuda:0"):
        self._gym_env = gym_env
        # Unwrap to the raw GoalkeeperEnv (DirectRLEnv)
        self._inner_env = gym_env.unwrapped  # type: ignore
        self.device = device

        self._last_obs: dict | None = None

    # ------------------------------------------------------------------
    # Static dimensions — read from inner env after first reset
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._inner_env.num_envs

    @property
    def num_obs(self) -> int:
        return self._inner_env.cfg.observation_space  # 960

    @property
    def num_privileged_obs(self) -> int:
        return self._inner_env.cfg.num_privileged_obs  # 113

    @property
    def num_one_step_obs(self) -> int:
        return self._inner_env.cfg.num_one_step_observations  # 96

    @property
    def actor_history_length(self) -> int:
        return self._inner_env.cfg.num_actor_history  # 10

    @property
    def num_actions(self) -> int:
        return self._inner_env.cfg.action_space  # 29

    @property
    def num_amp_obs(self) -> int:
        # The concatenated 2-frame dim stored in the rollout buffer
        return self._inner_env.cfg.amp_num_obs  # 42  (21 joints × 2 frames)

    @property
    def motions(self):
        return self._inner_env.motion_libs  # dict[str, MotionLib]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._inner_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._inner_env.episode_length_buf = value

    @property
    def max_episode_length(self) -> int:
        return self._inner_env.max_episode_length

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def reset(self):
        obs, info = self._gym_env.reset()
        self._last_obs = obs
        return obs, info

    def get_observations(self) -> torch.Tensor:
        assert self._last_obs is not None, "Call reset() before get_observations()"
        return self._last_obs["policy"].to(self.device)

    def get_privileged_observations(self) -> torch.Tensor:
        assert self._last_obs is not None, "Call reset() before get_privileged_observations()"
        return self._last_obs["critic"].to(self.device)

    def get_amp_observations(self) -> torch.Tensor:
        """Return per-frame AMP obs: current robot dof_pos at motion joints (N, 21)."""
        return self._inner_env.get_amp_observations().to(self.device)

    def step(self, actions: torch.Tensor):
        """Return 7-tuple matching the original HIM env interface."""
        obs_dict, rewards, terminated, truncated, infos = self._gym_env.step(actions)
        self._last_obs = obs_dict

        obs = obs_dict["policy"].to(self.device)
        priv_obs = obs_dict["critic"].to(self.device)
        rewards = rewards.to(self.device)
        dones = (terminated | truncated).float().to(self.device)

        # Indices of newly terminated envs + their (post-reset) critic obs.
        # For HIM smoothness loss these are masked out (cont_batch = 1 - dones = 0),
        # so the actual values don't matter — we just provide the post-reset priv_obs.
        term_ids = terminated.nonzero(as_tuple=False).squeeze(-1).to(self.device)
        term_priv_obs = priv_obs[term_ids] if term_ids.numel() > 0 else priv_obs[:0]

        return obs, priv_obs, rewards, dones, infos, term_ids, term_priv_obs

    def close(self):
        self._gym_env.close()
