# Ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/actor_critic.py
# No changes except removing unused imports.

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_one_step_obs,
                 actor_history_length,
                 num_actions=19,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected kwargs: " + str(list(kwargs.keys())))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.num_actions = num_actions

        self.history_latent_dim = 16
        self.estimate_ball_dim = 6
        self.num_regions = 6

        # actor input: current_obs(96) + history_latent(16) + estimate_ball(6) + region_argmax(1) = 119
        mlp_input_dim_a = num_one_step_obs + self.history_latent_dim + self.estimate_ball_dim + 1
        self.num_actor_input = mlp_input_dim_a

        mlp_input_dim_c = num_critic_obs
        mlp_input_dim_h = num_one_step_obs * actor_history_length

        self.history_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.history_latent_dim),
        )

        self.ball_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.estimate_ball_dim),
        )

        self.region_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_regions),
        )

        # Actor
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"History encoder: {self.history_encoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.estimate_ball = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        history_latent = self.history_encoder(obs_history)
        self.estimate_ball = self.ball_estimator(obs_history)
        self.estimate_region = self.region_estimator(obs_history)
        actor_input = torch.cat((
            obs_history[:, -self.num_one_step_obs:],
            history_latent,
            self.estimate_ball,
            torch.argmax(self.estimate_region, dim=-1, keepdim=True),
        ), dim=-1)
        action_mean = self.actor(actor_input)
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample(), self.estimate_ball, self.estimate_region

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        history_latent = self.history_encoder(obs_history)
        estimate_ball = self.ball_estimator(obs_history)
        estimate_region = self.region_estimator(obs_history)
        actor_input = torch.cat((
            obs_history[:, -self.num_one_step_obs:],
            history_latent,
            estimate_ball,
            torch.argmax(estimate_region, dim=-1, keepdim=True),
        ), dim=-1)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
