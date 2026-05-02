# Ported from Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py
# Unchanged except removing unused imports.

import torch
import torch.nn as nn
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
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


class AMP(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs, amp_coef, hidden_dims=[512, 256], activation='relu',
                 init_noise_std=1.0, device='cuda:0', **kwargs):
        if kwargs:
            print("AMP.__init__ got unexpected kwargs: " + str(list(kwargs.keys())))
        super(AMP, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim = num_obs

        disc_layers = []
        disc_layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        disc_layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                ln = nn.Linear(hidden_dims[l], 1)
                torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(ln.bias)
                self.amp_linear = ln
            else:
                ln = nn.Linear(hidden_dims[l], hidden_dims[l + 1])
                torch.nn.init.uniform_(ln.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(ln.bias)
                disc_layers.append(ln)
                disc_layers.append(activation)
        self.trunk = nn.Sequential(*disc_layers)

        print(f"Discriminator MLP: {self.trunk}")
        self.device = device
        self.amp_coef = amp_coef

    def compute_grad_pen(self, expert_state, policy_state, lambda_=5):
        expert_state.requires_grad = True
        disc_demo_logit = self.trunk(expert_state)
        disc_demo_logit = self.amp_linear(disc_demo_logit)
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit, expert_state,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True, retain_graph=True, only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        return disc_grad_penalty * lambda_

    def forward(self, x):
        return self.amp_linear(self.trunk(x))

    def compute_loss(self, agent_obs, expert_obs):
        policy_d = self.amp_linear(self.trunk(agent_obs))
        expert_d = self.amp_linear(self.trunk(expert_obs))

        expert_loss = (expert_d - 1).pow(2).mean()
        policy_loss = (policy_d + 1).pow(2).mean()
        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen(expert_obs, agent_obs) * 0.1

        return gail_loss + grad_pen, expert_loss, policy_loss

    def predict_reward(self, agent_obs, normalizer, num_samples=20, sigma=0.3):
        with torch.no_grad():
            self.eval()
            agent_obs = normalizer.normalize_torch(agent_obs, self.device)
            noise = torch.randn((*agent_obs.shape[:-1], num_samples, agent_obs.shape[-1]),
                                device=self.device) * sigma
            perturbed_obs = agent_obs.unsqueeze(-2) + noise
            original_shape = perturbed_obs.shape
            perturbed_obs = perturbed_obs.view(-1, original_shape[-1])

            d_all = self.amp_linear(self.trunk(perturbed_obs))
            d_all = d_all.view(*original_shape[:-1])

            squared_errors = torch.square(d_all - 1)
            reward = torch.clamp(1 - 0.25 * squared_errors.min(dim=-1).values, min=0).unsqueeze(-1)
            self.train()
            return reward
