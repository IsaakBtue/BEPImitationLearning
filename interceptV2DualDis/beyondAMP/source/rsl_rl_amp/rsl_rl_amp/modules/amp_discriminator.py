import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl_amp.utils import utils

# FIX 2026-07-08: match G1 exactly (Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/
# amp.py:9,100-118). Independently re-verified: G1 leaves the FIRST trunk
# layer (input_dim -> hidden_dims[0]) at PyTorch default init, but explicitly
# initializes every subsequent hidden layer AND the final amp_linear output
# layer with uniform(-DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE) weights
# and zero bias. This class previously left ALL layers at PyTorch default --
# including amp_linear, the layer that directly produces the discriminator
# logit d. PyTorch's default bound for a Linear(256,1) is 1/sqrt(256)=0.0625,
# ~16x smaller than G1's uniform(-1,1), plausibly producing a systematically
# muted initial logit/reward magnitude. See docs/BugFixes.md.
DISC_LOGIT_INIT_SCALE = 1.0


class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layer_sizes):
            linear = nn.Linear(curr_in_dim, hidden_dim)
            if i > 0:
                # First layer matches G1's default-init first layer; every
                # subsequent trunk layer gets G1's explicit init.
                torch.nn.init.uniform_(linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                torch.nn.init.zeros_(linear.bias)
            amp_layers.append(linear)
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)
        torch.nn.init.uniform_(self.amp_linear.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.amp_linear.bias)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None, num_samples=20, sigma=0.3):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            x = torch.cat([state, next_state], dim=-1)
            d = self.amp_linear(self.trunk(x))

            # FIX 2026-07-08: match G1 exactly (amp.py:185-206 predict_reward).
            # Perturbs the input with num_samples Gaussian noise draws (sigma)
            # and takes the MIN squared-error-from-1 over those samples before
            # computing the clamped reward, instead of a raw single-sample
            # evaluation -- min-over-samples systematically raises amp_reward
            # relative to a raw single-point read for the same underlying
            # discriminator. Independently re-verified present in G1, absent
            # here previously. `d`/d_logits (used for Train/mean_discri_logits
            # logging) stays the raw, unperturbed sample for interpretability.
            # See docs/BugFixes.md.
            noise = torch.randn((*x.shape[:-1], num_samples, x.shape[-1]), device=self.device) * sigma
            perturbed = x.unsqueeze(-2) + noise
            original_shape = perturbed.shape
            perturbed = perturbed.view(-1, original_shape[-1])
            d_all = self.amp_linear(self.trunk(perturbed))
            d_all = d_all.view(*original_shape[:-1])
            squared_errors = torch.square(d_all - 1)
            amp_reward = self.amp_reward_coef * torch.clamp(
                1 - (1/4) * squared_errors.min(dim=-1).values, min=0
            ).unsqueeze(-1)

            reward = amp_reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d.squeeze(), amp_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r