"""Utility functions and MotionLib for the Goalkeeper environment.

Ported from:
  Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_utils.py
  Humanoid-Goalkeeper/legged_gym/legged_gym/utils/math.py

Key change: quaternion convention is wxyz here (Isaac Lab), not xyzw (Isaac Gym).
"""
from __future__ import annotations

import math
import os
import random

import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Math utilities (wxyz convention — Isaac Lab)
# ---------------------------------------------------------------------------

def euler_from_quaternion_wxyz(quat_wxyz: torch.Tensor):
    """Convert wxyz quaternion to (roll, pitch, yaw)."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_imitation_dataset(folder: str, mapping: str, suffix: str = ".pt"):
    """Load all .pt motion files from folder and return (multidataset, joint_id_dict)."""
    filenames = [name for name in os.listdir(folder) if name.endswith(suffix)]

    multidataset = {}
    for filename in tqdm(filenames, desc="Loading motion dataset"):
        try:
            data = torch.load(os.path.join(folder, filename), weights_only=False)
            dataset_list = [data]
            random.shuffle(dataset_list)
            multidataset[filename[: -len(suffix)]] = dataset_list
        except Exception as e:
            print(f"[MotionLib] {filename} load failed: {e}")
            continue

    with open(mapping, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}

    return multidataset, joint_id_dict


def load_single_motion(pt_path: str):
    """Load a single .pt motion file; returns the raw data dict or None."""
    try:
        return torch.load(pt_path, weights_only=False)
    except Exception as e:
        print(f"[MotionLib] Failed to load {pt_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# MotionLib
# ---------------------------------------------------------------------------

class MotionLib:
    def __init__(
        self,
        datasets,
        mapping,
        dof_names,
        keyframe_names,
        fps=30,
        min_dt=0.1,
        device="cpu",
        amp_obs_type="dof",
        num_steps=2,
    ):
        self.fps = fps
        self.device = device
        self.amp_obs_type = amp_obs_type
        self.num_steps = num_steps
        self.min_frames = int(min_dt * fps)

        self.dof_names = dof_names
        self.keyframe_names = keyframe_names
        self.mapping = mapping

        # Build list of valid (dataset_key, motion_tensor) pairs
        self.motion_list = []
        for key, dataset_list in datasets.items():
            for motion in dataset_list:
                if isinstance(motion, dict):
                    frames = self._load_motion(motion)
                    if frames is not None and frames.shape[0] >= self.min_frames:
                        self.motion_list.append(frames)
                else:
                    frames = self._load_motion(motion)
                    if frames is not None and frames.shape[0] >= self.min_frames:
                        self.motion_list.append(frames)

        if len(self.motion_list) == 0:
            raise ValueError("No valid motions found in dataset!")

        print(f"[MotionLib] Loaded {len(self.motion_list)} motion clips.")

    def _load_motion(self, data):
        """Convert raw data dict/tensor to per-frame dof tensor (num_frames, num_robot_dof).

        Matches original g1_utils.py: allocates a (T, len(dof_names)) tensor and maps
        the 21 motion joints from joint_id.txt into their BFS-ordered robot positions,
        leaving the 8 non-motion joints as zero.  This produces 58-dim AMP obs (29×2)
        identical to the original discriminator input.
        """
        try:
            raw = None
            if isinstance(data, dict):
                for key in ["joint_position", "dof_pos", "joint_pos"]:
                    if key in data:
                        tensor = data[key]
                        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                            raw = tensor.float()
                            break
                if raw is None:
                    for val in data.values():
                        if isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[1] >= 6:
                            raw = val.float()
                            break
            elif isinstance(data, torch.Tensor):
                if data.ndim == 2:
                    raw = data.float()

            if raw is None:
                return None

            # Zero-pad to all robot joints (identical to original MotionLib.__init__ logic)
            T = raw.shape[0]
            full = torch.zeros(T, len(self.dof_names), dtype=torch.float)
            for j, name in enumerate(self.dof_names):
                if name in self.mapping:
                    col = self.mapping[name]
                    if col < raw.shape[1]:
                        full[:, j] = raw[:, col]
            return full
        except Exception:
            pass
        return None

    def sample_motion(self, n_envs: int):
        """Sample random starting frames from random motion clips."""
        indices = torch.randint(0, len(self.motion_list), (n_envs,))
        frames = []
        for idx in indices:
            motion = self.motion_list[idx.item()]
            start = random.randint(0, max(0, motion.shape[0] - self.num_steps - 1))
            frames.append(motion[start: start + self.num_steps])
        try:
            return torch.stack(frames, dim=0).to(self.device)
        except Exception:
            return None

    def get_amp_obs(self, n_envs: int):
        """Return AMP observation tensor (n_envs, num_steps * num_dof)."""
        frames = self.sample_motion(n_envs)
        if frames is None:
            return torch.zeros(n_envs, self.num_steps * len(self.dof_names), device=self.device)
        return frames.view(n_envs, -1)

    def get_expert_obs(self, batch_size: int):
        """Alias for get_amp_obs — matches the original HIM-PPO MotionLib interface."""
        return self.get_amp_obs(batch_size)


# ---------------------------------------------------------------------------
# AMP Discriminator
# ---------------------------------------------------------------------------

class AmpDiscriminator(nn.Module):
    """GAIL-style discriminator for AMP motion priors.

    Architecture matches Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py:
    plain Linear layers (no spectral norm), uniform weight init, separate trunk + head.
    Spectral norm is intentionally omitted — it conflicts with create_graph=True in
    the gradient penalty (in-place buffer updates break the autograd graph).
    """

    def __init__(self, input_dim: int = 58, hidden_dims: tuple = (512, 256)):
        super().__init__()
        _scale = 1.0

        # Trunk: input → all hidden layers
        disc_layers: list[nn.Module] = []
        first = nn.Linear(input_dim, hidden_dims[0])
        nn.init.uniform_(first.weight, -_scale, _scale)
        nn.init.zeros_(first.bias)
        disc_layers.extend([first, nn.ReLU()])
        for i in range(len(hidden_dims) - 1):
            ln = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            nn.init.uniform_(ln.weight, -_scale, _scale)
            nn.init.zeros_(ln.bias)
            disc_layers.extend([ln, nn.ReLU()])
        self.trunk = nn.Sequential(*disc_layers)

        # Head: last hidden → scalar logit
        self.amp_linear = nn.Linear(hidden_dims[-1], 1)
        nn.init.uniform_(self.amp_linear.weight, -_scale, _scale)
        nn.init.zeros_(self.amp_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.amp_linear(self.trunk(x))

    def predict_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute reward: clamp(1 - 0.25*(d-1)^2, min=0) from original."""
        with torch.no_grad():
            d = self.forward(obs)
        return torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0).squeeze(-1)

    def compute_loss(
        self,
        expert_obs: torch.Tensor,
        policy_obs: torch.Tensor,
        grad_penalty_coef: float = 5.0,
    ) -> torch.Tensor:
        """GAIL loss + gradient penalty (matches original amp.py)."""
        policy_d = self.forward(policy_obs)
        expert_d = self.forward(expert_obs)
        expert_loss = (expert_d - 1.0).pow(2).mean()
        policy_loss = (policy_d + 1.0).pow(2).mean()
        gail_loss = expert_loss + policy_loss

        # Gradient penalty on expert samples — penalises large input gradients
        exp_for_gp = expert_obs.detach().requires_grad_(True)
        disc_logit = self.forward(exp_for_gp)
        grads = torch.autograd.grad(
            outputs=disc_logit,
            inputs=exp_for_gp,
            grad_outputs=torch.ones_like(disc_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = torch.mean(torch.sum(grads.pow(2), dim=-1)) * grad_penalty_coef * 0.1

        return gail_loss + grad_pen


# ---------------------------------------------------------------------------
# AMP Normalizer (running mean/std for AMP observations)
# ---------------------------------------------------------------------------

class AmpNormalizer:
    """Online running mean/variance normalizer for AMP observations."""

    def __init__(self, dim: int, device: str, epsilon: float = 1e-4, clip_obs: float = 10.0):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = torch.tensor(1.0, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def update(self, x: torch.Tensor):
        """Update running statistics from a batch of observations."""
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0) if x.shape[0] > 1 else torch.zeros_like(batch_mean)
        batch_count = float(x.shape[0])
        total = self.count + batch_count
        self.mean = (self.count * self.mean + batch_count * batch_mean) / total
        self.var = (self.count * self.var + batch_count * batch_var) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.mean) / torch.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )
