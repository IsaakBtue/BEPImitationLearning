#!/usr/bin/env python3
"""Export goalkeeper checkpoint to TorchScript for deployment.

Usage:
    # Export a specific checkpoint (always writes goalkeeper_t1_latest.pt):
    python export_model.py --checkpoint path/to/model_200.pt

    # Re-export the default (model_2000 from 2026-05-23 run):
    python export_model.py

Output: goalkeeper_deploy/tasks/goalkeeper/models/goalkeeper_t1_latest.pt
"""
import argparse
import os
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

DEFAULT_CHECKPOINT = os.path.join(
    _REPO,
    "my_mjlab_project_booster_t1",
    "logs", "rsl_rl", "g1_goalkeeper",
    "2026-05-23_18-35-15", "model_2000.pt",
)
OUTPUT_PATH = os.path.join(
    _HERE, "tasks", "goalkeeper", "models", "goalkeeper_t1_latest.pt",
)

OBS_DIM = 87
HISTORY = 10
INPUT_DIM = OBS_DIM * HISTORY   # 870
OUTPUT_DIM = 23
HIDDEN_DIMS = (512, 256, 128)
EPS = 1e-2


class GoalkeeperActor(nn.Module):
    """Deterministic goalkeeper actor for TorchScript deployment.

    Wraps EmpiricalNormalization + MLP extracted from the rsl_rl checkpoint.
    Input:  obs_history flattened: (870,)  [oldest ... newest obs]
    Output: raw joint actions:     (23,)   [in training MuJoCo joint order]
    """

    def __init__(self, obs_mean: torch.Tensor, obs_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.eps: float = EPS
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIMS[0]), nn.ELU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]), nn.ELU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]), nn.ELU(),
            nn.Linear(HIDDEN_DIMS[2], OUTPUT_DIM),
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        normalized = (obs_flat - self.obs_mean) / (self.obs_std + self.eps)
        return self.mlp(normalized)


def export(checkpoint_path: str) -> None:
    print(f"Loading  ← {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    obs_mean = sd["obs_normalizer._mean"].squeeze(0)
    obs_std = sd["obs_normalizer._std"].squeeze(0)

    model = GoalkeeperActor(obs_mean, obs_std)
    model.mlp[0].weight.data = sd["mlp.0.weight"]
    model.mlp[0].bias.data = sd["mlp.0.bias"]
    model.mlp[2].weight.data = sd["mlp.2.weight"]
    model.mlp[2].bias.data = sd["mlp.2.bias"]
    model.mlp[4].weight.data = sd["mlp.4.weight"]
    model.mlp[4].bias.data = sd["mlp.4.bias"]
    model.mlp[6].weight.data = sd["mlp.6.weight"]
    model.mlp[6].bias.data = sd["mlp.6.bias"]
    model.eval()

    scripted = torch.jit.script(model)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    scripted.save(OUTPUT_PATH)
    print(f"Exported → {OUTPUT_PATH}")

    loaded = torch.jit.load(OUTPUT_PATH)
    dummy = torch.zeros(INPUT_DIM)
    with torch.no_grad():
        out = loaded(dummy)
    assert out.shape == (OUTPUT_DIM,), f"Unexpected output shape: {out.shape}"
    print(f"Verified output shape: {out.shape}  ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c",
        default=DEFAULT_CHECKPOINT,
        help="Path to rsl_rl model_N.pt checkpoint (absolute, or relative to current working directory).",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.abspath(ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    export(ckpt)
