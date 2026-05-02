"""Quick environment smoke test — runs a few steps headless.

Usage:
    conda activate /home/isaak/miniconda3/envs/env_isaaclab
    python scripts/test_env.py --headless --num_envs=16 --steps=50
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Goalkeeper env smoke test")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=50, help="Steps to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Suppress noisy log channels that clutter terminal output
try:
    import omni.log
    omni.log.get_log().set_channel_level("omni.usd", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
except Exception:
    pass

import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import goalkeeper  # noqa: F401

from goalkeeper.goalkeeper_env_cfg import GoalkeeperEnvCfg


def main():
    env_cfg = GoalkeeperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    print(f"[test] Creating env with {args_cli.num_envs} envs...")
    env = gym.make("Isaac-Goalkeeper-Direct-v0", cfg=env_cfg)

    print("[test] Resetting env...")
    obs, info = env.reset()
    print(f"[test] policy obs shape: {obs['policy'].shape}")
    print(f"[test] critic obs shape: {obs['critic'].shape}")

    print(f"[test] Running {args_cli.steps} steps...")
    for step in range(args_cli.steps):
        actions = torch.zeros(args_cli.num_envs, env_cfg.action_space, device="cuda:0")
        obs, reward, terminated, truncated, info = env.step(actions)
        if step % 10 == 0:
            print(f"  step {step}: reward={reward.mean():.4f}, terminated={terminated.sum().item()}")

    print("[test] PASSED — env steps completed without error.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
