"""Evaluate a trained Goalkeeper policy (HIM-PPO checkpoint).

Usage:
    conda activate isaak_isaaclab
    python scripts/play.py --headless --num_envs=4 --checkpoint=logs/him_ppo/goalkeeper/.../model_*.pt
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Goalkeeper with HIM-PPO")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from goalkeeper.agents.him_ppo_cfg import get_goalkeeper_him_cfg
from goalkeeper.him_ppo import HIMOnPolicyRunner, GoalkeeperHimWrapper


def main():
    device = args_cli.device

    env_cfg = GoalkeeperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.play = True

    train_cfg = get_goalkeeper_him_cfg()

    gym_env = gym.make("Isaac-Goalkeeper-Direct-v0", cfg=env_cfg)
    env = GoalkeeperHimWrapper(gym_env, device=device)

    runner = HIMOnPolicyRunner(env, train_cfg=train_cfg, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    obs_tensor = obs["policy"].to(device)

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs_tensor)
        obs, _, _, _, _, _, _ = env.step(actions)
        obs_tensor = obs.to(device)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
