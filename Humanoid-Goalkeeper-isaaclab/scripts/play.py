"""Evaluate a trained Goalkeeper policy.

Usage:
    conda activate isaak_isaaclab
    python scripts/play.py --headless --num_envs=4 --checkpoint=logs/rsl_rl/goalkeeper/.../model_*.pt
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Goalkeeper with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import goalkeeper  # noqa: F401

from goalkeeper.goalkeeper_env_cfg import GoalkeeperEnvCfg
from goalkeeper.agents.rsl_rl_ppo_cfg import GoalkeeperPPORunnerCfg


def main():
    env_cfg = GoalkeeperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.play = True

    agent_cfg = GoalkeeperPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = gym.make("Isaac-Goalkeeper-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
