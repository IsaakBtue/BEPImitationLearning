"""Train the Goalkeeper policy with HIM-PPO (full equivalent of the original).

Usage:
    conda activate isaak_isaaclab
    python -u scripts/train.py --headless --num_envs=512 --max_iterations=200000

Algorithm: HIM-PPO — exact port of Humanoid-Goalkeeper/rsl_rl (history encoder + AMP + auxiliary tasks)
Environment: isaak_isaaclab (Isaac Sim 5.1.0 / Isaac Lab 0.54.3)
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Goalkeeper with HIM-PPO")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=200000)
parser.add_argument("--seed", type=int, default=42)
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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import goalkeeper  # noqa: F401

from goalkeeper.goalkeeper_env_cfg import GoalkeeperEnvCfg
from goalkeeper.agents.him_ppo_cfg import get_goalkeeper_him_cfg
from goalkeeper.him_ppo import HIMOnPolicyRunner, GoalkeeperHimWrapper


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    device = args_cli.device

    env_cfg = GoalkeeperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    train_cfg = get_goalkeeper_him_cfg(
        max_iterations=args_cli.max_iterations,
        seed=args_cli.seed,
    )

    log_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "him_ppo", "goalkeeper",
    )
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] Logging to: {log_dir}")

    gym_env = gym.make("Isaac-Goalkeeper-Direct-v0", cfg=env_cfg)
    env = GoalkeeperHimWrapper(gym_env, device=device)

    runner = HIMOnPolicyRunner(env, train_cfg=train_cfg, log_dir=log_dir, device=device)
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"],
                 init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
