"""Train the Goalkeeper policy with RSL-RL (OnPolicyRunner / PPO).

Usage:
    conda activate isaak_isaaclab
    python -u scripts/train.py --headless --num_envs=512 --max_iterations=200000

Note: Isaac Sim must be launched via AppLauncher BEFORE any other imports.
Environment: isaak_isaaclab (Isaac Sim 5.1.0 / Isaac Lab 0.54.3 / rsl_rl 5.0.1)
"""
import argparse
import sys
import os

# ---- Launch Isaac Sim first (must come before all other imports) ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Goalkeeper with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=200000, help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device, etc.
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Now safe to import Isaac Lab / torch ----
import torch
import gymnasium as gym
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Add parent dir so `goalkeeper` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import goalkeeper  # noqa: F401 — registers Isaac-Goalkeeper-Direct-v0

from goalkeeper.goalkeeper_env_cfg import GoalkeeperEnvCfg
from goalkeeper.agents.rsl_rl_ppo_cfg import GoalkeeperPPORunnerCfg


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # Build configs
    env_cfg = GoalkeeperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = GoalkeeperPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Logging directory
    log_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name,
    )
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] Logging to: {log_dir}")

    # Create env
    env = gym.make("Isaac-Goalkeeper-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Resume if requested
    if agent_cfg.resume:
        from isaaclab_tasks.utils import get_checkpoint_path
        ckpt = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[train] Resuming from: {ckpt}")
        runner.load(ckpt)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
