"""Play a trained SimpleGoalKeeperHim policy (HIM-PPO, 2-disc AMP, 21-DOF).

Usage:
    # Zero-action sanity check (1 env, headless):
    uv run sgk_him_play simple_goalkeeper_him --agent zero --num-envs 1

    # Trained checkpoint:
    uv run sgk_him_play simple_goalkeeper_him \\
        --checkpoint-file logs/rsl_rl/simple_goalkeeper_him/<run>/model_500.pt
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from simple_goalkeeper_him.rsl_rl_amp.runners.him_amp_runner import GoalkeeperAmpRunner
from simple_goalkeeper_him.tasks.goalkeeper_amp_ppo_cfg import goalkeeper_amp_ppo_runner_cfg


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper_him  # noqa: F401

    import mjlab

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg = tyro.cli(
        PlayConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()
    os.environ["MUJOCO_GL"] = "egl"
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(chosen_task, play=True)
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    rl_cfg = goalkeeper_amp_ppo_runner_cfg()
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner = GoalkeeperAmpRunner(env, asdict(rl_cfg), log_dir=None, device=device)

    if cfg.agent == "trained":
        if cfg.checkpoint_file is None:
            raise ValueError("Provide --checkpoint-file for trained agent.")
        print(f"[sgk_him_play] loading checkpoint: {cfg.checkpoint_file}")
        runner.load(cfg.checkpoint_file)

    print(f"[sgk_him_play] task={chosen_task}  agent={cfg.agent}  device={device}")
    runner.play()


if __name__ == "__main__":
    main()
