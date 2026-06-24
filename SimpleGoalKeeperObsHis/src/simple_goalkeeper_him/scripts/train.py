"""Train the SimpleGoalKeeperHim policy (HIM-PPO, 2-disc AMP, 21-DOF).

Usage:
    # Single GPU:
    uv run sgk_him_train simple_goalkeeper_him --num-envs 4096

    # Resume from checkpoint:
    uv run sgk_him_train simple_goalkeeper_him \\
        --agent.resume --agent.load-run <run-name> --agent.load-checkpoint best
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from simple_goalkeeper_him.rsl_rl_amp.runners.him_amp_runner import GoalkeeperAmpRunner


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    num_envs: int | None = None
    log_root: str = "logs/rsl_rl"
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else f"cuda:{local_rank}"

    configure_torch_backends()
    if cfg.num_envs is not None:
        cfg.env.scene.num_envs = cfg.num_envs
    cfg.env.seed = cfg.agent.seed

    print(f"[sgk_him_train] task={task_id}  device={device}  envs={cfg.env.scene.num_envs}")
    print(f"[sgk_him_train] log_dir={log_dir}")

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner = GoalkeeperAmpRunner(env, agent_cfg, log_dir=str(log_dir), device=device)

    if cfg.agent.resume:
        log_root_path = log_dir.parent
        resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
        print(f"[sgk_him_train] loading checkpoint: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper_him  # noqa: F401 — registers simple_goalkeeper_him task

    import mjlab

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, _ = select_gpus(args.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    run_train(chosen_task, args, log_dir)


if __name__ == "__main__":
    main()
