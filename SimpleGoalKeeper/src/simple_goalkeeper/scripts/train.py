"""Train the beyondAMP goalkeeper policy on Booster T1 (mjlab backend).

Usage:
    # Single GPU:
    uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096

    # Multi-GPU (torchrunx):
    TORCHRUNX_HOSTS=localhost uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096 --gpu-ids all

    # Resume from checkpoint:
    uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 \\
        --agent.resume --agent.load-run <run-name> --agent.load-checkpoint best

    # Override motion files at runtime:
    uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 \\
        "--agent.amp-data.motion-files=[src/simple_goalkeeper/motions/data/1-1_booster_t1.npz]"
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: AMPRunnerCfg
    num_envs: int | None = None
    log_root: str = "logs/rsl_rl"
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, AMPRunnerCfg), (
            f"Task '{task_id}' is not an AMP task — got {type(agent_cfg).__name__}."
        )
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else f"cuda:{local_rank}"

    # Each rank writes to its own subdirectory to avoid checkpoint conflicts.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        log_dir = log_dir / f"rank_{local_rank}"

    configure_torch_backends()
    if cfg.num_envs is not None:
        cfg.env.scene.num_envs = cfg.num_envs
    cfg.env.seed = cfg.agent.seed

    print(f"[INFO] Training beyondAMP goalkeeper: task={task_id} device={device}")
    print(f"[INFO] Logging to: {log_dir}")

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    env = AMPEnvWrapper(env, clip_actions=cfg.agent.clip_actions, motion_dataset=cfg.agent.amp_data)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    runner = AMPOnPolicyRunner(env, asdict(cfg.agent), log_dir=str(log_dir), device=device)

    if cfg.agent.resume:
        log_root_path = log_dir.parent
        resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def main() -> None:
    maybe_print_top_level_help("train")

    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper.tasks  # noqa: F401 — registers goalkeeper task

    import mjlab

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]
    if not amp_tasks:
        raise RuntimeError("No AMP tasks registered.")

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(amp_tasks),
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

    if "TORCHRUNX_HOSTS" in os.environ:
        import torchrunx
        hosts = os.environ["TORCHRUNX_HOSTS"].split(",")
        num_gpus = len(selected_gpus) if selected_gpus else 1
        torchrunx.Launcher(
            hostnames=hosts,
            workers_per_host=num_gpus,
        ).run(run_train, chosen_task, args, log_dir)
    else:
        run_train(chosen_task, args, log_dir)


if __name__ == "__main__":
    main()
