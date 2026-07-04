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
from pathlib import Path

# Load W&B API key from config file if it exists
_wandb_key_file = Path.home() / ".wandb_api_key"
if _wandb_key_file.exists():
    os.environ["WANDB_API_KEY"] = _wandb_key_file.read_text().strip()

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


# Scalar/config keys shared between the multi-disc runner's plain-dict config
# (goalkeeper_multidisc_amp_runner_cfg -> dict) and AMPRunnerCfg's dataclass
# fields. The multi-disc task's rl_cfg is intentionally a dict (its runner reads
# it via subscripting), but tyro needs a dataclass to build the `--agent.*` CLI.
# So we surface these overridable keys through an AMPRunnerCfg placeholder for
# the CLI, then fold any overrides back into the real dict at run_train time.
_MULTIDISC_AGENT_KEYS: tuple[str, ...] = (
    "num_steps_per_env",
    "max_iterations",
    "save_interval",
    "experiment_name",
    "run_name",
    "empirical_normalization",
    "use_wandb",
    "wandb_project",
    "amp_discr_hidden_dims",
    "amp_reward_coef",
    "amp_task_reward_lerp",
    "amp_min_normalized_std",
)


def _amp_runner_cfg_from_multidisc(agent_dict: dict) -> AMPRunnerCfg:
    """Build an AMPRunnerCfg carrying the multi-disc dict's overridable scalar
    fields so tyro can expose them as `--agent.*` flags. The nested policy/
    algorithm/amp_data structures live only in the dict (rebuilt in run_train)."""
    kwargs = {k: agent_dict[k] for k in _MULTIDISC_AGENT_KEYS if k in agent_dict}
    return AMPRunnerCfg(**kwargs)


def _multidisc_train_cfg(task_id: str, agent: AMPRunnerCfg) -> dict:
    """Fresh multi-disc dict config with CLI overrides (carried on `agent`)
    folded back in over the registered defaults.

    load_rl_cfg() already deepcopies the registry's stored config (see
    mjlab.tasks.registry.load_rl_cfg), so mutating the returned dict below is
    safe and does not touch the registered defaults.
    """
    train_cfg = load_rl_cfg(task_id)
    assert isinstance(train_cfg, dict)
    for key in _MULTIDISC_AGENT_KEYS:
        if key in train_cfg and hasattr(agent, key):
            train_cfg[key] = getattr(agent, key)
    return train_cfg


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
        # Multi-disc tasks register a plain dict rl_cfg (see Task 7); wrap its
        # overridable scalars in an AMPRunnerCfg so the shared CLI/log plumbing
        # below keeps working. The real dict is rebuilt in run_train.
        if isinstance(agent_cfg, dict):
            agent_cfg = _amp_runner_cfg_from_multidisc(agent_cfg)
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

    # A registered custom runner_cls (currently only the multi-disc goalkeeper
    # task) selects the HIM/multi-discriminator path; everything else keeps the
    # single-disc AMPOnPolicyRunner behavior byte-for-byte.
    runner_cls = load_runner_cls(task_id)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))

    if runner_cls is not None:
        # Multi-disc: rl_cfg is a plain dict; its RSI/motion loading does not use
        # AMPEnvWrapper.motion_dataset (verified: no `.motion_dataset` reads in
        # this project's mdp code), and its amp_data is a dict of 4 cfgs that the
        # single-dataset wrapper path cannot consume — so pass motion_dataset=None.
        env = AMPEnvWrapper(env, clip_actions=cfg.agent.clip_actions, motion_dataset=None)
        train_cfg = _multidisc_train_cfg(task_id, cfg.agent)
        dumpable = dict(train_cfg)
        dumpable["amp_data"] = {
            name: dataclasses.asdict(c) for name, c in train_cfg["amp_data"].items()
        }
        dump_yaml(log_dir / "params" / "agent.yaml", dumpable)
        runner = runner_cls(env, train_cfg, log_dir=str(log_dir), device=device)
    else:
        env = AMPEnvWrapper(
            env, clip_actions=cfg.agent.clip_actions, motion_dataset=cfg.agent.amp_data
        )
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

    # AMPRunnerCfg → single-disc AMP tasks; plain dict → the multi-disc task
    # (its rl_cfg is intentionally a dict, see Task 7 / from_task).
    amp_tasks = [
        t for t in list_tasks() if isinstance(load_rl_cfg(t), (AMPRunnerCfg, dict))
    ]
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
