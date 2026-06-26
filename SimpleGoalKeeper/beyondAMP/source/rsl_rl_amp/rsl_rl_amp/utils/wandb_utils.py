from __future__ import annotations

import os
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb is required. Install with: pip install wandb")


class WandbSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg: dict):
        super().__init__(log_dir, flush_secs)

        run_name = os.path.basename(log_dir)
        wandb.init(
            project=cfg.get("wandb_project", "SimpleGoalKeeper"),
            entity=cfg.get("wandb_entity"),
            name=run_name,
            group=cfg.get("experiment_name", "exp"),
            tags=[cfg.get("run_name", "")],
            config={
                "algorithm": cfg.get("algorithm"),
                "policy": cfg.get("policy"),
                "amp_reward_coef": cfg.get("amp_reward_coef"),
                "amp_task_reward_lerp": cfg.get("amp_task_reward_lerp"),
                "amp_discr_hidden_dims": cfg.get("amp_discr_hidden_dims"),
                "num_steps_per_env": cfg.get("num_steps_per_env"),
                "max_iterations": cfg.get("max_iterations"),
            },
            dir=os.path.dirname(log_dir),
        )

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        try:
            wandb.log({tag: scalar_value}, step=global_step)
        except Exception as e:
            print(f"[WARN] W&B logging failed for {tag}: {e}")

    def stop(self):
        wandb.finish()
