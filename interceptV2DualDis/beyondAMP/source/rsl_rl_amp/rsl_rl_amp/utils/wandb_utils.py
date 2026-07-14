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
        # Buffer all add_scalar calls for the same step and flush them in a
        # single wandb.log() call.  Separate wandb.log() calls for the same
        # iteration would auto-increment WandB's internal step counter, making
        # metrics for the same training iteration appear at different x positions
        # in the WandB UI when the default "Step" axis is used.
        self._wandb_buffer: dict = {}
        self._wandb_step: int | None = None
        # him_amp_on_policy_runner.py logs some scalars keyed by training
        # iteration (int) and some ("*/time" series) keyed by self.tot_time
        # (float, wall-clock seconds) via the same add_scalar/global_step
        # path. wandb.log requires an int step, so tracking the last INT step
        # separately (instead of reusing whatever global_step this call used)
        # keeps every flush wandb-valid regardless of which series triggered
        # it -- TensorBoard is unaffected since it uses global_step directly.
        self._last_int_step: int | None = None

    def _flush_wandb(self) -> None:
        if self._wandb_buffer:
            try:
                wandb.log(self._wandb_buffer, step=self._last_int_step)
            except Exception as e:
                print(f"[WARN] W&B logging failed: {e}")
            self._wandb_buffer = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        if isinstance(global_step, int):
            self._last_int_step = global_step
        if global_step != self._wandb_step:
            self._flush_wandb()
            self._wandb_step = global_step
        self._wandb_buffer[tag] = scalar_value

    def stop(self):
        self._flush_wandb()
        wandb.finish()
