"""One-off continuation of run 2026-07-05_12-16-46_intercept_phase1 from
model_5500.pt (the last known-good checkpoint), with region_estimator split
into its own optimizer param group (see CLAUDE.md's "Multi-disc PPO
schedule" divergence row and docs/BugFixes.md's 2026-07-05 entries for the
full history).

Attempt 1 (load_optimizer=False, shared lr=1.0e-3 fixed, resumed IN PLACE
into the original run directory with current_learning_iteration=5750):
blew the policy up within ~5 iterations (reward to -1e8, episode length
~12). Worse: since 5750 is an exact multiple of save_interval=250, the
very first loop iteration's periodic-save check fired immediately and
OVERWROTE model_5750.pt with this already-blown-up post-update state --
renamed on disk to model_5750.pt.CORRUPTED-by-resume-attempt-do-not-use.
model_5500.pt (250 iterations earlier) and model_5250.pt (pushed to git,
commit 666f2f5) were saved before that attempt started and are unaffected.

Attempt 2 (load_optimizer=True, shared lr=2.0e-4 fixed, ~2.5x bump): no
catastrophic blowup, but episode length stuck around ~22 steps and never
recovered over 200 iterations -- even a modest bump on the SHARED group
was enough to knock the converged policy off its optimum.

Attempt 3: same design as this file, but unknowingly resumed from the
already-corrupted model_5750.pt (main lr baked in at 1.0e-3 from attempt
1's crash) -- blew up again for the same reason as attempt 1, not because
of anything wrong with the split-param-group design itself.

This attempt (4th): resumes from the verified-clean model_5500.pt, into a
SEPARATE run directory (RUN_DIR_CONTINUED, not the original) so there is
no possibility of iteration-number collision with any existing checkpoint
ever again. region_estimator trains through its own optimizer param group
(multi_disc_amp_ppo.py), fully decoupled from actor/critic/history_encoder/
ball_estimator: main group gets the lr read directly from model_5500.pt's
own saved optimizer_state_dict (zero intended change to the already-good
policy), region_estimator gets a much higher, independent lr (config
default 3.0e-3). load_optimizer=False because the param-group structure
changed (old checkpoint has one combined "actor_critic" group; new code
has two), so position-based state_dict restore isn't valid -- a fresh Adam
optimizer at the settled main lr should be safe given how small that lr
already is.

Usage:
    cd /home/ibouwmeest/BEPImitationLearning/interceptV2DualDis
    uv run python resume_region_fix.py
"""
from __future__ import annotations

import os
from pathlib import Path

_wandb_key_file = Path.home() / ".wandb_api_key"
if _wandb_key_file.exists():
    os.environ["WANDB_API_KEY"] = _wandb_key_file.read_text().strip()

import dataclasses

import torch

import mjlab.tasks  # noqa: F401
import simple_goalkeeper.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper

TASK = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
ORIGINAL_RUN_DIR = Path(
    "logs/rsl_rl/intercept_simple_goalkeeper_multidisc/"
    "2026-07-05_12-16-46_intercept_phase1"
)
CHECKPOINT = ORIGINAL_RUN_DIR / "model_5500.pt"
RESUME_ITERATION = 5500
NUM_ENVS = 4096

# Deliberately a different directory from ORIGINAL_RUN_DIR -- guarantees no
# filename can ever collide with an existing checkpoint, regardless of what
# RESUME_ITERATION or save_interval end up being.
RUN_DIR = Path(
    "logs/rsl_rl/intercept_simple_goalkeeper_multidisc/"
    "2026-07-05_12-16-46_intercept_phase1_continued_region_fix"
)
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "params").mkdir(exist_ok=True)

configure_torch_backends()
device = "cuda:0"
os.environ["MUJOCO_GL"] = "egl"

assert CHECKPOINT.exists(), f"missing {CHECKPOINT}"

# Read the settled main-group lr straight from the checkpoint rather than
# hardcoding a value -- avoids a silent mismatch if it differs from what was
# reported earlier, and would have caught attempt 3's mistake immediately
# (it would have printed 0.001 from the corrupted file instead of ~7.6e-5).
_ckpt_probe = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
_settled_main_lr = _ckpt_probe["optimizer_state_dict"]["param_groups"][0]["lr"]
_ckpt_saved_iter = _ckpt_probe["iter"]
del _ckpt_probe
print(f"[INFO] checkpoint: {CHECKPOINT}")
print(f"[INFO] settled main-group lr from checkpoint: {_settled_main_lr}")
print(f"[INFO] checkpoint's own saved iter field: {_ckpt_saved_iter} (expect 0 -- pre-existing bug, harmless here since we set RESUME_ITERATION explicitly)")
assert _settled_main_lr < 1e-3, (
    f"settled_main_lr={_settled_main_lr} is not below the base 1.0e-3 config "
    "value -- this checkpoint may be corrupted (see attempt 3). Aborting."
)

env_cfg = load_env_cfg(TASK)
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.seed = 42

train_cfg = load_rl_cfg(TASK)
assert isinstance(train_cfg, dict)
train_cfg["algorithm"]["learning_rate"] = _settled_main_lr
print(f"[INFO] schedule = {train_cfg['algorithm']['schedule']!r} (expect 'fixed')")
print(f"[INFO] main learning_rate = {train_cfg['algorithm']['learning_rate']}")
print(f"[INFO] region_estimator_learning_rate = {train_cfg['algorithm']['region_estimator_learning_rate']}")

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

dump_yaml(RUN_DIR / "params" / "env_resume.yaml", dataclasses.asdict(env_cfg))
dumpable = dict(train_cfg)
dumpable["amp_data"] = {
    name: dataclasses.asdict(c) for name, c in train_cfg["amp_data"].items()
}
dump_yaml(RUN_DIR / "params" / "agent_resume.yaml", dumpable)

runner_cls = load_runner_cls(TASK)
runner = runner_cls(env, train_cfg, log_dir=str(RUN_DIR), device=device)

print(f"[INFO] Loading checkpoint: {CHECKPOINT}")
runner.load(str(CHECKPOINT), load_optimizer=False)
runner.current_learning_iteration = RESUME_ITERATION
print(f"[INFO] Resuming from iteration {runner.current_learning_iteration}, saving into {RUN_DIR}")
for pg in runner.alg.optimizer.param_groups:
    print(f"  optimizer group {pg.get('name')}: lr={pg['lr']}")

remaining_iterations = train_cfg["max_iterations"] - RESUME_ITERATION
print(f"[INFO] Training {remaining_iterations} more iterations "
      f"(target max_iterations={train_cfg['max_iterations']})")

runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)
env.close()
