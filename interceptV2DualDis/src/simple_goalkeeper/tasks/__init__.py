"""Register SimpleGoalKeeper tasks.

Train:
    uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096

Play (zero policy):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1

Play with ghost overlay (trained checkpoint):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt

Play with specific motion file:
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --checkpoint-file <ckpt> --motion-file src/simple_goalkeeper/motions/data/1-1_booster_t1.npz
"""
from mjlab.tasks.registry import register_mjlab_task

from .goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg
from .goalkeeper_env_cfg import goalkeeper_env_cfg, goalkeeper_env_cfg_withoverlay
from .goalkeeper_multidisc_amp_cfg import (
    goalkeeper_multidisc_amp_runner_cfg,
    goalkeeper_multidisc_env_cfg,
)
from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
    HimAMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-BeyondAMP-Goalkeeper-T1",
    env_cfg=goalkeeper_env_cfg(),
    play_env_cfg=goalkeeper_env_cfg(play=True),
    rl_cfg=goalkeeper_amp_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay",
    env_cfg=goalkeeper_env_cfg(),
    play_env_cfg=goalkeeper_env_cfg_withoverlay(),
    rl_cfg=goalkeeper_amp_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc",
    env_cfg=goalkeeper_multidisc_env_cfg(),
    play_env_cfg=goalkeeper_multidisc_env_cfg(play=True),
    rl_cfg=goalkeeper_multidisc_amp_runner_cfg(),
    runner_cls=HimAMPOnPolicyRunner,
)
