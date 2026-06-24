from mjlab.tasks.registry import register_mjlab_task

from simple_goalkeeper_him.tasks.goalkeeper_env_cfg import (
    goalkeeper_env_cfg,
    goalkeeper_env_cfg_withoverlay,
)
from simple_goalkeeper_him.tasks.goalkeeper_amp_ppo_cfg import goalkeeper_amp_ppo_runner_cfg
from simple_goalkeeper_him.rsl_rl_amp.runners.him_amp_runner import GoalkeeperAmpRunner


def register_all() -> None:
    """Register simple_goalkeeper_him tasks (HIM-PPO, 2-disc AMP, 21-DOF feet-only)."""
    from mjlab.tasks.registry import list_tasks

    existing_tasks = set(list_tasks())
    rl_cfg = goalkeeper_amp_ppo_runner_cfg()

    if "simple_goalkeeper_him" not in existing_tasks:
        register_mjlab_task(
            task_id="simple_goalkeeper_him",
            env_cfg=goalkeeper_env_cfg(play=False),
            play_env_cfg=goalkeeper_env_cfg(play=True),
            rl_cfg=rl_cfg,
            runner_cls=GoalkeeperAmpRunner,
        )

    if "simple_goalkeeper_him_withoverlay" not in existing_tasks:
        register_mjlab_task(
            task_id="simple_goalkeeper_him_withoverlay",
            env_cfg=goalkeeper_env_cfg(play=False),
            play_env_cfg=goalkeeper_env_cfg_withoverlay(),
            rl_cfg=rl_cfg,
            runner_cls=GoalkeeperAmpRunner,
        )
