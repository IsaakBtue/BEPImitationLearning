from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from my_mjlab_project_booster_t1.tasks.goalkeeper_env_cfg import (
    goalkeeper_env_cfg,
    goalkeeper_play_env_cfg,
    goalkeeper_play_withoverlay_env_cfg,
)
from my_mjlab_project_booster_t1.tasks.goalkeeper_ppo_cfg import goalkeeper_ppo_runner_cfg


def register_all() -> None:
    rl_cfg = goalkeeper_ppo_runner_cfg()
    num_steps = rl_cfg.num_steps_per_env

    register_mjlab_task(
        task_id="goalkeeper_booster_t1",
        env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
        play_env_cfg=goalkeeper_play_env_cfg(num_steps_per_env=num_steps),
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )
    register_mjlab_task(
        task_id="goalkeeper_booster_t1_withoverlay",
        env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
        play_env_cfg=goalkeeper_play_withoverlay_env_cfg(num_steps_per_env=num_steps),
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )
