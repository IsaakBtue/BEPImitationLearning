from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from my_mjlab_project.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg, goalkeeper_play_env_cfg
from my_mjlab_project.tasks.goalkeeper_ppo_cfg import goalkeeper_ppo_runner_cfg


def register_all() -> None:
    register_mjlab_task(
        task_id="goalkeeper",
        env_cfg=goalkeeper_env_cfg(),
        play_env_cfg=goalkeeper_play_env_cfg(),
        rl_cfg=goalkeeper_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
