from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from imitationlearningbooster.tasks.goalkeeper_env_cfg import (
    goalkeeper_env_cfg,
    goalkeeper_play_env_cfg,
    goalkeeper_play_withoverlay_env_cfg,
    goalkeeper_play_withoverlay_legacy_env_cfg,
    goalkeeper_play_withoverlay_corporate_env_cfg,
)
from imitationlearningbooster.tasks.goalkeeper_ppo_cfg import goalkeeper_ppo_runner_cfg
from imitationlearningbooster.tasks.goalkeeper_amp_env_cfg import goalkeeper_amp_env_cfg
from imitationlearningbooster.tasks.goalkeeper_amp_ppo_cfg import goalkeeper_amp_ppo_runner_cfg
from imitationlearningbooster.rsl_rl_amp.runners.him_amp_runner import GoalkeeperAmpRunner


def register_all() -> None:
    """Register all Goalkeeper tasks for Booster T1.

    Includes:
      - goalkeeper_booster_t1: Motion tracking + task reward baseline
      - goalkeeper_booster_t1_withoverlay: With motion reference overlay for analysis
      - goalkeeper_booster_t1_legacy: For legacy checkpoints without ball/hand obs
      - goalkeeper_booster_t1_corporate: For checkpoints with history_length=1
      - goalkeeper_booster_t1_amp: AMP-augmented with 6 motion discriminators

    Tasks that are already registered (e.g., from another entry point) are skipped.
    """
    from mjlab.tasks.registry import list_tasks

    existing_tasks = set(list_tasks())

    rl_cfg = goalkeeper_ppo_runner_cfg()
    num_steps = rl_cfg.num_steps_per_env

    # Base motion tracking + task task (may already be registered by my_mjlab_project_booster_t1)
    if "goalkeeper_booster_t1" not in existing_tasks:
        register_mjlab_task(
            task_id="goalkeeper_booster_t1",
            env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
            play_env_cfg=goalkeeper_play_env_cfg(num_steps_per_env=num_steps),
            rl_cfg=rl_cfg,
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    if "goalkeeper_booster_t1_withoverlay" not in existing_tasks:
        register_mjlab_task(
            task_id="goalkeeper_booster_t1_withoverlay",
            env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
            play_env_cfg=goalkeeper_play_withoverlay_env_cfg(num_steps_per_env=num_steps),
            rl_cfg=rl_cfg,
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    if "goalkeeper_booster_t1_legacy" not in existing_tasks:
        register_mjlab_task(
            task_id="goalkeeper_booster_t1_legacy",
            env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
            play_env_cfg=goalkeeper_play_withoverlay_legacy_env_cfg(num_steps_per_env=num_steps),
            rl_cfg=rl_cfg,
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    if "goalkeeper_booster_t1_corporate" not in existing_tasks:
        register_mjlab_task(
            task_id="goalkeeper_booster_t1_corporate",
            env_cfg=goalkeeper_env_cfg(num_steps_per_env=num_steps),
            play_env_cfg=goalkeeper_play_withoverlay_corporate_env_cfg(num_steps_per_env=num_steps),
            rl_cfg=rl_cfg,
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    # AMP-augmented Goalkeeper: 6 motion discriminators + task rewards
    # This is new and should always be registered
    if "goalkeeper_booster_t1_amp" not in existing_tasks:
        amp_rl_cfg = goalkeeper_amp_ppo_runner_cfg()
        amp_num_steps = amp_rl_cfg.num_steps_per_env
        register_mjlab_task(
            task_id="goalkeeper_booster_t1_amp",
            env_cfg=goalkeeper_amp_env_cfg(num_steps_per_env=amp_num_steps),
            play_env_cfg=goalkeeper_amp_env_cfg(play=True, num_steps_per_env=amp_num_steps),
            rl_cfg=amp_rl_cfg,
            runner_cls=GoalkeeperAmpRunner,
        )
