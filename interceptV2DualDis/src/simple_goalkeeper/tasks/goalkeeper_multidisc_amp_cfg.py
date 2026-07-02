"""Env config for the 4-region, multi-discriminator AMP variant of the
goalkeeper task. Starts from goalkeeper_env_cfg() and layers on: a genuine
single-step actor_current observation group (history_length=0), ball/region
ground-truth critic obs terms, and region-conditioned ball spawn + static
region assignment events. See docs/superpowers/plans/2026-07-02-multi-
discriminator-amp-implementation-plan.md.
"""
from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from simple_goalkeeper.mdp import regions as gk_regions
from simple_goalkeeper.tasks.goalkeeper_env_cfg import BALL_NAME, goalkeeper_env_cfg


def goalkeeper_multidisc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Multi-discriminator goalkeeper env cfg: goalkeeper_env_cfg() + region
    ground truth, static region assignment, and region-conditioned ball spawn.
    """
    cfg = goalkeeper_env_cfg(play=play)

    # (a) Single-step actor observation group. The base goalkeeper_env_cfg()
    # already sets cfg.observations["actor"].history_length = 10, so "actor"
    # is the 10-step history source; there is otherwise no group anywhere in
    # this config providing a genuine single-step observation, which
    # HimActorCritic.act(obs_current, obs_history) needs. Build fresh
    # ObservationTermCfg clones (not references to "actor"'s term objects,
    # which are already bound to the group's history_length override) via
    # dataclasses.replace(..., history_length=0) so this group's terms are
    # decoupled from the history-stacked ones.
    current_terms = {
        name: dataclasses.replace(term_cfg, history_length=0)
        for name, term_cfg in cfg.observations["actor"].terms.items()
    }
    cfg.observations["actor_current"] = ObservationGroupCfg(
        terms=current_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )

    # (b) Ball/region ground-truth terms, critic-only (privileged).
    cfg.observations["critic"].terms["ball_gt"] = ObservationTermCfg(
        func=gk_regions.ball_state_gt,
        params={"ball_name": BALL_NAME},
    )
    cfg.observations["critic"].terms["region_gt"] = ObservationTermCfg(
        func=gk_regions.region_id_gt,
        params={},
    )

    # (c) Static region assignment, once at startup.
    cfg.events["assign_static_regions"] = EventTermCfg(
        func=gk_regions.assign_static_regions,
        mode="startup",
        params={},
    )

    # (d) Replace the shared-range ball spawn with the region-conditioned one.
    # Reuses the existing reset_ball event's ball_name/dist_range/
    # t_flight_range/spawn_z; drops y_start_range/y_end_range since those are
    # now resolved per-region inside reset_ball_rolling_by_region.
    existing = cfg.events["reset_ball"].params
    cfg.events["reset_ball"] = EventTermCfg(
        func=gk_regions.reset_ball_rolling_by_region,
        mode="reset",
        params={
            "ball_name": existing["ball_name"],
            "dist_range": existing["dist_range"],
            "t_flight_range": existing["t_flight_range"],
            "spawn_z": existing["spawn_z"],
        },
    )

    return cfg


from pathlib import Path

from beyondAMP.motion.motion_dataset import MotionDatasetCfg
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import (
    GOALKEEPER_ANCHOR_NAME,
    GOALKEEPER_KEY_BODY_NAMES,
)
from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"

REGION_MOTION_FILES: dict[str, str] = {
    "left_near": str(_MOTIONS_DIR / "LeftStep_own_booster_t1.npz"),
    "left_far": str(_MOTIONS_DIR / "LeftDoubleStep_own_booster_t1.npz"),
    "right_near": str(_MOTIONS_DIR / "Rightstep_own_booster_t1.npz"),
    "right_far": str(_MOTIONS_DIR / "RightDoubleStep_own_booster_t1.npz"),
}


def goalkeeper_multidisc_amp_runner_cfg() -> dict:
    amp_data = {
        name: MotionDatasetCfg(
            motion_files=[path],
            body_names=GOALKEEPER_KEY_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name=GOALKEEPER_ANCHOR_NAME,
        )
        for name, path in REGION_MOTION_FILES.items()
    }
    return {
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "amp_replay_buffer_size": 250_000,
        },
        "amp_data": amp_data,
        "num_steps_per_env": 24,
        "max_iterations": 50_000,
        "save_interval": 250,
        "experiment_name": "simple_goalkeeper_multidisc",
        "run_name": "phase1",
        "empirical_normalization": True,
        "use_wandb": True,
        "wandb_project": "SimpleGoalKeeper-MultiDisc",
        "amp_discr_hidden_dims": [256, 256],
        "amp_reward_coef": 0.5,
        "amp_task_reward_lerp": 0.6,
        "amp_min_normalized_std": 0.05,
    }
