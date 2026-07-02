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
