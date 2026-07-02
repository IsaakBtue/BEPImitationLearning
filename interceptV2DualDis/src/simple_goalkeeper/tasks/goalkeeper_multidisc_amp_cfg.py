"""Env config for the 4-region, multi-discriminator AMP variant of the
goalkeeper task. Starts from goalkeeper_env_cfg() and layers on: the
actor_history observation group (history_length=10), ball/region
ground-truth critic obs terms, and region-conditioned ball spawn + static
region assignment events. See docs/superpowers/plans/2026-07-02-multi-
discriminator-amp-implementation-plan.md.
"""
from __future__ import annotations

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

    # (a) History-stacked actor observation group — same terms dict as
    # "actor" (ObservationManager deep-copies its whole cfg + every term_cfg
    # on construction, so two groups sharing one terms dict is safe — see
    # mjlab.managers.observation_manager.ObservationManager.__init__/
    # _prepare_terms), 10-step history, term-major-flattened.
    cfg.observations["actor_history"] = ObservationGroupCfg(
        terms=cfg.observations["actor"].terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=10,
        flatten_history_dim=True,
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
