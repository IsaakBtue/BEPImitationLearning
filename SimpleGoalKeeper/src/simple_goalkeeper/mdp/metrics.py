"""Diagnostic metrics for the blue-ball landing gate (2026-07-04).

Not rewards -- no weight, no dt scaling (mjlab.managers.metrics_manager:
"Unlike rewards, metrics have no weight, no dt scaling, and no normalization
by episode length"). Distinguishes a landing the policy caused from one the
RSI reset pose handed it for free, since the 2026-07-04 hard-gate + RSI-
rebalance design bundles three changes into one training run and needs a way
to tell their effects apart afterward. See
docs/superpowers/specs/2026-07-04-blue-ball-hard-gate-rsi-rebalance-design.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import _DEFAULT_FEET_CFG, _get_reach_target_y

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def blue_landed_genuine(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """1.0 while env._blue_landed is true and it was NOT an RSI-assisted
    landing (env._blue_airborne_at_reset is false), else 0.0.

    Wired with reduce="last" in MetricsTermCfg (goalkeeper_env_cfg.py), so the
    logged Episode_Metrics value is this exact per-episode 0/1 outcome, not a
    mean over the episode. See _get_reach_target_y for both latches.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure latches are fresh
    return (env._blue_landed & ~env._blue_airborne_at_reset).float()


def blue_landed_rsi_assisted(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """1.0 while env._blue_landed is true and the assigned foot's first
    airborne transition happened within 2 steps of reset (env.
    _blue_airborne_at_reset), else 0.0. Mutually exclusive with
    blue_landed_genuine given _blue_landed true; both zero when _blue_landed
    is false. See blue_landed_genuine.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)
    return (env._blue_landed & env._blue_airborne_at_reset).float()
