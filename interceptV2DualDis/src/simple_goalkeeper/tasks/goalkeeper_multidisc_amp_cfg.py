"""Env config for the 4-region, multi-discriminator AMP variant of the
goalkeeper task. Starts from goalkeeper_env_cfg() and layers on: a genuine
single-step actor_current observation group (history_length=0), ball/region
ground-truth critic obs terms, and region-conditioned ball spawn + static
region assignment events. See docs/superpowers/plans/2026-07-02-multi-
discriminator-amp.md and docs/superpowers/specs/2026-07-02-multi-
discriminator-amp-design.md.
"""
from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from simple_goalkeeper.mdp import regions as gk_regions
from simple_goalkeeper.mdp.events import track_blue_landing_success
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

    # (b2) FIX 2026-07-08: G1's AMP discriminator observes joint POSITIONS
    # ONLY (Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py:
    # get_amp_observations() returns self.dof_pos.clone(), nothing else --
    # confirmed by an independent verification pass after an initial AMP-
    # parity fix this same day was found to have missed this specific
    # divergence). The base goalkeeper_env_cfg()'s shared "amp" group (used by
    # SimpleGoalKeeper's single-disc track too -- not modified here) includes
    # joint_vel alongside joint_pos, a real content divergence from G1, not an
    # equivalent restatement. Overriding just this task's "amp" group to
    # match G1 exactly; MotionDatasetCfg's amp_obs_terms is overridden to
    # match below (goalkeeper_multidisc_amp_runner_cfg), keeping expert/policy
    # obs dimensions in sync. See docs/BugFixes.md.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={"joint_pos": cfg.observations["amp"].terms["joint_pos"]},
        concatenate_terms=True,
        enable_corruption=False,
    )

    # (c) Region assignment. Training keeps the permanent per-env split (the
    # region_estimator is trained assuming a stable, balanced ground-truth
    # distribution across the parallel batch). Play re-randomizes on every
    # reset instead -- assign_static_regions degenerates at num_envs=1 (the
    # play default), permanently pinning the single env to one region for the
    # whole session; randomize_region_on_reset lets a single agent's episodes
    # cycle through all 4 regions over time. FIX 2026-07-07, docs/BugFixes.md.
    #
    # Must execute before "reset_ball" (reset_ball_rolling_by_region reads
    # env._region_id) and "reset_ball" must in turn stay before
    # "reset_from_motion_data" (its tier-routing depends on reset_ball's
    # freshly-computed trajectory, an existing convention from goalkeeper_
    # env_cfg). For mode="reset" events, mjlab's EventManager executes terms
    # in dict/registration order, not by key. "reset_ball" (and, if present,
    # "reset_from_motion_data"/"tick_catchstep") already exist in cfg.events
    # from the base goalkeeper_env_cfg() call above, inserted *before* this
    # function adds "assign_static_regions" -- a plain `cfg.events[key] = ...`
    # reassignment updates the value in place but does NOT move an existing
    # key's position, so simply reassigning "reset_ball" below would leave it
    # in its too-early original slot. Pop the order-sensitive keys first so
    # they get fresh (later) insertion positions once re-added, in the
    # correct relative order.
    reset_ball_cfg = cfg.events.pop("reset_ball")
    reset_from_motion_data_cfg = cfg.events.pop("reset_from_motion_data", None)
    tick_catchstep_cfg = cfg.events.pop("tick_catchstep", None)

    if play:
        cfg.events["assign_static_regions"] = EventTermCfg(
            func=gk_regions.randomize_region_on_reset,
            mode="reset",
            params={},
        )
    else:
        cfg.events["assign_static_regions"] = EventTermCfg(
            func=gk_regions.assign_static_regions,
            mode="startup",
            params={},
        )

    # (d) Replace the shared-range ball spawn with the region-conditioned one.
    # Reuses the existing reset_ball event's ball_name/dist_range/
    # t_flight_range/spawn_z; drops y_start_range/y_end_range since those are
    # now resolved per-region inside reset_ball_rolling_by_region.
    existing = reset_ball_cfg.params
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
    if reset_from_motion_data_cfg is not None:
        cfg.events["reset_from_motion_data"] = reset_from_motion_data_cfg
    if tick_catchstep_cfg is not None:
        cfg.events["tick_catchstep"] = tick_catchstep_cfg

    # FEAT 2026-07-11: rolling blue-landing success rate, read by
    # stopball/softstop to scale wide-crossing payoff -- see
    # track_blue_landing_success's docstring (mdp/events.py) and
    # docs/BugFixes.md. No ordering dependency on the events popped/
    # re-added above (only reads env._blue_wide/_blue_landed, set by reward
    # computation on the outgoing episode's last step, not by any other
    # reset event).
    if not play:
        cfg.events["track_blue_landing_success"] = EventTermCfg(
            func=track_blue_landing_success,
            mode="reset",
            params={},
        )

    return cfg


from pathlib import Path

from beyondAMP.motion.motion_dataset import MotionDatasetCfg
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import (
    GOALKEEPER_ANCHOR_NAME,
    GOALKEEPER_KEY_BODY_NAMES,
)
# FIX 2026-07-08: G1's AMP discriminator observes joint positions only (see
# the "amp" observation group override above) -- a task-local override, NOT
# a change to the shared AMPObsBaiscTerms constant (["joint_pos","joint_vel"]),
# which SimpleGoalKeeper's single-disc AMP track (goalkeeper_amp_cfg.py) and
# several other beyondAMP example tasks still use unmodified.
_MULTIDISC_AMP_OBS_TERMS: list[str] = ["joint_pos"]

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"

REGION_MOTION_FILES: dict[str, str] = {
    "left_near": str(_MOTIONS_DIR / "LeftStep_own_booster_t1.npz"),
    "left_far": str(_MOTIONS_DIR / "LeftDoubleStep_own_booster_t1.npz"),
    "right_near": str(_MOTIONS_DIR / "Rightstep_own_booster_t1.npz"),
    "right_far": str(_MOTIONS_DIR / "RightDoubleStep_own_booster_t1.npz"),
}


def goalkeeper_multidisc_env_cfg_withoverlay(
    motion_file: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Play-mode multi-disc config with ghost-robot overlay.

    Mirrors goalkeeper_env_cfg.goalkeeper_env_cfg_withoverlay(), but cycles
    through this task's own 4-file AMP dataset (REGION_MOTION_FILES) instead
    of SimpleGoalKeeper's full 14-file motions/data/ set -- those files are
    what this track's 4 discriminators were actually trained against.
    """
    from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
    from simple_goalkeeper.mdp.commands import (
        CyclingGhostMotionCommandCfg,
        GhostMotionCommandCfg,
    )
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import _T1_HEADLESS_BODY_NAMES

    cfg = goalkeeper_multidisc_env_cfg(play=True)
    cfg.scene.num_envs = 1

    if motion_file is not None:
        cfg.commands["motion_ghost"] = GhostMotionCommandCfg(
            motion_file=motion_file,
            anchor_body_name="Trunk",
            body_names=_T1_HEADLESS_BODY_NAMES,
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
        )
    else:
        npz_files = list(REGION_MOTION_FILES.values())
        cmd = CyclingGhostMotionCommandCfg(
            motion_file=npz_files[0],  # required by parent cfg; overridden at build
            anchor_body_name="Trunk",
            body_names=_T1_HEADLESS_BODY_NAMES,
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
        )
        cmd.motion_files = npz_files
        cfg.commands["motion_ghost"] = cmd

    return cfg


def goalkeeper_multidisc_amp_runner_cfg() -> dict:
    amp_data = {
        name: MotionDatasetCfg(
            motion_files=[path],
            body_names=GOALKEEPER_KEY_BODY_NAMES,
            amp_obs_terms=_MULTIDISC_AMP_OBS_TERMS,
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
            # 2026-07-06: reverted to "adaptive" to match G1's actual,
            # effective config (g1_29_config.py inherits schedule="adaptive"
            # from legged_robot_config.py:326, unmodified) -- the prior switch
            # to "fixed" here was based on the false premise that G1 uses a
            # constant LR. region_estimator shares this single schedule/LR
            # with the rest of actor_critic, also matching G1
            # (him_ppo.py:101-116, no separate group). See docs/BugFixes.md.
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
        # 2026-07-05: "intercept_" prefix on both experiment_name (top-level
        # local log folder + W&B group) and run_name (per-run folder suffix +
        # W&B run name/tags) so intercept runs are clearly labeled everywhere
        # while sharing the same W&B project as SimpleGoalKeeper itself
        # (wandb_project below), rather than a separate "-MultiDisc" project.
        "experiment_name": "intercept_simple_goalkeeper_multidisc",
        "run_name": "intercept_phase1",
        "empirical_normalization": True,
        "use_wandb": True,
        "wandb_project": "SimpleGoalKeeper",
        # FIX 2026-07-08: match G1's discriminator width exactly
        # (Humanoid-Goalkeeper/rsl_rl/rsl_rl/modules/amp.py:87, AMP.__init__'s
        # hidden_dims=[512, 256] default, which G1 never overrides). Was
        # [256, 256] -- half the trunk capacity, an undocumented divergence.
        "amp_discr_hidden_dims": [512, 256],
        "amp_reward_coef": 0.5,
        "amp_task_reward_lerp": 0.6,
        # NOT a G1-equivalent AMP parameter despite the name -- this is the
        # actor's action-noise std floor (min_std), consumed by
        # MultiDiscAMPPPO/HimAmpOnPolicyRunner, unrelated to the AMP
        # discriminator/reward mechanism G1 uses. G1 has no equivalent; left
        # unchanged as an intentional, already-documented, non-AMP addition.
        "amp_min_normalized_std": 0.05,
    }
