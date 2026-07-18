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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from simple_goalkeeper.mdp import events as gk_mdp
from simple_goalkeeper.mdp import observations as gk_obs
from simple_goalkeeper.mdp import regions as gk_regions
from simple_goalkeeper.tasks.goalkeeper_env_cfg import BALL_NAME, goalkeeper_env_cfg

# FIX 2026-07-15: the 13 non-arm T1 joints (Waist + 6 hip + 2 knee + 4 ankle),
# excluding the 8 arm joints (4 per side: Shoulder_Pitch/Roll, Elbow_Pitch/Yaw).
# Arms have no task-grounding reward (no equivalent of G1's hand-reach eereach
# for this feet-only design -- see docs/BugFixes.md, 2026-07-10 entry).
# NOT currently used for the "amp" observation group override (see FIX
# 2026-07-16 there) -- kept available since it's still a generically useful
# joint subset.
NON_ARM_JOINT_NAMES: tuple[str, ...] = (
    "Waist",
    "Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Right_Hip_Roll", "Right_Hip_Yaw",
    "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll",
)
ARM_JOINT_NAMES: tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
)


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
    #
    # FIX 2026-07-15: additionally restricted to NON_ARM_JOINT_NAMES (13 of
    # 21 joints) for ALL regions -- made arm swinging WORSE per user report
    # (zero AMP constraint on arms at all gave them even less reason to
    # behave). REVERTED 2026-07-16: back to the full 21-DOF vector, but now
    # region-conditionally masked (joint_pos_abs_arms_masked_by_region) --
    # near regions keep real arm motion in AMP, far regions get arm columns
    # frozen to default pose (uninformative to the discriminator) instead
    # of sliced out entirely, keeping the tensor shape uniform at 21 (the
    # multi-disc architecture routes one shared amp_obs tensor to whichever
    # per-region discriminator matches each env). See that function's
    # docstring (observations.py) and goalkeeper_multidisc_amp_runner_cfg's
    # matching mask on the motion-dataset (expert) side.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "joint_pos": ObservationTermCfg(
                func=gk_obs.joint_pos_abs_arms_masked_by_region,
                noise=None,
                params={},
            ),
        },
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

    # (e) Far-region required-travel-distance curriculum (2026-07-18,
    # research-doc recommendation B) -- only meaningful for this
    # region-conditioned task (reset_ball_rolling_by_region is the only
    # caller that ever sets use_far_travel_curriculum=True), so registered
    # here rather than in the shared goalkeeper_env_cfg(). Play mode has no
    # active CurriculumManager (base goalkeeper_env_cfg() only populates
    # cfg.curriculum when not play), so this mirrors that gating: without it,
    # reset_ball_rolling's use_far_travel_curriculum branch falls back to its
    # own getattr(..., 1.0) default (full far-travel range), matching how
    # env._ball_difficulty already behaves in play.
    if not play:
        cfg.curriculum["far_travel"] = CurriculumTermCfg(
            func=gk_mdp.far_travel_curriculum,
            params={
                "update_interval": 500,
                "ep_len_divisor":  50,
                # FIX 2026-07-18: was 0.004, sized for the old design where
                # far_travel_frac traveled the full [0,1] range. After the
                # G1-step redesign seeds far_inner/far_outer 65-67% of the
                # way to their bounds already, that step_size emptied the
                # much shorter remaining distance far faster than
                # ball_difficulty (confirmed live: far_inner 47% done at
                # iter ~296 while ball_difficulty was only 12% done) -- the
                # opposite of the intended lag. 0.0013 targets far_outer (the
                # binding constraint) needing ~2.5x ball_difficulty's
                # cu-units to saturate. See far_travel_curriculum's docstring.
                "step_size":       0.0013,
            },
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

REGION_MOTION_FILES: dict[str, list[str]] = {
    "left_near": [str(_MOTIONS_DIR / "LeftStep_own_booster_t1.npz")],
    "left_far": [
        # 2026-07-18: replaced the single synthetic 2.5x-retimed clip with
        # three real-mocap clips (raw pkl -> npz, cropped/leveled/hip-
        # straightened -- see commit 36fc902) spanning short/base/wide
        # distances (~0.7m/1.0m/1.3m, ~0.7s/1.0s/1.2s) that line up with the
        # far region's own y_end range (0.5-1.3m, regions.py). Addresses the
        # dataset concern flagged in docs/superpowers/specs/
        # 2026-07-18-doublestep-research-and-plan.md (finding #2): the prior
        # reference motion had no raw mocap source and its pace was reverse-
        # engineered from the ball-flight budget via a flat playback-speed
        # multiplier rather than being a genuinely-paced capture.
        str(_MOTIONS_DIR / "new_doublestepleft_short_booster_t1.npz"),
        str(_MOTIONS_DIR / "new_doublestepleft_booster_t1.npz"),
        str(_MOTIONS_DIR / "new_doublestepleft_wide_booster_t1.npz"),
    ],
    "right_near": [str(_MOTIONS_DIR / "Rightstep_own_booster_t1.npz")],
    "right_far": [
        str(_MOTIONS_DIR / "new_doublestepright_short_booster_t1.npz"),
        str(_MOTIONS_DIR / "new_doublestepright_booster_t1.npz"),
        str(_MOTIONS_DIR / "new_doublestepright_wide_booster_t1.npz"),
    ],
}

# 2026-07-18: the three far-region clips differ in duration/frame count
# (short ~0.7s < base ~1.0s < wide ~1.2s at the same fps) -- without an
# explicit weight, MotionDataset's per-transition-by-frame-count sampling
# would over-represent "wide" and under-represent "short" purely because of
# clip length, not because either distance matters more. Weighting equally
# per-file (not per-frame) so the far discriminator treats all three
# distances as equally valid style targets.
_FAR_REGION_MOTION_WEIGHTS: list[float] = [1.0, 1.0, 1.0]


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
        # Flatten: REGION_MOTION_FILES values are now per-region lists (2026-07-12,
        # 1.5x-retimed variants added alongside the originals for left_far/right_far)
        # -- the ghost overlay just needs a flat set of files to cycle through.
        npz_files = [p for paths in REGION_MOTION_FILES.values() for p in paths]
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
    # FIX 2026-07-16: only far regions get their reference clip's arm
    # columns frozen (matches the "amp" observation group's region-
    # conditional masking above) -- near regions keep real arm motion
    # capture data, both sides (expert and policy) must agree per region.
    _FAR_REGION_NAMES = {"left_far", "right_far"}
    amp_data = {
        name: MotionDatasetCfg(
            motion_files=paths,
            body_names=GOALKEEPER_KEY_BODY_NAMES,
            amp_obs_terms=_MULTIDISC_AMP_OBS_TERMS,
            anchor_name=GOALKEEPER_ANCHOR_NAME,
            freeze_joint_names=list(ARM_JOINT_NAMES) if name in _FAR_REGION_NAMES else None,
            motion_weights=list(_FAR_REGION_MOTION_WEIGHTS) if name in _FAR_REGION_NAMES else None,
        )
        for name, paths in REGION_MOTION_FILES.items()
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
