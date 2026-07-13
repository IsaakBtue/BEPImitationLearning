"""AMP runner configuration for the goalkeeper task."""
from __future__ import annotations

import os
from pathlib import Path

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import (
    AMPPPOAlgorithmCfg,
    AMPRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"

GOALKEEPER_ANCHOR_NAME: str = "Trunk"

# Full-body key points for the AMP discriminator: torso + limb extremities.
# Using the full body gives the discriminator a complete picture of motion
# naturalness — not just lower-body, but also arm swing and trunk posture.
# Follows KaydenKnapik/BoosterT1mjlab (proven working) extended to cover all
# major segments: torso (Trunk, Waist), hands, shanks, feet.
GOALKEEPER_KEY_BODY_NAMES: list[str] = [
    "Trunk",
    "Waist",
    "left_hand_link",
    "right_hand_link",
    "Shank_Left",
    "Shank_Right",
    "left_foot_link",
    "right_foot_link",
]


def _motion_files() -> list[str]:
    """2026-07-13: reverted back to the full 14-file glob. The 2026-07-12
    "only 2.5x DoubleStep clips" experiment (docs/BugFixes.md) was ported
    from blue-ball-waypoint's AMP-dilution fix, but that fix targeted a
    *region-conditioned* discriminator that already only serves one specific
    crossing type -- this branch's single discriminator serves EVERY
    crossing (single-step, double-step, triple-step, recovery poses), so
    cutting it down to 2 files removed the only source of "natural motion"
    grounding for everything except double-stepping. The very next training
    run (`green_2p5x_tflight_widen_2026-07-12`) saw its raw actions explode
    into a jerky, high-frequency oscillation (`action_acc_l2`/`action_rate_l2`
    diverging by 3-4 orders of magnitude and never recovering) -- a
    discriminator that's stopped constraining most of the task's motion is a
    plausible driver of exactly that kind of unconstrained actor behavior.
    Reverted to the full dataset per user request.
    """
    if not _MOTIONS_DIR.is_dir():
        return []
    return sorted(str(p) for p in _MOTIONS_DIR.glob("*.npz"))


# Relative sampling weight given to double/triple-step motions in the AMP
# discriminator dataset. These are otherwise represented only in proportion to
# their frame count alongside every other motion (~36% of all frames as of
# 2026-07), so this boosts their share of AMP transitions independent of length.
_DOUBLE_TRIPLE_STEP_WEIGHT = 4.0


def _motion_weights(files: list[str], boost: float = _DOUBLE_TRIPLE_STEP_WEIGHT) -> list[float]:
    """Per-motion AMP discriminator sampling weight: boost double/triple-step files."""
    return [
        boost if ("DoubleStep" in f or "TripleStep" in f) else 1.0
        for f in files
    ]


def goalkeeper_amp_runner_cfg() -> AMPRunnerCfg:
    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=250,
        experiment_name="simple_goalkeeper",
        run_name="phase1",
        empirical_normalization=True,
        use_wandb=True,
        wandb_project="SimpleGoalKeeper",
        wandb_entity="i-p-b-bouwmeester-eindhoven-university-of-technology",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=AMPPPOAlgorithmCfg(
            class_name="AMPPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            amp_replay_buffer_size=250_000,
        ),
        amp_data=MotionDatasetCfg(
            motion_files=_motion_files(),
            motion_weights=_motion_weights(_motion_files()),
            body_names=GOALKEEPER_KEY_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name=GOALKEEPER_ANCHOR_NAME,
        ),
        amp_discr_hidden_dims=[256, 256],
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.6,
        amp_min_normalized_std=0.05,
        video_interval=0,
        video_n_steps=250,
    )
