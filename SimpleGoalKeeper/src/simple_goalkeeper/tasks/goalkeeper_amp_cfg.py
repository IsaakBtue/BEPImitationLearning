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
