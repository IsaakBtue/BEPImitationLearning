"""PPO config with AMP discriminators for Goalkeeper AMP training.

Extends RslRlOnPolicyRunnerCfg with:
  - amp_coef: blending factor (0.4 = 40% AMP reward + 60% task reward)
  - amp_discr_hidden_dims: hidden layer sizes for 6 discriminators
"""
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config with AMP hyperparameters."""
    amp_coef: float = 0.4
    amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    amp_disc_mini_batch_size: int = 4096  # per-discriminator cap; G1 upstream gets ~17k but uses 8GB GPU


def goalkeeper_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create AMP-augmented PPO runner config for Goalkeeper.

    Returns:
        RslRlAmpRunnerCfg with actor, critic, and PPO algorithm configured.
    """
    return RslRlAmpRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,   # matches G1 upstream (g1_29_config.py:369); entropy explosion risk gone since disc overtake fixed via lr+steps
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="goalkeeper_amp_booster_t1",
        save_interval=200,
        num_steps_per_env=100,
        max_iterations=40_000,
        amp_coef=0.2,  # lowered from 0.4: reduces AMP adversarial pressure on arm deviations
        amp_discr_hidden_dims=[512, 256],
        amp_disc_mini_batch_size=4096,
    )
