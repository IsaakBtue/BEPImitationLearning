"""RSL-RL OnPolicyRunner configuration for the Goalkeeper environment.

Uses Isaac Lab's rsl_rl wrapper with standard PPO hyperparameters
matching the original G1 HIM-PPO configuration.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class GoalkeeperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard PPO runner config — mirrors g1_29 HIM-PPO hyperparameters."""

    num_steps_per_env: int = 100
    max_iterations: int = 200000
    save_interval: int = 200
    experiment_name: str = "goalkeeper"
    run_name: str = "g1_isaaclab"
    logger: str = "tensorboard"
    wandb_project: str = "goalkeeper"

    clip_actions: float = 100.0

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    # Policy network: stochastic actor + deterministic critic
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )

    # PPO algorithm configuration
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
