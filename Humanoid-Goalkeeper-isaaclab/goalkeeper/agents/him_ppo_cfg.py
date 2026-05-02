"""HIM-PPO training configuration — matches g1_29_config.py hyperparameters exactly."""


def get_goalkeeper_him_cfg(max_iterations: int = 200000, seed: int = 42) -> dict:
    """Return the train_cfg dict expected by HIMOnPolicyRunner."""
    return {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": 100,
            "save_interval": 200,
            "logger": "tensorboard",
            "max_iterations": max_iterations,
            "seed": seed,
        },
        "policy": {
            # Hidden dims from g1_29_config G129CfgPPO
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        "algorithm": {
            # Hyperparameters from g1_29_config G129CfgPPO / algorithm block
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            # Smoothness loss coefficients from HIM-PPO
            "value_smoothness_coef": 0.1,
            "smoothness_upper_bound": 1.0,
            "smoothness_lower_bound": 0.1,
        },
        "amp": {
            # 29 robot joints × 2 consecutive frames = 58-dim (matches original g1_29_config.py)
            # Non-motion joints are zero-padded in expert data — discriminator learns to keep them still.
            "num_obs": 58,
            "amp_coef": 0.4,   # 0.4 × AMP + 0.6 × task  (from original)
            "num_steps": 2,
        },
    }
