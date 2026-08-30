"""rsl_rl PPO runner configuration for the asymmetric actor-critic G1 policy."""

from __future__ import annotations


def build_train_cfg(
    num_steps_per_env: int = 24,
    save_interval: int = 50,
    use_wandb: bool = False,
    wandb_project: str = "g1_robust_locomotion",
) -> dict:
    """Build the rsl_rl ``train_cfg`` dict consumed by ``OnPolicyRunner``."""
    cfg = {
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
        "obs_groups": {"actor": ["actor_obs"], "critic": ["actor_obs", "critic_obs"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        },
    }
    if use_wandb:
        # WandbLogWriter derives the run name from the log_dir's basename, so callers should
        # pass log_dir=".../<run_name>" (see train.py) to get e.g. "asymmetric_payload_run".
        cfg["logger"] = {"class_name": "WandbLogWriter", "project_name": wandb_project}
    return cfg
