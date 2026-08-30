"""Train the G1 asymmetric actor-critic locomotion policy with rsl_rl PPO.

Phase 1 (local, this machine): num_envs=2, device=cpu -- sanity-check that the env resets,
steps, and computes rewards without shape errors.
    python train.py --num_envs 2 --device cpu --iterations 20 --run_name smoke_test

Phase 2 (Colab, T4 GPU): num_envs=1024, device=cuda, with WandB logging.
    python train.py --num_envs 1024 --device cuda --iterations 1500 --wandb --run_name asymmetric_payload_run
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from g1_locomotion.config import EnvCfg  # noqa: E402
from g1_locomotion.ppo_cfg import build_train_cfg  # noqa: E402
from g1_locomotion.rl_env import G1VecEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="asymmetric_payload_run")
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Where checkpoints/logs are written. Defaults to <repo>/logs/<run_name> (ephemeral on "
        "Colab!). On Colab, point this at a mounted Drive path, e.g. "
        "/content/drive/MyDrive/g1_checkpoints/<run_name>, so a disconnect doesn't lose progress.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases instead of TensorBoard.")
    parser.add_argument("--wandb_project", type=str, default="g1_robust_locomotion")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file to resume from.")
    return parser.parse_args()


def _patch_wandb_settings_compat() -> None:
    """rsl_rl (as of 5.5.0) still constructs wandb.Settings(start_method="thread"), a kwarg modern
    wandb (0.17+, pydantic-based Settings) rejects outright with a ValidationError. Strip it before
    it reaches wandb's strict validator -- rsl_rl's own package, not something we can configure away."""
    import wandb

    original_init = wandb.Settings.__init__

    def patched_init(self, **kwargs):
        kwargs.pop("start_method", None)
        original_init(self, **kwargs)

    wandb.Settings.__init__ = patched_init


def main() -> None:
    args = parse_args()

    if args.wandb:
        _patch_wandb_settings_compat()

    env_cfg = EnvCfg(
        scene_xml=os.path.join(PROJECT_ROOT, "assets", "g1", "scene_train.xml"),
        num_envs=args.num_envs,
        device=args.device,
    )
    env = G1VecEnv(env_cfg)

    train_cfg = build_train_cfg(
        num_steps_per_env=args.num_steps_per_env,
        save_interval=args.save_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    log_dir = args.log_dir or os.path.join(PROJECT_ROOT, "logs", args.run_name)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
