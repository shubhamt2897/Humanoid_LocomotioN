"""Train the G1 locomotion policy on the MJX (GPU-batched) physics backend.

Separate from train.py ON PURPOSE. train.py drives the MuJoCo backend and is the known-good path;
nothing here can break it. The two scripts differ ONLY in which physics implementation they
construct -- they share the same environment wrapper, observations, actions, rewards, terminations,
gait clock, domain randomization and PPO config, so a MuJoCo run and an MJX run of the same config
are directly comparable. If an MJX run behaves differently, the backend is the only variable.

WHY MJX
-------
The MuJoCo backend steps N independent MjModel/MjData pairs in a serial Python loop. Measured on
the v10 run: 256 envs = 7.03 s/iteration = ~874 timesteps/s, and 128 envs gave the same ~878
timesteps/s. Throughput is pinned by the loop, so raising num_envs buys data-per-update but never
speed, and `--device cuda` there only accelerates the (tiny) policy network.

It is also memory-bound: each env holds its own copy of the model, including 185,482 mesh vertices,
at ~77 MB per env. 512 envs needs 39 GB; a 1024-env job was OOMKilled on a 32 GB flavor.

MJX uploads ONE shared model and vmaps the physics across environments on the GPU:

    envs      MuJoCo backend      MJX backend
     256          19.7 GB           0.02 GB
    1024          78.6 GB           0.07 GB
    4096         314.6 GB           0.26 GB      (measured; excludes XLA workspace)

INSTALL (not in the default environment -- the MuJoCo backend needs neither)
    pip install 'jax[cuda12]' mujoco-mjx      # GPU
    pip install 'jax[cpu]'    mujoco-mjx      # CPU, for tools/check_mjx_parity.py only

USAGE
    python train_mjx.py --num_envs 4096 --device cuda --iterations 5000 \
        --wandb --run_name g1_mjx

MIGRATE DELIBERATELY. Get a result you trust on the MuJoCo backend first, then re-run the same
config here and compare. Run tools/check_mjx_parity.py before a long job -- it rolls both backends
out on the SAME scene and checks they agree.

SCENE. This forces assets/g1/scene_mjx.xml regardless of what EnvCfg defaults to: MJX has no
heightfield collision, so scene_train.xml's hfield floor cannot be stepped. Terrain domain
randomization is therefore unavailable on this backend and terrain_enabled=True is rejected.
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
    # Defaults are deliberately larger than train.py's: the whole point of this backend is that env
    # count is no longer limited by host RAM. 4096 matches Unitree's own G1 training scale.
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--run_name", type=str, default="g1_mjx")
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Where checkpoints/logs are written. Defaults to <repo>/logs/<run_name>. On HF Jobs "
        "point this at the mounted bucket, e.g. /checkpoints/<run_name>, so a crash does not lose "
        "progress -- and resume from the last model_<N>.pt rather than restarting at iteration 0.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases instead of TensorBoard.")
    parser.add_argument("--wandb_project", type=str, default="g1_robust_locomotion")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file to resume from.")
    return parser.parse_args()


def _patch_wandb_settings_compat() -> None:
    """rsl_rl (as of 5.5.0) still constructs wandb.Settings(start_method="thread"), a kwarg modern
    wandb (0.17+, pydantic-based Settings) rejects outright with a ValidationError. Strip it before
    it reaches wandb's strict validator -- rsl_rl's own package, not something we can configure away.

    Duplicated from train.py rather than shared, to keep the two entry points independent.
    """
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
        # scene_mjx.xml, not scene_train.xml -- see the SCENE note in the module docstring.
        scene_xml=os.path.join(PROJECT_ROOT, "assets", "g1", "scene_mjx.xml"),
        num_envs=args.num_envs,
        device=args.device,
    )
    if env_cfg.domain_rand.terrain_enabled:
        raise SystemExit(
            "terrain_enabled=True is not supported on the MJX backend (no heightfield collision). "
            "Train terrain curricula with train.py on the MuJoCo backend."
        )

    env = G1VecEnv(env_cfg, backend="mjx")

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

    print(
        f"[mjx] {args.num_envs} envs x {args.num_steps_per_env} steps = "
        f"{args.num_envs * args.num_steps_per_env:,} timesteps/iteration, "
        f"{args.num_envs * args.num_steps_per_env * args.iterations:,} total"
    )
    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
