"""Visually roll out a trained (or freshly-initialized) G1 policy with the passive MuJoCo viewer.

This is the Phase 1 "visual debugging" step from the project spec: confirm the robot's
physics, domain randomization, and policy behavior look correct before scaling up on Colab.

    python play.py --checkpoint logs/asymmetric_payload_run/model_20.pt
    python play.py   # no checkpoint: rolls out a random-initialized policy, just to sanity-check the sim
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco.viewer  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from g1_locomotion.config import EnvCfg  # noqa: E402
from g1_locomotion.ppo_cfg import build_train_cfg  # noqa: E402
from g1_locomotion.rl_env import G1VecEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--duration_s", type=float, default=60.0)
    parser.add_argument(
        "--auto_reset", action="store_true",
        help="Respawn on fall/timeout like training does. Default is off: watch one continuous "
        "attempt (with a fixed command) for the full --duration_s, including lying there if it falls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_cfg = EnvCfg(
        scene_xml=os.path.join(PROJECT_ROOT, "assets", "g1", "scene_train.xml"),
        num_envs=1,
        device="cpu",
    )
    env = G1VecEnv(env_cfg, auto_reset=args.auto_reset)

    train_cfg = build_train_cfg()
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
    if args.checkpoint:
        runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cpu")

    obs = env.get_observations()
    model, data = env.sim.models[0], env.sim.datas[0]
    print(f"Command (vx, vy, wz): {env.commands[0]}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < args.duration_s:
            step_start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            viewer.sync()
            time.sleep(max(0.0, env_cfg.control_dt - (time.time() - step_start)))


if __name__ == "__main__":
    main()
