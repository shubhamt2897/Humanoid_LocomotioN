"""Export a trained actor checkpoint to ONNX for on-robot deployment.

Only the actor (proprioceptive-input) network is exported -- the critic and its privileged
observations never leave simulation, per the asymmetric actor-critic design.

    python export_policy.py --checkpoint logs/asymmetric_payload_run/model_1500.pt --out models/g1_policy.onnx
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "models", "g1_policy.onnx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_cfg = EnvCfg(
        scene_xml=os.path.join(PROJECT_ROOT, "assets", "g1", "scene_train.xml"),
        num_envs=1,
        device="cpu",
    )
    env = G1VecEnv(env_cfg)
    train_cfg = build_train_cfg()
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
    runner.load(args.checkpoint)

    out_dir = os.path.dirname(args.out)
    out_name = os.path.basename(args.out)
    runner.export_policy_to_onnx(out_dir, filename=out_name)
    print(f"Exported ONNX policy to {args.out}")


if __name__ == "__main__":
    main()
