"""Visually roll out a trained (or freshly-initialized) G1 policy with the passive MuJoCo viewer.

This is the Phase 1 "visual debugging" step from the project spec: confirm the robot's
physics, domain randomization, and policy behavior look correct before scaling up on Colab.

    python play.py --checkpoint logs/asymmetric_payload_run/model_20.pt
    python play.py   # no checkpoint: rolls out a random-initialized policy, just to sanity-check the sim

With --record, press R in the viewer window to start/stop capturing a clip to media/ -- nothing
is written to disk until you press R the first time, so you control exactly what gets saved (e.g.
a "random action" baseline now, to compare against later checkpoints):

    python play.py --record
    python play.py --record --checkpoint logs/short_cpu_run/model_299.pt
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from g1_locomotion.config import EnvCfg  # noqa: E402
from g1_locomotion.ppo_cfg import build_train_cfg  # noqa: E402
from g1_locomotion.rl_env import G1VecEnv  # noqa: E402

RECORD_KEY = ord("R")  # GLFW uses ASCII codes for printable keys


class VideoRecorder:
    """Starts/stops writing MJPEG-rendered frames to a timestamped .mp4 on demand."""

    def __init__(self, model, video_dir: str, tag: str, fps: float):
        import imageio

        self._imageio = imageio
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.video_dir = video_dir
        self.tag = tag
        self.fps = fps
        self.writer = None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def toggle(self, data) -> None:
        if self.is_recording:
            self.stop()
        else:
            self.start(data)

    def start(self, data) -> None:
        os.makedirs(self.video_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.video_dir, f"{self.tag}_{stamp}.mp4")
        self.writer = self._imageio.get_writer(path, fps=self.fps, macro_block_size=None)
        print(f"[record] started -> {path}")

    def stop(self) -> None:
        self.writer.close()
        print("[record] stopped")
        self.writer = None

    def capture_frame(self, data) -> None:
        if not self.is_recording:
            return
        self.renderer.update_scene(data)
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        if self.is_recording:
            self.stop()
        self.renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--duration_s", type=float, default=60.0)
    parser.add_argument(
        "--auto_reset", action="store_true",
        help="Respawn on fall/timeout like training does. Default is off: watch one continuous "
        "attempt (with a fixed command) for the full --duration_s, including lying there if it falls.",
    )
    parser.add_argument("--record", action="store_true", help="Enable the R-key video-capture toggle.")
    parser.add_argument("--video_dir", type=str, default=os.path.join(PROJECT_ROOT, "media"))
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

    recorder = None
    if args.record:
        tag = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else "random_init"
        recorder = VideoRecorder(model, args.video_dir, tag, fps=1.0 / env_cfg.control_dt)
        print(f"[record] press R in the viewer window to start/stop capturing a clip to {args.video_dir}")

    def key_callback(keycode: int) -> None:
        if recorder is not None and keycode == RECORD_KEY:
            recorder.toggle(data)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            start = time.time()
            while viewer.is_running() and (time.time() - start) < args.duration_s:
                step_start = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                viewer.sync()
                if recorder is not None:
                    recorder.capture_frame(data)
                time.sleep(max(0.0, env_cfg.control_dt - (time.time() - step_start)))
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
