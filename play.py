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

A green arrow shows the commanded (target) ground velocity and an orange arrow shows the robot's
actual measured ground velocity, both live in the viewer and baked into any recorded video --
pass --no_arrows to turn this off. Both are flattened to the ground plane (the target uses the
robot's heading only, not torso lean), so they're directly comparable. Before training, expect
the two arrows to look unrelated; a well-trained policy should show them roughly aligned.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from g1_locomotion.config import EnvCfg  # noqa: E402
from g1_locomotion.ppo_cfg import build_train_cfg  # noqa: E402
from g1_locomotion.rl_env import G1VecEnv  # noqa: E402

RECORD_KEY = ord("R")  # GLFW uses ASCII codes for printable keys

ARROW_ORIGIN_HEIGHT = 0.9  # m above the pelvis, so arrows float clear of the robot's body
ARROW_WIDTH = 0.02  # m
ARROW_MAX_LEN = 2.0  # m, purely a rendering clamp so a fall-induced velocity spike doesn't draw a huge arrow
TARGET_RGBA = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)  # green: commanded velocity
ACTUAL_RGBA = np.array([1.0, 0.5, 0.0, 0.9], dtype=np.float32)  # orange: actual measured velocity


def _heading_from_quat(quat: np.ndarray) -> float:
    """Yaw (rotation about world Z) from a wxyz quaternion -- ignores roll/pitch (torso lean)."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _velocity_arrow_vectors(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, target_vec, actual_vec) in world frame for env 0.

    Both vectors are kept flat (z=0) so they represent ground-plane movement: the target
    is the commanded (vx, vy) rotated by the robot's heading only, so it always lies flat
    on the floor instead of tilting with the torso's instantaneous lean.
    """
    state = env._state  # noqa: SLF001 -- play.py is a debug tool, reaching into the wrapper's cached state is fine
    pelvis_pos = env.sim.datas[0].qpos[env.sim.free_qpos_adr : env.sim.free_qpos_adr + 3]
    origin = pelvis_pos + np.array([0.0, 0.0, ARROW_ORIGIN_HEIGHT])

    heading = _heading_from_quat(state["base_quat"][0])
    cmd_x, cmd_y = env.commands[0, 0], env.commands[0, 1]
    target_vec = np.array(
        [np.cos(heading) * cmd_x - np.sin(heading) * cmd_y, np.sin(heading) * cmd_x + np.cos(heading) * cmd_y, 0.0]
    )

    actual_vec = state["base_lin_vel"][0].copy()
    actual_vec[2] = 0.0

    for vec in (target_vec, actual_vec):
        norm = np.linalg.norm(vec)
        if norm > ARROW_MAX_LEN:
            vec *= ARROW_MAX_LEN / norm
    return origin, target_vec, actual_vec


def _draw_velocity_arrows(scene, start_idx: int, origin: np.ndarray, target_vec: np.ndarray, actual_vec: np.ndarray) -> None:
    idx = start_idx
    for vec, rgba in ((target_vec, TARGET_RGBA), (actual_vec, ACTUAL_RGBA)):
        if np.linalg.norm(vec) < 1e-3:
            continue
        geom = scene.geoms[idx]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba
        )
        # mjv_initGeom leaves category=0, which the renderer's category mask then filters out --
        # must be set explicitly or the geom is computed correctly but silently never drawn.
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH, origin, origin + vec)
        idx += 1
    scene.ngeom = idx


class VideoRecorder:
    """Starts/stops writing MJPEG-rendered frames to a timestamped .mp4 on demand.

    Uses a body-tracking camera (follows the pelvis) rather than mujoco.Renderer's static
    default -- otherwise a robot that walks away from the origin (or a velocity arrow pointing
    away from a fixed viewing angle) drifts out of frame over a longer clip.

    The R-key toggle fires on the viewer's own render thread while capture_frame() runs on the
    main loop thread; a lock around every read/write of self.writer prevents the two racing (the
    race previously surfaced as a stray "I/O operation on closed Writer" crash).
    """

    def __init__(self, model, video_dir: str, tag: str, fps: float, track_body_id: int):
        import imageio

        self._imageio = imageio
        self._lock = threading.Lock()
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = track_body_id
        self.camera.distance = 3.0
        self.camera.azimuth = 120.0
        self.camera.elevation = -15.0
        self.video_dir = video_dir
        self.tag = tag
        self.fps = fps
        self.writer = None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def toggle(self, data) -> None:
        with self._lock:
            if self.writer is not None:
                self._stop_locked()
            else:
                self._start_locked(data)

    def _start_locked(self, data) -> None:
        os.makedirs(self.video_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.video_dir, f"{self.tag}_{stamp}.mp4")
        self.writer = self._imageio.get_writer(path, fps=self.fps, macro_block_size=None)
        print(f"[record] started -> {path}")

    def _stop_locked(self) -> None:
        self.writer.close()
        print("[record] stopped")
        self.writer = None

    def capture_frame(self, data, arrow_vecs: tuple | None = None) -> None:
        with self._lock:
            if self.writer is None:
                return
            self._render_and_write_locked(data, arrow_vecs)

    def _render_and_write_locked(self, data, arrow_vecs: tuple | None) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        if arrow_vecs is not None:
            origin, target_vec, actual_vec = arrow_vecs
            _draw_velocity_arrows(self.renderer.scene, self.renderer.scene.ngeom, origin, target_vec, actual_vec)
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        with self._lock:
            if self.writer is not None:
                self._stop_locked()
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
    parser.add_argument(
        "--no_arrows", action="store_true",
        help="Disable the target-velocity (green) / actual-velocity (orange) arrow overlay.",
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
        # map_location: rsl_rl's runner.load() leaves torch.load's device mapping at its default
        # (whatever the checkpoint was saved on), so a GPU-trained checkpoint (e.g. from Colab)
        # fails to deserialize on this CPU-only viewer without this override.
        runner.load(args.checkpoint, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    obs = env.get_observations()
    model, data = env.sim.models[0], env.sim.datas[0]
    print(f"Command (vx, vy, wz): {env.commands[0]}")

    recorder = None
    if args.record:
        tag = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else "random_init"
        recorder = VideoRecorder(
            model, args.video_dir, tag, fps=1.0 / env_cfg.control_dt, track_body_id=env.sim.pelvis_body_id
        )
        print(f"[record] press R in the viewer window to start/stop capturing a clip to {args.video_dir}")

    if not args.no_arrows:
        print("[arrows] green = target velocity, orange = actual velocity")

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

                arrow_vecs = None if args.no_arrows else _velocity_arrow_vectors(env)
                if arrow_vecs is not None:
                    _draw_velocity_arrows(viewer.user_scn, 0, *arrow_vecs)
                else:
                    viewer.user_scn.ngeom = 0
                viewer.sync()

                if recorder is not None:
                    recorder.capture_frame(data, arrow_vecs)
                time.sleep(max(0.0, env_cfg.control_dt - (time.time() - step_start)))
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
