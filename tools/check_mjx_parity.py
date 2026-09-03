"""Compare the MJX backend against the MuJoCo backend on an identical deterministic rollout.

Run this before trusting an MJX training run, and after any change to either backend:

    # MuJoCo backend env has no jax; the MJX-capable env has no torch. Run with the jax one:
    <jax env>/python.exe tools/check_mjx_parity.py

Both backends load their own scene (hfield floor vs plane floor), have their mesh collision
configured differently, run different solvers at different precision (float64 vs float32), and
MJX's collision set is deliberately reduced -- so bit-identical output is NOT the goal and would be
suspicious. What this checks is that they agree closely enough that reward terms computed from
either one mean the same thing: the robot settles to the same height, the joints hold the same
pose, and the torso stays upright the same way.

All domain randomization is switched off so the only differences are the physics backends.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from g1_locomotion import robot  # noqa: E402
from g1_locomotion.config import EnvCfg  # noqa: E402
from g1_locomotion.mujoco_env import G1MultiEnv  # noqa: E402
from g1_locomotion.mjx_env import G1MjxEnv  # noqa: E402

STEPS = 100
NUM_ENVS = 2


def deterministic_cfg(scene: str) -> EnvCfg:
    cfg = EnvCfg(scene_xml=os.path.join(ROOT, *scene.split("/")), num_envs=NUM_ENVS, device="cpu")
    dr = cfg.domain_rand
    dr.payload_mass_range = (0.0, 0.0)
    dr.payload_com_offset_range = (0.0, 0.0)
    dr.friction_range = (0.8, 0.8)
    dr.push_lin_vel_range = (0.0, 0.0)
    dr.push_interval_range = (1.0e9, 1.0e9)  # never
    dr.init_joint_noise = 0.0
    dr.init_yaw_range = (0.0, 0.0)
    dr.terrain_enabled = False
    return cfg


def rollout(env, cfg, actions) -> dict[str, np.ndarray]:
    env.reset_idx(np.arange(NUM_ENVS), cfg.domain_rand)
    traj = {"base_height": [], "joint_pos": [], "projected_gravity": [], "contacts": []}
    for t in range(STEPS):
        torques = env.compute_torques(actions)
        env.step(torques, cfg.domain_rand)
        s = env.gather_state()
        traj["base_height"].append(np.asarray(s["base_height"], dtype=np.float64).copy())
        traj["joint_pos"].append(np.asarray(s["joint_pos"], dtype=np.float64).copy())
        traj["projected_gravity"].append(np.asarray(s["projected_gravity"], dtype=np.float64).copy())
        traj["contacts"].append(np.asarray(s["foot_contact"]).sum(axis=-1).copy())
    return {k: np.stack(v) for k, v in traj.items()}


def main() -> int:
    actions = np.zeros((NUM_ENVS, robot.NUM_JOINTS))  # hold the default pose

    print(f"rolling out {STEPS} steps, {NUM_ENVS} envs, zero actions, all DR off\n")
    mj = rollout(G1MultiEnv(deterministic_cfg("assets/g1/scene_train.xml")),
                 deterministic_cfg("assets/g1/scene_train.xml"), actions)
    mjx = rollout(G1MjxEnv(deterministic_cfg("assets/g1/scene_mjx.xml")),
                  deterministic_cfg("assets/g1/scene_mjx.xml"), actions)

    print(f"{'quantity':26s} {'mujoco':>12s} {'mjx':>12s} {'abs diff':>12s}")
    rows = []

    h_mj, h_mjx = mj["base_height"][-1].mean(), mjx["base_height"][-1].mean()
    rows.append(("final base height (m)", h_mj, h_mjx, abs(h_mj - h_mjx), 0.02))

    s_mj = mj["base_height"][-20:].mean()
    s_mjx = mjx["base_height"][-20:].mean()
    rows.append(("settled height, last 20", s_mj, s_mjx, abs(s_mj - s_mjx), 0.02))

    j_mj = mj["joint_pos"][-1].mean(axis=0)
    j_mjx = mjx["joint_pos"][-1].mean(axis=0)
    rows.append(("max |joint pos| diff (rad)", np.abs(j_mj).max(), np.abs(j_mjx).max(),
                 np.abs(j_mj - j_mjx).max(), 0.10))

    g_mj = mj["projected_gravity"][-1].mean(axis=0)[2]
    g_mjx = mjx["projected_gravity"][-1].mean(axis=0)[2]
    rows.append(("proj gravity z (upright=-1)", g_mj, g_mjx, abs(g_mj - g_mjx), 0.05))

    c_mj = mj["contacts"].mean()
    c_mjx = mjx["contacts"].mean()
    rows.append(("mean feet in contact", c_mj, c_mjx, abs(c_mj - c_mjx), 0.35))

    ok = True
    for name, a, b, diff, tol in rows:
        flag = "OK" if diff <= tol else "FAIL"
        if diff > tol:
            ok = False
        print(f"{name:26s} {a:12.4f} {b:12.4f} {diff:12.4f}   {flag} (tol {tol})")

    print()
    if ok:
        print("PARITY OK -- the two backends agree within tolerance on a held default pose.")
    else:
        print("PARITY FAILED -- do not train on MJX until this is understood.")
    print("\nNote: tolerances are loose on purpose (different solvers, float32 vs float64,")
    print("reduced MJX collision set). This checks 'same physical behaviour', not bitwise equality.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
