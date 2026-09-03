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

# MUST stay short enough that the robot has not fallen yet.
#
# This harness holds the default pose with zero actions, which is an UNSTABLE equilibrium -- the
# robot topples on its own in roughly 23-42 control steps (measured across v10 checkpoints and an
# untrained policy). Once it hits the ground the dynamics are chaotic, and two backends that agree
# to 4 decimal places beforehand will land in completely different heaps afterwards. Comparing
# endpoints at 100 steps therefore measures the fall, not the backends: the first version of this
# file did exactly that, reported a spurious MJX failure, and sent me tuning integrators, armature
# and PD gains for a bug that did not exist. 20 steps is comfortably pre-fall.
STEPS = 20
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

    # BOTH backends load scene_mjx.xml. Pointing MuJoCo at scene_train.xml instead would compare a
    # hfield floor with full mesh collision against a plane floor with simplified collision -- two
    # deliberately different models -- and report their (expected) differences as an MJX failure.
    # This must isolate the backend, so the model has to be identical on both sides.
    print(f"rolling out {STEPS} steps, {NUM_ENVS} envs, zero actions, all DR off")
    print("both backends on assets/g1/scene_mjx.xml, so only the physics implementation differs\n")
    mj = rollout(G1MultiEnv(deterministic_cfg("assets/g1/scene_mjx.xml")),
                 deterministic_cfg("assets/g1/scene_mjx.xml"), actions)
    mjx = rollout(G1MjxEnv(deterministic_cfg("assets/g1/scene_mjx.xml")),
                  deterministic_cfg("assets/g1/scene_mjx.xml"), actions)

    # STATISTICAL comparison, not trajectory matching.
    #
    # Do not "tighten" these into a per-step trajectory diff -- that test cannot pass, for reasons
    # that are a property of the system rather than of either backend. A 29-DoF biped holding an
    # unstable equilibrium on four 5 mm contact spheres is chaotic: the two wrappers agree EXACTLY
    # at reset (max diff 0.000000 on height/joints/gravity; 2e-6 on torque, pure float32 rounding),
    # and that 2e-6 amplifies at roughly 1.9x per control step, reaching ~0.65 rad within 20 steps.
    # Any epsilon does this -- float32 vs float64 alone guarantees one exists.
    #
    # So what is verified here is that both backends produce the same physical BEHAVIOUR in
    # aggregate: the robot ends up at a comparable height, stays comparably upright, and makes
    # ground contact a comparable fraction of the time. That is what has to hold for a reward
    # computed on either backend to mean the same thing.
    print(f"{'quantity':30s} {'mujoco':>11s} {'mjx':>11s} {'diff':>10s}")
    rows = []

    h_mj, h_mjx = mj["base_height"].mean(), mjx["base_height"].mean()
    rows.append(("mean base height (m)", h_mj, h_mjx, abs(h_mj - h_mjx), 0.05))

    lo_mj, lo_mjx = mj["base_height"].min(), mjx["base_height"].min()
    rows.append(("min base height (m)", lo_mj, lo_mjx, abs(lo_mj - lo_mjx), 0.05))

    g_mj = mj["projected_gravity"][..., 2].mean()
    g_mjx = mjx["projected_gravity"][..., 2].mean()
    rows.append(("mean proj gravity z", g_mj, g_mjx, abs(g_mj - g_mjx), 0.05))

    j_mj = np.abs(mj["joint_pos"]).mean()
    j_mjx = np.abs(mjx["joint_pos"]).mean()
    rows.append(("mean |joint pos| (rad)", j_mj, j_mjx, abs(j_mj - j_mjx), 0.05))

    c_mj, c_mjx = mj["contacts"].mean(), mjx["contacts"].mean()
    rows.append(("mean feet in contact", c_mj, c_mjx, abs(c_mj - c_mjx), 0.6))

    for name, a, b, _d, _t in rows:
        if not (np.isfinite(a) and np.isfinite(b)):
            print(f"  NON-FINITE VALUE in {name} -- a backend is numerically broken.")

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
