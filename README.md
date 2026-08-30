# Robust Humanoid Locomotion (Unitree G1)

Asymmetric actor-critic PPO locomotion policy with domain randomization, per
[Project Specification.md](Project%20Specification.md). Actor sees only deployable
proprioception; critic additionally sees privileged simulator state (true base velocity,
ground reaction forces, payload mass/CoM, friction).

## Setup

```
conda activate humanoid_locomotion
python check_env.py   # verify torch/mujoco/rsl_rl/wandb/numpy import cleanly
```

## Layout

- `assets/g1/scene_train.xml` -- G1 29-DOF scene with a heightfield terrain that gets
  regenerated (Perlin noise) every episode reset.
- `src/g1_locomotion/`
  - `robot.py` -- joint order, PD gains, action scale.
  - `mujoco_env.py` -- N independent MjModel/MjData instances (CPU-parallel Phase 1 backend),
    domain randomization, PD control, sensor/contact readout.
  - `terrain.py` -- procedural Perlin-noise heightfield generation.
  - `rewards.py` -- velocity tracking, contact timing, ZMP margin, torque regularization.
  - `rl_env.py` -- `rsl_rl`-compatible `VecEnv`; builds the `actor_obs`/`critic_obs`
    TensorDict split.
  - `config.py` / `ppo_cfg.py` -- env and PPO hyperparameters.
- `train.py`, `play.py`, `export_policy.py` -- entry points (see next section).

## Usage

**Phase 1 -- local sanity check (CPU, num_envs=2):**
```
python train.py --num_envs 2 --device cpu --iterations 20 --run_name smoke_test
python play.py --checkpoint logs/smoke_test/model_20.pt   # visual check via mujoco viewer
```

**Phase 2 -- Colab (GPU, num_envs=1024):** upload the repo, `pip install mujoco torch rsl-rl-lib
wandb imageio`, then:
```
wandb login
python train.py --num_envs 1024 --device cuda --iterations 1500 --wandb --run_name asymmetric_payload_run
```
Note: this repo's physics backend (`mujoco_env.py`) parallelizes envs with a Python loop over
independent CPU simulations. It is correct at any `num_envs`, but Phase 2's GPU-batched
`mujoco-warp` throughput described in the spec is not implemented -- swapping in a warp-batched
backend behind the same `step()`/`reset_idx()` contract is the remaining Phase 2 work.

**Export for deployment:**
```
python export_policy.py --checkpoint logs/asymmetric_payload_run/model_1500.pt --out models/g1_policy.onnx
```

## Design notes / approximations

- PD gains in `robot.py` are reasonable defaults, not manufacturer-tuned values.
- The ZMP margin penalty (`rewards.py`) approximates the support polygon as a fixed-radius
  circle/segment around foot contact points rather than the true convex hull.
- Ground reaction forces exposed to the critic are per-foot normal/tangential magnitudes from
  `mj_contactForce`, not full 3D world-frame vectors.
