# Robust Humanoid Locomotion (Unitree G1)

## What this is

A walking policy for the Unitree G1 humanoid, trained in MuJoCo with PPO, whose whole point is
**robustness the policy has to earn through its own senses** -- not a vanilla "walk forward on
flat ground" controller. During training the robot gets hit with random pushes, carries random
extra payload mass shifted around its torso, and (eventually) walks on procedurally bumpy
terrain. It never gets to see any of that directly. All it has at deployment time is
proprioception -- joint angles, joint velocities, an IMU, and its own recent actions -- and it
has to *infer* what's throwing it off balance from how its own joints react, the same way the
real hardware would have to. See [Project Specification.md](Project%20Specification.md) for the
full design brief this implementation follows.

## The core idea: asymmetric actor-critic

Two networks are trained together, but they don't see the same things:

- **The actor** (the one that actually gets deployed) sees only what a real G1 could sense
  onboard: joint positions/velocities, gravity direction + angular velocity from the IMU, its own
  previous action, and the commanded velocity. No cameras, no ground-truth physics.
- **The critic** (training-time only, thrown away afterward) additionally sees privileged
  simulator state that doesn't exist on real hardware: the robot's *true* velocity, exact ground
  reaction forces, and the exact payload mass/CoM/friction that were randomized this episode.

The critic's job is just to give the actor a better learning signal during training -- since it
knows *why* the robot is being pushed off balance this episode, it can judge the actor's actions
more accurately than if it only saw the same limited view the actor does. The actor still has to
solve the hard problem alone: react correctly to a payload it can't see, using only the way its
own joints are behaving.

## Reward function: what is the robot actually being rewarded for?

Every simulation step produces a handful of independent signals, each capturing one specific
idea, which get scaled and summed into a single number PPO optimizes. All of them live in
[`rewards.py`](src/g1_locomotion/rewards.py); weights live in
[`config.py::RewardScales`](src/g1_locomotion/config.py).

| Term | Weight | Physically, what is this rewarding or punishing? |
|---|---:|---|
| `lin_vel_tracking` | +1.5 | Walking at the commanded speed, in the commanded direction. This is the actual goal -- everything else exists to make it achievable without falling over. |
| `ang_vel_tracking` | +0.75 | Turning at the commanded rate. |
| `contact_timing` | +0.3 | Rewards a foot touching down after a proper swing phase (~0.2-0.5s in the air), not an instant tap -- encourages a real walking gait over shuffling. |
| `double_flight` | (shares `contact_timing`'s weight, negated) | Penalizes both feet being off the ground at once -- no hopping/jumping in place of walking. |
| `zmp_margin` | -0.2 | Soft penalty as the center of pressure drifts toward the edge of the foot/feet currently in contact -- a cheap approximation of "don't overbalance your stance." |
| `torque_penalty` | -1e-5 | Penalizes `Σ torque²` across all 29 joints -- energy efficiency, and keeps the policy from relying on violent, hardware-damaging corrections. |
| `action_rate_penalty` | -0.01 | Penalizes jerky, fast-changing actions frame to frame -- favors smooth control. |
| `orientation` | -1.0 | **Penalizes torso tilt, continuously, every single step.** Computed from `projected_gravity` (which way "down" appears in the robot's own body frame -- (0,0,-1) exactly means perfectly upright). This is the one that actually teaches "leaning = bad" while there's still time to correct, rather than only finding out once it's already fallen past the point of no return. |
| `base_height` | -10.0 | Penalizes the pelvis straying from its normal standing height -- keeps posture close to "standing normally" instead of gaming other rewards by crouching or launching upward. |
| `lin_vel_z` | -2.0 | Penalizes vertical bounce/bob in the pelvis. A good gait keeps torso height roughly steady. |
| `ang_vel_xy` | -0.05 | Penalizes roll/pitch *rotation rate* -- catches the robot actively tipping over as it's happening, complementing `orientation`'s "already tilted" check. |
| `alive_bonus` | +0.1 | A flat reward every step the episode hasn't ended. |

**Why the last four exist:** early versions of this reward function had no continuous "stay
upright" signal at all -- only the flat `alive_bonus` (which doesn't care *how* upright the robot
is, only whether the episode is still running) and a hard termination once the robot had *already*
tipped past a fall threshold. That's enough of a gap that a partially-trained policy could
actually end up walking (well, falling) worse than random joint noise: fighting to stay balanced
costs `torque_penalty`/`action_rate_penalty` every step with nothing offsetting it, so an
under-trained policy can find it "cheaper" to just go limp and fall quickly. `orientation`,
`base_height`, `lin_vel_z`, and `ang_vel_xy` close that gap, matching Unitree's own official G1 RL
config and ETH Zurich's `legged_gym` convention that this project's reward style otherwise follows:
https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py

Termination itself is separate from all of the above: an episode ends early if the pelvis drops
below 0.5m or `projected_gravity`'s vertical component exceeds 0.6 (i.e. the robot has actually
tipped over) -- both judged relative to gravity/world-vertical, not local terrain slope, so the
correct learned behavior is "torso stays level, legs and feet adapt to the ground," not "match
whatever angle the ground happens to be."

## PD gains

Leg joints (hip/knee/ankle) use Unitree's own official values for this robot, not guessed
constants:
https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py

| Joint group | Kp | Kd | Action scale |
|---|---:|---:|---:|
| Hip (pitch/roll/yaw) | 100 | 2.0 | 0.25 |
| Knee | 150 | 4.0 | 0.25 |
| Ankle (pitch/roll) | 40 | 2.0 | 0.25 |
| Waist | 150 | 3.0 | 0.15 |
| Arm | 40 | 1.0 | 0.30 |
| Wrist | 15 | 0.3 | 0.30 |

Unitree's reference config only actuates the 12 leg joints; this robot has 29 DOF (waist and
arms too), so those extra groups have no official counterpart and remain reasonable
starting-point estimates -- see [`robot.py`](src/g1_locomotion/robot.py).

## Domain randomization

Applied at every episode reset, so the actor is forced to adapt within an episode rather than
memorize one fixed physical setup:

- **Payload mass**: 0-5 kg added to the torso.
- **Payload CoM offset**: shifted up to ±5cm per axis, simulating carrying the load on the chest
  vs. the back vs. off-center.
- **Floor friction**: randomized between 0.4 and 1.2.
- **Push perturbations**: random impulse kicks to the base every 3-5 seconds.
- **Procedural terrain** (Perlin-noise heightfield): implemented, currently disabled
  (`terrain_enabled=False`) until flat-ground standing/tracking is solid.

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
  - `rewards.py` -- velocity tracking, stability (orientation/height/bounce), contact timing,
    ZMP margin, torque/action-rate regularization.
  - `rl_env.py` -- `rsl_rl`-compatible `VecEnv`; builds the `actor_obs`/`critic_obs`
    TensorDict split.
  - `config.py` / `ppo_cfg.py` -- env and PPO hyperparameters.
- `train.py`, `play.py`, `export_policy.py` -- entry points (see next section).

## Usage

**Phase 1 -- local sanity check (CPU, num_envs=2):**
```
python train.py --num_envs 2 --device cpu --iterations 20 --run_name smoke_test
python play.py --checkpoint logs/smoke_test/model_19.pt   # visual check via mujoco viewer
```
(`rsl_rl`'s training loop is 0-indexed, so a 20-iteration run's last checkpoint is `model_19.pt`,
not `model_20.pt` -- check `logs/smoke_test/` for whichever `model_<N>.pt` is actually highest.)

**Phase 2 -- Colab (GPU):** open `colab_train.ipynb`, which handles mounting Drive, cloning
fresh each session, installing dependencies, `wandb login`, training with checkpoints written
straight to Drive (so a disconnect doesn't lose progress), and a resume cell for continuing from
the latest checkpoint.
```
wandb login
python train.py --num_envs 64 --device cuda --iterations 500 --save_interval 100 \
  --wandb --run_name asymmetric_payload_run_v2
```
Note: this repo's physics backend (`mujoco_env.py`) parallelizes envs with a Python loop over
independent CPU simulations. It is correct at any `num_envs`, but Phase 2's GPU-batched
`mujoco-warp` throughput described in the spec is not implemented -- swapping in a warp-batched
backend behind the same `step()`/`reset_idx()` contract is the remaining Phase 2 work. It also
means every env carries its own full copy of the robot's mesh geometry, which in practice caps
free-tier Colab (~12-13GB RAM) at roughly 64-128 envs today, well short of the spec's `num_envs
= 1024` target -- that scale needs the GPU-batched backend above, not just more patience.

**Export for deployment:**
```
python export_policy.py \
  --checkpoint /content/drive/MyDrive/g1_checkpoints/asymmetric_payload_run_v2/model_499.pt \
  --out models/g1_policy.onnx
```
(same 0-indexing applies -- check the checkpoint directory for the actual highest `model_<N>.pt`
rather than assuming it matches `--iterations` exactly.)

## Design notes / approximations

- The ZMP margin penalty (`rewards.py`) approximates the support polygon as a fixed-radius
  circle/segment around foot contact points rather than the true convex hull.
- Ground reaction forces exposed to the critic are per-foot normal/tangential magnitudes from
  `mj_contactForce`, not full 3D world-frame vectors.
