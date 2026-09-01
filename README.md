# Robust Humanoid Locomotion (Unitree G1)

[![Checkpoints on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-shubhamt0802%2Fhmn__loc__RL-yellow)](https://huggingface.co/buckets/shubhamt0802/hmn_loc_RL)
[![Training runs on W&B](https://img.shields.io/badge/W%26B-training%20runs-ff7043?logo=weightsandbiases&logoColor=white)](https://wandb.ai/shubhamt2897-hochschule-schmalkalden/g1_robust_locomotion)
[![Technical docs](https://img.shields.io/badge/docs-implementation%20details-blue)](src/README.md)

## Objective

A walking policy for the Unitree G1 humanoid (29-DOF), trained in MuJoCo with PPO. The end goal
is a policy that's **robust to disturbances it has to infer from its own senses** -- pushes,
shifted payload, uneven ground -- using only proprioception (joint angles/velocities, IMU, its
own recent actions), the same limited information the real hardware would have. No cameras, no
privileged state at deployment time.

## Current status

**Not walking yet.** Domain randomization (payload mass/CoM, pushes) is implemented and already
active during training, but the project is still at the earlier milestone of learning to stand
and take a single step from scratch on flat ground -- robustness can't be meaningfully evaluated
before basic locomotion works. Latest run (`asymmetric_payload_run_v7`) adds a reward specifically
for attempting to lift a foot, after the prior run showed the policy had learned to stand rigidly
and never step at all. Full run-by-run diagnostic history: `PROGRESS.md` (local working log,
gitignored, not in this repository's git history).

## Method, in brief

PPO (`rsl_rl`), with an asymmetric actor-critic split: the deployed policy sees only what real
hardware could sense, while the critic (training-time only) additionally sees privileged
simulator state -- true velocity, ground-reaction forces, the exact randomized payload/friction --
to give it a sharper learning signal without leaking into the deployed policy. Full architecture,
reward function, PPO hyperparameters, and the complete diagnostic history of what's been tried and
why: **[src/README.md](src/README.md)**.

## Results

### Training progression

Single continuous 6-second rollouts (no reset), same fixed camera, same MuJoCo viewer, captured
via `play.py --headless`. Not curated highlights -- the actual, typical behavior at each
checkpoint, looping automatically below (higher-res `.mp4` originals linked underneath each):

<table>
<tr>
<td align="center" width="50%"><b>Random init</b></td>
<td align="center" width="50%"><b><code>v3</code> (5000 iters, old reward)</b></td>
</tr>
<tr>
<td align="center"><img src="media/progression/01_random_init.gif" width="300"/></td>
<td align="center"><img src="media/progression/02_v3_5000iter_old_reward.gif" width="300"/></td>
</tr>
<tr>
<td align="center">No training at all -- collapses within about a second.<br/><a href="media/progression/01_random_init.mp4">full-res .mp4</a></td>
<td align="center">Stands, but braces in a stiff, defensive "arms-forward" stance before eventually toppling forward.<br/><a href="media/progression/02_v3_5000iter_old_reward.mp4">full-res .mp4</a></td>
</tr>
<tr><td colspan="2">&nbsp;</td></tr>
<tr>
<td align="center" colspan="2"><b><code>v6</code> (2500 iters, rebalanced reward)</b></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="media/progression/03_v6_2500iter_rebalanced.gif" width="300"/></td>
</tr>
<tr>
<td align="center" colspan="2">Stands with arms hanging naturally rather than braced -- direct result of
rebalancing <code>alive_bonus</code> so survival stops dominating the reward (see
<a href="src/README.md">src/README.md</a>). Still falls; doesn't yet attempt to lift a foot.<br/>
<a href="media/progression/03_v6_2500iter_rebalanced.mp4">full-res .mp4</a></td>
</tr>
</table>

### Training curves

Two metrics, three runs -- read together, not separately. `v3` "wins" on raw survival time, but
that's mostly a reward-shaping artifact (a large `alive_bonus` rewarding survival regardless of
tracking quality). `v6` and `v7` -- with that bonus rebalanced down to Unitree's own official
value -- reach *better* velocity-tracking quality in under half the iterations, which is the
metric that actually reflects walking ability:

<table>
<tr>
<td width="50%"><img src="media/results/episode_length_comparison.png"/></td>
<td width="50%"><img src="media/results/tracking_quality_comparison.png"/></td>
</tr>
</table>

Live, explorable versions of every logged metric (not just these two) are on
[Weights & Biases](https://wandb.ai/shubhamt2897-hochschule-schmalkalden/g1_robust_locomotion).

## Checkpoints

All checkpoints (`model_<N>.pt`, `rsl_rl`/PPO format) and each run's TensorBoard event file are in
the public HF bucket, organized by run name:

**[huggingface.co/buckets/shubhamt0802/hmn_loc_RL](https://huggingface.co/buckets/shubhamt0802/hmn_loc_RL)**

```
hf buckets ls shubhamt0802/hmn_loc_RL/asymmetric_payload_run_v7
hf buckets cp hf://buckets/shubhamt0802/hmn_loc_RL/asymmetric_payload_run_v7/model_<N>.pt ./model_<N>.pt
```

## Quickstart

```
conda activate humanoid_locomotion
python check_env.py                                        # verify the environment
python train.py --num_envs 2 --device cpu --iterations 20 --run_name smoke_test
python play.py --checkpoint logs/smoke_test/model_19.pt     # visual check via mujoco viewer
```

Full setup (Colab/HF Jobs cloud training, headless checkpoint review, ONNX export) is in
**[src/README.md](src/README.md)**.
