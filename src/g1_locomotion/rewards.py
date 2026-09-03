"""Composite reward terms for the G1 asymmetric-actor-critic locomotion task.

Each term returns a raw non-negative "error"/"activity" magnitude per env; the caller
(``rl_env.py``) multiplies by the signed weight in ``RewardScales`` and sums.

The velocity-tracking and stability terms below mirror the convention used by ETH Zurich's
``legged_gym`` (the reward-shaping style this project follows) and Unitree's own official RL
config for this exact robot:
https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
"""

from __future__ import annotations

import numpy as np

from . import robot
from .config import RewardScales
from .math_utils import quat_rotate_inverse
from .robot import STANDING_BASE_HEIGHT

SUPPORT_RADIUS = 0.12  # m, approximate single-foot support margin for the ZMP penalty

# Global multiplier on every reward weight. 1.0 = OFF, which is v9's behaviour and the setting
# with evidence behind it.
#
# v11 set this to control_dt (0.02) to mirror legged_gym's `self.reward_scales[key] *= self.dt`
# and IsaacLab's `* env.step_dt`. That is a faithful description of what those frameworks do, and
# it still broke training. Measured across runs:
#
#   run              reward median   Loss/value median   Policy/mean_std end
#   v3                    399.4            438.19              2.309
#   v7                     23.9             23.01              1.424
#   v9                     59.9             26.82              1.443
#   v10                   -44.6             39.33              1.181
#   v11 (dt + clamp)        0.0000           0.0000            1.500  <- std_range ceiling
#
# Why it breaks, and why "PPO normalizes advantages so a uniform scale is harmless" (my own earlier
# reasoning) is wrong: advantage normalization makes the SURROGATE loss scale-invariant, but the
# value loss goes as (returns - values)^2, so a 0.02x reward scale shrinks it ~2500x. Meanwhile
# entropy_coef (0.01) is an absolute term that does not scale at all. The total loss therefore
# flips from value-dominated to entropy-dominated, and the policy maximizes entropy -- straight to
# the std_range ceiling, exactly as observed. Every run that learned had a value loss in the tens
# to hundreds; v11's was zero.
#
# The reference frameworks apply dt AND tune value_loss_coef / entropy_coef against the resulting
# scale as a package. Porting the dt factor alone while keeping entropy_coef from a 50x-larger
# regime is what broke it. If this is ever revisited, entropy_coef and value_loss_coef in
# ppo_cfg.py have to move with it.
REWARD_DT = 1.0


def lin_vel_tracking(state: dict, commands: np.ndarray) -> np.ndarray:
    """How closely the robot's horizontal velocity (in its own body frame) matches the
    commanded (vx, vy). This is the main "walk where you're told, at the speed you're told"
    signal -- everything else in this file exists to keep the robot upright long enough, and
    efficiently enough, for this term to matter."""
    local_vel = quat_rotate_inverse(state["base_quat"], state["base_lin_vel"])
    err = np.sum((commands[:, :2] - local_vel[:, :2]) ** 2, axis=-1)
    return np.exp(-err / 0.25)


def ang_vel_tracking(state: dict, commands: np.ndarray) -> np.ndarray:
    """Same idea as lin_vel_tracking, but for the commanded turning rate (yaw angular velocity)."""
    err = (commands[:, 2] - state["ang_vel"][:, 2]) ** 2
    return np.exp(-err / 0.25)


def orientation_penalty(state: dict) -> np.ndarray:
    """Penalizes the torso tilting away from upright, continuously, every step -- not just
    when it falls over. ``projected_gravity`` is "which way is down, as seen in the robot's own
    body frame": (0, 0, -1) means perfectly upright, and any nonzero x/y component means the
    torso is leaning. Squaring and summing just those two components gives a single "how tilted
    am I right now" number that grows the more the robot leans in *any* direction.

    Before this term existed, the only anti-falling pressure was the flat per-step alive_bonus
    (which doesn't care how upright the robot is, only whether the episode is still running) plus
    hard termination once it had *already* tipped past the fall threshold. This term is what
    actually teaches the robot "lean = bad" while it still has time to correct, instead of only
    finding out after it's too late.
    """
    return np.sum(state["projected_gravity"][:, :2] ** 2, axis=-1)


def base_height_penalty(state: dict) -> np.ndarray:
    """Penalizes the pelvis straying from its normal standing height (squatting, sinking, or
    launching itself into the air). ``STANDING_BASE_HEIGHT`` is the robot's calibrated standing
    pelvis height from the MJCF, so this keeps the robot's overall posture close to "standing
    normally" rather than, say, learning to walk in a permanent crouch to game the other rewards.
    """
    return (state["base_height"] - STANDING_BASE_HEIGHT) ** 2


def lin_vel_z_penalty(state: dict) -> np.ndarray:
    """Penalizes vertical (up/down) velocity of the pelvis -- i.e. bouncing or bobbing while
    walking. A smooth, efficient gait keeps the torso's height roughly constant; a bouncy one
    wastes energy and is harder to balance and to track velocity commands with.
    """
    return state["base_lin_vel"][:, 2] ** 2


def ang_vel_xy_penalty(state: dict) -> np.ndarray:
    """Penalizes roll/pitch angular velocity -- i.e. the torso actively rotating end-over-end or
    side-to-side, as opposed to just yawing (turning) to follow a commanded direction. This is
    the "don't be actively tipping over" signal, complementing orientation_penalty's "don't
    already be tipped over": one catches the tipping motion as it starts, the other penalizes
    the resulting lean.
    """
    return np.sum(state["ang_vel"][:, :2] ** 2, axis=-1)


def clocked_contact(foot_contact: np.ndarray, expected_stance: np.ndarray) -> np.ndarray:
    """Counts how many feet (0, 1, or 2) are currently in the contact state the gait clock says
    they should be in. Mirrors unitree_rl_gym's G1 ``_reward_contact``.

    ``expected_stance`` comes from the fixed alternating clock (see EnvCfg.gait_period_s /
    gait_stance_duty): True where that leg is scheduled to be planted, False where it is scheduled
    to be swinging. A foot planted during scheduled stance scores, and so does a foot lifted during
    scheduled swing.

    This REPLACES the old air-time-based ``contact_timing``. The critical difference is that the
    schedule is external and visible: the policy also observes sin/cos of the clock phase, so it
    can actually learn the mapping from "where am I in the cycle" to "which foot should be down".
    A clocked reward the policy cannot observe would just be unlearnable noise, which is why the
    observation terms and this reward have to land together.
    """
    return np.sum(foot_contact == expected_stance, axis=-1).astype(np.float64)


# Target swing-foot height.
#
# v12: back to v9's 0.08. v11 moved this to 0.10 to match unitree_rl_lab's 29dof feet_clearance
# ("target_height": 0.1), but 0.08 is what v9 -- the only policy that visibly strides -- was
# trained with, and this revision holds every reward value at v9's except the gait clock.
#
# NOTE what this is measured on: our foot_pos is the ankle_roll_link BODY ORIGIN, not the sole. On
# this model the body origin sits 0.0350 m above the sole (measured by FK: standing on flat ground,
# sole z = 0.0000, body origin z = 0.0348). So a foot resting flat reads 0.035, not 0, and a 0.10
# target means ~0.065 m of TRUE sole clearance. Both references measure on the same link of the same
# robot, so their targets should carry the same meaning, but only our offset is measured here.
FOOT_CLEARANCE_TARGET = 0.08  # m, on the foot body origin (~0.045 m of true sole clearance)
FOOT_CLEARANCE_STD = 0.05  # m, width of the exp kernel below


def foot_clearance(foot_pos: np.ndarray, foot_contact: np.ndarray) -> np.ndarray:
    """Reward a lifted (non-contact) foot for being near the target swing clearance.

    v11 restores this to its POSITIVE form. v10 had inverted it into a -20.0 `feet_swing_height`
    penalty, taken from unitree_rl_gym (the 12-DoF build). unitree_rl_lab's 29dof config -- our
    exact robot -- keeps it a positive reward at weight +1.0, so the inversion was following the
    wrong reference. The v10 run does not defend the penalty form either: feet_swing_height went
    -9.20 -> -0.08 while episodes lasted only 0.47 s, i.e. it went quiet because there was no swing
    phase to charge for, not because foot-drag had been solved.

    Difference from IsaacLab's `foot_clearance_reward` worth knowing: theirs additionally weights
    the height error by tanh(foot horizontal speed), so a foot is only asked to be at clearance
    while it is actually travelling. Ours gates on contact only, which asks a foot lifted in place
    to reach clearance too. Closest form without importing their velocity shaping.

    Gated on *actual* contact, not the clock's expected swing: any airborne foot is expected near
    clearance regardless of what the clock says it should be doing. Assumes flat ground (world
    z=0); needs a ground-height offset if terrain is re-enabled.
    """
    reward = np.exp(-((foot_pos[..., 2] - FOOT_CLEARANCE_TARGET) ** 2) / (2 * FOOT_CLEARANCE_STD**2))
    return np.sum(np.where(foot_contact, 0.0, reward), axis=-1)


def double_flight_penalty(foot_contact: np.ndarray) -> np.ndarray:
    """1.0 whenever both feet are simultaneously airborne (unstable jumping/shuffling)."""
    return (~foot_contact.any(axis=-1)).astype(np.float64)


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.sum((p - a) * ab, axis=-1) / np.clip(np.sum(ab * ab, axis=-1), 1e-8, None)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)


def zmp_margin_violation(state: dict) -> np.ndarray:
    """Soft support-polygon margin penalty, approximated from the foot contact points."""
    cop = state["cop_xy"]
    foot_xy = state["foot_pos"][..., :2]
    contact = state["foot_contact"]
    n = cop.shape[0]
    violation = np.zeros(n)

    both = contact[:, 0] & contact[:, 1]
    dist_seg = _point_to_segment_distance(cop, foot_xy[:, 0], foot_xy[:, 1])
    violation = np.where(both, np.clip(dist_seg - SUPPORT_RADIUS, 0.0, None) / SUPPORT_RADIUS, violation)

    for foot_idx in (0, 1):
        only_this = contact[:, foot_idx] & ~contact[:, 1 - foot_idx]
        dist_pt = np.linalg.norm(cop - foot_xy[:, foot_idx], axis=-1)
        violation = np.where(only_this, np.clip(dist_pt - SUPPORT_RADIUS, 0.0, None) / SUPPORT_RADIUS, violation)

    # Cap grounded excursions at the same 1.0 ceiling as "airborne" below, so a badly-balanced but
    # still-grounded stance can never out-penalize losing ground contact entirely.
    violation = np.clip(violation, 0.0, 1.0)

    airborne = ~contact.any(axis=-1)
    violation = np.where(airborne, 1.0, violation)
    return violation


def torque_penalty(joint_torque: np.ndarray) -> np.ndarray:
    return np.sum(joint_torque**2, axis=-1)


def action_rate_penalty(actions: np.ndarray, prev_actions: np.ndarray) -> np.ndarray:
    return np.sum((actions - prev_actions) ** 2, axis=-1)


def joint_deviation(joint_pos: np.ndarray, default_pos: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Squared deviation from the default pose over a subset of joints.

    Backs three separate terms that differ only in which joints they look at and how hard they are
    weighted: ``hip_pos`` (hip roll/yaw), ``waist_deviation``, and ``arm_deviation``. Measured
    against ``default_pos``, not against zero -- the crouched default means those are no longer the
    same thing for the leg joints, and hard-coding zero here would quietly penalize the neutral pose.
    """
    idx = np.asarray(indices, dtype=int)
    return np.sum((joint_pos[:, idx] - default_pos[None, idx]) ** 2, axis=-1)


def dof_pos_limit_violation(
    joint_pos: np.ndarray, soft_lower: np.ndarray, soft_upper: np.ndarray
) -> np.ndarray:
    """How far (in radians, summed over joints) the pose reaches outside the soft joint limits.

    Linear, not squared, matching legged_gym: the point is a hard wall just inside the mechanical
    limit, not a gentle basin. Zero everywhere inside the soft band, so at the crouched default this
    reads exactly 0 -- which is only true *because* the crouch landed with it (at the old all-zero
    default the knee sat 0.061 rad outside its own soft lower bound and this term would have charged
    the neutral pose every step).
    """
    below = np.clip(soft_lower[None, :] - joint_pos, 0.0, None)
    above = np.clip(joint_pos - soft_upper[None, :], 0.0, None)
    return np.sum(below + above, axis=-1)


def dof_acc_penalty(joint_acc: np.ndarray) -> np.ndarray:
    return np.sum(joint_acc**2, axis=-1)


def dof_vel_penalty(joint_vel: np.ndarray) -> np.ndarray:
    return np.sum(joint_vel**2, axis=-1)


def energy_penalty(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Mechanical power drawn at the joints, sum |tau * omega| over all 29.

    unitree_rl_lab 29dof only (`energy`, weight -2e-5); no unitree_rl_gym counterpart. Distinct
    from torque_penalty (tau^2, which charges for holding a static load) -- this charges only for
    torque applied while actually moving, which is what costs a real battery.
    """
    return np.sum(np.abs(joint_torque * joint_vel), axis=-1)


def contact_no_vel_penalty(foot_lin_vel: np.ndarray, foot_contact: np.ndarray) -> np.ndarray:
    """Squared velocity of feet that are currently planted -- the anti-slip / anti-scuff term.

    A planted foot should be stationary; any velocity while in contact means it is sliding or
    scuffing along the ground. Primarily a sim2real term (scuffing wrecks real hardware and does not
    transfer), but it also bears directly on the trailing-foot drag we are trying to remove.
    """
    vel = foot_lin_vel * foot_contact[..., None]
    return np.sum(vel**2, axis=(1, 2))


def compute_reward(
    state: dict,
    commands: np.ndarray,
    actions: np.ndarray,
    prev_actions: np.ndarray,
    expected_stance: np.ndarray,
    joint_acc: np.ndarray,
    default_dof_pos: np.ndarray,
    soft_dof_lower: np.ndarray,
    soft_dof_upper: np.ndarray,
    scales: RewardScales,
    only_positive_rewards: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sum every weighted reward term for one control step.

    Returns ``(total, terms)`` where ``terms`` holds each already-weighted component, so the caller
    can log per-term attribution -- which is the whole point of running this as one change set
    rather than three staged ones.

    REWARD-SCALE NOTE. Measured per-step balance on this config (300 steps, disturbances off):
      holding the crouched default (stand still):  pos +1.38 / neg -0.35  ->  NET +1.02
      random actions (PPO's starting regime):      pos +1.00 / neg -36.3  ->  NET -35.3
    The standing case is healthy: a robot that just stays up scores clearly positive, so there is a
    positive baseline for the policy to climb toward. The random case is dominated by three terms --
    ang_vel_xy (-12.3), dof_acc (-11.1), dof_vel (-8.5), together 84% of the negative mass -- all of
    which are unbounded quadratics over quantities that only get large while the robot is tumbling
    violently, and all of which are near zero (-0.05 combined) in the standing case. They are
    therefore loud exactly when the policy is worst and quiet once it is upright.
    See EnvCfg.only_positive_rewards for why clamping the total is NOT the fix here.
    Per-term values in the returned dict are always pre-clamp, so logging shows each term's true
    signed contribution even on a clamped step.
    """
    terms = {
        # --- task ---
        "lin_vel_tracking": scales.lin_vel_tracking * lin_vel_tracking(state, commands),
        "ang_vel_tracking": scales.ang_vel_tracking * ang_vel_tracking(state, commands),
        "alive_bonus": np.full(commands.shape[0], scales.alive_bonus),
        # --- gait structure ---
        "contact": scales.contact * clocked_contact(state["foot_contact"], expected_stance),
        "feet_clearance": scales.feet_clearance
        * foot_clearance(state["foot_pos"], state["foot_contact"]),
        "double_flight": scales.double_flight * double_flight_penalty(state["foot_contact"]),
        "zmp_margin": scales.zmp_margin * zmp_margin_violation(state),
        "contact_no_vel": scales.contact_no_vel
        * contact_no_vel_penalty(state["foot_lin_vel"], state["foot_contact"]),
        # --- stability ---
        "orientation": scales.orientation * orientation_penalty(state),
        "base_height": scales.base_height * base_height_penalty(state),
        "lin_vel_z": scales.lin_vel_z * lin_vel_z_penalty(state),
        "ang_vel_xy": scales.ang_vel_xy * ang_vel_xy_penalty(state),
        # --- joint regularization ---
        "hip_pos": scales.hip_pos
        * joint_deviation(state["joint_pos"], default_dof_pos, robot.HIP_ROLL_YAW_INDICES),
        "waist_deviation": scales.waist_deviation
        * joint_deviation(state["joint_pos"], default_dof_pos, robot.WAIST_INDICES),
        "arm_deviation": scales.arm_deviation
        * joint_deviation(state["joint_pos"], default_dof_pos, robot.ARM_INDICES),
        "dof_pos_limits": scales.dof_pos_limits
        * dof_pos_limit_violation(state["joint_pos"], soft_dof_lower, soft_dof_upper),
        "dof_acc": scales.dof_acc * dof_acc_penalty(joint_acc),
        "dof_vel": scales.dof_vel * dof_vel_penalty(state["joint_vel"]),
        # --- effort / smoothness ---
        "torque_penalty": scales.torque_penalty * torque_penalty(state["joint_torque"]),
        "action_rate_penalty": scales.action_rate_penalty * action_rate_penalty(actions, prev_actions),
        "energy": scales.energy * energy_penalty(state["joint_torque"], state["joint_vel"]),
    }
    # Every weight is a PER-SECOND rate and has to be multiplied by the control timestep, which is
    # what both reference frameworks do and what this code was missing until v11:
    #   legged_gym, LeggedRobot._prepare_reward_function:  self.reward_scales[key] *= self.dt
    #     with self.dt = cfg.control.decimation * sim_params.dt = 4 * 0.005 = 0.02
    #   IsaacLab, RewardManager.compute:                   value * term_cfg.weight * env.step_dt
    # Without it every weight borrowed from those configs was 50x too large in absolute terms.
    # Corroborated by the v10 run: Loss/value started at 100023, a pathological value scale.
    #
    # A uniform factor does not change which policy is optimal, so this is not by itself the fix
    # for v10's suicide policy (only_positive_rewards is) -- but it puts the value function, the
    # advantage magnitudes and the adaptive-LR / desired_kl machinery back in the range the
    # reference hyperparameters were actually tuned for.
    terms = {k: v * REWARD_DT for k, v in terms.items()}
    total = sum(terms.values())
    if only_positive_rewards:
        total = np.clip(total, 0.0, None)
    return total, terms
