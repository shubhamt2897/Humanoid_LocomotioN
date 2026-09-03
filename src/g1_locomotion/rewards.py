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


# Target swing-foot height, verified against unitree_rl_gym's _reward_feet_swing_height, which uses
# a literal 0.08 in `square(feet_pos[:, :, 2] - 0.08) * ~contact`.
#
# NOTE what this is measured on: like theirs, our foot_pos is the ankle_roll_link BODY ORIGIN, not
# the sole. On this model the body origin sits 0.0350 m above the sole (measured by FK: standing on
# flat ground, sole z = 0.0000, body origin z = 0.0348). So a foot resting flat reads 0.035, not 0,
# and a 0.08 target means ~0.045 m of TRUE sole clearance -- a realistic walking foot lift, whereas
# 0.08 m of actual sole clearance would be closer to a march. Unitree measures on the same link of
# the same robot, so their 0.08 should carry the same meaning, but only our offset is measured here.
FOOT_CLEARANCE_TARGET = 0.08  # m, on the foot body origin (~0.045 m of true sole clearance)


def feet_swing_height(foot_pos: np.ndarray, foot_contact: np.ndarray) -> np.ndarray:
    """Squared error between a lifted foot's height and the target swing clearance.

    This is the INVERTED form of the old ``foot_clearance`` reward (+1.0, exp kernel). Same physical
    quantity, opposite sign and Unitree's -20.0 weight. The reason for the inversion: a reward for
    lifting can simply be declined -- a policy that drags its trailing foot forfeits it and keeps
    everything else -- whereas a penalty is charged every step the foot is off the ground and not
    at clearance. Foot-drag is the specific behavior this is aimed at.

    Gated on *actual* contact rather than the clock's expected swing, matching Unitree: any airborne
    foot is expected to be at clearance height, regardless of whether the clock agrees it should be
    airborne right now. Assumes flat ground (world z=0); needs a ground-height offset if terrain is
    re-enabled.
    """
    error = (foot_pos[..., 2] - FOOT_CLEARANCE_TARGET) ** 2
    return np.sum(np.where(foot_contact, 0.0, error), axis=-1)


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
        "feet_swing_height": scales.feet_swing_height
        * feet_swing_height(state["foot_pos"], state["foot_contact"]),
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
    }
    total = sum(terms.values())
    if only_positive_rewards:
        total = np.clip(total, 0.0, None)
    return total, terms
