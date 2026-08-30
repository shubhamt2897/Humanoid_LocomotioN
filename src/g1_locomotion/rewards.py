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


def contact_timing(feet_air_time: np.ndarray, new_contact: np.ndarray) -> np.ndarray:
    """Reward feet that touched down after a healthy swing phase (~0.2-0.5s), like legged_gym."""
    reward = np.sum((feet_air_time - 0.2) * new_contact, axis=-1)
    return np.clip(reward, 0.0, None)


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

    airborne = ~contact.any(axis=-1)
    violation = np.where(airborne, 1.0, violation)
    return violation


def torque_penalty(joint_torque: np.ndarray) -> np.ndarray:
    return np.sum(joint_torque**2, axis=-1)


def action_rate_penalty(actions: np.ndarray, prev_actions: np.ndarray) -> np.ndarray:
    return np.sum((actions - prev_actions) ** 2, axis=-1)


def compute_reward(
    state: dict,
    commands: np.ndarray,
    actions: np.ndarray,
    prev_actions: np.ndarray,
    feet_air_time: np.ndarray,
    new_contact: np.ndarray,
    scales: RewardScales,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    terms = {
        "lin_vel_tracking": scales.lin_vel_tracking * lin_vel_tracking(state, commands),
        "ang_vel_tracking": scales.ang_vel_tracking * ang_vel_tracking(state, commands),
        "contact_timing": scales.contact_timing * contact_timing(feet_air_time, new_contact),
        "double_flight": scales.contact_timing * -double_flight_penalty(state["foot_contact"]),
        "zmp_margin": scales.zmp_margin * zmp_margin_violation(state),
        "torque_penalty": scales.torque_penalty * torque_penalty(state["joint_torque"]),
        "action_rate_penalty": scales.action_rate_penalty * action_rate_penalty(actions, prev_actions),
        "orientation": scales.orientation * orientation_penalty(state),
        "base_height": scales.base_height * base_height_penalty(state),
        "lin_vel_z": scales.lin_vel_z * lin_vel_z_penalty(state),
        "ang_vel_xy": scales.ang_vel_xy * ang_vel_xy_penalty(state),
        "alive_bonus": np.full(commands.shape[0], scales.alive_bonus),
    }
    total = sum(terms.values())
    return total, terms
