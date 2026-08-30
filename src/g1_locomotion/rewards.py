"""Composite reward terms for the G1 asymmetric-actor-critic locomotion task.

Each term returns a raw non-negative "error"/"activity" magnitude per env; the caller
(``rl_env.py``) multiplies by the signed weight in ``RewardScales`` and sums.
"""

from __future__ import annotations

import numpy as np

from .config import RewardScales
from .math_utils import quat_rotate_inverse

SUPPORT_RADIUS = 0.12  # m, approximate single-foot support margin for the ZMP penalty


def lin_vel_tracking(state: dict, commands: np.ndarray) -> np.ndarray:
    local_vel = quat_rotate_inverse(state["base_quat"], state["base_lin_vel"])
    err = np.sum((commands[:, :2] - local_vel[:, :2]) ** 2, axis=-1)
    return np.exp(-err / 0.25)


def ang_vel_tracking(state: dict, commands: np.ndarray) -> np.ndarray:
    err = (commands[:, 2] - state["ang_vel"][:, 2]) ** 2
    return np.exp(-err / 0.25)


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
        "alive_bonus": np.full(commands.shape[0], scales.alive_bonus),
    }
    total = sum(terms.values())
    return total, terms
