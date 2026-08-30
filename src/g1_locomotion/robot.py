"""Static description of the Unitree G1 (29-DOF) actuated joints and default pose.

Leg PD gains (hip/knee/ankle) match Unitree's own official RL config for this robot:
https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
That reference only actuates the 12 leg joints; waist/arm/wrist gains here have no official
counterpart and remain reasonable-starting-point estimates -- tune them once training is running.
"""

from __future__ import annotations

from dataclasses import dataclass

# Order matches the <actuator> block in assets/g1/g1_29dof.xml exactly.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NUM_JOINTS = len(JOINT_NAMES)  # 29

ACTUATOR_NAMES: tuple[str, ...] = tuple(name.removesuffix("_joint") for name in JOINT_NAMES)

# Per-joint-group (kp, kd, action_scale). "action_scale" bounds how far a unit action
# can move the PD target off the default pose, in radians.
# Hip/knee/ankle kp+kd are Unitree's official values (see module docstring); action_scale
# there is a uniform 0.25 across all 12 leg joints in that reference, matched here too.
_LEG_GAINS = (100.0, 2.0, 0.25)
_KNEE_GAINS = (150.0, 4.0, 0.25)
_ANKLE_GAINS = (40.0, 2.0, 0.25)
_WAIST_GAINS = (150.0, 3.0, 0.15)
_ARM_GAINS = (40.0, 1.0, 0.30)
_WRIST_GAINS = (15.0, 0.3, 0.30)

JOINT_GROUP_GAINS: dict[str, tuple[float, float, float]] = {
    "left_hip_pitch_joint": _LEG_GAINS,
    "left_hip_roll_joint": _LEG_GAINS,
    "left_hip_yaw_joint": _LEG_GAINS,
    "left_knee_joint": _KNEE_GAINS,
    "left_ankle_pitch_joint": _ANKLE_GAINS,
    "left_ankle_roll_joint": _ANKLE_GAINS,
    "right_hip_pitch_joint": _LEG_GAINS,
    "right_hip_roll_joint": _LEG_GAINS,
    "right_hip_yaw_joint": _LEG_GAINS,
    "right_knee_joint": _KNEE_GAINS,
    "right_ankle_pitch_joint": _ANKLE_GAINS,
    "right_ankle_roll_joint": _ANKLE_GAINS,
    "waist_yaw_joint": _WAIST_GAINS,
    "waist_roll_joint": _WAIST_GAINS,
    "waist_pitch_joint": _WAIST_GAINS,
    "left_shoulder_pitch_joint": _ARM_GAINS,
    "left_shoulder_roll_joint": _ARM_GAINS,
    "left_shoulder_yaw_joint": _ARM_GAINS,
    "left_elbow_joint": _ARM_GAINS,
    "left_wrist_roll_joint": _WRIST_GAINS,
    "left_wrist_pitch_joint": _WRIST_GAINS,
    "left_wrist_yaw_joint": _WRIST_GAINS,
    "right_shoulder_pitch_joint": _ARM_GAINS,
    "right_shoulder_roll_joint": _ARM_GAINS,
    "right_shoulder_yaw_joint": _ARM_GAINS,
    "right_elbow_joint": _ARM_GAINS,
    "right_wrist_roll_joint": _WRIST_GAINS,
    "right_wrist_pitch_joint": _WRIST_GAINS,
    "right_wrist_yaw_joint": _WRIST_GAINS,
}

# All-zero pose is the model's calibrated standing pose (MJCF zero configuration).
DEFAULT_JOINT_ANGLES: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}

FOOT_BODY_NAMES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_BODY_NAME = "torso_link"
PELVIS_BODY_NAME = "pelvis"
IMU_SITE_NAME = "imu"
FREE_JOINT_NAME = "floating_base_joint"
FLOOR_GEOM_NAME = "floor"

STANDING_BASE_HEIGHT = 0.793  # from the pelvis body's default pos in g1_29dof.xml


@dataclass(frozen=True)
class ActionScaling:
    """Per-actuator PD gains and action scale, ordered to match JOINT_NAMES."""

    kp: tuple[float, ...]
    kd: tuple[float, ...]
    action_scale: tuple[float, ...]

    @staticmethod
    def build() -> "ActionScaling":
        kp, kd, scale = zip(*(JOINT_GROUP_GAINS[name] for name in JOINT_NAMES))
        return ActionScaling(kp=kp, kd=kd, action_scale=scale)
