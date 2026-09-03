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

# Default ("neutral") pose. This is the reference the PD targets are built off, so action=0 holds
# exactly this pose -- changing it changes what every action means, and invalidates any checkpoint
# trained against a different default.
#
# Was all-zero (the MJCF zero configuration, straight-legged) through run v9_smoke. Now a slight
# crouch matching Unitree's official G1 config: hip_pitch -0.1, knee +0.3, ankle_pitch -0.2.
# Two reasons this matters beyond "Unitree does it":
#   1. Lower CoM / pre-bent knees is a more stable equilibrium and puts the PD targets already near
#      a realistic mid-stride knee bend, so the +-0.25 rad action range can reach useful stepping
#      poses instead of spending its whole budget just getting out of full extension.
#   2. It is a hard prerequisite for the dof_pos_limits penalty (see RewardScales.dof_pos_limits).
#      The knee's MJCF range is [-0.0873, +2.8798]; at soft_dof_pos_limit=0.9 its soft lower bound
#      is +0.0611 rad, so a knee at 0.0 is already *outside* the soft limit -- the limit penalty
#      would have punished the neutral pose itself. Crouched knee=0.3 clears it comfortably.
_CROUCH_HIP_PITCH = -0.1
_CROUCH_KNEE = 0.3
_CROUCH_ANKLE_PITCH = -0.2

DEFAULT_JOINT_ANGLES: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
for _name in JOINT_NAMES:
    if "hip_pitch" in _name:
        DEFAULT_JOINT_ANGLES[_name] = _CROUCH_HIP_PITCH
    elif "knee" in _name:
        DEFAULT_JOINT_ANGLES[_name] = _CROUCH_KNEE
    elif "ankle_pitch" in _name:
        DEFAULT_JOINT_ANGLES[_name] = _CROUCH_ANKLE_PITCH

# Joint-index groups used by the regularization reward terms (indices into JOINT_NAMES order).
# Hip roll/yaw only -- hip *pitch* is excluded on purpose, since swinging the leg fore/aft is the
# whole point of walking; roll/yaw are what produce the splayed-leg gait RL likes to find.
HIP_ROLL_YAW_INDICES: tuple[int, ...] = tuple(
    i for i, n in enumerate(JOINT_NAMES) if "hip_roll" in n or "hip_yaw" in n
)
WAIST_INDICES: tuple[int, ...] = tuple(i for i, n in enumerate(JOINT_NAMES) if "waist" in n)
ARM_INDICES: tuple[int, ...] = tuple(
    i for i, n in enumerate(JOINT_NAMES)
    if any(k in n for k in ("shoulder", "elbow", "wrist"))
)

FOOT_BODY_NAMES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_BODY_NAME = "torso_link"
PELVIS_BODY_NAME = "pelvis"
IMU_SITE_NAME = "imu"
FREE_JOINT_NAME = "floating_base_joint"
FLOOR_GEOM_NAME = "floor"

# Pelvis height when standing at DEFAULT_JOINT_ANGLES with the feet exactly on flat ground (z=0).
# This is the target of the base_height reward penalty. It is NOT the spawn height -- see
# SPAWN_HEIGHT_CLEARANCE below, which is added on top of it at reset.
#
# Recomputed by forward kinematics on assets/g1/scene_train.xml (place the base high, set the
# default joint angles, mj_forward, then solve pelvis_z so the lowest foot collision geom sits at
# z=0). Was 0.793 for the old straight-legged all-zero default (FK confirms 0.7919 for that pose);
# the crouched default above lowers it to 0.7842. This constant MUST be recomputed whenever
# DEFAULT_JOINT_ANGLES changes -- otherwise base_height (-10.0) actively fights the default pose,
# penalizing the robot for standing exactly where the PD targets put it.
STANDING_BASE_HEIGHT = 0.784

# Extra height added to STANDING_BASE_HEIGHT when spawning, so the feet start just ABOVE the floor
# and settle into contact instead of starting inside it.
#
# Unitree does the same thing: their G1 config spawns at pos=[0,0,0.8] against a base_height_target
# of 0.78 -- a deliberate 2 cm gap, not a rounding difference.
#
# Measured on this model, spawning at exactly STANDING_BASE_HEIGHT (i.e. no clearance): with
# DomainRandCfg.init_joint_noise = 0.1 rad perturbing the leg joints at reset, the leg can extend
# past its nominal length, and 98.2% of resets started with a foot up to 19.5 mm BELOW the floor
# (400 samples). MuJoCo resolves that interpenetration with a large contact impulse -- the first
# steps showed 596-1079 N of foot normal force against a ~343 N body weight, i.e. 2-3x weight,
# still oscillating a dozen steps later rather than settling. That is a corrupted first observation
# on essentially every episode.
#
# Clearance sweep inside the real env (8 envs x 6 resets x 60 steps, zero actions, body weight
# ~344 N), measuring peak foot normal force and how many feet are already touching at reset:
#
#   clearance   peak N   peak/W   feet in contact at reset
#     0.000       1537    4.47      1.75  <- spawns embedded in the floor
#     0.005       1572    4.57      1.25
#     0.010       1517    4.41      0.38
#     0.020       1693    4.92      0.00
#     0.030       1769    5.14      0.00
#
# Two things to read off this. First, the fix works: at >= 0.02 no foot starts in contact. Second,
# the peak impulse is ~4.5x body weight even at zero clearance, and clearance only moves it ~15%
# across the whole range -- the transient is dominated by the PD controller pulling the
# noise-perturbed joints back to the crouched default, not by the drop. So buying clean
# non-penetrating resets costs very little.
#
# 0.03 rather than Unitree's 0.02 because our reset noise is a different scheme: we add uniform
# +-0.1 rad to all 29 joints, they multiply the default angles by 0.5-1.5x. Our measured worst-case
# leg extension puts a sole 19.5 mm below the floor (400 FK samples), so 0.02 clears it by only
# 0.5 mm while 0.03 leaves real headroom for the tail, at +4% peak force.
SPAWN_HEIGHT_CLEARANCE = 0.03


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
