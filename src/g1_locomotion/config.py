"""Configuration dataclasses for the G1 locomotion environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandCfg:
    """Ranges for the reset()-time domain-randomization pipeline."""

    payload_mass_range: tuple[float, float] = (0.0, 5.0)  # kg, added to torso
    payload_com_offset_range: tuple[float, float] = (-0.05, 0.05)  # m, per-axis torso CoM shift
    friction_range: tuple[float, float] = (0.4, 1.2)
    terrain_enabled: bool = False  # flat floor while debugging/early training; flip on once it can stand/walk
    terrain_res: int = 8  # Perlin gradient-cell count; higher = bumpier terrain (must divide 128)
    push_interval_range: tuple[float, float] = (3.0, 5.0)  # seconds between pushes
    push_lin_vel_range: tuple[float, float] = (-0.5, 0.5)  # m/s impulse kick per axis (x, y)
    init_joint_noise: float = 0.1  # rad, uniform noise added to default pose on reset
    init_yaw_range: tuple[float, float] = (-3.14159, 3.14159)


@dataclass
class RewardScales:
    lin_vel_tracking: float = 1.5
    ang_vel_tracking: float = 0.75
    contact_timing: float = 0.3
    double_flight: float = -0.3
    zmp_margin: float = -0.2
    torque_penalty: float = -1.0e-5
    action_rate_penalty: float = -0.01
    # asymmetric_payload_run_v2 (5000 iters) plateaued at ~64/1000 step episodes (6.4% of max)
    # and never improved past iteration ~1000 -- alive_bonus was too small relative to the
    # penalty terms below (0.1 vs. base_height's -10.0) to give PPO much incentive to fight for
    # extra survival time. A comparable open-source G1/PPO/rsl_rl-style policy that *did* learn
    # stable standing (836/1000 step episodes) used alive_bonus=5.0:
    # https://huggingface.co/hardware-pathon-ai/unitree-g1-phase1-locomotion
    alive_bonus: float = 5.0
    # Stability terms below match Unitree's official G1 RL config (see rewards.py for the
    # per-term explanation of what each one physically means for the robot):
    # https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
    orientation: float = -1.0
    base_height: float = -10.0
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05


@dataclass
class EnvCfg:
    scene_xml: str = "assets/g1/scene_train.xml"
    num_envs: int = 2
    device: str = "cpu"
    control_dt: float = 0.02  # 50 Hz policy rate
    sim_dt: float = 0.005  # 200 Hz physics rate (4 substeps per control step)
    max_episode_length_s: float = 20.0
    fall_height_threshold: float = 0.5  # pelvis z below this => terminate
    fall_gravity_threshold: float = 0.6  # projected gravity z-component above this => terminate
    command_lin_vel_range: tuple[float, float] = (-1.0, 1.0)
    command_ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
    reward_scales: RewardScales = field(default_factory=RewardScales)

    @property
    def substeps(self) -> int:
        return max(1, round(self.control_dt / self.sim_dt))

    @property
    def max_episode_length_steps(self) -> int:
        return max(1, round(self.max_episode_length_s / self.control_dt))
