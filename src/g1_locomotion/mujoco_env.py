"""Low-level, CPU-parallel MuJoCo physics manager for N independent G1 instances.

Each environment owns its own ``MjModel``/``MjData`` pair (compiled independently from the
same XML) so that domain-randomized parameters -- payload mass, CoM offset, floor friction,
terrain heightfield -- can differ per environment. This is the Phase 1 (local, CPU,
``num_envs`` small) backend described in the project spec; Phase 2's GPU-batched
``mujoco-warp`` backend on Colab is a drop-in replacement for this module's ``step``/``reset``
contract, not implemented here.
"""

from __future__ import annotations

import numpy as np
import mujoco

from . import robot
from .config import DomainRandCfg, EnvCfg
from .math_utils import quat_rotate_inverse
from .terrain import apply_terrain_to_model, flatten_terrain, terrain_height_at_center


def _sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found in model.")
    return int(model.sensor_adr[sid])


class G1MultiEnv:
    """Owns and steps ``num_envs`` independent G1 MuJoCo simulations."""

    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.rng = np.random.default_rng()

        self.models = [mujoco.MjModel.from_xml_path(cfg.scene_xml) for _ in range(cfg.num_envs)]
        for m in self.models:
            m.opt.timestep = cfg.sim_dt
        self.datas = [mujoco.MjData(m) for m in self.models]

        base = self.models[0]
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in robot.ACTUATOR_NAMES]
        )
        self.pos_adr = np.array([_sensor_adr(base, f"{n}_pos") for n in robot.ACTUATOR_NAMES])
        self.vel_adr = np.array([_sensor_adr(base, f"{n}_vel") for n in robot.ACTUATOR_NAMES])
        self.torque_adr = np.array([_sensor_adr(base, f"{n}_torque") for n in robot.ACTUATOR_NAMES])
        self.gyro_adr = _sensor_adr(base, "imu_gyro")
        self.quat_adr = _sensor_adr(base, "imu_quat")
        self.linvel_adr = _sensor_adr(base, "frame_vel")

        self.torso_body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, robot.TORSO_BODY_NAME)
        self.pelvis_body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, robot.PELVIS_BODY_NAME)
        self.floor_geom_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_GEOM, robot.FLOOR_GEOM_NAME)

        free_joint_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, robot.FREE_JOINT_NAME)
        self.free_qpos_adr = int(base.jnt_qposadr[free_joint_id])
        self.free_qvel_adr = int(base.jnt_dofadr[free_joint_id])

        self.joint_qpos_adr = np.array(
            [base.jnt_qposadr[mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in robot.JOINT_NAMES]
        )

        self.foot_geom_ids: list[np.ndarray] = []
        self.foot_body_ids: list[int] = []
        for body_name in robot.FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.foot_body_ids.append(body_id)
            start, num = base.body_geomadr[body_id], base.body_geomnum[body_id]
            geom_ids = np.arange(start, start + num)
            geom_ids = geom_ids[base.geom_contype[geom_ids] != 0]
            self.foot_geom_ids.append(geom_ids)

        self.default_torso_mass = float(base.body_mass[self.torso_body_id])
        self.default_torso_ipos = base.body_ipos[self.torso_body_id].copy()

        gains = robot.ActionScaling.build()
        self.kp = np.array(gains.kp)
        self.kd = np.array(gains.kd)
        self.action_scale = np.array(gains.action_scale)
        self.torque_limits = np.abs(base.actuator_ctrlrange[self.actuator_ids]).max(axis=1)

        self.default_dof_pos = np.array([robot.DEFAULT_JOINT_ANGLES[n] for n in robot.JOINT_NAMES])

        n = self.num_envs
        self.push_timer = np.zeros(n)
        self.next_push_time = np.zeros(n)
        self.payload_mass = np.zeros(n)
        self.payload_com = np.zeros((n, 3))
        self.floor_friction = np.zeros(n)

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids: np.ndarray, dr: DomainRandCfg) -> None:
        for i in env_ids:
            model, data = self.models[i], self.datas[i]

            payload = self.rng.uniform(*dr.payload_mass_range)
            com_offset = self.rng.uniform(dr.payload_com_offset_range[0], dr.payload_com_offset_range[1], size=3)
            friction = self.rng.uniform(*dr.friction_range)

            model.body_mass[self.torso_body_id] = self.default_torso_mass + payload
            model.body_ipos[self.torso_body_id] = self.default_torso_ipos + com_offset
            model.geom_friction[self.floor_geom_id, 0] = friction
            if dr.terrain_enabled:
                apply_terrain_to_model(model, "terrain", self.rng, res=dr.terrain_res)
            else:
                flatten_terrain(model, "terrain")
            ground_z = terrain_height_at_center(model, "terrain")

            self.payload_mass[i] = payload
            self.payload_com[i] = com_offset
            self.floor_friction[i] = friction

            mujoco.mj_resetData(model, data)
            data.qpos[self.free_qpos_adr : self.free_qpos_adr + 3] = [
                0.0, 0.0, ground_z + robot.STANDING_BASE_HEIGHT
            ]
            yaw = self.rng.uniform(*dr.init_yaw_range)
            data.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7] = [
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
            ]
            noise = self.rng.uniform(-dr.init_joint_noise, dr.init_joint_noise, size=robot.NUM_JOINTS)
            data.qpos[self.joint_qpos_adr] = self.default_dof_pos + noise
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)

            self.push_timer[i] = 0.0
            self.next_push_time[i] = self.rng.uniform(*dr.push_interval_range)

    # ------------------------------------------------------------------- step
    def step(self, torques: np.ndarray, dr: DomainRandCfg) -> None:
        """Apply ``torques`` (num_envs, 29), integrate physics, and handle scheduled pushes."""
        for i in range(self.num_envs):
            model, data = self.models[i], self.datas[i]
            data.ctrl[self.actuator_ids] = torques[i]
            for _ in range(self.cfg.substeps):
                mujoco.mj_step(model, data)

            self.push_timer[i] += self.cfg.control_dt
            if self.push_timer[i] >= self.next_push_time[i]:
                impulse = self.rng.uniform(dr.push_lin_vel_range[0], dr.push_lin_vel_range[1], size=2)
                data.qvel[self.free_qvel_adr : self.free_qvel_adr + 2] += impulse
                self.push_timer[i] = 0.0
                self.next_push_time[i] = self.rng.uniform(*dr.push_interval_range)

    # --------------------------------------------------------------- gather
    def compute_torques(self, actions: np.ndarray) -> np.ndarray:
        """PD control: actions (num_envs, 29) in [-1, 1] -> torque command, clipped to limits."""
        targets = self.default_dof_pos[None, :] + actions * self.action_scale[None, :]
        joint_pos = self._batch_sensor(self.pos_adr)
        joint_vel = self._batch_sensor(self.vel_adr)
        torque = self.kp[None, :] * (targets - joint_pos) - self.kd[None, :] * joint_vel
        return np.clip(torque, -self.torque_limits[None, :], self.torque_limits[None, :])

    def _batch_sensor(self, adr: np.ndarray) -> np.ndarray:
        return np.stack([self.datas[i].sensordata[adr] for i in range(self.num_envs)], axis=0)

    def gather_state(self) -> dict[str, np.ndarray]:
        joint_pos = self._batch_sensor(self.pos_adr)
        joint_vel = self._batch_sensor(self.vel_adr)
        joint_torque = self._batch_sensor(self.torque_adr)
        ang_vel = self._batch_sensor(np.arange(self.gyro_adr, self.gyro_adr + 3))
        base_quat = self._batch_sensor(np.arange(self.quat_adr, self.quat_adr + 4))
        base_lin_vel = self._batch_sensor(np.arange(self.linvel_adr, self.linvel_adr + 3))

        world_gravity = np.tile(np.array([0.0, 0.0, -1.0]), (self.num_envs, 1))
        projected_gravity = quat_rotate_inverse(base_quat, world_gravity)

        base_height = np.array([self.datas[i].qpos[self.free_qpos_adr + 2] for i in range(self.num_envs)])
        foot_pos = np.stack(
            [np.stack([self.datas[i].xpos[bid] for bid in self.foot_body_ids]) for i in range(self.num_envs)]
        )

        foot_normal_force = np.zeros((self.num_envs, 2))
        foot_tangential_force = np.zeros((self.num_envs, 2))
        foot_contact = np.zeros((self.num_envs, 2), dtype=bool)
        cop_xy = np.zeros((self.num_envs, 2))

        for i in range(self.num_envs):
            data = self.datas[i]
            weighted_pos = np.zeros(3)
            total_normal = 0.0
            for foot_idx, geom_ids in enumerate(self.foot_geom_ids):
                for c in range(data.ncon):
                    con = data.contact[c]
                    if con.geom1 != self.floor_geom_id and con.geom2 != self.floor_geom_id:
                        continue
                    other = con.geom2 if con.geom1 == self.floor_geom_id else con.geom1
                    if other not in geom_ids:
                        continue
                    force6 = np.zeros(6)
                    mujoco.mj_contactForce(self.models[i], data, c, force6)
                    normal = abs(force6[0])
                    tangential = float(np.hypot(force6[1], force6[2]))
                    foot_normal_force[i, foot_idx] += normal
                    foot_tangential_force[i, foot_idx] += tangential
                    foot_contact[i, foot_idx] = True
                    weighted_pos += np.array(con.pos) * normal
                    total_normal += normal
            if total_normal > 1e-6:
                cop_xy[i] = (weighted_pos / total_normal)[:2]

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
            "ang_vel": ang_vel,
            "projected_gravity": projected_gravity,
            "base_quat": base_quat,
            "base_lin_vel": base_lin_vel,
            "base_height": base_height,
            "foot_normal_force": foot_normal_force,
            "foot_tangential_force": foot_tangential_force,
            "foot_contact": foot_contact,
            "cop_xy": cop_xy,
            "foot_pos": foot_pos,
            "payload_mass": self.payload_mass.copy(),
            "payload_com": self.payload_com.copy(),
            "floor_friction": self.floor_friction.copy(),
        }
