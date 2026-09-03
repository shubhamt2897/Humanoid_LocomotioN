"""GPU-batched MJX physics backend -- a drop-in replacement for :mod:`mujoco_env`.

WHY THIS EXISTS
---------------
``mujoco_env.G1MultiEnv`` owns ``num_envs`` independent ``MjModel``/``MjData`` pairs and steps them
in a serial Python loop. It is correct at any ``num_envs`` but does not parallelize: measured on the
v10 run, 256 envs on a T4 ran 7.03 s/iteration = ~874 timesteps/s, and 128 envs gave the *same*
~878 timesteps/s. Throughput is pinned by the Python loop, so `num_envs` only trades iteration
latency against data per update -- adding envs never adds throughput, and `--device cuda` was only
ever accelerating the (tiny) policy network. That is the actual ceiling on this project, not any
reward weight: reference Isaac Gym setups run this task at tens of thousands of steps/s.

MJX compiles the whole physics step to XLA and vmaps it across environments on the GPU, so envs are
batched rather than looped.

CONTRACT
--------
Mirrors ``G1MultiEnv`` exactly -- ``reset_idx``, ``step``, ``compute_torques``, ``gather_state``,
plus the ``default_dof_pos`` / ``rng`` / ``pelvis_body_id`` / ``foot_body_ids`` attributes that
``rl_env.G1VecEnv`` and ``play.py`` read. ``gather_state`` returns the same dict keys with the same
shapes and units, as NumPy arrays, so ``rewards.py`` and ``rl_env.py`` need no changes.

DIFFERENCES FROM THE MUJOCO BACKEND -- read before trusting a comparison between the two
-----------------------------------------------------------------------------------------
1. **No terrain DR.** MJX has no heightfield collision, so this backend loads
   ``assets/g1/scene_mjx.xml`` (plane floor) instead of ``scene_train.xml`` (hfield floor).
   ``DomainRandCfg.terrain_enabled=True`` is rejected outright rather than silently ignored.
2. **Mesh collision is disabled** (see :func:`simplify_collisions`). Ground contact is carried
   entirely by the 8 sphere geoms already present on the two ankle_roll links, which is what makes
   this model cheap for MJX. Consequences:
     - no self-collision (arms can pass through the torso). The MuJoCo path had it, but no reward
       term used it -- ``collision`` is absent here and 0.0 in Unitree's own config.
     - **pelvis-contact termination cannot fire on this backend.** ``gather_state`` returns
       ``pelvis_contact`` as all-False. Acceptable on evidence, not by assumption: it fired exactly
       0 times across all 2000 iterations of v10, because the 0.8 rad tilt bound always terminates
       a topple first. Asserted at init so it cannot rot silently.
3. **Contact is force-thresholded the same way** (``CONTACT_FORCE_THRESHOLD``, 1 N) but the force
   comes from MJX's contact solver rather than ``mj_contactForce``. Solver-level differences mean
   per-step contact booleans will not be bit-identical to the MuJoCo backend.
4. **float32.** MJX is single-precision by default; the MuJoCo backend runs float64.

STATUS -- NOT READY TO TRAIN ON. READ THIS BEFORE LAUNCHING A JOB.
------------------------------------------------------------------
The plumbing works: the backend constructs, resets, steps, and returns every state key with the
right shape and units. The PHYSICS does not. ``tools/check_mjx_parity.py`` currently FAILS.

Symptom (1 env, zero actions, all DR off, holding the default pose -- it should just stand):

    step   base_h   |qvel|max
       3  +0.7806        0.78     <- free fall from spawn, still fine
       4  +0.7732        6.67     <- first ground contact
      10  +0.7860       48.68     <- joint velocities exploding
      20  +0.7748       77.24
      50  -0.4814       60.90     <- fallen through the floor, torso inverted

Energy is injected at first contact and never dissipates. What has been ruled out:

  * Not the model-batching / vmap in_axes wrapper in this file. A completely shared, unbatched,
    single-env MJX model reproduces the divergence BIT-IDENTICALLY (same 0.8115/0.8052/0.7949/
    0.7806 descent, same 6.67 at step 4). The bug is below this module, in the MJX model setup.
  * Not the integrator timestep. Diverges at sim_dt 0.005, 0.002 (the model's own native value)
    and 0.001 alike, peak |qvel| 104 and 91 respectively.
  * Not solver budget. The model already carries solver=Newton, iterations=100, ls_iterations=50,
    and MJX inherits those.

Still to check, in the order I would try them: the foot spheres' contact solref/solimp against
MJX's solver; condim/friction handling on the reduced sphere-vs-plane contact set; and float32
conditioning of the stiff PD gains (knee kp=150) which the float64 MuJoCo backend absorbs. Note
the MuJoCo backend also emitted a QACC instability warning on the same rollout, so sim_dt=0.005
against a model authored at 0.002 may be marginal for BOTH backends and is worth revisiting
independently of MJX.

One thing this port did establish: ``simplify_collisions`` is not an optimization, it is mandatory.
``mjx.put_model`` on the unmodified model raises outright:
    NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_MESH) collisions not implemented.

GPU throughput is also unmeasured -- the local GPU is a 4 GB GTX 1050, and the CPU parity harness
is for correctness only (~67 env-steps/s at 4 envs, meaningless as a performance number).
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    # contact_force lives in the private _src.support module, not the public mjx namespace (checked
    # against mujoco 3.12.0 / jax 0.10.2: `dir(mjx)` exposes no contact-force helper at all). Import
    # it explicitly so a future rename fails loudly here instead of at the first contact.
    from mujoco.mjx._src.support import contact_force as _mjx_contact_force
except ImportError as exc:  # pragma: no cover - surfaced as a clear message, not a stack trace
    raise ImportError(
        "The MJX backend needs `jax` and `mujoco-mjx`, which the default environment does not "
        "install (the MuJoCo backend does not need them). Install with:\n"
        "    pip install 'jax[cuda12]' mujoco-mjx    # GPU\n"
        "    pip install 'jax[cpu]'    mujoco-mjx    # CPU, for the parity check only\n"
        f"Original error: {exc}"
    ) from exc

from . import robot
from .config import DomainRandCfg, EnvCfg
from .math_utils import quat_rotate_inverse
from .mujoco_env import CONTACT_FORCE_THRESHOLD

# Geom types whose collision is switched off by simplify_collisions(). Meshes are the expensive
# ones (this model's pelvis mesh has 18k vertices and the torso 22k -- MJX would have to convexify
# and pair them every step); the shoulder cylinders are self-collision only and equally pointless
# once the meshes are gone.
# Stored as plain ints, and compared against int(geom_type). Do NOT compare the raw numpy value
# against the mjtGeom enum members directly: that silently matched NOTHING on the training
# container (mujoco 3.12.0 there too), disabling 0 geoms and leaving all 38 collidable, while
# working locally. int-vs-int has no such ambiguity across mujoco/numpy versions.
_DISABLED_COLLISION_TYPES = (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_CYLINDER))


def simplify_collisions(model: mujoco.MjModel) -> int:
    """Disable every mesh/cylinder collision geom in place; return how many were switched off.

    Leaves exactly the 8 foot spheres and the ground plane collidable, i.e. 8 possible contact
    pairs per environment. Verified on this model: 29 geoms disabled, 38 collidable geoms -> 9.

    This is done in Python rather than in a hand-edited MJCF so that ``scene_mjx.xml`` can keep
    ``<include file="g1_29dof.xml"/>`` and never drift from the robot definition the MuJoCo backend
    uses. The foot spheres are what the whole backend's performance rests on, so their survival is
    asserted rather than assumed.
    """
    disabled = 0
    for g in range(model.ngeom):
        if int(model.geom_type[g]) in _DISABLED_COLLISION_TYPES and (
            model.geom_contype[g] or model.geom_conaffinity[g]
        ):
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
            disabled += 1

    survivors = [
        g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    spheres = [g for g in survivors if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)]
    planes = [g for g in survivors if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    if len(spheres) != 8 or len(planes) != 1 or len(survivors) != 9:
        from collections import Counter

        breakdown = Counter(int(model.geom_type[g]) for g in survivors)
        raise RuntimeError(
            "simplify_collisions expected exactly 8 foot spheres + 1 ground plane to remain "
            f"collidable, got {len(spheres)} spheres / {len(planes)} planes / {len(survivors)} "
            f"total after disabling {disabled} geoms.\n"
            f"  surviving geom_type -> count: {dict(breakdown)}\n"
            f"  types this disables: {_DISABLED_COLLISION_TYPES} "
            f"(MESH={int(mujoco.mjtGeom.mjGEOM_MESH)}, CYLINDER={int(mujoco.mjtGeom.mjGEOM_CYLINDER)})\n"
            "If `disabled` is 0, the type comparison is not matching in this mujoco build. "
            "Otherwise the robot MJCF's foot collision geometry has changed -- re-check which geoms "
            "carry ground contact before using this backend."
        )
    return disabled


class G1MjxEnv:
    """``num_envs`` G1 simulations stepped as one batched MJX computation."""

    def __init__(self, cfg: EnvCfg):
        if cfg.domain_rand.terrain_enabled:
            raise ValueError(
                "terrain_enabled=True is not supported by the MJX backend: MJX has no heightfield "
                "collision, and scene_mjx.xml uses a plane floor. Use the MuJoCo backend for "
                "terrain DR, or keep terrain off (it is off by default and stays off until "
                "flat-ground walking works)."
            )

        self.cfg = cfg
        self.num_envs = n = cfg.num_envs
        self.rng = np.random.default_rng()

        scene = cfg.scene_xml
        if scene.endswith("scene_train.xml"):
            scene = scene.replace("scene_train.xml", "scene_mjx.xml")
        print(f"[mjx] loading {scene}")
        self.mj_model = mujoco.MjModel.from_xml_path(scene)
        self.mj_model.opt.timestep = cfg.sim_dt
        disabled = simplify_collisions(self.mj_model)
        print(f"[mjx] disabled {disabled} mesh/cylinder collision geoms -> 8 foot-sphere contact pairs")

        # ---- index bookkeeping (identical semantics to mujoco_env) ----
        m = self.mj_model
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in robot.ACTUATOR_NAMES]
        )
        self.torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot.TORSO_BODY_NAME)
        self.pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot.PELVIS_BODY_NAME)
        self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, robot.FLOOR_GEOM_NAME)
        free_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, robot.FREE_JOINT_NAME)
        self.free_qpos_adr = int(m.jnt_qposadr[free_jid])
        self.free_qvel_adr = int(m.jnt_dofadr[free_jid])
        self.joint_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x)] for x in robot.JOINT_NAMES]
        )
        self.joint_qvel_adr = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x)] for x in robot.JOINT_NAMES]
        )
        self.foot_body_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in robot.FOOT_BODY_NAMES
        ]
        # Which surviving collision geoms belong to which foot, for contact attribution.
        self.foot_geom_ids = []
        for bid in self.foot_body_ids:
            ids = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            self.foot_geom_ids.append(np.array(ids))

        # Soft joint limits for the dof_pos_limits reward. Computed identically to
        # mujoco_env.G1MultiEnv so the term means exactly the same thing on both backends.
        jnt_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x) for x in robot.JOINT_NAMES]
        )
        lower, upper = m.jnt_range[jnt_ids, 0], m.jnt_range[jnt_ids, 1]
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        self.soft_dof_lower = mid - half * cfg.soft_dof_pos_limit
        self.soft_dof_upper = mid + half * cfg.soft_dof_pos_limit

        self.default_torso_mass = float(m.body_mass[self.torso_body_id])
        self.default_torso_ipos = m.body_ipos[self.torso_body_id].copy()

        gains = robot.ActionScaling.build()
        self.kp = np.array(gains.kp)
        self.kd = np.array(gains.kd)
        self.action_scale = np.array(gains.action_scale)
        self.torque_limits = np.abs(m.actuator_ctrlrange[self.actuator_ids]).max(axis=1)
        self.default_dof_pos = np.array([robot.DEFAULT_JOINT_ANGLES[x] for x in robot.JOINT_NAMES])

        # ---- MJX model, batched over the fields domain randomization touches ----
        # put_model uploads one shared model; body_mass / body_ipos / geom_friction are then given
        # a leading env axis so each environment can carry its own payload and floor friction. Every
        # other field stays shared (in_axes=None) so nothing is duplicated num_envs times on device.
        self.mjx_model = mjx.put_model(m)
        self._batched_fields = ("body_mass", "body_ipos", "geom_friction")
        self.mjx_model = self.mjx_model.tree_replace(
            {f: jnp.broadcast_to(getattr(self.mjx_model, f), (n, *getattr(self.mjx_model, f).shape))
             for f in self._batched_fields}
        )

        data = mjx.make_data(m)
        self.mjx_data = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), data)

        # Per-env DR bookkeeping mirrored on the host, so gather_state's privileged critic terms
        # match the MuJoCo backend exactly.
        self.payload_mass = np.zeros(n)
        self.payload_com = np.zeros((n, 3))
        self.floor_friction = np.zeros(n)
        self.push_timer = np.zeros(n)
        self.next_push_time = np.zeros(n)

        self._substeps = cfg.substeps
        self._build_jitted()

    # ------------------------------------------------------------------ jit
    def _build_jitted(self) -> None:
        """Compile the batched multi-substep step, the forward pass, and contact-force extraction.

        vmap in_axes for the model is expressed as a pytree of the same structure carrying 0 for
        the three domain-randomized fields and None everywhere else, so only those three are
        materialized per-env on device and the rest of the model stays shared.
        """
        in_axes_model = jax.tree.map(lambda _: None, self.mjx_model)
        in_axes_model = in_axes_model.tree_replace({f: 0 for f in self._batched_fields})
        self._in_axes_model = in_axes_model

        def _one_env_step(model, data, ctrl):
            data = data.replace(ctrl=ctrl)
            for _ in range(self._substeps):
                data = mjx.step(model, data)
            return data

        # contact_force indexes `contact.efc_address[contact_id]` with a plain Python int, so it
        # cannot be vmapped over contact_id (a traced index raises TracerArrayConversionError).
        # The buffer size is static and small on this model -- collision is reduced to 8
        # foot-sphere-vs-plane pairs -- so unroll over it at trace time instead.
        self._ncon = int(self.mjx_data.contact.dist.shape[-1])

        def _one_env_contact_forces(model, data):
            return jnp.stack([_mjx_contact_force(model, data, i) for i in range(self._ncon)])

        self._step_fn = jax.jit(jax.vmap(_one_env_step, in_axes=(in_axes_model, 0, 0)))
        self._forward_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(in_axes_model, 0)))
        self._contact_force_fn = jax.jit(
            jax.vmap(_one_env_contact_forces, in_axes=(in_axes_model, 0))
        )

    # ---------------------------------------------------------------- reset
    def reset_idx(self, env_ids: np.ndarray, dr: DomainRandCfg) -> None:
        """Re-randomize and re-pose the given environments.

        Host-side construction then a single device write, rather than a jitted scatter: resets are
        infrequent (a few per step at most) and this keeps the randomization logic byte-identical to
        ``mujoco_env.reset_idx`` -- same rng calls in the same order, so the two backends see the
        same DR draws for a given seed.
        """
        env_ids = np.atleast_1d(np.asarray(env_ids, dtype=int))
        if env_ids.size == 0:
            return
        k = env_ids.size

        payload = self.rng.uniform(*dr.payload_mass_range, size=k)
        com = self.rng.uniform(dr.payload_com_offset_range[0], dr.payload_com_offset_range[1], size=(k, 3))
        friction = self.rng.uniform(*dr.friction_range, size=k)
        yaw = self.rng.uniform(*dr.init_yaw_range, size=k)
        noise = self.rng.uniform(-dr.init_joint_noise, dr.init_joint_noise, size=(k, robot.NUM_JOINTS))

        self.payload_mass[env_ids] = payload
        self.payload_com[env_ids] = com
        self.floor_friction[env_ids] = friction
        self.push_timer[env_ids] = 0.0
        self.next_push_time[env_ids] = self.rng.uniform(*dr.push_interval_range, size=k)

        # --- model fields ---
        # .copy() is required: np.asarray on a JAX array gives a read-only view of device memory.
        mass = np.asarray(self.mjx_model.body_mass).copy()
        ipos = np.asarray(self.mjx_model.body_ipos).copy()
        fric = np.asarray(self.mjx_model.geom_friction).copy()
        mass[env_ids, self.torso_body_id] = self.default_torso_mass + payload
        ipos[env_ids, self.torso_body_id] = self.default_torso_ipos + com
        fric[env_ids, self.floor_geom_id, 0] = friction
        self.mjx_model = self.mjx_model.tree_replace(
            {"body_mass": jnp.asarray(mass), "body_ipos": jnp.asarray(ipos),
             "geom_friction": jnp.asarray(fric)}
        )

        # --- state ---
        qpos = np.asarray(self.mjx_data.qpos).copy()
        qvel = np.asarray(self.mjx_data.qvel).copy()
        qpos[env_ids] = self.mj_model.qpos0
        # Ground is a plane at z=0 on this backend, so ground_z is always 0 (the MuJoCo backend
        # queries the hfield for it). Same SPAWN_HEIGHT_CLEARANCE, for the same reason.
        qpos[env_ids, self.free_qpos_adr + 0] = 0.0
        qpos[env_ids, self.free_qpos_adr + 1] = 0.0
        qpos[env_ids, self.free_qpos_adr + 2] = robot.STANDING_BASE_HEIGHT + robot.SPAWN_HEIGHT_CLEARANCE
        qpos[env_ids, self.free_qpos_adr + 3] = np.cos(yaw / 2)
        qpos[env_ids, self.free_qpos_adr + 4] = 0.0
        qpos[env_ids, self.free_qpos_adr + 5] = 0.0
        qpos[env_ids, self.free_qpos_adr + 6] = np.sin(yaw / 2)
        qpos[np.ix_(env_ids, self.joint_qpos_adr)] = self.default_dof_pos[None, :] + noise
        qvel[env_ids] = 0.0

        self.mjx_data = self.mjx_data.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
        self.mjx_data = self._forward_fn(self.mjx_model, self.mjx_data)

    # ----------------------------------------------------------------- step
    def step(self, torques: np.ndarray, dr: DomainRandCfg) -> None:
        """Apply ``torques`` (num_envs, 29), advance ``substeps`` physics steps, apply pushes."""
        ctrl = np.zeros((self.num_envs, self.mj_model.nu))
        ctrl[:, self.actuator_ids] = torques
        self.mjx_data = self._step_fn(self.mjx_model, self.mjx_data, jnp.asarray(ctrl))

        # Scheduled velocity pushes. Host-side like the MuJoCo backend: same rng draw order, and it
        # only touches the 6 free-joint dofs.
        self.push_timer += self.cfg.control_dt
        due = np.nonzero(self.push_timer >= self.next_push_time)[0]
        if due.size:
            impulse = self.rng.uniform(dr.push_lin_vel_range[0], dr.push_lin_vel_range[1], size=(due.size, 2))
            qvel = np.asarray(self.mjx_data.qvel).copy()
            qvel[due, self.free_qvel_adr : self.free_qvel_adr + 2] += impulse
            self.mjx_data = self.mjx_data.replace(qvel=jnp.asarray(qvel))
            self.push_timer[due] = 0.0
            self.next_push_time[due] = self.rng.uniform(*dr.push_interval_range, size=due.size)

    # --------------------------------------------------------------- torques
    def compute_torques(self, actions: np.ndarray) -> np.ndarray:
        """PD control, identical formula to the MuJoCo backend."""
        targets = self.default_dof_pos[None, :] + actions * self.action_scale[None, :]
        joint_pos = np.asarray(self.mjx_data.qpos)[:, self.joint_qpos_adr]
        joint_vel = np.asarray(self.mjx_data.qvel)[:, self.joint_qvel_adr]
        torque = self.kp[None, :] * (targets - joint_pos) - self.kd[None, :] * joint_vel
        return np.clip(torque, -self.torque_limits[None, :], self.torque_limits[None, :])

    # ---------------------------------------------------------------- state
    def gather_state(self) -> dict[str, np.ndarray]:
        """Same keys, shapes and units as ``G1MultiEnv.gather_state``, as host NumPy arrays.

        Read straight off qpos/qvel/xpos/xquat/cvel rather than through MuJoCo sensors: MJX's
        sensor coverage is partial, and this model carries 95 sensors. The quantities are the same
        ones the sensors would have reported.
        """
        d = self.mjx_data
        n = self.num_envs

        qpos = np.asarray(d.qpos)
        qvel = np.asarray(d.qvel)
        joint_pos = qpos[:, self.joint_qpos_adr]
        joint_vel = qvel[:, self.joint_qvel_adr]
        # actuator_force is MJX's applied actuator force, the analogue of the torque sensors.
        joint_torque = np.asarray(d.actuator_force)[:, self.actuator_ids]

        base_quat = qpos[:, self.free_qpos_adr + 3 : self.free_qpos_adr + 7]
        base_lin_vel = qvel[:, self.free_qvel_adr : self.free_qvel_adr + 3]
        # Free-joint angular velocity is in the WORLD frame in qvel, but the gyro sensor the MuJoCo
        # backend reads reports body frame -- rotate so the two backends agree.
        ang_vel_world = qvel[:, self.free_qvel_adr + 3 : self.free_qvel_adr + 6]
        ang_vel = quat_rotate_inverse(base_quat, ang_vel_world)

        world_gravity = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
        projected_gravity = quat_rotate_inverse(base_quat, world_gravity)
        base_height = qpos[:, self.free_qpos_adr + 2]

        xpos = np.asarray(d.xpos)
        foot_pos = np.stack([xpos[:, bid] for bid in self.foot_body_ids], axis=1)
        # cvel is [angular(3), linear(3)] per body, in the world frame.
        cvel = np.asarray(d.cvel)
        foot_lin_vel = np.stack([cvel[:, bid, 3:6] for bid in self.foot_body_ids], axis=1)

        normal, tangential, cop_xy = self._contact_forces()
        foot_contact = normal > CONTACT_FORCE_THRESHOLD

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
            "ang_vel": ang_vel,
            "projected_gravity": projected_gravity,
            "base_quat": base_quat,
            "base_lin_vel": base_lin_vel,
            "base_height": base_height,
            "foot_normal_force": normal,
            "foot_tangential_force": tangential,
            "foot_contact": foot_contact,
            # Always False: mesh collision is off on this backend, so the pelvis has no collision
            # geometry to register a floor contact with. See the module docstring -- this fired 0
            # times in 2000 iterations of v10, so it is a documented no-op rather than a silent one.
            "pelvis_contact": np.zeros(n, dtype=bool),
            "cop_xy": cop_xy,
            "foot_pos": foot_pos,
            "foot_lin_vel": foot_lin_vel,
            "payload_mass": self.payload_mass.copy(),
            "payload_com": self.payload_com.copy(),
            "floor_friction": self.floor_friction.copy(),
        }

    def _contact_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-foot normal / tangential force and centre of pressure, from MJX's contact array.

        MJX keeps a fixed-size contact buffer; inactive slots carry ``dist >= 0`` and zero force.
        ``contact_force`` returns a 6-vector per contact in the contact frame, same convention as
        ``mj_contactForce`` (index 0 normal, 1-2 tangential), so the thresholding and CoP weighting
        match the MuJoCo backend term for term.
        """
        d = self.mjx_data
        n = self.num_envs
        normal = np.zeros((n, 2))
        tangential = np.zeros((n, 2))
        cop_num = np.zeros((n, 2))

        forces = np.asarray(self._contact_force_fn(self.mjx_model, d))  # (n, ncon, 6)
        geom = np.asarray(d.contact.geom)  # (n, ncon, 2)
        pos = np.asarray(d.contact.pos)  # (n, ncon, 3)
        active = np.asarray(d.contact.dist) < 0  # (n, ncon)

        for foot_idx, gids in enumerate(self.foot_geom_ids):
            mine = active & (np.isin(geom[..., 0], gids) | np.isin(geom[..., 1], gids))
            f_n = np.abs(forces[..., 0]) * mine
            f_t = np.hypot(forces[..., 1], forces[..., 2]) * mine
            normal[:, foot_idx] = f_n.sum(axis=-1)
            tangential[:, foot_idx] = f_t.sum(axis=-1)
            # CoP is the normal-force-weighted mean contact position across BOTH feet, matching
            # mujoco_env (which accumulates weighted_pos/total_normal over every floor contact).
            cop_num += (pos[..., :2] * f_n[..., None]).sum(axis=1)

        total = normal.sum(axis=-1, keepdims=True)
        cop_xy = np.where(total > 1e-6, cop_num / np.maximum(total, 1e-6), 0.0)
        return normal, tangential, cop_xy
