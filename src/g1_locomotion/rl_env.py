"""rsl_rl-compatible VecEnv wrapper around :class:`~g1_locomotion.mujoco_env.G1MultiEnv`.

Implements the asymmetric actor-critic observation split from the project spec by returning
two TensorDict groups: ``actor_obs`` (deployable, proprioceptive) and ``critic_obs``
(privileged-only extras). The training config maps ``obs_groups = {"actor": ["actor_obs"],
"critic": ["actor_obs", "critic_obs"]}`` so the critic sees both concatenated.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from . import robot, rewards
from .config import EnvCfg
from .mujoco_env import G1MultiEnv


class G1VecEnv:
    """VecEnv adapter matching ``rsl_rl.env.VecEnv``'s duck-typed contract."""

    def __init__(self, cfg: EnvCfg, auto_reset: bool = True, backend: str = "mujoco"):
        """``backend`` selects the physics implementation ONLY.

        Everything above the physics -- observations, actions, rewards, terminations, the gait
        clock, domain randomization ranges, the asymmetric actor/critic split, and the PPO config --
        is shared verbatim between the two. That is deliberate: if an MJX run behaves differently
        from a MuJoCo one, the difference is the backend, not a silently forked reward or obs
        definition. Both sims expose the same reset_idx/step/compute_torques/gather_state contract,
        so this class never learns which one it is driving.

        "mujoco" -- mujoco_env.G1MultiEnv. Serial Python loop over N independent MjModel/MjData
                    pairs. Correct and well-tested, but throughput is pinned at ~874 timesteps/s
                    regardless of num_envs, and memory is ~77 MB PER ENV (each env holds its own
                    copy of the 185k-vertex mesh set), so 512 envs already needs 39 GB.
        "mjx"    -- mjx_env.G1MjxEnv. One shared model, vmapped over envs on GPU. ~0.064 MB per env
                    (1200x less), so env count stops being a memory question. Requires jax +
                    mujoco-mjx, which the default environment does not install.
        """
        if backend not in ("mujoco", "mjx"):
            raise ValueError(f"backend must be 'mujoco' or 'mjx', got {backend!r}")
        self.backend = backend
        self.cfg = cfg
        # rewards.REWARD_DT is applied to every reward weight and is hard-coded so the scaling
        # lives in one place; if control_dt is ever retuned, that constant has to move with it.
        assert abs(rewards.REWARD_DT - cfg.control_dt) < 1e-12, (
            f"rewards.REWARD_DT ({rewards.REWARD_DT}) must equal EnvCfg.control_dt "
            f"({cfg.control_dt}) -- reward weights are per-second rates scaled by the control step."
        )
        if backend == "mjx":
            # Imported lazily: mjx_env raises ImportError without jax/mujoco-mjx, and the MuJoCo
            # path must keep working in an environment that has neither.
            from .mjx_env import G1MjxEnv

            self.sim = G1MjxEnv(cfg)
        else:
            self.sim = G1MultiEnv(cfg)
        self.num_envs = cfg.num_envs
        self.num_actions = robot.NUM_JOINTS
        self.max_episode_length = cfg.max_episode_length_steps
        self.device = cfg.device
        # Whether a fallen/timed-out env is respawned inside step(). Training always needs this on;
        # play.py can turn it off to watch one continuous attempt instead of being silently reset.
        self.auto_reset = auto_reset

        n = self.num_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=self.device)
        self.commands = np.zeros((n, 3))
        self.prev_actions = np.zeros((n, robot.NUM_JOINTS))
        self.prev_joint_vel = np.zeros((n, robot.NUM_JOINTS))
        self.command_resample_steps = round(10.0 / cfg.control_dt)

        # Per-episode reward-term sums, mean episode length, and termination-reason counts. Kept so
        # a single run gives per-term attribution instead of needing staged runs to tell which term
        # did what. `_last_ep_sums` holds the most recent completed-episode means so the log keys
        # stay present on steps where no episode happened to finish.
        self.episode_sums: dict[str, np.ndarray] = {}
        self._last_ep_sums: dict[str, float] = {}
        self._last_ep_length = 0.0

        all_ids = np.arange(n)
        self.sim.reset_idx(all_ids, cfg.domain_rand)
        self._resample_commands(all_ids)
        self._state = self.sim.gather_state()

    # --------------------------------------------------------------- helpers
    def _resample_commands(self, env_ids: np.ndarray) -> None:
        lo, hi = self.cfg.command_lin_vel_range
        alo, ahi = self.cfg.command_ang_vel_range
        self.commands[env_ids, 0] = self.sim.rng.uniform(lo, hi, size=len(env_ids))
        self.commands[env_ids, 1] = self.sim.rng.uniform(lo, hi, size=len(env_ids)) * 0.5
        self.commands[env_ids, 2] = self.sim.rng.uniform(alo, ahi, size=len(env_ids))

    def _gait_phase(self) -> np.ndarray:
        """Normalized gait-clock phase in [0, 1), derived from the episode step counter.

        Deriving it from ``episode_length_buf`` rather than keeping separate state means it resets
        with the episode by construction and cannot drift out of sync with it.
        """
        t = self.episode_length_buf.cpu().numpy() * self.cfg.control_dt
        return (t / self.cfg.gait_period_s) % 1.0

    def _expected_stance(self, phase: np.ndarray) -> np.ndarray:
        """(num_envs, 2) bool: which feet the clock says should be planted right now.

        Left leg runs at ``phase``, right leg at ``phase + 0.5`` (exact antiphase). A leg is
        scheduled for stance over the first ``gait_stance_duty`` of its own cycle.
        """
        duty = self.cfg.gait_stance_duty
        legs = np.stack([phase, (phase + 0.5) % 1.0], axis=-1)
        return legs < duty

    def _build_obs(self) -> TensorDict:
        s = self._state
        # Gait-clock phase as sin/cos, so it is continuous across the 1->0 wrap. These two terms are
        # what make the clocked `contact` reward learnable at all -- without them the policy is
        # being scored against a schedule it cannot see. They bring actor_obs from 96 to 98 dims
        # (and critic, which concatenates actor_obs + privileged extras, from 108 to 110); nothing
        # hard-codes those numbers, rsl_rl infers them from these tensors.
        phase = self._gait_phase()
        clock = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=-1)

        actor_obs = np.concatenate(
            [
                s["joint_pos"] - self.sim.default_dof_pos[None, :],
                s["joint_vel"],
                s["projected_gravity"],
                s["ang_vel"],
                self.prev_actions,
                self.commands,
                clock,
            ],
            axis=-1,
        ).astype(np.float32)

        local_lin_vel = rewards.quat_rotate_inverse(s["base_quat"], s["base_lin_vel"])
        critic_extra = np.concatenate(
            [
                local_lin_vel,
                s["foot_normal_force"],
                s["foot_tangential_force"],
                s["payload_mass"][:, None],
                s["payload_com"],
                s["floor_friction"][:, None],
            ],
            axis=-1,
        ).astype(np.float32)

        return TensorDict(
            {
                "actor_obs": torch.from_numpy(actor_obs).to(self.device),
                "critic_obs": torch.from_numpy(critic_extra).to(self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    # ----------------------------------------------------------------- VecEnv API
    def get_observations(self) -> TensorDict:
        return self._build_obs()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().astype(np.float64)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        torques = self.sim.compute_torques(actions_np)
        self.sim.step(torques, self.cfg.domain_rand)
        self._state = self.sim.gather_state()

        self.episode_length_buf += 1

        # Joint acceleration by finite difference over one control step, for the dof_acc penalty.
        joint_acc = (self._state["joint_vel"] - self.prev_joint_vel) / self.cfg.control_dt

        expected_stance = self._expected_stance(self._gait_phase())
        reward, reward_terms = rewards.compute_reward(
            self._state, self.commands, actions_np, self.prev_actions,
            expected_stance, joint_acc,
            self.sim.default_dof_pos, self.sim.soft_dof_lower, self.sim.soft_dof_upper,
            self.cfg.reward_scales, self.cfg.only_positive_rewards,
        )

        # Termination reasons are tracked separately (and can overlap on the same step) so the logs
        # attribute *why* episodes are ending, not just that they are.
        height_fail = self._state["base_height"] < self.cfg.fall_height_threshold
        tilt_fail = self._state["projected_gravity"][:, 2] > -self.cfg.fall_gravity_threshold
        pelvis_fail = (
            self._state["pelvis_contact"]
            if self.cfg.terminate_on_pelvis_contact
            else np.zeros(self.num_envs, dtype=bool)
        )
        fail = height_fail | tilt_fail | pelvis_fail
        timeout = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        done = fail | timeout

        if not self.episode_sums:
            self.episode_sums = {k: np.zeros(self.num_envs) for k in reward_terms}
            self._last_ep_sums = {k: 0.0 for k in reward_terms}
        for k, v in reward_terms.items():
            self.episode_sums[k] += v

        self.prev_actions = actions_np.copy()
        self.prev_joint_vel = self._state["joint_vel"].copy()

        done_ids = np.nonzero(done)[0] if self.auto_reset else np.empty(0, dtype=np.int64)
        if len(done_ids) > 0:
            # Snapshot the finished episodes' term sums and lengths before zeroing them.
            for k, buf in self.episode_sums.items():
                self._last_ep_sums[k] = float(np.mean(buf[done_ids]))
                buf[done_ids] = 0.0
            self._last_ep_length = float(
                np.mean(self.episode_length_buf.cpu().numpy()[done_ids])
            )

            self.sim.reset_idx(done_ids, self.cfg.domain_rand)
            self._resample_commands(done_ids)
            self.prev_actions[done_ids] = 0.0
            self.prev_joint_vel[done_ids] = 0.0
            self.episode_length_buf[torch.from_numpy(done_ids).to(self.device)] = 0
            self._state = self.sim.gather_state()

        if self.auto_reset:
            resample_ids = np.nonzero(
                (self.episode_length_buf.cpu().numpy() % self.command_resample_steps == 0) & ~done
            )[0]
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids)

        obs = self._build_obs()
        reward_t = torch.from_numpy(reward.astype(np.float32)).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)
        log = {f"reward/{k}": float(np.mean(v)) for k, v in reward_terms.items()}
        # Per-episode sum of each term -- the attribution signal. Per-step means (above) say which
        # term is loudest right now; these say what each term actually contributed over a whole
        # episode, which is what makes terms of very different firing rates comparable (e.g. the
        # clocked `contact` reward fires every step, base_height only bites near the extremes).
        log.update({f"ep_sum/{k}": v for k, v in self._last_ep_sums.items()})
        log["episode/length_mean"] = self._last_ep_length
        # Termination attribution: mean count per step of each reason. Reasons can co-occur on one
        # step (e.g. a topple trips orientation and pelvis contact together), so these are raw flag
        # counts and do not sum to the termination count.
        log["terminations/height"] = float(np.mean(height_fail))
        log["terminations/orientation"] = float(np.mean(tilt_fail))
        log["terminations/pelvis_contact"] = float(np.mean(pelvis_fail))
        log["terminations/timeout"] = float(np.mean(timeout))

        extras = {
            "time_outs": torch.from_numpy(timeout.astype(np.float32)).to(self.device),
            "log": log,
        }
        return obs, reward_t, done_t, extras
