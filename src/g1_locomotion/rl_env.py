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

    def __init__(self, cfg: EnvCfg, auto_reset: bool = True):
        self.cfg = cfg
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
        self.feet_air_time = np.zeros((n, 2))
        self.last_contact = np.zeros((n, 2), dtype=bool)
        self.command_resample_steps = round(10.0 / cfg.control_dt)

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

    def _build_obs(self) -> TensorDict:
        s = self._state
        actor_obs = np.concatenate(
            [
                s["joint_pos"] - self.sim.default_dof_pos[None, :],
                s["joint_vel"],
                s["projected_gravity"],
                s["ang_vel"],
                self.prev_actions,
                self.commands,
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

        foot_contact = self._state["foot_contact"]
        self.feet_air_time += self.cfg.control_dt
        new_contact = foot_contact & (~self.last_contact)
        reward, reward_terms = rewards.compute_reward(
            self._state, self.commands, actions_np, self.prev_actions,
            self.feet_air_time, new_contact, self.cfg.reward_scales,
        )
        self.feet_air_time = np.where(foot_contact, 0.0, self.feet_air_time)
        self.last_contact = foot_contact

        height_fail = self._state["base_height"] < self.cfg.fall_height_threshold
        tilt_fail = self._state["projected_gravity"][:, 2] > -self.cfg.fall_gravity_threshold
        fail = height_fail | tilt_fail
        timeout = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        done = fail | timeout

        self.prev_actions = actions_np.copy()

        done_ids = np.nonzero(done)[0] if self.auto_reset else np.empty(0, dtype=np.int64)
        if len(done_ids) > 0:
            self.sim.reset_idx(done_ids, self.cfg.domain_rand)
            self._resample_commands(done_ids)
            self.prev_actions[done_ids] = 0.0
            self.feet_air_time[done_ids] = 0.0
            self.last_contact[done_ids] = False
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
        extras = {
            "time_outs": torch.from_numpy(timeout.astype(np.float32)).to(self.device),
            "log": {f"reward/{k}": float(np.mean(v)) for k, v in reward_terms.items()},
        }
        return obs, reward_t, done_t, extras
