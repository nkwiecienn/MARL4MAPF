"""Adapter exposing WarehouseEnv to Stable-Baselines3.

Why not SuperSuit?
------------------
The usual recipe is
``supersuit.pettingzoo_env_to_vec_env_v1(...)`` + SB3. That combination is
currently uninstallable here: ``pogema==1.4.0`` pins ``gymnasium==0.28.1``
exactly, while ``supersuit>=3.11`` requires ``gymnasium>=1.0``. Rather than
fight it, this module implements the same idea directly -- it is a small,
readable file with no extra dependencies.

What it does
------------
Each *vehicle* becomes one "environment" in an SB3 ``VecEnv`` of size
``num_vehicles``. SB3's PPO then trains a **single shared policy** on
transitions pooled from all vehicles. That is parameter-shared independent
PPO (IPPO): agents act on their own local observations with no
communication, but learn one common policy. It's the standard first
baseline for homogeneous-agent MAPF problems.

Important detail: all vehicles share one episode boundary. WarehouseEnv
already reports ``terminated``/``truncated`` identically for every vehicle
(finished vehicles are frozen, not removed), so all sub-envs reset together
and stay in lockstep -- which is exactly what SB3's VecEnv assumes.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from warehouse_marl.env.warehouse_env import WarehouseEnv


class WarehouseVecEnv(VecEnv):
    """Presents the N vehicles of one WarehouseEnv as an N-env SB3 VecEnv."""

    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.vehicle_ids: List[str] = env.possible_agents
        sample = self.vehicle_ids[0]
        super().__init__(
            num_envs=len(self.vehicle_ids),
            observation_space=env.observation_space(sample),
            action_space=env.action_space(sample),
        )
        self._actions: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None

    # -- VecEnv API -----------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._last_obs = self._stack(obs)
        return self._last_obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        action_dict = {v: int(self._actions[i]) for i, v in enumerate(self.vehicle_ids)}
        obs, rewards, terminated, truncated, infos = self.env.step(action_dict)

        obs_arr = self._stack(obs)
        reward_arr = np.array([rewards[v] for v in self.vehicle_ids], dtype=np.float32)
        done_arr = np.array(
            [terminated[v] or truncated[v] for v in self.vehicle_ids], dtype=bool
        )

        info_list: List[Dict[str, Any]] = []
        for i, v in enumerate(self.vehicle_ids):
            info = dict(infos[v])
            if done_arr[i]:
                # SB3 convention: stash the true final observation, because
                # `obs` returned below is the first obs of the NEXT episode.
                info["terminal_observation"] = obs_arr[i]
                info["TimeLimit.truncated"] = bool(truncated[v]) and not bool(terminated[v])
            info_list.append(info)

        if done_arr.all():
            reset_obs, _ = self.env.reset()
            obs_arr = self._stack(reset_obs)

        self._last_obs = obs_arr
        return obs_arr, reward_arr, done_arr, info_list

    def close(self) -> None:
        self.env.close()

    # -- introspection required by the VecEnv interface -----------------------

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self.env, attr_name) for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        method = getattr(self.env, method_name)
        return [method(*args, **kwargs) for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._indices(indices)]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None for _ in range(self.num_envs)]

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return [seed for _ in range(self.num_envs)]

    # -- helpers ---------------------------------------------------------------

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def _stack(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(obs_dict[v], dtype=np.float32) for v in self.vehicle_ids])
