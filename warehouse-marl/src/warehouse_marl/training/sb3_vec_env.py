from typing import Any, Optional, Sequence

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from warehouse_marl.env.warehouse_env import WarehouseEnv


class WarehouseVecEnv(VecEnv):
    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.vehicle_ids = env.possible_agents
        first_vehicle = self.vehicle_ids[0]
        super().__init__(
            num_envs=len(self.vehicle_ids),
            observation_space=env.observation_space(first_vehicle),
            action_space=env.action_space(first_vehicle),
        )
        self._actions: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return self._stack(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        actions = {vehicle: int(self._actions[i]) for i, vehicle in enumerate(self.vehicle_ids)}
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        observations = self._stack(obs)
        reward_array = np.array([rewards[v] for v in self.vehicle_ids], dtype=np.float32)
        dones = np.array(
            [terminated[v] or truncated[v] for v in self.vehicle_ids], dtype=bool
        )

        info_list: list[dict[str, Any]] = []
        for i, vehicle in enumerate(self.vehicle_ids):
            info = dict(infos[vehicle])
            if dones[i]:
                info["terminal_observation"] = observations[i]
                info["TimeLimit.truncated"] = truncated[vehicle] and not terminated[vehicle]
            info_list.append(info)

        if dones.all():
            reset_obs, _ = self.env.reset()
            observations = self._stack(reset_obs)

        return observations, reward_array, dones, info_list

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self.env, attr_name) for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        method = getattr(self.env, method_name)
        return [method(*args, **kwargs) for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._indices(indices)]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None for _ in range(self.num_envs)]

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        return [seed for _ in range(self.num_envs)]

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def _stack(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(obs[v], dtype=np.float32) for v in self.vehicle_ids])
