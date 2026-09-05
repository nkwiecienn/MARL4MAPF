import warnings
from typing import Dict, List, Optional, Sequence
from warehouse_marl.env.depots import allocate_depot_cells
from warehouse_marl.env.routing import Coord, build_sequence
from pogema import GridConfig, pogema_v0


class WarehouseEnv:
    render_mode = None

    def __init__(
        self,
        grid_map: str,
        orders: Dict[str, Sequence[Coord]],
        depot_zones: Dict[str, Sequence[Coord]],
        vehicle_depot_zone: Dict[str, str],
        order_strategy: str = "nearest",
        obs_radius: int = 5,
        collision_system: str = "soft",
        max_episode_steps: int = 512,
        goal_reward: float = 1.0,
        step_penalty: float = 0.01,
        completion_bonus: float = 5.0,
        seed: Optional[int] = None,
        **grid_kwargs,
    ) -> None:
        self.vehicle_ids: List[str] = list(vehicle_depot_zone.keys())
        self.depot_zones = {zid: [tuple(c) for c in cells] for zid, cells in depot_zones.items()}
        self.vehicle_depot_zone = dict(vehicle_depot_zone)

        self.depots = allocate_depot_cells(
            depot_zones=self.depot_zones,
            vehicle_depot_zone=self.vehicle_depot_zone,
            seed=seed,
        )
        self.vehicle_depot_cell = self.depots

        self.sequences: Dict[str, List[List[int]]] = {
            v: build_sequence(self.depots[v], orders[v], order_strategy) for v in self.vehicle_ids
        }

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus

        self.grid_config = GridConfig(
            map=grid_map,
            num_agents=len(self.vehicle_ids),
            agents_xy=[self.depots[v] for v in self.vehicle_ids],
            targets_xy=[self.sequences[v] for v in self.vehicle_ids],
            on_target="restart",
            collision_system=collision_system,
            obs_radius=obs_radius,
            max_episode_steps=max_episode_steps,
            seed=seed,
            integration="PettingZoo",
            **grid_kwargs,
        )
        self._env = pogema_v0(grid_config=self.grid_config)

        self._vehicle_to_player = {v: f"player_{i}" for i, v in enumerate(self.vehicle_ids)}
        self._player_to_vehicle = {p: v for v, p in self._vehicle_to_player.items()}

        self._goal_hits: Dict[str, int] = {}
        self._finished: Dict[str, bool] = {}
        self._steps = 0

    @property
    def possible_agents(self) -> List[str]:
        return list(self.vehicle_ids)

    @property
    def agents(self) -> List[str]:
        return list(self.vehicle_ids)

    def observation_space(self, vehicle_id: str):
        return self._env.observation_space(self._vehicle_to_player[vehicle_id])

    def action_space(self, vehicle_id: str):
        return self._env.action_space(self._vehicle_to_player[vehicle_id])

    def tour_length(self, vehicle_id: str) -> int:
        """Number of goals (orders + depot return) this vehicle must complete."""
        return len(self.sequences[vehicle_id])

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._goal_hits = {v: 0 for v in self.vehicle_ids}
        self._finished = {v: False for v in self.vehicle_ids}
        self._steps = 0
        return self._remap(obs), self._remap(info)

    def step(self, actions: Dict[str, int]):
        pz_actions = {
            self._vehicle_to_player[v]: (0 if self._finished[v] else int(actions.get(v, 0)))
            for v in self.vehicle_ids
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cycling back to the beginning.*")
            obs, raw_rewards, _, pz_truncated, infos = self._env.step(pz_actions)

        self._steps += 1
        obs = self._remap(obs)
        raw_rewards = self._remap(raw_rewards)
        pz_truncated = self._remap(pz_truncated)
        infos = self._remap(infos)

        rewards: Dict[str, float] = {}
        for v in self.vehicle_ids:
            if self._finished[v]:
                rewards[v] = 0.0
                continue

            reward = -self.step_penalty
            if raw_rewards[v] > 0:
                self._goal_hits[v] += 1
                reward += self.goal_reward
                if self._goal_hits[v] >= len(self.sequences[v]):
                    self._finished[v] = True
                    reward += self.completion_bonus
            rewards[v] = reward

        episode_over = all(self._finished.values())
        timed_out = bool(pz_truncated) and all(pz_truncated.values())

        terminated = {v: episode_over for v in self.vehicle_ids}
        truncated = {v: (timed_out and not episode_over) for v in self.vehicle_ids}

        for v in self.vehicle_ids:
            infos[v] = dict(infos[v])
            infos[v].update(
                finished=self._finished[v],
                goals_completed=self._goal_hits[v],
                tour_length=len(self.sequences[v]),
            )
        if episode_over or timed_out:
            for v in self.vehicle_ids:
                infos[v]["episode_solved"] = episode_over
                infos[v]["episode_steps"] = self._steps

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    def _remap(self, d: dict) -> dict:
        return {self._player_to_vehicle[k]: v for k, v in d.items()}
