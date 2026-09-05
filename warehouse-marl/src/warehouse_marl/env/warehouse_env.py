import warnings
from typing import Optional, Sequence

from pogema import GridConfig, pogema_v0

from warehouse_marl.env.depots import allocate_depot_cells
from warehouse_marl.env.routing import Coord, build_sequence


class WarehouseEnv:
    render_mode = None

    def __init__(
        self,
        grid_map: str,
        orders: dict[str, Sequence[Coord]],
        depot_zones: dict[str, Sequence[Coord]],
        vehicle_depot_zone: dict[str, str],
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
        self.vehicle_ids = list(vehicle_depot_zone)
        self.depot_zones = {zone: [tuple(cell) for cell in cells] for zone, cells in depot_zones.items()}
        self.vehicle_depot_zone = dict(vehicle_depot_zone)

        self.depots = allocate_depot_cells(
            depot_zones=self.depot_zones,
            vehicle_depot_zone=self.vehicle_depot_zone,
            seed=seed,
        )
        self.sequences = {
            vehicle: build_sequence(self.depots[vehicle], orders[vehicle], order_strategy)
            for vehicle in self.vehicle_ids
        }

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus

        self.grid_config = GridConfig(
            map=grid_map,
            num_agents=len(self.vehicle_ids),
            agents_xy=[self.depots[vehicle] for vehicle in self.vehicle_ids],
            targets_xy=[self.sequences[vehicle] for vehicle in self.vehicle_ids],
            on_target="restart",
            collision_system=collision_system,
            obs_radius=obs_radius,
            max_episode_steps=max_episode_steps,
            seed=seed,
            integration="PettingZoo",
            **grid_kwargs,
        )
        self._env = pogema_v0(grid_config=self.grid_config)

        self._player_of = {vehicle: f"player_{i}" for i, vehicle in enumerate(self.vehicle_ids)}
        self._vehicle_of = {player: vehicle for vehicle, player in self._player_of.items()}

        self._goal_hits: dict[str, int] = {}
        self._finished: dict[str, bool] = {}
        self._steps = 0

    @property
    def possible_agents(self) -> list[str]:
        return list(self.vehicle_ids)

    @property
    def agents(self) -> list[str]:
        return list(self.vehicle_ids)

    @property
    def goal_hits(self) -> dict[str, int]:
        return dict(self._goal_hits)

    def observation_space(self, vehicle_id: str):
        return self._env.observation_space(self._player_of[vehicle_id])

    def action_space(self, vehicle_id: str):
        return self._env.action_space(self._player_of[vehicle_id])

    def tour_length(self, vehicle_id: str) -> int:
        """Number of goals (orders + depot return) this vehicle must complete."""
        return len(self.sequences[vehicle_id])

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._goal_hits = {vehicle: 0 for vehicle in self.vehicle_ids}
        self._finished = {vehicle: False for vehicle in self.vehicle_ids}
        self._steps = 0
        return self._by_vehicle(obs), self._by_vehicle(info)

    def step(self, actions: dict[str, int]):
        player_actions = {
            self._player_of[vehicle]: 0 if self._finished[vehicle] else int(actions.get(vehicle, 0))
            for vehicle in self.vehicle_ids
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cycling back to the beginning.*")
            obs, raw_rewards, _, raw_truncated, infos = self._env.step(player_actions)

        self._steps += 1
        obs = self._by_vehicle(obs)
        raw_rewards = self._by_vehicle(raw_rewards)
        raw_truncated = self._by_vehicle(raw_truncated)
        infos = self._by_vehicle(infos)

        rewards = {}
        for vehicle in self.vehicle_ids:
            if self._finished[vehicle]:
                rewards[vehicle] = 0.0
                continue

            reward = -self.step_penalty
            if raw_rewards[vehicle] > 0:
                self._goal_hits[vehicle] += 1
                reward += self.goal_reward
                if self._goal_hits[vehicle] >= self.tour_length(vehicle):
                    self._finished[vehicle] = True
                    reward += self.completion_bonus
            rewards[vehicle] = reward

        episode_over = all(self._finished.values())
        timed_out = bool(raw_truncated) and all(raw_truncated.values())

        terminated = {vehicle: episode_over for vehicle in self.vehicle_ids}
        truncated = {vehicle: timed_out and not episode_over for vehicle in self.vehicle_ids}

        for vehicle in self.vehicle_ids:
            info = dict(infos[vehicle])
            info.update(
                finished=self._finished[vehicle],
                goals_completed=self._goal_hits[vehicle],
                tour_length=self.tour_length(vehicle),
            )
            if episode_over or timed_out:
                info["episode_solved"] = episode_over
                info["episode_steps"] = self._steps
            infos[vehicle] = info

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    def _by_vehicle(self, by_player: dict) -> dict:
        return {self._vehicle_of[player]: value for player, value in by_player.items()}
