import warnings
from typing import Optional, Sequence

from pogema import GridConfig, pogema_v0

from warehouse_marl.env.depots import allocate_depot_cells
from warehouse_marl.env.routing import Coord, build_sequence


class WarehouseEnv:
    """Episodic warehouse VRP on top of POGEMA's lifelong (`on_target="restart"`) mode.

    POGEMA's lifelong mode is built for *endless* operation: when an agent
    exhausts its goal sequence, POGEMA silently wraps back to the first goal
    and keeps paying +1 forever. Left alone, a policy would learn to loop its
    tour for unbounded reward and the episode would never end.

    So this class owns the terminal condition: it counts goal completions per
    vehicle, freezes a vehicle once it has completed its sequence (zero reward,
    forced idle), and reports termination when every vehicle is done. Frozen
    vehicles are *not* removed -- they still occupy a cell and can block
    others, like a real parked vehicle.

    The API is PettingZoo-parallel-shaped (dicts keyed by vehicle id) but this
    is not a registered ParallelEnv; notably `agents` keeps listing finished
    vehicles, because they are still physically present.
    """

    # POGEMA renders through its own wrapper rather than a Gymnasium render
    # mode; the attribute exists only because SB3 looks it up on the env.
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
        """Snapshot of goals completed per vehicle.

        A fresh dict each call, so callers can log successive values without
        every entry aliasing the same live counter.
        """
        return dict(self._goal_hits)

    @property
    def pogema_grid(self):
        """The POGEMA grid underneath.

        Exposed so the renderer can wrap it in a recorder without reaching
        through this class's private attributes.
        """
        return self._env.pogema

    @pogema_grid.setter
    def pogema_grid(self, grid) -> None:
        self._env.pogema = grid

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
            # A finished vehicle keeps stepping (frozen), so POGEMA keeps
            # wrapping its exhausted goal list and warning about it. That is
            # expected here -- this class, not POGEMA, ends the episode.
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
            if raw_rewards[vehicle] > 0:  # POGEMA pays out iff a goal was reached
                self._goal_hits[vehicle] += 1
                reward += self.goal_reward
                if self._goal_hits[vehicle] >= self.tour_length(vehicle):
                    self._finished[vehicle] = True
                    reward += self.completion_bonus
            rewards[vehicle] = reward

        episode_over = all(self._finished.values())
        timed_out = bool(raw_truncated) and all(raw_truncated.values())

        # One episode boundary for everyone, not per-vehicle: a vehicle that
        # finishes early is frozen rather than removed, so the whole fleet
        # starts and stops together. WarehouseVecEnv relies on this to keep its
        # sub-envs in lockstep.
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
