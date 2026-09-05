"""Episodic multi-vehicle warehouse pathfinding on top of POGEMA.

* Each vehicle has its own depot -- they need not share one.
* Each vehicle has a *fixed, predefined* set of order nodes to visit.
  Nothing new arrives during an episode.
* A vehicle is finished once it has visited all its orders **and**
  returned to its depot.
* The episode ends when every vehicle is finished (or on timeout).
* Collisions between vehicles are handled by POGEMA.
"""

import warnings
from typing import Dict, List, Optional, Sequence

from warehouse_marl.env.routing import Coord, build_sequence


class WarehouseEnv:
    # Queried by SB3's VecEnv wrapper via get_attr("render_mode").
    render_mode = None

    def __init__(
        self,
        grid_map: str,
        depots: Dict[str, Coord],
        orders: Dict[str, Sequence[Coord]],
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
        """
        Parameters
        ----------
        grid_map:
            POGEMA ASCII map: '.' free, '#' obstacle, one row per line.
        depots:
            vehicle_id -> (row, col) start and return position.
        orders:
            vehicle_id -> list of (row, col) nodes that vehicle must visit.
            Every vehicle in `depots` needs at least one order.
        order_strategy:
            "nearest" (greedy nearest-neighbour tour) or "as_given".
        collision_system:
            "soft" (POGEMA's own LMAPF experiment setting) or "priority"
            (POGEMA's default).
        goal_reward:
            Awarded each time a vehicle reaches its current target node.
        step_penalty:
            Subtracted each step from vehicles that are not yet finished,
            so shorter tours score higher. Set to 0 to disable.
        completion_bonus:
            One-off reward when a vehicle completes its tour and is home.
        """
        from pogema import GridConfig, pogema_v0

        self.vehicle_ids: List[str] = list(depots.keys())
        missing = [v for v in self.vehicle_ids if not orders.get(v)]
        if missing:
            raise ValueError(f"vehicles with no orders: {missing}")

        self.depots = {v: tuple(depots[v]) for v in self.vehicle_ids}
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

    # -- spaces ---------------------------------------------------------------

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

    # -- core loop ------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._goal_hits = {v: 0 for v in self.vehicle_ids}
        self._finished = {v: False for v in self.vehicle_ids}
        self._steps = 0
        return self._remap(obs), self._remap(info)

    def step(self, actions: Dict[str, int]):
        # Finished vehicles are frozen in place at their depot. They still
        # occupy a cell (so they can still block others, as a real parked
        # vehicle would), but they no longer act or score.
        pz_actions = {
            self._vehicle_to_player[v]: (0 if self._finished[v] else int(actions.get(v, 0)))
            for v in self.vehicle_ids
        }

        # POGEMA warns every time a goal sequence wraps. We deliberately let
        # it wrap and handle termination ourselves, so the warning is noise.
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
            if raw_rewards[v] > 0:  # POGEMA gives 1.0 on reaching the current goal
                self._goal_hits[v] += 1
                reward += self.goal_reward
                if self._goal_hits[v] >= len(self.sequences[v]):
                    self._finished[v] = True
                    reward += self.completion_bonus
            rewards[v] = reward

        episode_over = all(self._finished.values())
        timed_out = bool(pz_truncated) and all(pz_truncated.values())

        # A single, shared episode boundary for every vehicle. Vehicles that
        # finish early are frozen rather than removed, which keeps all agents
        # in lockstep -- much easier to batch for a shared-policy learner.
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
        return {self._player_to_vehicle[k]: v for k, v in d.items() if k in self._player_to_vehicle}
