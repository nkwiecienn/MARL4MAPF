"""Greedy and policy evaluation helpers.

`evaluate_policy` reports the two numbers that actually matter for this
problem -- how often every vehicle completes its tour and gets home, and
how many steps that took -- rather than raw reward, which is only a proxy.

`astar_baseline` runs a collision-blind shortest-path controller. It is
the number to beat: it shows what the task looks like with perfect
sequencing but no collision reasoning at all, so a policy that can't beat
it isn't earning its keep.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from warehouse_marl.env.warehouse_env import WarehouseEnv
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv


def evaluate_policy(model, env: WarehouseEnv, n_episodes: int = 20, deterministic: bool = True) -> Dict:
    vec = WarehouseVecEnv(env)
    solved, lengths, returns = 0, [], []
    for _ in range(n_episodes):
        obs = vec.reset()
        steps, ep_return = 0, 0.0
        for _ in range(env.grid_config.max_episode_steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec.step(action)
            ep_return += float(np.sum(rewards))
            steps += 1
            if dones.all():
                if infos[0].get("episode_solved"):
                    solved += 1
                    lengths.append(steps)
                break
        returns.append(ep_return)
    return {
        "solve_rate": solved / n_episodes,
        "mean_steps_when_solved": float(np.mean(lengths)) if lengths else None,
        "mean_return": float(np.mean(returns)),
        "episodes": n_episodes,
    }


def _astar_step(start, goal, obstacles) -> int:
    """First action of a shortest path from `start` to `goal`.

    Full-map A*: it knows the static layout but ignores other vehicles, so
    it is *collision-blind*. That is deliberate -- it isolates how much of
    the difficulty comes from inter-vehicle conflicts rather than from
    navigation, which is exactly the gap the RL policy has to close.

    POGEMA MOVES = [[0,0], [-1,0], [1,0], [0,-1], [0,1]].
    """
    from heapq import heappop, heappush

    if start == goal:
        return 0
    rows, cols = obstacles.shape
    moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    frontier = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
    came_from = {start: (None, 0)}
    while frontier:
        _, cost, node = heappop(frontier)
        if node == goal:
            break
        for action, (dr, dc) in moves.items():
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if obstacles[nxt[0], nxt[1]]:
                continue
            new_cost = cost + 1
            if nxt not in came_from or new_cost < came_from[nxt][1]:
                came_from[nxt] = (node, new_cost)
                priority = new_cost + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heappush(frontier, (priority, new_cost, nxt))

    if goal not in came_from:
        return 0  # unreachable; sit still rather than thrash

    node = goal
    while came_from[node][0] is not None and came_from[node][0] != start:
        node = came_from[node][0]
    if came_from[node][0] is None:
        return 0
    dr, dc = node[0] - start[0], node[1] - start[1]
    for action, delta in moves.items():
        if delta == (dr, dc):
            return action
    return 0


def astar_baseline(env: WarehouseEnv, n_episodes: int = 20) -> Dict:
    """Per-vehicle shortest-path controller, replanned every step.

    This is the number your policy must beat. It has perfect static-map
    knowledge and optimal single-agent routing, but no notion of the other
    vehicles -- so any shortfall here is caused purely by congestion.
    """
    solved, lengths = 0, []
    for _ in range(n_episodes):
        env.reset()
        obstacles = env._env.pogema.get_obstacles(ignore_borders=True).astype(bool)
        steps = 0
        for _ in range(env.grid_config.max_episode_steps + 1):
            pogema = env._env.pogema
            positions = pogema.get_agents_xy(ignore_borders=True)
            targets = pogema.get_targets_xy(ignore_borders=True)
            actions = {
                v: _astar_step(tuple(positions[i]), tuple(targets[i]), obstacles)
                for i, v in enumerate(env.possible_agents)
            }
            _, _, terminated, truncated, infos = env.step(actions)
            steps += 1
            if all(terminated.values()) or all(truncated.values()):
                if infos[env.possible_agents[0]].get("episode_solved"):
                    solved += 1
                    lengths.append(steps)
                break
    return {
        "solve_rate": solved / n_episodes,
        "mean_steps_when_solved": float(np.mean(lengths)) if lengths else None,
        "episodes": n_episodes,
    }
