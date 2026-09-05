"""Build a WarehouseEnv from a configs/env.yaml-style file."""

from pathlib import Path
from typing import Any, Dict

import yaml

from warehouse_marl.env.warehouse_env import WarehouseEnv


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(config: Dict[str, Any], repo_root: str = ".", **overrides) -> WarehouseEnv:
    grid_map = (Path(repo_root) / config["map_path"]).read_text()

    depots = {k: tuple(v) for k, v in config["depots"].items()}
    orders = {k: [tuple(node) for node in v] for k, v in config["orders"].items()}

    kwargs = dict(
        grid_map=grid_map,
        depots=depots,
        orders=orders,
        order_strategy=config.get("order_strategy", "nearest"),
        obs_radius=config.get("obs_radius", 5),
        collision_system=config.get("collision_system", "soft"),
        max_episode_steps=config.get("max_episode_steps", 512),
        goal_reward=config.get("goal_reward", 1.0),
        step_penalty=config.get("step_penalty", 0.01),
        completion_bonus=config.get("completion_bonus", 5.0),
        seed=config.get("seed"),
    )
    kwargs.update(overrides)
    return WarehouseEnv(**kwargs)
