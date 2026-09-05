from pathlib import Path
from typing import Any

import yaml

from warehouse_marl.env.warehouse_env import WarehouseEnv


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(config: dict[str, Any], repo_root: str = ".", **overrides) -> WarehouseEnv:
    settings: dict[str, Any] = dict(
        grid_map=(Path(repo_root) / config["map_path"]).read_text(),
        orders={vehicle: [tuple(node) for node in nodes] for vehicle, nodes in config["orders"].items()},
        depot_zones={zone: [tuple(cell) for cell in cells] for zone, cells in config["depot_zones"].items()},
        vehicle_depot_zone=dict(config["vehicle_depot_zone"]),
        order_strategy=config.get("order_strategy", "nearest"),
        obs_radius=config.get("obs_radius", 5),
        collision_system=config.get("collision_system", "soft"),
        max_episode_steps=config.get("max_episode_steps", 512),
        goal_reward=config.get("goal_reward", 1.0),
        step_penalty=config.get("step_penalty", 0.01),
        completion_bonus=config.get("completion_bonus", 5.0),
        seed=config.get("seed"),
    )
    settings.update(overrides)
    return WarehouseEnv(**settings)
