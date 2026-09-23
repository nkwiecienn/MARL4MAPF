"""Turning a YAML scenario file into a WarehouseEnv.

Scenario files and maps ship *inside* the package (`warehouse_marl/configs`,
`warehouse_marl/maps`), so an installed copy can find its own defaults without
any caller having to know where the project lives on disk.

This module deliberately declares no default *values* of its own. Every key it
recognises is simply forwarded, so `WarehouseEnv.__init__`'s signature stays
the single place where a default is written down.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any, Union

import yaml

from warehouse_marl.env.warehouse_env import WarehouseEnv

# `files()` locates the installed package wherever it ended up. It returns a
# Traversable; this package ships as plain files (never a zip), so converting
# to a real Path is safe and keeps the rest of the module ordinary.
PACKAGE_ROOT = Path(str(files("warehouse_marl")))

#: The scenario bundled with the package, used when no config is given.
DEFAULT_ENV_CONFIG = PACKAGE_ROOT / "configs" / "env.yaml"

#: Config keys forwarded verbatim to WarehouseEnv. A key absent from the YAML
#: is simply not passed, so the constructor's own default applies.
PASSTHROUGH_KEYS = (
    "orders",
    "depot_zones",
    "vehicle_depot_zone",
    "order_strategy",
    "obs_radius",
    "collision_system",
    "max_episode_steps",
    "goal_reward",
    "step_penalty",
    "completion_bonus",
    "seed",
)


def load_config(path: Union[str, Path] = DEFAULT_ENV_CONFIG) -> dict[str, Any]:
    """Load a scenario YAML.

    `map_path` is interpreted relative to the config file that names it, and
    comes back absolute. That one rule covers both the bundled scenario and an
    experiment YAML you keep anywhere else on disk.
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    config["map_path"] = str((path.parent / config["map_path"]).resolve())
    return config


def build_env(config: dict[str, Any], **overrides) -> WarehouseEnv:
    """Build the env described by `config`; `overrides` win over the file.

    Coordinates stay as the plain lists PyYAML produced -- WarehouseEnv and
    routing.order_nodes both coerce them to tuples themselves.
    """
    settings = {key: config[key] for key in PASSTHROUGH_KEYS if key in config}
    settings["grid_map"] = Path(config["map_path"]).read_text()
    settings.update(overrides)
    return WarehouseEnv(**settings)
