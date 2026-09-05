"""Assigning each vehicle one concrete, fixed cell within its depot zone.

A depot *zone* is a set of cells that one or more vehicles may return to.
Vehicles sharing a zone still each get their own distinct concrete cell,
chosen once (deterministically, given a seed) when the environment is
built -- not re-picked per episode, and not "any free cell at arrival
time". See warehouse_env.WarehouseEnv for how the resolved cell is then
used exactly like today's flat `depots` mapping.
"""

import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int]


def parse_free_cells(grid_map: str) -> set:
    """Cells that are not '#' in the ASCII map, as (row, col)."""
    lines = grid_map.splitlines()
    return {(r, c) for r, line in enumerate(lines) for c, ch in enumerate(line) if ch != "#"}


def _zone_seed(seed: Optional[int], zone_id: str) -> int:
    # zlib.crc32, NOT builtin hash() -- str hashing is randomized per
    # process (PYTHONHASHSEED) and would break reproducibility across runs.
    base = 0 if seed is None else int(seed)
    return (base * 2_147_483_647 + zlib.crc32(zone_id.encode())) & 0xFFFFFFFF


def allocate_depot_cells(
    grid_map: str,
    depot_zones: Dict[str, Sequence[Coord]],
    vehicle_depot_zone: Dict[str, str],
    seed: Optional[int] = None,
) -> Dict[str, Coord]:
    """Deterministically assign each vehicle exactly one free cell in its zone.

    Raises ValueError if a vehicle references an unknown zone id, a zone
    contains a cell that is an obstacle in `grid_map`, or a zone's free-cell
    count is smaller than the number of vehicles assigned to it.
    """
    free_cells = parse_free_cells(grid_map)

    zone_to_vehicles: Dict[str, List[str]] = {}
    for v in sorted(vehicle_depot_zone):  # sorted -> deterministic grouping order
        zone_id = vehicle_depot_zone[v]
        if zone_id not in depot_zones:
            raise ValueError(f"vehicle {v!r} references unknown depot zone {zone_id!r}")
        zone_to_vehicles.setdefault(zone_id, []).append(v)

    result: Dict[str, Coord] = {}
    for zone_id, vehicles in zone_to_vehicles.items():
        zone_cells = [tuple(c) for c in depot_zones[zone_id]]
        bad = [c for c in zone_cells if c not in free_cells]
        if bad:
            raise ValueError(f"depot zone {zone_id!r} contains non-free cell(s): {bad}")
        if len(zone_cells) < len(vehicles):
            raise ValueError(
                f"depot zone {zone_id!r} has capacity {len(zone_cells)} cells but "
                f"{len(vehicles)} vehicle(s) are assigned to it: {vehicles}"
            )
        rng = np.random.default_rng(_zone_seed(seed, zone_id))
        order = rng.permutation(len(zone_cells))
        chosen = [zone_cells[i] for i in order[: len(vehicles)]]
        for v, cell in zip(vehicles, chosen):  # `vehicles` already sorted -> reproducible
            result[v] = cell
    return result
