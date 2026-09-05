from typing import Optional, Sequence

import numpy as np

Coord = tuple[int, int]


def allocate_depot_cells(
    depot_zones: dict[str, Sequence[Coord]],
    vehicle_depot_zone: dict[str, str],
    seed: Optional[int] = None,
) -> dict[str, Coord]:
    vehicles_by_zone: dict[str, list[str]] = {}
    for vehicle in sorted(vehicle_depot_zone):
        zone = vehicle_depot_zone[vehicle]
        vehicles_by_zone.setdefault(zone, []).append(vehicle)

    rng = np.random.default_rng(seed)
    depots: dict[str, Coord] = {}
    for zone, vehicles in sorted(vehicles_by_zone.items()):
        cells = [tuple(cell) for cell in depot_zones[zone]]
        shuffled = rng.permutation(len(cells))
        for vehicle, index in zip(vehicles, shuffled):
            depots[vehicle] = cells[index]
    return depots
