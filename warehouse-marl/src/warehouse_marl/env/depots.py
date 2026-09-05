from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

Coord = Tuple[int, int]


def allocate_depot_cells(
    depot_zones: Dict[str, Sequence[Coord]],
    vehicle_depot_zone: Dict[str, str],
    seed: Optional[int] = None,
) -> Dict[str, Coord]:
    zone_to_vehicles: Dict[str, List[str]] = {}
    for v in sorted(vehicle_depot_zone):
        zone_to_vehicles.setdefault(vehicle_depot_zone[v], []).append(v)

    rng = np.random.default_rng(seed)
    result: Dict[str, Coord] = {}
    for zone_id in sorted(zone_to_vehicles):
        vehicles = zone_to_vehicles[zone_id]
        zone_cells = [tuple(c) for c in depot_zones[zone_id]]
        order = rng.permutation(len(zone_cells))
        for v, i in zip(vehicles, order):
            result[v] = zone_cells[i]
    return result
