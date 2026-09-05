import pytest

from warehouse_marl.env.depots import allocate_depot_cells

GRID = ".....\n.....\n.....\n"


def test_single_vehicle_zone_returns_its_only_cell():
    result = allocate_depot_cells(
        grid_map=GRID,
        depot_zones={"z": [(0, 0)]},
        vehicle_depot_zone={"a": "z"},
        seed=0,
    )
    assert result == {"a": (0, 0)}


def test_shared_zone_assigns_distinct_cells_to_each_vehicle():
    result = allocate_depot_cells(
        grid_map=GRID,
        depot_zones={"z": [(0, 0), (0, 1), (0, 2)]},
        vehicle_depot_zone={"a": "z", "b": "z"},
        seed=0,
    )
    assert result["a"] != result["b"]
    assert set(result.values()) <= {(0, 0), (0, 1), (0, 2)}


def test_allocation_is_reproducible_given_same_seed():
    kwargs = dict(
        grid_map=GRID,
        depot_zones={"z": [(0, 0), (0, 1), (0, 2)]},
        vehicle_depot_zone={"a": "z", "b": "z", "c": "z"},
        seed=42,
    )
    assert allocate_depot_cells(**kwargs) == allocate_depot_cells(**kwargs)


def test_capacity_exceeded_raises():
    with pytest.raises(ValueError, match="capacity"):
        allocate_depot_cells(
            grid_map=GRID,
            depot_zones={"z": [(0, 0)]},
            vehicle_depot_zone={"a": "z", "b": "z"},
            seed=0,
        )


def test_obstacle_cell_in_zone_rejected():
    grid = "..#..\n.....\n"
    with pytest.raises(ValueError, match="non-free"):
        allocate_depot_cells(
            grid_map=grid,
            depot_zones={"z": [(0, 2)]},
            vehicle_depot_zone={"a": "z"},
            seed=0,
        )


def test_unknown_zone_reference_rejected():
    with pytest.raises(ValueError, match="unknown depot zone"):
        allocate_depot_cells(
            grid_map=GRID,
            depot_zones={"z": [(0, 0)]},
            vehicle_depot_zone={"a": "not_z"},
            seed=0,
        )
