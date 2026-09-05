from typing import Sequence

Coord = tuple[int, int]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_nodes(depot: Coord, orders: Sequence[Coord], strategy: str = "nearest") -> list[Coord]:
    remaining = [tuple(order) for order in orders]
    if strategy == "as_given":
        return remaining

    tour: list[Coord] = []
    current = tuple(depot)
    while remaining:
        nearest = min(remaining, key=lambda node: _manhattan(current, node))
        remaining.remove(nearest)
        tour.append(nearest)
        current = nearest
    return tour


def build_sequence(depot: Coord, orders: Sequence[Coord], strategy: str = "nearest") -> list[list[int]]:
    tour = order_nodes(depot, orders, strategy) + [tuple(depot)]
    return [[int(row), int(col)] for row, col in tour]
