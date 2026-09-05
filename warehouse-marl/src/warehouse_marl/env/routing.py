from typing import List, Sequence, Tuple

Coord = Tuple[int, int]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_nodes(depot: Coord, orders: Sequence[Coord], strategy: str = "nearest") -> List[Coord]:
    orders = [tuple(o) for o in orders]
    if strategy == "as_given":
        return list(orders)

    remaining = list(orders)
    tour: List[Coord] = []
    current = tuple(depot)
    while remaining:
        nxt = min(remaining, key=lambda node: _manhattan(current, node))
        remaining.remove(nxt)
        tour.append(nxt)
        current = nxt
    return tour


def build_sequence(depot: Coord, orders: Sequence[Coord], strategy: str = "nearest") -> List[List[int]]:
    tour = order_nodes(depot, orders, strategy)
    return [[int(x), int(y)] for x, y in list(tour) + [tuple(depot)]]
