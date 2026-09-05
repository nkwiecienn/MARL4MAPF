"""Turning a vehicle's *set* of orders into an ordered visit sequence.

POGEMA's lifelong mode consumes a fixed, ordered list of goals per agent.
Your problem gives each vehicle an unordered *set* of nodes, so something
has to choose the visiting order.

In v1 that choice is made once, up front, by this module -- the RL policy
learns navigation and collision avoidance, not sequencing. That is a real
simplification of the VRP: the tour order is fixed, so the policy cannot
trade a longer tour for fewer conflicts. It keeps the first baseline
tractable, and letting the policy (or a planner) choose the next node is
the natural v2 extension.

Two strategies are provided:

``as_given``
    Visit orders in exactly the order supplied. Use when the upstream
    system already decides sequencing and you just want it executed.
``nearest``
    Greedy nearest-neighbour tour from the depot using Manhattan distance
    (ignoring obstacles). Not optimal, but it removes the arbitrariness of
    an accidental input ordering and gives a much saner baseline tour.
"""

from typing import List, Sequence, Tuple

Coord = Tuple[int, int]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_nodes(depot: Coord, orders: Sequence[Coord], strategy: str = "nearest") -> List[Coord]:
    """Return the orders arranged into a visiting order (depot NOT included)."""
    orders = [tuple(o) for o in orders]
    if strategy == "as_given":
        return list(orders)
    if strategy != "nearest":
        raise ValueError(f"unknown ordering strategy: {strategy!r}")

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
    """Full POGEMA goal sequence for one vehicle: ordered orders, then home.

    The trailing depot entry is what makes "return to the depot when done"
    a goal the environment can actually detect and reward.
    """
    if len(orders) == 0:
        raise ValueError(
            "each vehicle needs at least one order; a vehicle with an empty "
            "order set has nothing to do and cannot form a valid POGEMA "
            "goal sequence (POGEMA requires >= 2 goals per agent)"
        )
    tour = order_nodes(depot, orders, strategy)
    return [[int(x), int(y)] for x, y in list(tour) + [tuple(depot)]]
