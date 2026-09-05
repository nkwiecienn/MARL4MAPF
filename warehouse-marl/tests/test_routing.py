import pytest

from warehouse_marl.env.routing import build_sequence, order_nodes


def test_as_given_preserves_input_order():
    orders = [(5, 5), (0, 1), (3, 3)]
    assert order_nodes((0, 0), orders, "as_given") == orders


def test_nearest_visits_closest_first():
    # (0,1) is 1 away from the depot, (3,3) is 6, (5,5) is 10.
    tour = order_nodes((0, 0), [(5, 5), (0, 1), (3, 3)], "nearest")
    assert tour == [(0, 1), (3, 3), (5, 5)]


def test_sequence_ends_at_depot():
    seq = build_sequence((2, 2), [(0, 0), (4, 4)], "nearest")
    assert seq[-1] == [2, 2], "vehicle must be sent home as its final goal"
    assert len(seq) == 3


def test_sequence_is_plain_ints():
    # POGEMA's pydantic GridConfig rejects numpy scalars here.
    seq = build_sequence((0, 0), [(1, 1)], "nearest")
    assert all(isinstance(c, int) for node in seq for c in node)


def test_empty_order_set_rejected():
    with pytest.raises(ValueError, match="at least one order"):
        build_sequence((0, 0), [], "nearest")


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown ordering strategy"):
        order_nodes((0, 0), [(1, 1)], "travelling_salesman")
