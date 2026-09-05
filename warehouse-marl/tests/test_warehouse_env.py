"""Behavioural tests for the episodic wrapper.

The reward-farming test is the important one: POGEMA's lifelong mode
silently wraps an exhausted goal sequence back to its start and keeps
paying out, so without the wrapper's own terminal condition an agent
could loop its tour forever for unbounded reward.
"""

import numpy as np

from warehouse_marl.env import WarehouseEnv

GRID = ".....\n.....\n.....\n"
DEPOTS = {"a": (0, 0), "b": (2, 4)}
ORDERS = {"a": [(0, 2), (0, 4)], "b": [(2, 2)]}


def _make(**kwargs):
    params = dict(
        grid_map=GRID,
        depots=DEPOTS,
        orders=ORDERS,
        order_strategy="as_given",
        obs_radius=2,
        max_episode_steps=40,
    )
    params.update(kwargs)
    return WarehouseEnv(**params)


def _greedy(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if dx:
        return 1 if dx < 0 else 2
    if dy:
        return 3 if dy < 0 else 4
    return 0


def _run_greedy(env, max_steps=40):
    env.reset()
    for step in range(max_steps):
        pogema = env._env.pogema
        pos = pogema.get_agents_xy(ignore_borders=True)
        tgt = pogema.get_targets_xy(ignore_borders=True)
        actions = {v: _greedy(pos[i], tgt[i]) for i, v in enumerate(env.possible_agents)}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        if all(terminated.values()) or all(truncated.values()):
            return step + 1, rewards, terminated, truncated, infos
    return None, None, None, None, None


def test_tour_includes_orders_plus_depot_return():
    env = _make()
    assert env.tour_length("a") == 3  # 2 orders + depot
    assert env.tour_length("b") == 2  # 1 order  + depot
    assert env.sequences["a"][-1] == [0, 0]


def test_episode_terminates_when_all_vehicles_home():
    env = _make()
    steps, _, terminated, truncated, infos = _run_greedy(env)
    assert steps is not None, "episode never ended"
    assert all(terminated.values())
    assert not any(truncated.values())
    assert infos["a"]["episode_solved"] is True


def test_finished_vehicle_stops_earning_reward():
    """Regression guard against POGEMA's sequence wrap-around."""
    env = _make()
    env.reset()
    totals = {v: 0.0 for v in env.possible_agents}
    finished_at = {}
    for step in range(40):
        pogema = env._env.pogema
        pos = pogema.get_agents_xy(ignore_borders=True)
        tgt = pogema.get_targets_xy(ignore_borders=True)
        actions = {v: _greedy(pos[i], tgt[i]) for i, v in enumerate(env.possible_agents)}
        _, rewards, terminated, truncated, infos = env.step(actions)
        for v, r in rewards.items():
            if infos[v]["finished"] and v in finished_at:
                assert r == 0.0, f"{v} earned reward after finishing"
            totals[v] += r
        for v in env.possible_agents:
            if infos[v]["finished"] and v not in finished_at:
                finished_at[v] = step
        if all(terminated.values()) or all(truncated.values()):
            break
    # Each vehicle scores its goals once, not repeatedly.
    for v in env.possible_agents:
        max_possible = env.tour_length(v) * env.goal_reward + env.completion_bonus
        assert totals[v] <= max_possible


def test_truncation_on_timeout():
    env = _make(max_episode_steps=15)
    rng = np.random.default_rng(0)
    env.reset()
    for step in range(30):
        actions = {v: int(rng.integers(0, 5)) for v in env.possible_agents}
        _, _, terminated, truncated, infos = env.step(actions)
        if all(truncated.values()):
            assert not all(terminated.values())
            assert infos["a"]["episode_solved"] is False
            return
    raise AssertionError("episode was never truncated")


def test_reset_clears_progress():
    env = _make()
    _run_greedy(env)
    obs, _ = env.reset()
    assert not any(env._finished.values())
    assert all(hits == 0 for hits in env._goal_hits.values())
    assert set(obs) == set(env.possible_agents)


def test_vehicle_without_orders_rejected():
    import pytest

    with pytest.raises(ValueError, match="no orders"):
        _make(orders={"a": [(0, 2)], "b": []})


def test_legacy_depots_kwarg_still_works():
    env = _make()
    assert env.depots == DEPOTS
    assert env.depot_zones == {"a": [(0, 0)], "b": [(2, 4)]}
    assert env.vehicle_depot_zone == {"a": "a", "b": "b"}


def test_zone_kwargs_build_valid_env():
    env = _make(
        depots=None,
        depot_zones={"shared": [(0, 0), (0, 2), (0, 4)]},
        vehicle_depot_zone={"a": "shared", "b": "shared"},
    )
    assert env.depots["a"] != env.depots["b"]
    assert set(env.depots.values()) <= {(0, 0), (0, 2), (0, 4)}
    assert env.depot_zones == {"shared": [(0, 0), (0, 2), (0, 4)]}
    assert env.vehicle_depot_zone == {"a": "shared", "b": "shared"}


def test_shared_zone_capacity_error_surfaces_from_constructor():
    import pytest

    with pytest.raises(ValueError, match="capacity"):
        _make(
            depots=None,
            depot_zones={"shared": [(0, 0)]},
            vehicle_depot_zone={"a": "shared", "b": "shared"},
        )


def test_conflicting_depot_kwargs_rejected():
    import pytest

    with pytest.raises(ValueError, match="not both"):
        _make(depot_zones={"shared": [(0, 0)]}, vehicle_depot_zone={"a": "shared"})
