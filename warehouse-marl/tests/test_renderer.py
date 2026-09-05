from pogema.svg_animation.animation_drawer import AnimationConfig, SvgSettings
from pogema.wrappers.persistence import AgentState

from warehouse_marl.env import WarehouseEnv
from warehouse_marl.viz.renderer import (
    Diamond,
    WarehouseAnimationDrawer,
    WarehouseAnimationMonitor,
    WarehouseGridHolder,
    _diamond_points,
)

SETTINGS = SvgSettings()


def _holder(**kwargs):
    base = dict(
        width=2,
        height=2,
        obstacles=[[0, 0], [0, 0]],
        episode_length=1,
        history=[[AgentState(0, 0, 0, 1, 0, True)]],
        obs_radius=3,
        on_target="restart",
        colors={0: SETTINGS.colors[0]},
        config=AnimationConfig(),
        svg_settings=SETTINGS,
        vehicle_depot_cell={"a": (0, 0)},
        sequences={"a": [[0, 0], [0, 1]]},
        goal_hits_history={"a": [0]},
        vehicle_order=["a"],
    )
    base.update(kwargs)
    return WarehouseGridHolder(**base)


def test_diamond_points_are_negated_and_centered():
    points = _diamond_points(cx=100.0, cy=200.0, half_diagonal=20.0)
    pairs = [tuple(map(float, p.split(","))) for p in points.split(" ")]
    assert pairs == [
        (100.0, -180.0),  # top:    (cx, cy - h) -> y negated
        (120.0, -200.0),  # right:  (cx + h, cy) -> y negated
        (100.0, -220.0),  # bottom: (cx, cy + h) -> y negated
        (80.0, -200.0),   # left:   (cx - h, cy) -> y negated
    ]


def test_diamond_is_a_polygon_element():
    d = Diamond(cx=0, cy=0, half_diagonal=10, fill="none", stroke="#c1433c")
    assert d.tag == "polygon"
    assert d.render().startswith("<polygon")
    assert 'stroke="#c1433c"' in d.render()


def test_create_depot_markers_places_one_filled_square_per_vehicle():
    drawer = WarehouseAnimationDrawer()
    rects = drawer.create_depot_markers(_holder())
    assert len(rects) == 1
    # row=0,col=0 -> cx=100, cy=200 -> x=cx-r, y=cy-r
    assert rects[0].attributes["x"] == 100 - SETTINGS.r
    assert rects[0].attributes["width"] == SETTINGS.r * 2
    assert rects[0].attributes["fill"] == SETTINGS.colors[0]
    assert rects[0].attributes["fill_opacity"] < 1  # lighter, so a same-colored agent stands out on top


def test_create_depot_markers_uses_each_vehicles_own_color():
    drawer = WarehouseAnimationDrawer()
    gh = _holder(
        colors={0: SETTINGS.colors[0], 1: SETTINGS.colors[1]},
        vehicle_depot_cell={"a": (0, 0), "b": (1, 1)},
        sequences={"a": [[0, 0], [0, 1]], "b": [[1, 0], [1, 1]]},
        goal_hits_history={"a": [0], "b": [0]},
        vehicle_order=["a", "b"],
        history=[[AgentState(0, 0, 0, 1, 0, True)], [AgentState(1, 1, 1, 0, 0, True)]],
    )
    rects = drawer.create_depot_markers(gh)
    assert [r.attributes["fill"] for r in rects] == [SETTINGS.colors[0], SETTINGS.colors[1]]


def test_create_targets_returns_three_shapes_per_waypoint():
    drawer = WarehouseAnimationDrawer()
    shapes = drawer.create_targets(_holder())
    assert len(shapes) == 6  # 2 waypoints * 3 shapes each
    assert [kind for _, _, kind in drawer._waypoint_meta] == [
        "visited", "next", "remaining",
        "visited", "next", "remaining",
    ]

    visited, next_ring, remaining = shapes[:3]
    assert visited.attributes["r"] == SETTINGS.r * 0.35
    assert visited.attributes["fill"] == SETTINGS.colors[0]

    assert next_ring.attributes["r"] == SETTINGS.r
    assert next_ring.attributes["fill"] == "none"
    assert next_ring.attributes["stroke"] == SETTINGS.colors[0]

    assert isinstance(remaining, Diamond)
    assert remaining.attributes["fill"] == "none"
    assert remaining.attributes["stroke"] == SETTINGS.colors[0]


def test_animate_targets_shows_exactly_one_shape_per_waypoint_at_each_step():
    drawer = WarehouseAnimationDrawer()
    gh = _holder(goal_hits_history={"a": [0, 0, 1, 1]})
    shapes = drawer.create_targets(gh)
    drawer.animate_targets(shapes, gh)

    def expected(opacity_tokens):
        return drawer.compressed_anim("opacity", opacity_tokens, gh.svg_settings.time_scale).attributes["values"]

    # wp_idx 0: current ("next") at steps 0-1, visited at steps 2-3.
    assert shapes[0].animations[0].attributes["values"] == expected(["0", "0", "1", "1"])  # visited
    assert shapes[1].animations[0].attributes["values"] == expected(["1", "1", "0", "0"])  # next
    assert shapes[2].animations[0].attributes["values"] == expected(["0", "0", "0", "0"])  # remaining

    # wp_idx 1: remaining at steps 0-1, current ("next") at steps 2-3.
    assert shapes[3].animations[0].attributes["values"] == expected(["0", "0", "0", "0"])  # visited
    assert shapes[4].animations[0].attributes["values"] == expected(["0", "0", "1", "1"])  # next
    assert shapes[5].animations[0].attributes["values"] == expected(["1", "1", "0", "0"])  # remaining


def test_create_animation_paints_depot_markers_underneath_agents_and_targets():
    drawer = WarehouseAnimationDrawer()
    drawing = drawer.create_animation(_holder())
    depot_markers = drawer.create_depot_markers(_holder())
    assert len(depot_markers) == 1
    # The single depot-marker Rectangle must be the very first element, so
    # it paints underneath everything super().create_animation() added.
    assert drawing.elements[0].attributes["fill"] == SETTINGS.colors[0]
    assert drawing.elements[0].tag == "rect"


def test_create_animation_svg_contains_expected_markers():
    drawer = WarehouseAnimationDrawer()
    svg = drawer.create_animation(_holder()).render()
    assert "<polygon" in svg  # remaining-waypoint diamond
    assert f'fill="{SETTINGS.colors[0]}"' in svg  # depot square + visited dot


# -- end-to-end -------------------------------------------------------------

GRID = ".....\n.....\n.....\n"
ORDERS = {"a": [(0, 2), (0, 4)], "b": [(2, 2)]}


def _greedy(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if dx:
        return 1 if dx < 0 else 2
    if dy:
        return 3 if dy < 0 else 4
    return 0


def test_end_to_end_render_produces_expected_svg_markers(tmp_path):
    env = WarehouseEnv(
        grid_map=GRID,
        orders=ORDERS,
        depot_zones={"shared": [(0, 0), (2, 4), (2, 0)]},
        vehicle_depot_zone={"a": "shared", "b": "shared"},
        order_strategy="as_given",
        obs_radius=2,
        max_episode_steps=40,
    )
    env._env.pogema = WarehouseAnimationMonitor(env._env.pogema, warehouse_env=env)

    env.reset()
    goal_hits_log = [dict(env._goal_hits)]
    for _ in range(40):
        pogema = env._env.pogema
        pos = pogema.get_agents_xy(ignore_borders=True)
        tgt = pogema.get_targets_xy(ignore_borders=True)
        actions = {v: _greedy(pos[i], tgt[i]) for i, v in enumerate(env.possible_agents)}
        _, _, terminated, truncated, _ = env.step(actions)
        goal_hits_log.append(dict(env._goal_hits))
        if all(terminated.values()) or all(truncated.values()):
            break

    out = tmp_path / "episode.svg"
    env._env.pogema.save_animation(str(out), goal_hits_log=goal_hits_log)

    svg = out.read_text()
    assert svg
    assert "<polygon" in svg  # remaining-waypoint diamonds
    assert svg.count('rx="') >= 2  # depot squares (one per vehicle)
