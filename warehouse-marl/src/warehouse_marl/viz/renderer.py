"""Project-owned SVG renderer for WarehouseEnv episodes.

Extends the *installed* pogema 1.4.0 SVG animation classes
(`pogema.svg_animation.animation_wrapper.AnimationMonitor` and
`pogema.svg_animation.animation_drawer.AnimationDrawer`) to also draw:

* each vehicle's depot cell as a filled square in that vehicle's color, and
* every node in each vehicle's tour, all the time, as one of three shapes
  in that vehicle's color: a small dot (visited), a hollow ring (next),
  or a hollow diamond (remaining) -- distinguished by shape, never by
  fading opacity.

The vendored pogema copy at the repo root (a newer 2.0.0a pre-release kept
purely for reference) is never imported here -- everything comes from the
`pogema` package actually installed in the environment.
"""

from dataclasses import dataclass
from itertools import cycle
from typing import Dict, List, Optional, Tuple

from pogema.svg_animation.animation_drawer import (
    AnimationConfig,
    AnimationDrawer,
    GridHolder,
    SvgSettings,
)
from pogema.svg_animation.animation_wrapper import AnimationMonitor
from pogema.svg_animation.svg_objects import Circle, Rectangle, SvgObject

Coord = Tuple[int, int]


def _pad_to(values: List[int], length: int) -> List[int]:
    """Truncate/repeat the last value so `values` has exactly `length` entries."""
    if not values:
        return [0] * length
    if len(values) >= length:
        return values[:length]
    return values + [values[-1]] * (length - len(values))


def _diamond_points(cx: float, cy: float, half_diagonal: float) -> str:
    """SVG `points` string for a diamond centered on (cx, cy).

    cx/cy here are the same *positive*-space values used everywhere else
    in this file. Circle/Rectangle negate y internally per shape; a
    polygon has no such built-in, so each vertex's y is negated here, by
    hand, the same way.
    """
    pts = [
        (cx, cy - half_diagonal),
        (cx + half_diagonal, cy),
        (cx, cy + half_diagonal),
        (cx - half_diagonal, cy),
    ]
    return " ".join(f"{x},{-y}" for x, y in pts)


class Diamond(SvgObject):
    tag = "polygon"

    def __init__(self, cx: float, cy: float, half_diagonal: float, **kwargs):
        kwargs["points"] = _diamond_points(cx, cy, half_diagonal)
        super().__init__(**kwargs)


@dataclass
class WarehouseGridHolder(GridHolder):
    vehicle_depot_cell: Optional[Dict[str, Coord]] = None
    sequences: Optional[Dict[str, List[List[int]]]] = None
    goal_hits_history: Optional[Dict[str, List[int]]] = None
    vehicle_order: Optional[List[str]] = None


class WarehouseAnimationMonitor(AnimationMonitor):
    """`AnimationMonitor` that also threads depot-cell/full-tour data through.

    Only `save_animation` is overridden -- `__init__`/`step`/`reset`/
    `pick_name` are inherited unchanged. `save_animation`'s body is a
    copy-adapted version of `AnimationMonitor.save_animation`: that method
    hardcodes `GridHolder(...)`/`AnimationDrawer()` construction inline
    with no seam to override, so duplicating it (rather than the whole
    class hierarchy) is the smallest available diff. `pogema` is pinned to
    exactly 1.4.0 in requirements.txt/pyproject.toml, so this won't
    silently drift out of sync with the base method.
    """

    def __init__(self, env, warehouse_env, animation_config: AnimationConfig = AnimationConfig()):
        super().__init__(env, animation_config)
        self._vehicle_depot_cell = warehouse_env.vehicle_depot_cell
        self._sequences = warehouse_env.sequences
        self._vehicle_order = list(warehouse_env.vehicle_ids)

    def save_animation(
        self,
        name: str = "render.svg",
        animation_config: AnimationConfig = AnimationConfig(),
        goal_hits_log: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        wr = self._working_radius
        if wr > 0:
            obstacles = self.env.get_obstacles(ignore_borders=False)[wr:-wr, wr:-wr]
        else:
            obstacles = self.env.get_obstacles(ignore_borders=False)
        history = self.env.decompress_history(self.history)

        # AnimationMonitor crops the obstacle border down to working_radius =
        # obs_radius - 1, not obs_radius -- i.e. it deliberately leaves one
        # ring of boundary wall visible, and shifts every recorded agent/
        # target position by +1 (via PersistentWrapper's xy_offset) to match.
        # WarehouseEnv's own coordinates (vehicle_depot_cell, sequences) are
        # in the *unshifted* "true"/ignore_borders=True system, so they need
        # the same +1 applied here to land on the cells vehicles actually
        # walk to (verified empirically: true (4, 0) records as (5, 1)).
        border_offset = self.grid_config.obs_radius - wr
        vehicle_depot_cell = {
            v: (row + border_offset, col + border_offset) for v, (row, col) in self._vehicle_depot_cell.items()
        }
        sequences = {
            v: [[row + border_offset, col + border_offset] for row, col in seq]
            for v, seq in self._sequences.items()
        }

        svg_settings = SvgSettings(time_scale=0.5)  # default 0.25 read as too fast
        colors_cycle = cycle(svg_settings.colors)
        agents_colors = {index: next(colors_cycle) for index in range(self.grid_config.num_agents)}

        for agent_idx in range(self.grid_config.num_agents):
            history[agent_idx].append(history[agent_idx][-1])

        episode_length = len(history[0])
        if animation_config.egocentric_idx is not None and self.grid_config.on_target == "finish":
            episode_length = history[animation_config.egocentric_idx][-1].step + 1
            for agent_idx in range(self.grid_config.num_agents):
                history[agent_idx] = history[agent_idx][:episode_length]

        goal_hits_log = goal_hits_log or []
        goal_hits_history = {
            v: _pad_to([entry.get(v, 0) for entry in goal_hits_log], episode_length)
            for v in self._vehicle_order
        }

        grid_holder = WarehouseGridHolder(
            width=len(obstacles),
            height=len(obstacles[0]),
            obstacles=obstacles,
            episode_length=episode_length,
            history=history,
            obs_radius=self.grid_config.obs_radius,
            on_target=self.grid_config.on_target,
            colors=agents_colors,
            config=animation_config,
            svg_settings=svg_settings,
            vehicle_depot_cell=vehicle_depot_cell,
            sequences=sequences,
            goal_hits_history=goal_hits_history,
            vehicle_order=self._vehicle_order,
        )

        animation = WarehouseAnimationDrawer().create_animation(grid_holder)
        with open(name, "w") as f:
            f.write(animation.render())


class WarehouseAnimationDrawer(AnimationDrawer):
    """Draws a colored depot square per vehicle plus its full tour as shapes.

    Coordinate convention (verified against the installed animation_drawer.py
    / svg_objects.py, do not deviate from it): for a grid coordinate
    (row, col), the SVG *center* point is

        cx = draw_start + col * scale_size
        cy = draw_start + (width - row - 1) * scale_size

    exactly as `create_agents`/`create_field_of_view` already compute it.
    `Circle`/`Rectangle` negate cy/y internally at construction, so callers
    always pass this positive-space value (see `Diamond`/`_diamond_points`
    above for the same convention applied by hand). `fix_point()` is the
    *reverse* mapping (SVG loop index -> grid index), used only by
    `create_obstacles` for its obstacle-array lookup -- never for placement.

    Every tour waypoint is drawn as three superimposed, persistent shapes
    (visited dot / next ring / remaining diamond); only one is ever visible
    at a time, toggled by a hard 0/1 opacity animation -- there is no path
    morphing in SMIL, so "the shape changes over time" has to mean "three
    shapes exist, and visibility swaps between them."
    """

    def create_targets(self, grid_holder):
        gh: WarehouseGridHolder = grid_holder
        shapes = []
        self._waypoint_meta: List[Tuple[str, int, str]] = []  # (vehicle, wp_idx, kind)
        for agent_idx, v in enumerate(gh.vehicle_order):
            if not any(s.is_active() for s in gh.history[agent_idx]):
                continue
            color = gh.colors[agent_idx]
            for wp_idx, (row, col) in enumerate(gh.sequences[v]):
                cx = gh.svg_settings.draw_start + col * gh.svg_settings.scale_size
                cy = gh.svg_settings.draw_start + (gh.width - row - 1) * gh.svg_settings.scale_size

                visited = Circle(cx=cx, cy=cy, r=gh.svg_settings.r * 0.35, fill=color)
                next_ring = Circle(
                    cx=cx,
                    cy=cy,
                    r=gh.svg_settings.r,
                    fill="none",
                    stroke=color,
                    stroke_width=gh.svg_settings.stroke_width,
                )
                remaining = Diamond(
                    cx=cx,
                    cy=cy,
                    half_diagonal=gh.svg_settings.r * 0.6,
                    fill="none",
                    stroke=color,
                    stroke_width=gh.svg_settings.stroke_width,
                )
                shapes += [visited, next_ring, remaining]
                self._waypoint_meta += [
                    (v, wp_idx, "visited"),
                    (v, wp_idx, "next"),
                    (v, wp_idx, "remaining"),
                ]
        return shapes

    def animate_targets(self, targets, grid_holder):
        gh: WarehouseGridHolder = grid_holder
        for shape, (v, wp_idx, kind) in zip(targets, self._waypoint_meta):
            opacity = []
            for step_hits in gh.goal_hits_history[v]:
                if kind == "visited":
                    visible = wp_idx < step_hits
                elif kind == "next":
                    visible = wp_idx == step_hits
                else:  # "remaining"
                    visible = wp_idx > step_hits
                opacity.append("1" if visible else "0")
            shape.add_animation(self.compressed_anim("opacity", opacity, gh.svg_settings.time_scale))

    def create_depot_markers(self, grid_holder):
        gh: WarehouseGridHolder = grid_holder
        rects = []
        for agent_idx, v in enumerate(gh.vehicle_order):
            row, col = gh.vehicle_depot_cell[v]
            cx = gh.svg_settings.draw_start + col * gh.svg_settings.scale_size
            cy = gh.svg_settings.draw_start + (gh.width - row - 1) * gh.svg_settings.scale_size
            rects.append(
                Rectangle(
                    x=cx - gh.svg_settings.r,
                    y=cy - gh.svg_settings.r,
                    width=gh.svg_settings.r * 2,
                    height=gh.svg_settings.r * 2,
                    fill=gh.colors[agent_idx],
                    fill_opacity=0.3,  # lighter than the agent's own color so a same-colored agent stands out on top
                    rx=gh.svg_settings.rx,
                )
            )
        return rects

    def create_animation(self, grid_holder):
        drawing = super().create_animation(grid_holder)
        # Insert at the FRONT of the element list, not append -- these are
        # solid filled squares, and super().create_animation() already
        # added agents/targets to `drawing`. Appending would paint the
        # squares LAST (on top), burying every waypoint shape and any
        # agent parked in its depot cell under solid color. Inserting at
        # index 0 paints them first/underneath everything.
        drawing.elements[0:0] = self.create_depot_markers(grid_holder)
        return drawing
