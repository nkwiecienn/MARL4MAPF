"""Project-owned SVG renderer: depot squares plus each vehicle's full tour.

POGEMA's own animation draws one moving target per agent. This one draws the
*whole* tour up front and lets each waypoint change appearance as the vehicle
works through it, so a single frame shows where a vehicle has been, where it
is headed, and what it still owes.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from pathlib import Path
from typing import Optional

from pogema.svg_animation.animation_drawer import (
    AnimationConfig,
    AnimationDrawer,
    GridHolder,
    SvgSettings,
)
from pogema.svg_animation.animation_wrapper import AnimationMonitor
from pogema.svg_animation.svg_objects import Circle, Rectangle, SvgObject

Coord = tuple[int, int]


class WaypointState(Enum):
    """Where a waypoint sits relative to a vehicle's progress along its tour."""

    VISITED = "visited"
    NEXT = "next"
    REMAINING = "remaining"


def _pad_or_truncate(values: list[int], length: int) -> list[int]:
    """Fit `values` to exactly `length` entries, repeating the last one if short."""
    if not values:
        return [0] * length
    if len(values) >= length:
        return values[:length]
    return values + [values[-1]] * (length - len(values))


def _cell_center(settings: SvgSettings, grid_width: int, cell: Coord) -> tuple[float, float]:
    row, col = cell
    x = settings.draw_start + col * settings.scale_size
    y = settings.draw_start + (grid_width - row - 1) * settings.scale_size
    return x, y


def _is_visible(state: WaypointState, position: int, goal_hits: int) -> bool:
    """Whether the `state` look of waypoint `position` shows at `goal_hits` goals done."""
    if state is WaypointState.VISITED:
        return position < goal_hits
    if state is WaypointState.NEXT:
        return position == goal_hits
    if state is WaypointState.REMAINING:
        return position > goal_hits
    raise ValueError(f"unknown waypoint state: {state!r}")


class Diamond(SvgObject):
    tag = "polygon"

    def __init__(self, cx: float, cy: float, half_diagonal: float, **kwargs):
        corners = [
            (cx, cy - half_diagonal),
            (cx + half_diagonal, cy),
            (cx, cy + half_diagonal),
            (cx - half_diagonal, cy),
        ]
        # POGEMA draws in a y-flipped frame: its Circle/Rectangle/Line each
        # negate y inside their own __init__. A hand-built polygon gets no such
        # treatment, so flip the points here to line up with everything else.
        kwargs["points"] = " ".join(f"{x},{-y}" for x, y in corners)
        super().__init__(**kwargs)


def _waypoint_shapes(
    settings: SvgSettings, cx: float, cy: float, color: str
) -> list[tuple[WaypointState, SvgObject]]:
    """The three looks one waypoint can have; exactly one is visible at a time.

    This is the only place the legend is defined -- filled dot for a waypoint
    already served, hollow ring for the one being approached, diamond for one
    still owed. `create_targets` and `animate_targets` both read the pairing
    from here, so the shapes and their meanings cannot drift apart.
    """
    return [
        (WaypointState.VISITED, Circle(cx=cx, cy=cy, r=settings.r * 0.35, fill=color)),
        (
            WaypointState.NEXT,
            Circle(
                cx=cx,
                cy=cy,
                r=settings.r,
                fill="none",
                stroke=color,
                stroke_width=settings.stroke_width,
            ),
        ),
        (
            WaypointState.REMAINING,
            Diamond(
                cx=cx,
                cy=cy,
                half_diagonal=settings.r * 0.6,
                fill="none",
                stroke=color,
                stroke_width=settings.stroke_width,
            ),
        ),
    ]


@dataclass
class WarehouseGridHolder(GridHolder):
    # These default to None only because the base dataclass gives its own
    # fields defaults; WarehouseAnimationMonitor always supplies all four.
    depots: Optional[dict[str, Coord]] = None
    sequences: Optional[dict[str, list[list[int]]]] = None
    goal_hits_history: Optional[dict[str, list[int]]] = None
    vehicle_order: Optional[list[str]] = None


class WarehouseAnimationMonitor(AnimationMonitor):
    """Records an episode and renders it with tours and depots drawn in."""

    def __init__(self, env, warehouse_env, animation_config: AnimationConfig = AnimationConfig()):
        super().__init__(env, animation_config)
        self._depots = warehouse_env.depots
        self._sequences = warehouse_env.sequences
        self._vehicle_order = list(warehouse_env.vehicle_ids)

    @classmethod
    def attach(
        cls, warehouse_env, animation_config: AnimationConfig = AnimationConfig()
    ) -> "WarehouseAnimationMonitor":
        """Wrap the env's POGEMA grid in this recorder, in place, and return it."""
        warehouse_env.pogema_grid = cls(warehouse_env.pogema_grid, warehouse_env, animation_config)
        return warehouse_env.pogema_grid

    def save_animation(
        self,
        name: str = "render.svg",
        animation_config: AnimationConfig = AnimationConfig(),
        goal_hits_log: Optional[list[dict[str, int]]] = None,
    ) -> None:
        obstacles = self._cropped_obstacles()
        history = self._padded_history()
        episode_length = len(history[0])

        if animation_config.egocentric_idx is not None and self.grid_config.on_target == "finish":
            episode_length = history[animation_config.egocentric_idx][-1].step + 1
            history = [agent_history[:episode_length] for agent_history in history]

        # Cropping the border shifts every cell; move our own coordinates to match.
        offset = self.grid_config.obs_radius - self._working_radius
        depots, sequences = self._to_render_frame(offset)

        svg_settings = SvgSettings(time_scale=0.5)
        grid_holder = WarehouseGridHolder(
            width=len(obstacles),
            height=len(obstacles[0]),
            obstacles=obstacles,
            episode_length=episode_length,
            history=history,
            obs_radius=self.grid_config.obs_radius,
            on_target=self.grid_config.on_target,
            colors=dict(zip(range(self.grid_config.num_agents), cycle(svg_settings.colors))),
            config=animation_config,
            svg_settings=svg_settings,
            depots=depots,
            sequences=sequences,
            goal_hits_history={
                vehicle: _pad_or_truncate(
                    [entry.get(vehicle, 0) for entry in goal_hits_log or []], episode_length
                )
                for vehicle in self._vehicle_order
            },
            vehicle_order=self._vehicle_order,
        )

        animation = WarehouseAnimationDrawer().create_animation(grid_holder)
        Path(name).write_text(animation.render())

    def _cropped_obstacles(self):
        """The obstacle grid with POGEMA's observation-padding border removed."""
        obstacles = self.env.get_obstacles(ignore_borders=False)
        if self._working_radius > 0:
            obstacles = obstacles[self._working_radius : -self._working_radius,
                                  self._working_radius : -self._working_radius]
        return obstacles

    def _padded_history(self) -> list[list]:
        """Recorded agent states, with the final state repeated once.

        The extra frame gives the last real step somewhere to hold, so the
        animation does not snap back the instant the episode ends.
        """
        history = self.env.decompress_history(self.history)
        for agent_history in history:
            agent_history.append(agent_history[-1])
        return history

    def _to_render_frame(self, offset: int):
        """Shift depots and tours from grid coordinates into render coordinates."""
        depots = {
            vehicle: (row + offset, col + offset)
            for vehicle, (row, col) in self._depots.items()
        }
        sequences = {
            vehicle: [[row + offset, col + offset] for row, col in sequence]
            for vehicle, sequence in self._sequences.items()
        }
        return depots, sequences


class WarehouseAnimationDrawer(AnimationDrawer):
    """Draws depot squares and every waypoint of every tour.

    Each waypoint contributes all three shapes from `_waypoint_shapes`; the
    animation then toggles their opacity so only the one matching the
    vehicle's current progress is on screen at any moment.
    """

    def __init__(self):
        super().__init__()
        # Filled by create_targets, consumed by animate_targets. The base class
        # calls those two in that order (AnimationDrawer.create_animation), and
        # each entry lines up with the shape at the same index.
        self._waypoints: list[tuple[str, int, WaypointState]] = []

    def create_targets(self, grid_holder: WarehouseGridHolder):
        settings = grid_holder.svg_settings
        shapes = []
        self._waypoints = []

        for agent_idx, vehicle in enumerate(grid_holder.vehicle_order):
            if not any(state.is_active() for state in grid_holder.history[agent_idx]):
                continue
            color = grid_holder.colors[agent_idx]

            for position, cell in enumerate(grid_holder.sequences[vehicle]):
                cx, cy = _cell_center(settings, grid_holder.width, cell)
                for state, shape in _waypoint_shapes(settings, cx, cy, color):
                    shapes.append(shape)
                    self._waypoints.append((vehicle, position, state))

        return shapes

    def animate_targets(self, targets, grid_holder: WarehouseGridHolder):
        for shape, (vehicle, position, state) in zip(targets, self._waypoints):
            opacity = [
                "1" if _is_visible(state, position, goal_hits) else "0"
                for goal_hits in grid_holder.goal_hits_history[vehicle]
            ]
            animation = self.compressed_anim("opacity", opacity, grid_holder.svg_settings.time_scale)
            shape.add_animation(animation)

    def create_depot_markers(self, grid_holder: WarehouseGridHolder):
        settings = grid_holder.svg_settings
        markers = []
        for agent_idx, vehicle in enumerate(grid_holder.vehicle_order):
            cx, cy = _cell_center(settings, grid_holder.width, grid_holder.depots[vehicle])
            markers.append(
                Rectangle(
                    x=cx - settings.r,
                    y=cy - settings.r,
                    width=settings.r * 2,
                    height=settings.r * 2,
                    fill=grid_holder.colors[agent_idx],
                    fill_opacity=0.3,  # lighter than the agent's own color so a same-colored agent stands out on top
                    rx=settings.rx,
                )
            )
        return markers

    def create_animation(self, grid_holder: WarehouseGridHolder):
        drawing = super().create_animation(grid_holder)
        # Prepend, so depot squares paint underneath the agents and waypoints.
        drawing.elements[0:0] = self.create_depot_markers(grid_holder)
        return drawing
