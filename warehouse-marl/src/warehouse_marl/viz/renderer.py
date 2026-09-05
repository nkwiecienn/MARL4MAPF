from dataclasses import dataclass
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

VISITED = "visited"
NEXT = "next"
REMAINING = "remaining"


def _pad_to(values: list[int], length: int) -> list[int]:
    """Truncate/repeat the last value so `values` has exactly `length` entries."""
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


def _is_visible(kind: str, position: int, goal_hits: int) -> bool:
    if kind == VISITED:
        return position < goal_hits
    if kind == NEXT:
        return position == goal_hits
    return position > goal_hits


class Diamond(SvgObject):
    tag = "polygon"

    def __init__(self, cx: float, cy: float, half_diagonal: float, **kwargs):
        corners = [
            (cx, cy - half_diagonal),
            (cx + half_diagonal, cy),
            (cx, cy + half_diagonal),
            (cx - half_diagonal, cy),
        ]
        kwargs["points"] = " ".join(f"{x},{-y}" for x, y in corners)
        super().__init__(**kwargs)


@dataclass
class WarehouseGridHolder(GridHolder):
    depots: Optional[dict[str, Coord]] = None
    sequences: Optional[dict[str, list[list[int]]]] = None
    goal_hits_history: Optional[dict[str, list[int]]] = None
    vehicle_order: Optional[list[str]] = None


class WarehouseAnimationMonitor(AnimationMonitor):

    def __init__(self, env, warehouse_env, animation_config: AnimationConfig = AnimationConfig()):
        super().__init__(env, animation_config)
        self._depots = warehouse_env.depots
        self._sequences = warehouse_env.sequences
        self._vehicle_order = list(warehouse_env.vehicle_ids)

    def save_animation(
        self,
        name: str = "render.svg",
        animation_config: AnimationConfig = AnimationConfig(),
        goal_hits_log: Optional[list[dict[str, int]]] = None,
    ) -> None:
        radius = self._working_radius
        obstacles = self.env.get_obstacles(ignore_borders=False)
        if radius > 0:
            obstacles = obstacles[radius:-radius, radius:-radius]

        offset = self.grid_config.obs_radius - radius
        depots = {
            vehicle: (row + offset, col + offset)
            for vehicle, (row, col) in self._depots.items()
        }
        sequences = {
            vehicle: [[row + offset, col + offset] for row, col in sequence]
            for vehicle, sequence in self._sequences.items()
        }

        history = self.env.decompress_history(self.history)
        for agent_history in history:
            agent_history.append(agent_history[-1])

        episode_length = len(history[0])
        if animation_config.egocentric_idx is not None and self.grid_config.on_target == "finish":
            episode_length = history[animation_config.egocentric_idx][-1].step + 1
            history = [agent_history[:episode_length] for agent_history in history]

        svg_settings = SvgSettings(time_scale=0.5)
        colors = dict(zip(range(self.grid_config.num_agents), cycle(svg_settings.colors)))

        goal_hits_log = goal_hits_log or []
        goal_hits_history = {
            vehicle: _pad_to([entry.get(vehicle, 0) for entry in goal_hits_log], episode_length)
            for vehicle in self._vehicle_order
        }

        grid_holder = WarehouseGridHolder(
            width=len(obstacles),
            height=len(obstacles[0]),
            obstacles=obstacles,
            episode_length=episode_length,
            history=history,
            obs_radius=self.grid_config.obs_radius,
            on_target=self.grid_config.on_target,
            colors=colors,
            config=animation_config,
            svg_settings=svg_settings,
            depots=depots,
            sequences=sequences,
            goal_hits_history=goal_hits_history,
            vehicle_order=self._vehicle_order,
        )

        animation = WarehouseAnimationDrawer().create_animation(grid_holder)
        Path(name).write_text(animation.render())


class WarehouseAnimationDrawer(AnimationDrawer):

    def create_targets(self, grid_holder: WarehouseGridHolder):
        settings = grid_holder.svg_settings
        shapes = []
        self._waypoints: list[tuple[str, int, str]] = []

        for agent_idx, vehicle in enumerate(grid_holder.vehicle_order):
            if not any(state.is_active() for state in grid_holder.history[agent_idx]):
                continue
            color = grid_holder.colors[agent_idx]

            for position, cell in enumerate(grid_holder.sequences[vehicle]):
                cx, cy = _cell_center(settings, grid_holder.width, cell)
                shapes.append(Circle(cx=cx, cy=cy, r=settings.r * 0.35, fill=color))
                shapes.append(
                    Circle(
                        cx=cx,
                        cy=cy,
                        r=settings.r,
                        fill="none",
                        stroke=color,
                        stroke_width=settings.stroke_width,
                    )
                )
                shapes.append(
                    Diamond(
                        cx=cx,
                        cy=cy,
                        half_diagonal=settings.r * 0.6,
                        fill="none",
                        stroke=color,
                        stroke_width=settings.stroke_width,
                    )
                )
                self._waypoints.append((vehicle, position, VISITED))
                self._waypoints.append((vehicle, position, NEXT))
                self._waypoints.append((vehicle, position, REMAINING))

        return shapes

    def animate_targets(self, targets, grid_holder: WarehouseGridHolder):
        for shape, (vehicle, position, kind) in zip(targets, self._waypoints):
            opacity = [
                "1" if _is_visible(kind, position, goal_hits) else "0"
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
        drawing.elements[0:0] = self.create_depot_markers(grid_holder)
        return drawing
