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

        border_offset = self.grid_config.obs_radius - wr
        vehicle_depot_cell = {
            v: (row + border_offset, col + border_offset) for v, (row, col) in self._vehicle_depot_cell.items()
        }
        sequences = {
            v: [[row + border_offset, col + border_offset] for row, col in seq]
            for v, seq in self._sequences.items()
        }

        svg_settings = SvgSettings(time_scale=0.5)
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
        drawing.elements[0:0] = self.create_depot_markers(grid_holder)
        return drawing
