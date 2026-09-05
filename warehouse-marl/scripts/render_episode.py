#!/usr/bin/env python3
"""Render an episode to an SVG animation you can open in a browser.

Uses a trained PPO model if you point at one, otherwise the collision-blind
A* baseline, so you can eyeball the difference in how they handle congestion.

Usage:
    python scripts/render_episode.py
    python scripts/render_episode.py --model checkpoints/ppo_warehouse_final.zip
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pogema import AnimationMonitor  # noqa: E402

from warehouse_marl.env import build_env, load_config  # noqa: E402
from warehouse_marl.training.evaluate import _astar_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "env.yaml"))
    parser.add_argument("--model", default=None, help="path to a saved SB3 model (.zip)")
    parser.add_argument("--out", default=str(REPO_ROOT / "episode.svg"))
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config, repo_root=str(REPO_ROOT))

    # AnimationMonitor wraps POGEMA's own env, which sits underneath our
    # PettingZoo-facing wrapper -- so attach it to the inner env.
    env._env.pogema = AnimationMonitor(env._env.pogema)

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        print(f"using policy from {args.model}")
    else:
        print("no --model given; using the collision-blind A* baseline")

    import numpy as np

    obs, _ = env.reset()
    obstacles = env._env.pogema.get_obstacles(ignore_borders=True).astype(bool)

    for _ in range(config.get("max_episode_steps", 256)):
        if model is not None:
            stacked = np.stack([obs[v] for v in env.possible_agents]).astype(np.float32)
            raw_actions, _ = model.predict(stacked, deterministic=True)
            actions = {v: int(raw_actions[i]) for i, v in enumerate(env.possible_agents)}
        else:
            pogema = env._env.pogema
            positions = pogema.get_agents_xy(ignore_borders=True)
            targets = pogema.get_targets_xy(ignore_borders=True)
            actions = {
                v: _astar_step(tuple(positions[i]), tuple(targets[i]), obstacles)
                for i, v in enumerate(env.possible_agents)
            }

        obs, _, terminated, truncated, infos = env.step(actions)
        if all(terminated.values()) or all(truncated.values()):
            print(f"episode ended: solved={infos[env.possible_agents[0]].get('episode_solved')}")
            break

    env._env.pogema.save_animation(args.out)
    print(f"saved animation to {args.out}")


if __name__ == "__main__":
    main()
