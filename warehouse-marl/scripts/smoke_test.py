#!/usr/bin/env python3
"""Run this FIRST. No RL library involved.

Builds the scenario from configs/env.yaml and checks the things that are
easy to get silently wrong: that tours are built correctly, that a greedy
controller can actually finish them, and that the episode terminates
instead of running forever.

Usage:
    python scripts/smoke_test.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from warehouse_marl.env import build_env, load_config  # noqa: E402
from warehouse_marl.training.evaluate import astar_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "env.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config, repo_root=str(REPO_ROOT))

    print("vehicles:", env.possible_agents)
    for v in env.possible_agents:
        print(f"  {v}: depot={env.depots[v]} tour={env.sequences[v]} ({env.tour_length(v)} goals)")
    print(f"observation space: {env.observation_space(env.possible_agents[0])}")
    print(f"action space:      {env.action_space(env.possible_agents[0])}")

    print("\nrunning A* (collision-blind) baseline...")
    stats = astar_baseline(env, n_episodes=args.episodes)
    print(f"  solve rate:  {stats['solve_rate']:.0%}")
    print(f"  mean steps:  {stats['mean_steps_when_solved']}")

    if stats["solve_rate"] == 0:
        print(
            "\nWARNING: A* solved nothing. Usually the map/orders are "
            "unreachable, or max_episode_steps is too low."
        )
    else:
        print("\nEnvironment looks healthy. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
