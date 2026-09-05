import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from warehouse_marl.env import build_env, load_config
from warehouse_marl.viz.renderer import WarehouseAnimationMonitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "env.yaml"))
    parser.add_argument("--model", required=True, help="path to a saved SB3 model (.zip)")
    parser.add_argument("--out", default=str(REPO_ROOT / "episode.svg"))
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_env(config, repo_root=str(REPO_ROOT))
    env._env.pogema = WarehouseAnimationMonitor(env._env.pogema, warehouse_env=env)
    model = PPO.load(args.model)

    vehicles = env.possible_agents
    obs, _ = env.reset()
    goal_hits_log = [dict(env._goal_hits)]

    for _ in range(config.get("max_episode_steps", 256)):
        obs_batch = np.stack([obs[v] for v in vehicles]).astype(np.float32)
        action_batch, _ = model.predict(obs_batch, deterministic=True)

        actions = {}
        for vehicle, action in zip(vehicles, action_batch):
            actions[vehicle] = int(action)

        obs, _, terminated, truncated, infos = env.step(actions)
        goal_hits_log.append(dict(env._goal_hits))

        episode_over = all(terminated.values()) or all(truncated.values())
        if episode_over:
            solved = infos[vehicles[0]].get("episode_solved")
            print(f"episode ended: solved={solved}")
            break

    env._env.pogema.save_animation(args.out, goal_hits_log=goal_hits_log)
    print(f"saved animation to {args.out}")


if __name__ == "__main__":
    main()
