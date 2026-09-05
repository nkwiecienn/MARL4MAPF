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
    monitor = WarehouseAnimationMonitor(env._env.pogema, warehouse_env=env)
    env._env.pogema = monitor
    model = PPO.load(args.model)

    vehicles = env.possible_agents
    obs, _ = env.reset()
    goal_hits_log = [env.goal_hits]

    for _ in range(config.get("max_episode_steps", 256)):
        observations = np.stack([obs[vehicle] for vehicle in vehicles]).astype(np.float32)
        predicted, _ = model.predict(observations, deterministic=True)
        actions = {vehicle: int(action) for vehicle, action in zip(vehicles, predicted)}

        obs, _, terminated, truncated, infos = env.step(actions)
        goal_hits_log.append(env.goal_hits)

        if all(terminated.values()) or all(truncated.values()):
            print(f"episode ended: solved={infos[vehicles[0]].get('episode_solved')}")
            break

    monitor.save_animation(args.out, goal_hits_log=goal_hits_log)
    print(f"saved animation to {args.out}")


if __name__ == "__main__":
    main()
