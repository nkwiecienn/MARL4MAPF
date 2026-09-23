"""Roll out one episode with a trained policy and save it as an SVG animation."""

import argparse

import numpy as np
from stable_baselines3 import PPO

from warehouse_marl.env import DEFAULT_ENV_CONFIG, build_env, load_config
from warehouse_marl.viz.renderer import WarehouseAnimationMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_ENV_CONFIG),
                        help="scenario YAML (default: the one bundled with the package)")
    parser.add_argument("--model", required=True, help="path to a saved SB3 model (.zip)")
    parser.add_argument("--out", default="episode.svg")
    args = parser.parse_args()

    env = build_env(load_config(args.config))
    monitor = WarehouseAnimationMonitor.attach(env)
    model = PPO.load(args.model)

    vehicles = env.possible_agents
    obs, _ = env.reset()
    goal_hits_log = [env.goal_hits]

    for _ in range(env.grid_config.max_episode_steps):
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
