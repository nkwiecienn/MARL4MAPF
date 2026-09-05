import numpy as np

from warehouse_marl.env.warehouse_env import WarehouseEnv
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv


def evaluate_policy(model, env: WarehouseEnv, n_episodes: int = 20, deterministic: bool = True) -> dict:
    vec_env = WarehouseVecEnv(env)
    solved = 0
    steps_when_solved = []
    returns = []

    for _ in range(n_episodes):
        obs = vec_env.reset()
        steps = 0
        episode_return = 0.0
        for _ in range(env.grid_config.max_episode_steps + 1):
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(actions)
            episode_return += float(np.sum(rewards))
            steps += 1
            if dones.all():
                if infos[0].get("episode_solved"):
                    solved += 1
                    steps_when_solved.append(steps)
                break
        returns.append(episode_return)

    return {
        "solve_rate": solved / n_episodes,
        "mean_steps_when_solved": float(np.mean(steps_when_solved)) if steps_when_solved else None,
        "mean_return": float(np.mean(returns)),
        "episodes": n_episodes,
    }
