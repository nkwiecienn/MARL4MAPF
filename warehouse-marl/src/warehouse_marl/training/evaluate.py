from typing import Dict, Optional, Tuple
import numpy as np
from warehouse_marl.env.warehouse_env import WarehouseEnv
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv


def evaluate_policy(model, env: WarehouseEnv, n_episodes: int = 20, deterministic: bool = True) -> Dict:
    vec = WarehouseVecEnv(env)
    solved, lengths, returns = 0, [], []
    for _ in range(n_episodes):
        obs = vec.reset()
        steps, ep_return = 0, 0.0
        for _ in range(env.grid_config.max_episode_steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec.step(action)
            ep_return += float(np.sum(rewards))
            steps += 1
            if dones.all():
                if infos[0].get("episode_solved"):
                    solved += 1
                    lengths.append(steps)
                break
        returns.append(ep_return)
    return {
        "solve_rate": solved / n_episodes,
        "mean_steps_when_solved": float(np.mean(lengths)) if lengths else None,
        "mean_return": float(np.mean(returns)),
        "episodes": n_episodes,
    }
