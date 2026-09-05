import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from warehouse_marl.env import build_env, load_config
from warehouse_marl.training.evaluate import evaluate_policy
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=str(REPO_ROOT / "configs" / "env.yaml"))
    parser.add_argument("--train-config", default=str(REPO_ROOT / "configs" / "training.yaml"))
    parser.add_argument("--timesteps", type=int, default=None, help="overrides the config")
    args = parser.parse_args()

    env_config = load_config(args.env_config)
    with open(args.train_config) as f:
        cfg = yaml.safe_load(f)
    total_timesteps = args.timesteps or cfg["total_timesteps"]

    train_env = WarehouseVecEnv(build_env(env_config, repo_root=str(REPO_ROOT)))

    model = PPO(
        cfg["policy"],
        train_env,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        learning_rate=cfg["learning_rate"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        seed=cfg["seed"],
        tensorboard_log=None,
        verbose=1,
    )

    checkpoint_dir = REPO_ROOT / cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(
        save_freq=max(1, cfg["checkpoint_every"] // train_env.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_warehouse",
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = checkpoint_dir / "ppo_warehouse_final.zip"
    model.save(str(model_path))
    print(f"\nsaved model to {model_path}")

    print("\n=== after training ===")
    trained = evaluate_policy(model, build_env(env_config, repo_root=str(REPO_ROOT)),
                              n_episodes=cfg["eval_episodes"])
    print(f"trained PPO:          solve_rate={trained['solve_rate']:.0%} "
          f"mean_steps={trained['mean_steps_when_solved']} "
          f"mean_return={trained['mean_return']:.2f}")


if __name__ == "__main__":
    main()
