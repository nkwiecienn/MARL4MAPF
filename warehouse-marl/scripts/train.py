import argparse
import sys
from pathlib import Path

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
    train_config = load_config(args.train_config)
    total_timesteps = args.timesteps or train_config["total_timesteps"]

    train_env = WarehouseVecEnv(build_env(env_config, repo_root=str(REPO_ROOT)))

    model = PPO(
        train_config["policy"],
        train_env,
        n_steps=train_config["n_steps"],
        batch_size=train_config["batch_size"],
        n_epochs=train_config["n_epochs"],
        gamma=train_config["gamma"],
        gae_lambda=train_config["gae_lambda"],
        clip_range=train_config["clip_range"],
        learning_rate=train_config["learning_rate"],
        ent_coef=train_config["ent_coef"],
        vf_coef=train_config["vf_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        seed=train_config["seed"],
        verbose=1,
    )

    checkpoint_dir = REPO_ROOT / train_config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointCallback(
        save_freq=max(1, train_config["checkpoint_every"] // train_env.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_warehouse",
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoints)

    model_path = checkpoint_dir / "ppo_warehouse_final.zip"
    model.save(str(model_path))
    print(f"\nsaved model to {model_path}")

    eval_env = build_env(env_config, repo_root=str(REPO_ROOT))
    metrics = evaluate_policy(model, eval_env, n_episodes=train_config["eval_episodes"])
    print("\n=== after training ===")
    print(
        f"trained PPO:          solve_rate={metrics['solve_rate']:.0%} "
        f"mean_steps={metrics['mean_steps_when_solved']} "
        f"mean_return={metrics['mean_return']:.2f}"
    )


if __name__ == "__main__":
    main()
