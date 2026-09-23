"""Train a single PPO policy shared by every vehicle (parameter-shared IPPO).

Hyperparameters live here as constants rather than in a YAML file: they are
tied to this script and nothing else reads them. The *scenario* -- map, depot
zones, orders, rewards -- is what varies between experiments, and that stays
in the packaged scenario file (warehouse_marl/configs/env.yaml).
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from warehouse_marl.env import DEFAULT_ENV_CONFIG, build_env, load_config
from warehouse_marl.training.evaluate import evaluate_policy
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv

POLICY = "MlpPolicy"
PPO_KWARGS = dict(
    n_steps=256,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    learning_rate=3e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

TOTAL_TIMESTEPS = 500_000
SEED = 0
# Outputs land where you run the script, not next to the installed package.
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_EVERY = 50_000
EVAL_EPISODES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default=str(DEFAULT_ENV_CONFIG),
                        help="scenario YAML (default: the one bundled with the package)")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    env_config = load_config(args.env_config)
    train_env = WarehouseVecEnv(build_env(env_config))

    model = PPO(POLICY, train_env, seed=SEED, verbose=1, **PPO_KWARGS)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointCallback(
        # CheckpointCallback counts calls, not env steps, and one call advances
        # every sub-env at once -- so divide to get the requested step interval.
        save_freq=max(1, CHECKPOINT_EVERY // train_env.num_envs),
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_warehouse",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoints)

    model_path = CHECKPOINT_DIR / "ppo_warehouse_final.zip"
    model.save(str(model_path))
    print(f"\nsaved model to {model_path}")

    eval_env = build_env(env_config)
    metrics = evaluate_policy(model, eval_env, n_episodes=EVAL_EPISODES)
    print("\n=== after training ===")
    print(
        f"trained PPO:          solve_rate={metrics['solve_rate']:.0%} "
        f"mean_steps={metrics['mean_steps_when_solved']} "
        f"mean_return={metrics['mean_return']:.2f}"
    )


if __name__ == "__main__":
    main()
