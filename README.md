# Warehouse path finding with reinforcement learning

Multi-agent warehouse pathfinding built on [POGEMA](https://github.com/Cognitive-AI-Systems/pogema):
- Each vehicle has a fixed, predefined set of order nodes to visit.
- Vehicles start and return at a depot zone.
- A vehicle finishes when it has visited all its orders and returned to its depot.
- The episode ends when every vehicle is finished.
- Collision avoidance between vehicles is handled by POGEMA.

Online order and policy-chosen routing will be added in the near future.

---

Each vehicle is a one sub-env in an SB3 `VecEnv`, so PPO trains one shared policy on transitions pooled from all vehicles (parameter-shared independent PPO - IPPO). Agents act on local observations with no communication.

All sub-envs share one episode boundary (vehicles that finish early are frozen, not removed), which keeps them in lockstep, since SB3 auto-resets sub-envs independently.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate

cd warehouse-marl
pip install -e .

python scripts/train.py         # train PPO
python scripts/render_episode.py --model checkpoints/ppo_warehouse_final.zip   # -> episode.svg
```

The scenario — map, depot zones, orders, rewards — lives in
`src/warehouse_marl/configs/env.yaml`, with its map alongside in
`src/warehouse_marl/maps/`. They sit inside the package so an installed copy
carries its own defaults; both scripts fall back to that bundled scenario when
no `--config` / `--env-config` is given.

To run a variant, copy the YAML anywhere and pass it. Its `map_path` is read
relative to the config file itself, so a scenario and its map travel together:

```bash
python scripts/train.py --env-config ~/experiments/dense_warehouse.yaml
```

Outputs (`checkpoints/`, `episode.svg`) are written relative to wherever you
run the script.