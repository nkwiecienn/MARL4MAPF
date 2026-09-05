import numpy as np

from warehouse_marl.env import WarehouseEnv
from warehouse_marl.training.sb3_vec_env import WarehouseVecEnv

GRID = ".....\n.....\n.....\n"


def _make_vec(max_episode_steps=20):
    env = WarehouseEnv(
        grid_map=GRID,
        depots={"a": (0, 0), "b": (2, 4)},
        orders={"a": [(0, 2), (0, 4)], "b": [(2, 2)]},
        obs_radius=2,
        max_episode_steps=max_episode_steps,
    )
    return WarehouseVecEnv(env)


def test_one_subenv_per_vehicle():
    vec = _make_vec()
    assert vec.num_envs == 2


def test_reset_shape_matches_num_envs():
    vec = _make_vec()
    obs = vec.reset()
    assert obs.shape[0] == vec.num_envs
    assert obs.dtype == np.float32


def test_all_subenvs_share_one_episode_boundary():
    """SB3 auto-resets per sub-env; if vehicles ended at different times the
    sub-envs would desync, so dones must always be uniform."""
    vec = _make_vec(max_episode_steps=10)
    vec.reset()
    rng = np.random.default_rng(0)
    saw_done = False
    for _ in range(40):
        _, _, dones, infos = vec.step(rng.integers(0, 5, size=vec.num_envs))
        assert dones.all() or not dones.any(), "sub-envs finished at different times"
        if dones.any():
            saw_done = True
            assert all("terminal_observation" in i for i in infos)
    assert saw_done
