import gym
import hydra
from hydra.experimental import compose, initialize
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import pytest

from vr_env.envs.play_table_env import PlayTableSimEnv


@pytest.fixture(scope="module")
def env() -> PlayTableSimEnv:
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config_data_collection.yaml",
            overrides=[
                "env.show_gui=False",
                "env.use_vr=False",
                "env.use_egl=True",
                "robot.euler_obs=True",
                "env.use_scene_info=True",
            ],
        )
        camera_conf = DictConfig(
            {
                "_target_": "vr_env.camera.static_camera.StaticCamera",
                "name": "static",
                "fov": 75,
                "aspect": 1,
                "nearval": 0.01,
                "farval": 2,
                "width": 200,
                "height": 200,
                "look_at": [0.1, 0.8, 0.7],
                "look_from": [0.3, -0.2, 1.3],
            }
        )
        cfg.env.cameras = ListConfig([camera_conf])
        assert isinstance(cfg, DictConfig)
        assert isinstance(cfg.env, DictConfig)
        assert isinstance(cfg.robot, DictConfig)
        assert isinstance(cfg.env.cameras, ListConfig)
        env = hydra.utils.instantiate(cfg.env)
        return env


def test_render(env):
    obs = env.reset()
    info = env.get_info()
    action = (info["robot_info"]["tcp_pos"], info["robot_info"]["tcp_orn"], info["robot_info"]["gripper_action"])
    for i in range(100):
        obs, r, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert 0 < np.mean(img) < 255


def test_reset(env):
    obs1 = env.reset()
    obs2 = env.reset(robot_obs=obs1["robot_obs"], scene_obs=obs1["scene_obs"])
