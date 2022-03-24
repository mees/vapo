import logging
import os.path

import hydra
from omegaconf import OmegaConf
from robot_io.cams.realsense.realsense import Realsense

from vapo.agent.vapo_agent import VAPOAgent
from vapo.wrappers.affordance.aff_wrapper_real_world import AffordanceWrapperRealWorld
from vapo.wrappers.real_world.panda_tabletop_wrapper import PandaEnvWrapper


@hydra.main(config_path="./config", config_name="cfg_tabletop_real")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    model_cfg = os.path.join(cfg.test.folder_name, ".hydra/config.yaml")
    run_cfg = OmegaConf.load(model_cfg)

    robot = hydra.utils.instantiate(run_cfg.robot)
    env = hydra.utils.instantiate(run_cfg.robot_env, robot=robot)
    env = PandaEnvWrapper(**run_cfg.panda_env_wrapper, env=env)

    env = AffordanceWrapperRealWorld(
        env,
        max_ts=run_cfg.agent.learn_config.max_episode_length,
        train=False,
        affordance_cfg=run_cfg.affordance,
        viz=cfg.viz_obs,
        save_images=cfg.save_images,
        real_world=True,
        **cfg.env_wrapper,
    )

    sac_cfg = {
        "env": env,
        "eval_env": None,
        "model_name": run_cfg.model_name,
        "save_dir": run_cfg.agent.save_dir,
        "net_cfg": run_cfg.agent.net_cfg,
        "log": log,
        **run_cfg.agent.hyperparameters,
    }

    run_cfg.save_images = cfg.save_images
    log.info("model: %s" % run_cfg.model_name)

    model = VAPOAgent(run_cfg, sac_cfg=sac_cfg, target_search_mode=run_cfg.target_search, rand_target=True)

    original_dir = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_dir, cfg.resume_model_path)
    path = "%s/trained_models/%s.pth" % (cfg.test.folder_name, cfg.test.model_name)
    if os.path.exists(path):
        model.load(path, load_replay_buffer=False)
    else:
        print("Model path does not exist: %s \n" % os.path.abspath(path))
        return
    # model.evaluate(env, deterministic=True,
    #                    **cfg.test.eval_cfg)
    model.tidy_up(env, deterministic=True, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
