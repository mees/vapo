import hydra
import logging
import os
from robot_io.cams.realsense.realsense import Realsense
from vapo.wrappers.affordance.aff_wrapper_real_world import AffordanceWrapperRealWorld
from vapo.wrappers.utils import get_name
from vapo.agent.vapo_real_world import VAPOAgent
from vapo.wrappers.real_world.panda_tabletop_wrapper import PandaEnvWrapper


@hydra.main(config_path="../config", config_name="cfg_real_world")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.robot_env, robot=robot)
    env = PandaEnvWrapper(**cfg.panda_env_wrapper, env=env)
    env = AffordanceWrapperRealWorld(env,
                                     max_ts=cfg.agent.learn_config.max_episode_length,
                                     train=True,
                                     affordance_cfg=cfg.affordance,
                                     save_images=cfg.save_images,
                                     real_world=True,
                                     **cfg.env_wrapper)
    sac_cfg = {"env": env,
               "eval_env": None,
               "model_name": cfg.model_name,
               "save_dir": cfg.agent.save_dir,
               "net_cfg": cfg.agent.net_cfg,
               "train_mean_n_ep": cfg.agent.train_mean_n_ep,
               "save_replay_buffer": cfg.agent.save_replay_buffer,
               "log": log,
               **cfg.agent.hyperparameters}

    log.info("model: %s" % cfg.model_name)
    model = VAPOAgent(cfg,
                      sac_cfg=sac_cfg,
                      target_search_mode=cfg.target_search,
                      wandb_login=cfg.wandb_login,
                      rand_target=True)
    if cfg.resume_training:
        original_dir = hydra.utils.get_original_cwd()
        model_path = os.path.join(original_dir, cfg.resume_model_path)
        path = "%s/trained_models/%s.pth" % (model_path,
                                             cfg.model_name + "_last")
        if os.path.exists(path):
            print("loading model: %s" % path)
            model.load(path)
        else:
            print("Model path does not exist: %s \n Training from start"
                  % os.path.abspath(path))
    model.learn(**cfg.agent.learn_config)
    env.close()


if __name__ == "__main__":
    main()
