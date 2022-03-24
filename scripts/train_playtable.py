import logging

import hydra

from vapo.agent.vapo_agent import VAPOAgent
from vapo.wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.wrappers.play_table_rl import PlayTableRL
from vapo.wrappers.utils import get_name


@hydra.main(config_path="../config", config_name="cfg_playtable")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    max_ts = cfg.agent.learn_config.max_episode_length
    env = PlayTableRL(viz=cfg.viz_obs, **cfg.env)
    training_env = AffordanceWrapperSim(
        env,
        max_ts,
        train=True,
        affordance_cfg=cfg.affordance,
        viz=cfg.viz_obs,
        save_images=cfg.save_images,
        **cfg.env_wrapper,
    )
    sac_cfg = {
        "env": training_env,
        "eval_env": None,
        "model_name": cfg.model_name,
        "save_dir": cfg.agent.save_dir,
        "net_cfg": cfg.agent.net_cfg,
        "train_mean_n_ep": cfg.agent.train_mean_n_ep,
        "save_replay_buffer": cfg.agent.save_replay_buffer,
        "log": log,
        **cfg.agent.hyperparameters,
    }

    log.info("model: %s" % cfg.model_name)
    cfg.target_search.mode = "affordance"
    model = VAPOAgent(cfg, sac_cfg=sac_cfg, wandb_login=cfg.wandb_login)
    model.learn(**cfg.agent.learn_config)
    training_env.close()


if __name__ == "__main__":
    main()
