import logging
import os
from pathlib import Path
import hydra

from vapo.affordance.dataset_creation.core.data_classifier import BaseDetector, TaskDetector
from vapo.affordance.dataset_creation.core.data_discovery import TasksDiscovery
from vapo.affordance.dataset_creation.data_labeler import DataLabeler
from vapo.affordance.utils.utils import get_abs_path

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    repo_dir = hydra.utils.get_original_cwd()
    parent_folder = os.path.abspath(os.path.join(repo_dir, cfg.paths.parent_folder))
    print(parent_folder)
    cfg.paths.parent_folder = parent_folder
    # Find different objects, trajectories, directions etc for clustering
    if cfg.output_cfg.multiclass:
        output_dir = get_abs_path(cfg.output_dir)
        cluster_info = os.path.join(output_dir, "classes_data.json")

        # 2nd priority
        cluster_info_available = False
        if cfg.task_detector.cluster_info_path is not None:
            cluster_info_path = get_abs_path(cfg.task_detector.cluster_info_path)
            cluster_info_path = os.path.join(cluster_info_path, "classes_data.json")
            cluster_info_available = os.path.isfile(cluster_info_path)

        if not (os.path.isfile(cluster_info) or cluster_info_available):
            log.info("No classes_data.json found... initiating tasks discovery")
            task_discovery = TasksDiscovery(cfg)
            # info_dct = task_discovery.iterate()
            task_discovery.iterate()
            log.info("Tasks discovery finished")

        # Initialize class detector
        classifier = TaskDetector(cfg.task_detector)
        classifier.find_clusters("grasps", k=cfg.task_detector.k_largest)
    else:
        used_episodes = []
        classifier = BaseDetector(cfg.task_detector)

    # Label data
    labeler = DataLabeler(cfg, classifier=classifier, discovery_episodes=used_episodes, new_cfg=False)
    # labeler.after_loop()
    labeler.iterate()


if __name__ == "__main__":
    main()
