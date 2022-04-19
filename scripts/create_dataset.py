import logging
import os
import hydra

from vapo.affordance.dataset_creation.core.data_classifier import BaseDetector
from vapo.affordance.dataset_creation.data_labeler import DataLabeler
from vapo.affordance.dataset_creation.merge_datasets import merge_datasets
import glob
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    repo_dir = hydra.utils.get_original_cwd()
    parent_folder = os.path.abspath(os.path.join(repo_dir, cfg.paths.parent_folder))
    print(parent_folder)
    cfg.paths.parent_folder = parent_folder

    # Label data
    play_data_directories = glob.glob(cfg.play_data_dir + "/*/")
    _output_dir = cfg.output_dir
    output_directories = []
    for data_dir in play_data_directories:
        play_data_name = os.path.basename(os.path.normpath(data_dir))
        cfg.output_dir = os.path.join(_output_dir, play_data_name)
        output_directories.append(cfg.output_dir)

        cfg.play_data_dir = data_dir
        used_episodes = []
        classifier = BaseDetector(cfg)
        labeler = DataLabeler(cfg, classifier=classifier, discovery_episodes=used_episodes)
        labeler.iterate()
    merge_datasets(_output_dir, output_directories)


if __name__ == "__main__":
    main()
