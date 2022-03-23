import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from vapo.affordance.dataset_creation.core.data_reader import DataReader
from vapo.affordance.utils.utils import get_abs_path

log = logging.getLogger(__name__)


class TasksDiscovery(DataReader):
    def __init__(self, cfg, *args, **kwargs):
        super(TasksDiscovery, self).__init__(cfg, *args, **kwargs)
        self._output_dir = get_abs_path(cfg.output_dir)
        self.trajectories = {}
        self.labels_lst = []

        self.dist_thresh = cfg.task_discovery.dist_thresh
        self.sample_freq = cfg.task_discovery.sample_freq
        self.frames_after_move = cfg.task_discovery.frames_after_move
        self.max_n_episodes = cfg.task_discovery.max_n_episodes

        self.interact_ts_counter = 0
        self.start_point = None
        self.curr_obj = None
        self.directions = []
        self.grasps = []
        self.normalized_directions = []
        self.used_episodes = set([])
        log.info("Writing to %s" % self.output_dir)

    @property
    def output_dir(self):
        return self._output_dir

    def after_loop(self, episode):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        save_file = os.path.join(self.output_dir, "classes_data.json")
        info_dct = {
            "start_to_end_frames": self.frames_after_move,
            "directions": self.directions,
            "trajectories": self.trajectories,
            "grasps": self.grasps,
            "normalized_directions": self.normalized_directions,
            "used_episodes": list(self.used_episodes),
            "output_dir": self.output_dir,
        }
        with open(save_file, "w") as json_file:
            print("saving to: %s" % save_file)
            json.dump(info_dct, json_file, indent=4, sort_keys=True)
        return info_dct

    def after_iteration(self, episode, ep_id, curr_folder):
        """
        :return stop: if true, will stop iterating files
        """
        if episode == self.max_n_episodes:
            return True
        self.used_episodes.add(episode)
        return False

    def closed_gripper(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        self.interact_ts_counter += 1
        if self.interact_ts_counter >= self.sample_freq:
            # print("labeling frames hist")
            self.trajectories[self.curr_obj].append(list(curr_robot_obs))
        if self.interact_ts_counter >= self.frames_after_move:
            curr_pt = curr_robot_obs[:3]
            direction = curr_pt - self.start_point
            self.directions.append(list(direction))
            self.normalized_directions.append(list(direction / np.linalg.norm(direction)))
        super().closed_gripper(dct)

    def closed_to_open(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "frame_idx": frame_idx}
        """
        curr_robot_obs = dct["robot_obs"]
        if self.curr_obj is not None:
            self.trajectories[self.curr_obj].append(list(curr_robot_obs))
        self.interact_ts_counter = 0
        self.grasps.append({"tcp_pos": list(curr_robot_obs[:3]), "tcp_orn": list(curr_robot_obs[3:6])})
        super().closed_to_open(dct)

    def open_to_closed(self, dct):
        """
        dct =  {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "frame_idx": frame_idx,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        self.interact_ts_counter += 1
        self.start_point = curr_robot_obs[:3].copy()

        # New object tracking
        self.curr_obj = self.find_tracked_objects(self.start_point)
        if self.curr_obj == -1:
            self.curr_obj = len(self.trajectories.items())
            self.trajectories[self.curr_obj] = [list(curr_robot_obs)]
        else:
            self.trajectories[self.curr_obj].append(list(curr_robot_obs))

        self.grasps.append({"tcp_pos": list(curr_robot_obs[:3]), "tcp_orn": list(curr_robot_obs[3:6])})
        super().open_to_closed(dct)

    def find_tracked_objects(self, query_point):
        # Last point in trajectory list is last known 3d position
        best_match = -1
        if len(self.trajectories) > 0:
            min_dist = 10000  # arbitrary high number

            for label, points in self.trajectories.items():
                # points (n,3), query_point = (3)
                curr_pt = np.expand_dims(query_point, 0)  # 1, 3
                if label in self.labels_lst:  # Track in ep after initialized
                    end_point = points[-1]
                    distance = np.linalg.norm(end_point - query_point)
                    self.labels_lst.append(label)
                else:
                    distance = np.linalg.norm(np.array(points)[:, :3] - curr_pt, axis=-1)

                if np.any(distance < self.dist_thresh):
                    dist = min(distance)
                    if dist < min_dist:
                        best_match = label
        return best_match


@hydra.main(config_path="../../config", config_name="cfg_datacollection")
def main(cfg):
    discovery = TasksDiscovery(cfg)
    discovery.iterate()


if __name__ == "__main__":
    main()
