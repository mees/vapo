import json
import os
import shutil

import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from vapo.affordance.utils.utils import get_abs_path


class BaseDetector:
    def __init__(self, cfg, *args, **kwargs):
        self.n_classes = 1
        cm = plt.get_cmap("jet")
        self._colors = cm(np.linspace(0, 1, self.n_classes))
        self.clusters = {}

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value

    def predict(self, new_point):
        return 0


class TaskDetector(BaseDetector):
    def __init__(self, cfg, data_info=None, *args, **kwargs):
        super().__init__()
        if cfg.clustering_method.lower() == "kmeans":
            _clustering_method = KMeans(**cfg.params)
        else:  # dbscan
            _clustering_method = DBSCAN(**cfg.params)
        self._clustering_method = _clustering_method
        self._clustering = None
        self.interact_ts_counter = 0
        self.start_point = None
        self.curr_obj = None
        self.directions = []
        if data_info is not None:
            self.data = data_info
        else:
            self.data = self.read_data(cfg.dataset_dir, cfg.cluster_info_path)
        self.dims = cfg.dims

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        self._clustering = value

    def read_data(self, data_path, cluster_info_path):
        """
        :param data_path(str): directory of the dataset

        :return data(dict):
            trajectories (list): trajectories that a given tracked object followed
            directions (list): directions in which the tcp moved after interaction after start_to_end_frames
            start_to_end_frames (int): Amount of frames from interaction_start to start_to_end_frames to compute the directions
        """
        data_dir = get_abs_path(data_path)
        data_filename = os.path.join(data_dir, "classes_data.json")
        if not os.path.isfile(data_filename):
            # Check other path
            print("No cluster info found in: %s" % data_filename)
            if cluster_info_path is not None:
                source_file = get_abs_path(cluster_info_path)
                source_file = os.path.join(source_file, "classes_data.json")
                if not os.path.isfile(source_file):
                    print("No cluster info found in: %s" % source_file)
                else:
                    os.makedirs(get_abs_path(data_dir), exist_ok=True)
                    shutil.copy2(source_file, data_filename)

        with open(data_filename) as json_file:
            data = json.load(json_file)
        return data

    def plot_clusters(self):
        # Plot clusters
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        viz_n = -1  # label = -1 if noise
        labels = []
        for label, info in self.clusters.items():
            cluster = info["data"]
            if label > -1:
                c = self.colors[label]
                ax.scatter(cluster[:viz_n, 0], cluster[:viz_n, 1], cluster[:viz_n, 2], c=c)
            labels.append(label)
        # Origin
        ax.scatter(0, 0, 0, c="black", s=20, marker="x")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(labels)
        return ax

    def find_clusters(self, dict_str="trajectories", k=-1):
        if dict_str == "trajectories":
            # Ignore z angle
            clustering_data = [np.array(v)[:, :6] for v in self.data[dict_str].values()]
            full_data = np.vstack(clustering_data)
            data = full_data[:, self.dims]
        elif dict_str == "grasps":
            clustering_data = [[*x["tcp_pos"], *x["tcp_orn"]] for x in self.data[dict_str]]
            full_data = np.vstack(clustering_data)
            data = full_data[:, self.dims]
        else:
            clustering_data = self.data[dict_str]
            data = np.vstack(clustering_data)
        self.clustering = self._clustering_method.fit(data)
        labels = self.clustering.labels_

        # color to label
        n_labels = np.unique(labels)
        clusters = {}
        for label in n_labels:
            indices = np.where(labels == label)
            average_orn = np.mean(full_data[indices][:, 3:], axis=0)
            clusters.update({label: {"data": data[indices], "orn": average_orn}})

        # Get n largest clusters
        clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]["data"]), reverse=True)[:k])
        if len(clusters) > k and k > -1:
            clusters = clusters[:k]

        if isinstance(self._clustering_method, DBSCAN):
            self.clustering = {n: value for n, (old_label, value) in enumerate(clusters)}
            self.n_classes = len(clusters)
        else:
            self.n_classes = len(n_labels)
        self.clusters = clusters
        self.set_color_map()
        return clusters

    def set_color_map(self):
        cm = plt.get_cmap("tab10")
        self.colors = cm(np.linspace(0, 1, self.n_classes))

    def predict(self, new_point):
        """
        :param new_point(np.ndarray): len=7 world coords point + orn euler + gripper_width
        """
        if self.n_classes <= 1:
            return 0

        if isinstance(self._clustering_method, KMeans):
            best_match = self.clustering.predict(np.expand_dims(new_point[self.dims], 0)).item()
        else:
            # dbscan
            best_match = 0
            min_dist = float("inf")  # arbitrary high number
            for label, points in self.clustering.items():
                # points (n,3), query_point = (3)
                # curr_pt = np.expand_dims(new_point, 0)[:, self.dims]  # 1, n
                distance = np.linalg.norm(points[:, :3] - new_point[:3], axis=-1)
                dist = np.mean(distance)
                if dist < min_dist:
                    best_match = label
                    min_dist = dist
        return best_match


class TrajectoriesClassifier(TaskDetector):
    def __init__(self, *args, **kwargs):
        super(TrajectoriesClassifier, self).__init__(*args, **kwargs)

    def plot_trajectories(self):
        clusters = self.data["trajectories"]
        cm = plt.get_cmap("tab10")
        colors = cm(np.linspace(0, 1, len(clusters.keys())))

        # Plot clusters
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        viz_n = -1  # label = -1 if noise
        labels = []
        for c, (label, cluster) in zip(colors, clusters.items()):
            labels.append(int(label))
            cluster = np.array(cluster)
            ax.scatter(cluster[:viz_n, 0], cluster[:viz_n, 1], cluster[:viz_n, 2], c=c)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(labels)
        plt.show()

    # Always classify as neighrest distance
    def predict(self, point):
        best_match = 0
        min_dist = 10000  # arbitrary high number
        for label, points in self.data["trajectories"].items():
            # points (n,3), query_point = (3)
            curr_pt = np.expand_dims(point, 0)  # 1, 3
            distance = np.linalg.norm(np.array(points) - curr_pt, axis=-1)
            dist = min(distance)
            if dist < min_dist:
                best_match = label
                min_dist = dist
        # Class 0 is background
        return int(best_match) + 1


@hydra.main(config_path="../../config", config_name="cfg_datacollection")
def main(cfg):
    classifier = TaskDetector(cfg.task_detector)
    classifier.find_clusters("grasps", k=2)
    ax = classifier.plot_clusters()
    # _data = np.vstack([x for x in classifier.data["trajectories"].values()])
    # for pt in _data:
    #     label = classifier.predict(pt)
    #     ax.scatter(pt[0], pt[1], pt[2], c=classifier.colors[label], s=30, marker="x")
    plt.show()
    # classifier = TrajectoriesClassifier(cfg.task_detector)
    # classifier.plot_trajectories()


if __name__ == "__main__":
    main()
