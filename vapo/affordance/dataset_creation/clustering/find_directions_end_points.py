import json
import os
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import tqdm

from vapo.affordance.dataset_creation.core.utils import check_file, get_files


def find_clusters(directions, root_dir):
    data = np.array(list(directions.values()))
    clustering = KMeans(n_clusters=6, random_state=0).fit(data)
    # clustering = DBSCAN(eps=0.03, min_samples=3).fit(data)
    labels = clustering.labels_
    # color to label
    n_labels = np.unique(labels)
    cm = plt.get_cmap("tab10")
    colors = cm(np.linspace(0, 1, len(n_labels)))

    # Plot clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # For each set of style and range settings, plot n random points in the box
    viz_n = -1
    for c, label in zip(colors, n_labels):
        indices = np.where(labels == label)
        cluster = data[indices]
        ax.scatter(cluster[:viz_n, 0], cluster[:viz_n, 1], cluster[:viz_n, 2], c=c)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    save_dir = os.path.join(root_dir, "k_means.pkl")
    with open(save_dir, "wb") as f:
        pickle.dump(clustering, f)
    plt.show()


# Find unit motion vectors from n episodes
def label_motion(cfg):
    # Episodes info
    ep_start_end_ids = np.load(os.path.join(cfg.play_data_dir, "ep_start_end_ids.npy"))
    end_ids = ep_start_end_ids[:, -1]

    # Iterate rendered_data
    # Sorted files
    files = get_files(cfg.play_data_dir, "npz")
    directions = {}
    frames_hist = []

    past_action = 1
    start_point = None

    episode = 0
    n_episodes = 3
    files = files[: end_ids[n_episodes - 1]]
    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = check_file(filename)
        if data is None:
            continue  # Skip file

        _, tail = os.path.split(filename)

        # Initialize img, mask, id
        frame_id = tail[:-4]
        point = data["robot_obs"][:3]
        frames_hist.append(frame_id)

        # Start of interaction
        ts_id = int(tail[:-4].split("_")[-1])
        end_of_ep = ts_id >= end_ids[0] + 1 and len(end_ids) > 1
        if data["actions"][-1] <= 0 or end_of_ep:  # closed gripper
            # Get mask for static images
            # open -> closed
            if past_action == 1:
                start_point = point
        # Open gripper a = 1
        else:
            # Closed -> open transition
            if past_action <= 0:
                direction = point - start_point
                direction = direction / np.linalg.norm(direction)
                start_point = point
                directions.update({frames_hist[-1]: list(direction)})
                frames_hist = []

        # Reset everything
        if end_of_ep:
            end_ids = end_ids[1:]
            past_action = 1  # Open
            episode += 1

        past_action = data["actions"][-1]

    save_dirs = os.path.join(cfg.output_dir, "motion_vectors_normalized.json")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with open(save_dirs, "w") as json_file:
        print("saving to: %s" % save_dirs)
        json.dump(directions, json_file, indent=4, sort_keys=True)
    return directions


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    dirs = label_motion(cfg)
    # dirs = load_json('%s/motion_vectors_normalized.json' % cfg.output_dir)
    find_clusters(dirs, cfg.output_dir)


if __name__ == "__main__":
    main()
