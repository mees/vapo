import json
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import tqdm

from vapo.affordance.dataset_creation.core.utils import check_file, get_files


def find_clusters(directions):
    data = np.array(list(directions.values()))
    clustering = KMeans(n_clusters=5, random_state=0).fit(data)
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
    plt.show()


def label_motion(cfg):
    # Episodes info
    # ep_lens = np.load(os.path.join(cfg.play_data_dir, "ep_lens.npy"))
    ep_start_end_ids = np.load(os.path.join(cfg.play_data_dir, "ep_start_end_ids.npy"))
    end_ids = ep_start_end_ids[:, -1]

    # Iterate rendered_data
    # Sorted files
    files = get_files(cfg.play_data_dir, "npz")
    directions = {}
    frames_hist = []

    start_counter = 0
    max_dir_ts = 4  # How many ts let pass before computing direction
    past_action = 1
    start_point = None

    episode = 0
    n_episodes = 1
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
        ep_id = int(tail[:-4].split("_")[-1])
        end_of_ep = ep_id >= end_ids[0] + 1 and len(end_ids) > 1
        if data["actions"][-1] <= 0 or end_of_ep:  # closed gripper
            # Get mask for static images
            # open -> closed
            if past_action == 1:
                start_counter += 1
                start_point = point
            else:
                # mask on close
                # Was closed and remained closed
                # Last element in gripper_hist is the newest
                start_counter += 1
                if start_counter >= max_dir_ts:
                    # print("labeling frames hist")
                    frames_dir = point - start_point
                    start_point = point
                    directions.update({frames_hist[-1]: list(frames_dir)})
                    frames_hist = []
        # Open gripper
        else:
            # Closed -> open transition
            start_counter = 0

        # Reset everything
        if end_of_ep:
            end_ids = end_ids[1:]
            past_action = 1  # Open
            episode += 1

        past_action = data["actions"][-1]

    save_dirs = os.path.join(cfg.output_dir, "motion_vectors_ts-%d.json" % max_dir_ts)
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
    # dirs = load_json('C:/Users/Jessica/Documents/Proyecto_ssd/datasets/tmp_test/motion_vectors_ts-4.json')
    find_clusters(dirs)


if __name__ == "__main__":
    main()
