import json
import os

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import tqdm

from vapo.affordance.dataset_creation.core.data_discovery import TaskDetector
from vapo.affordance.dataset_creation.core.utils import check_file, get_data


def plot_clusters(data):
    labels = list(data.keys())
    cm = plt.get_cmap("tab10")
    colors = cm(np.linspace(0, 1, len(labels)))

    # Plot clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # For each set of style and range settings, plot n random points in the box
    viz_n = -1
    for c, label in zip(colors, labels):
        cluster = np.array(data[label])
        ax.scatter(cluster[:viz_n, 0], cluster[:viz_n, 1], cluster[:viz_n, 2], color=c)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(labels)
    plt.show()


# If query_pt within neighborhood of any end_pt in trajectory
def find_tracked_objects(trajectories, query_point, labels_lst, neighborhood=0.04):
    # Last point in trajectory list is last known 3d position
    best_match = -1
    if len(trajectories) > 0:
        min_dist = 10000  # arbitrary high number

        for label, points in trajectories.items():
            # points (n,3), query_point = (3)
            curr_pt = np.expand_dims(query_point, 0)  # 1, 3
            if label in labels_lst:  # Track in ep after initialized
                end_point = points[-1]
                distance = np.linalg.norm(end_point - query_point)
                labels_lst.append(label)
            else:
                distance = np.linalg.norm(np.array(points) - curr_pt, axis=-1)

            if np.any(distance < neighborhood):
                dist = min(distance)
                if dist < min_dist:
                    best_match = label
    return best_match, labels_lst


def label_motion(cfg):
    files, end_ids = get_data(cfg.play_data_dir, cfg.labeling.real_world)

    start_counter = 0
    sample_freq = 10  # How many ts let pass before computing direction
    past_action = 1
    start_point = None
    trajectories = {}
    curr_obj = None

    n_episodes = 1
    episode = 0
    initialized_labels = []
    files = files[:4000]

    # Iterate rendered_data
    head, tail = os.path.split(files[0])
    ep_id = int(tail[:-4].split("_")[-1])
    # Multiple folders in real-robot data
    _, curr_folder = os.path.split(head)

    task_detector = TaskDetector(cfg)
    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = check_file(filename)
        if data is None:
            continue  # Skip file

        _, tail = os.path.split(filename)

        # Initialize img, mask, id
        robot_obs = task_detector.get_robot_state(data)
        point = robot_obs[:3]
        rgb_img = data["rgb_static"]
        rgb_img = cv2.resize(rgb_img, (300, 300))
        cv2.imshow("Static", rgb_img[:, :, ::-1])
        cv2.waitKey(1)

        if idx < len(files) - 1:
            next_folder = os.path.split(os.path.split(files[idx + 1])[0])[-1]
        else:
            next_folder = curr_folder

        # Start of interaction
        if not cfg.labeling.real_world:
            ep_id = int(tail[:-4].split("_")[-1])
            end_of_ep = ep_id >= end_ids[0] + 1 and len(end_ids) > 1
            gripper_action = data["actions"][-1]  # -1 -> closed, 1 -> open
        else:
            ep_id = int(tail[:-4].split("_")[-1])
            end_of_ep = (len(end_ids) > 1 and ep_id >= end_ids[0]) or curr_folder != next_folder
            gripper_action = robot_obs[-1] > 0.077  # Open
            gripper_action = (data["action"].item()["motion"][-1] + 1) / 2

        if gripper_action <= 0 or end_of_ep:  # open gripper
            # Get mask for static images
            # open -> closed
            if past_action == 1:
                start_counter += 1
                start_point = point

                # New object tracking
                curr_obj, initialized_labels = find_tracked_objects(trajectories, point, initialized_labels)
                if curr_obj == -1:
                    curr_obj = len(trajectories.items())
                    trajectories[curr_obj] = [list(start_point)]
                else:
                    trajectories[curr_obj].append(list(start_point))
            else:
                # mask on close
                # Was closed and remained closed
                # Last element in gripper_hist is the newest
                start_counter += 1
                if start_counter >= sample_freq:
                    # print("labeling frames hist")
                    direction = point - start_point
                    start_point = point
                    trajectories[curr_obj].append(list(point))
        # Open gripper
        else:
            # Closed -> open transition
            if past_action <= 0:
                if curr_obj is not None:
                    trajectories[curr_obj].append(list(point))
                start_counter = 0

        # Only collect trajectories for a given number of episodes
        if episode == n_episodes:
            break

        # Reset everything
        if end_of_ep:
            end_ids = end_ids[1:]
            past_action = 1  # Open
            episode += 1
            curr_folder = next_folder
            initialized_labels = []

        past_action = gripper_action

    save_dirs = os.path.join(task_detector.output_dir, "trajectories.json")
    if not os.path.exists(task_detector.output_dir):
        os.makedirs(task_detector.output_dir)
    with open(save_dirs, "w") as json_file:
        print("saving to: %s" % save_dirs)
        json.dump(trajectories, json_file, indent=4, sort_keys=True)
    task_detector.find_clusters(trajectories)
    return trajectories


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def find_most_sampled(trajectories):
    k = 3
    n_samples = {k: len(list(v)) for k, v in sorted(trajectories.items(), reverse=True, key=lambda item: len(item[1]))}
    new_dict = {}
    sorted_keys = list(n_samples.keys())[:k]
    for i in range(k):
        new_dict[i] = trajectories[sorted_keys[i]]
    return new_dict


@hydra.main(config_path="../../../config", config_name="cfg_datacollection")
def main(cfg):
    pos = label_motion(cfg)
    # pos = load_json('/mnt/ssd_shared/Users/Jessica/Documents/Proyecto_ssd/datasets/tmp_test/trajectories.json')
    # pos = load_json("C:/Users/Jessica/Documents/Proyecto_ssd/datasets/playtable_multiclass_200px_MoC/trajectories_3objs.json")
    # pos = find_most_sampled(pos)
    plot_clusters(pos)


if __name__ == "__main__":
    main()
