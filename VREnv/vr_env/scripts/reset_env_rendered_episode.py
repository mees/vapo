from copy import deepcopy
import glob
import os
from pathlib import Path
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from robot_io.utils.utils import matrix_to_orn, matrix_to_pos_orn, orn_to_matrix, pos_orn_to_matrix, quat_to_euler

np.set_printoptions(precision=6, suppress=True)
from vr_env.envs.tasks import Tasks
from vr_env.utils import utils

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


def noise(action, pos_std=0.01, rot_std=1):
    """
    adds gaussian noise to position and orientation.
    units are m for pos and degree for rot
    """
    pos, orn, gripper = action
    rot_std = np.radians(rot_std)
    pos_noise = np.random.normal(0, pos_std, 3)
    rot_noise = p.getQuaternionFromEuler(np.random.normal(0, rot_std, 3))
    pos, orn = p.multiplyTransforms(pos, orn, pos_noise, rot_noise)
    return pos, orn, gripper


# def to_world_frame(action , robot_obs):
#     pos, orn, gripper = np.split(action, [3, 6])
#     orn_mat = orn_to_matrix(orn)
#     T_robot_tcp = orn_to_matrix(robot_obs[3:6])
#
#     pos_w = T_robot_tcp @ pos
#     orn_w = quat_to_euler(matrix_to_orn(T_robot_tcp @ orn_mat))
#     orn_rel = orn_w - robot_obs[3:6]
#     orn_rel = np.where(orn_rel < -np.pi, orn_rel + 2 * np.pi, orn_rel)
#     orn_rel = np.where(orn_rel > np.pi, orn_rel - 2 * np.pi, orn_rel)
#
#     return np.array([*pos_w, *orn_rel, *gripper])
#
def to_world_frame(action, robot_obs):
    pos, orn_tcp_rel, gripper = np.split(action, [3, 6])
    T_world_tcp_old = orn_to_matrix(robot_obs[3:6])
    pos_w_rel = T_world_tcp_old[:3, :3] @ pos
    orn_tcp_rel *= 0.01
    T_tcp_new_tcp_old = orn_to_matrix(orn_tcp_rel)

    T_world_tcp_new = T_world_tcp_old @ np.linalg.inv(T_tcp_new_tcp_old)
    orn_w_new = matrix_to_orn(T_world_tcp_new)
    orn_w_new = quat_to_euler(orn_w_new)
    orn_w_rel = orn_w_new - robot_obs[3:6]
    orn_w_rel = np.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
    orn_w_rel = np.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
    orn_w_rel *= 100
    return np.array([*pos_w_rel, *orn_w_rel, *gripper])


# def to_tcp_frame(action, robot_obs):
#     pos_w, orn_w, gripper = np.split(action, [3, 6])
#     orn_w_mat = orn_to_matrix(orn_w)
#     T_tcp_world = np.linalg.inv(orn_to_matrix(robot_obs[3:6]))
#     pos_tcp = T_tcp_world @ pos_w
#     orn_tcp = quat_to_euler(matrix_to_orn(T_tcp_world @ np.linalg.inv(orn_w_mat)))
#     orn_rel = orn_tcp - quat_to_euler(matrix_to_orn(T_tcp_world ))
#     orn_rel = np.where(orn_rel < -np.pi, orn_rel + 2 * np.pi, orn_rel)
#     orn_rel = np.where(orn_rel > np.pi, orn_rel - 2 * np.pi, orn_rel)
#     return np.array([*pos_tcp, *orn_rel, *gripper])


def to_tcp_frame(action, robot_obs):
    pos_w, orn_w_rel, gripper = np.split(action, [3, 6])
    orn_w_rel *= 0.01
    T_tcp_world = np.linalg.inv(orn_to_matrix(robot_obs[3:6]))
    pos_tcp_rel = T_tcp_world @ pos_w
    new_orn_w = robot_obs[3:6] + orn_w_rel
    T_world_tcp = orn_to_matrix(robot_obs[3:6])
    T_world_tcp_new = orn_to_matrix(new_orn_w)
    T_tcp_new_tcp_old = np.linalg.inv(T_world_tcp_new) @ T_world_tcp
    orn_tcp_rel = quat_to_euler(matrix_to_orn(T_tcp_new_tcp_old))
    orn_tcp_rel = np.where(orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel)
    orn_tcp_rel = np.where(orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel)
    orn_tcp_rel *= 100
    return np.array([*pos_tcp_rel, *orn_tcp_rel, *gripper])


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)

    root_dir = Path("/home/hermannl/phd/repos/calvin_sandbox/calvin/dataset/task_D_D/training")

    ep_start_end_ids = np.sort(np.load(root_dir / "ep_start_end_ids.npy"), axis=0)
    rel_actions = []
    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()
    for s, e in ep_start_end_ids:
        print("new_episode")
        for i in range(s, e + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            data = np.load(file)
            # img = data["rgb_static"]
            # cv2.imshow("win2", cv2.resize(img[:, :, ::-1], (500, 500)))
            # cv2.waitKey(1)

            if (i - s) % 32 == 0:
                print(f"reset {i}")
                env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            action = data["rel_actions"]

            action = np.array([-0.0241, -0.0059, -0.0197, 0.0295, 0.0633, 0.1359, 1.0000])
            print(action)
            action_w = np.array([-0.0241, -0.0059, -0.0197, 0.0304, 0.0625, 0.1397, 1.0000])
            robot_obs = np.array([-0.0800, -0.1823, 0.5755, -3.1235, -0.0472, 1.6385])
            action_tcp = to_tcp_frame(action, robot_obs)
            action_world = to_world_frame(action_tcp, robot_obs)
            print(action_world)
            print(action_w)
            print()
            # action = np.split(data["actions"], [3, 6])
            # action = noise(action)

            # rel_actions.append(create_relative_action(data["actions"], data["robot_obs"][:6]))
            # action = utils.to_relative_action(data["actions"], data["robot_obs"], max_pos=0.04, max_orn=0.1)
            # tcp_pos, tcp_orn = p.getLinkState(env.robot.robot_uid, env.robot.tcp_link_id, physicsClientId=env.cid)[:2]
            # tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            # action2 = create_relative_action(data["actions"], np.concatenate([tcp_pos, tcp_orn]))
            o, _, _, info = env.step(action_world)
            # print(info["scene_info"]["lights"]["led"]["logical_state"])
            # if (i - s) % 32 != 0:
            #     print(tasks.get_task_info(prev_info, info))
            # else:
            #     prev_info = deepcopy(info)
            time.sleep(0.01)
    print(time.time() - t1)
    # rel_actions = np.array(rel_actions)
    # for j in range(rel_actions.shape[1]):
    #     plt.figure(j)
    #     plt.hist(rel_actions[:, j], bins=10)
    #     plt.plot()
    #     plt.show()


if __name__ == "__main__":
    run_env()
