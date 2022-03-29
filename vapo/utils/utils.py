import glob
import os

import gym
import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform.rotation import Rotation as R

from vapo.affordance.affordance_model import AffordanceModel
from vapo.agent.core.utils import set_init_pos


def get_files_regex(path, search_str, recursive):
    files = glob.glob(os.path.join(path, search_str), recursive=recursive)
    if not files:
        print("No *.%s files found in %s" % (search_str, path))
    files.sort()
    return files


# Ger valid numpy files with raw data
def get_files(path, extension, recursive=False):
    if not os.path.isdir(path):
        print("path does not exist: %s" % path)
    search_str = "*.%s" % extension if not recursive else "**/*.%s" % extension
    files = get_files_regex(path, search_str, recursive)
    return files


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


def init_aff_net(affordance_cfg, cam_str=None, in_channels=1):
    aff_net = None
    if affordance_cfg is not None:
        if cam_str is not None:
            aff_cfg = affordance_cfg["%s_cam" % cam_str]
        else:
            aff_cfg = affordance_cfg
        if "use" in aff_cfg and aff_cfg.use:
            path = aff_cfg.model_path
            path = get_abs_path(path)
            # Configuration of the model
            hp = {
                "cfg": aff_cfg.hyperparameters.cfg,
                "n_classes": aff_cfg.hyperparameters.n_classes,
                "input_channels": in_channels,
            }
            hp = OmegaConf.create(hp)
            # Create model
            if os.path.exists(path):
                aff_net = AffordanceModel.load_from_checkpoint(path, **hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: %s cam affordance model loaded" % cam_str)
            else:
                # affordance_cfg = None
                raise TypeError("Path does not exist: %s" % path)
    return aff_net


def change_project_path(cfg, run_cfg):
    net_cfg = run_cfg.agent.net_cfg
    run_cfg.paths.parent_folder = cfg.paths.parent_folder
    # Change affordance path to match current system
    static_cam_aff_path = net_cfg.affordance.static_cam.model_path
    static_cam_aff_path = static_cam_aff_path.replace(run_cfg.models_path, cfg.models_path)
    net_cfg.affordance.static_cam.model_path = static_cam_aff_path

    # Gripper cam
    gripper_cam_aff_path = net_cfg.affordance.gripper_cam.model_path
    gripper_cam_aff_path = gripper_cam_aff_path.replace(run_cfg.models_path, cfg.models_path)
    net_cfg.affordance.gripper_cam.model_path = gripper_cam_aff_path

    # Static cam target_search
    target_search = run_cfg.target_search.model_path
    target_search = target_search.replace(run_cfg.models_path, cfg.models_path)
    run_cfg.target_search.model_path = target_search

    # VREnv data path
    run_cfg.models_path = cfg.models_path
    run_cfg.data_path = run_cfg.data_path.replace(run_cfg.project_path, cfg.project_path)


def load_cfg(cfg_path, cfg, optim_res=False):
    if os.path.exists(cfg_path) and not optim_res:
        run_cfg = OmegaConf.load(cfg_path)
        net_cfg = run_cfg.agent.net_cfg
        env_wrapper = run_cfg.env_wrapper
        agent_cfg = run_cfg.agent.hyperparameters
        run_cfg.paths.parent_folder = cfg.paths.parent_folder
        # change_project_path(cfg, run_cfg)
    else:
        run_cfg = cfg
        net_cfg = cfg.agent.net_cfg
        env_wrapper = cfg.env_wrapper
        agent_cfg = cfg.agent.hyperparameters

    if "init_pos_near" in run_cfg:
        if run_cfg.init_pos_near:
            init_pos = run_cfg.env.robot_cfg.initial_joint_positions
            init_pos = set_init_pos(run_cfg.task, init_pos)
            run_cfg.env.robot_cfg.initial_joint_positions = init_pos
            run_cfg.eval_env.robot_cfg.initial_joint_positions = init_pos
    if "rand_init_state" in run_cfg.env:
        run_cfg.env.pop("rand_init_state")
        run_cfg.eval_env.pop("rand_init_state")
    return run_cfg, net_cfg, env_wrapper, agent_cfg


def get_3D_end_points(x, y, z, w, h, d):
    w = w / 2
    h = h / 2
    box_top_left = [x - w, y + h, z]
    box_bott_right = [x + w, y - h, z + d]
    return (box_top_left, box_bott_right)


def register_env():
    gym.envs.register(
        id="VREnv-v0",
        entry_point="VREnv.vr_env.envs.play_table_env:PlayTableSimEnv",
        max_episode_steps=200,
    )


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler("xyz", orn).as_matrix()
    mat[:3, 3] = pos
    return mat


def get_depth_around_point(point, depth):
    for width in range(5):
        area = depth[point[1] - width : point[1] + width + 1, point[0] - width : point[0] + width + 1]
        area[np.where(area == 0)] = np.inf
        if np.all(np.isinf(area)):
            continue
        new_point = np.array([point[1], point[0]]) + np.array(np.unravel_index(area.argmin(), area.shape)) - width

        assert depth[new_point[0], new_point[1]] != 0
        return (new_point[1], new_point[0]), True
    return None, False
