import gym
import cv2
import numpy as np
import torch
from vapo.affordance.utils.utils import get_transforms


def get_name(cfg, model_name):
    offset_dist = str(cfg.env.offset[-1]).split('.')[-1]
    model_name += "_%s" % offset_dist

    timesteps = str(cfg.agent.learn_config.max_episode_length)
    model_name += "_%sts" % timesteps
    if(cfg.env_wrapper.gripper_cam.use_img):
        model_name += "_img"
    if(cfg.env_wrapper.gripper_cam.use_depth):
        model_name += "_depth"
    if(cfg.affordance.gripper_cam.target_in_obs):
        model_name += "_target"
    if(cfg.affordance.gripper_cam.use_distance):
        model_name += "_dist"
    if(cfg.affordance.gripper_cam.use):
        model_name += "_affMask"
    if(cfg.affordance.gripper_cam.densify_reward):
        model_name += "_dense"
    else:
        model_name += "_sparse"
    return model_name


def find_cam_ids(cameras):
    static_id, gripper_id, render_cam_id = 0, 1, None
    for i, cam in enumerate(cameras):
        if "gripper" in cam.name:
            gripper_id = i
        elif "static" in cam.name:
            static_id = i
        elif "render" in cam.name:
            render_cam_id = i
    cameras = {"static": static_id,
               "gripper": gripper_id}
    if(render_cam_id):
        cameras.update({"render": render_cam_id})
    return cameras


def get_obs_space(affordance_cfg, gripper_cam_cfg, static_cam_cfg,
                  channels, img_size, use_robot_obs, task,
                  real_world=False, oracle=False):
    cfg_dict = {
        "gripper": [gripper_cam_cfg, affordance_cfg.gripper_cam],
        "static": [static_cam_cfg, affordance_cfg.static_cam]
    }
    obs_space_dict = {}
    for cam_type, config in cfg_dict.items():
        env_obs_cfg, aff_cfg = config
        if(env_obs_cfg.use_img):
            obs_space_dict["%s_img_obs" % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(channels, img_size, img_size))
        if(env_obs_cfg.use_depth):
            obs_space_dict['%s_depth_obs' % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(1, img_size, img_size))
        if(aff_cfg.use):
            obs_space_dict['%s_aff' % cam_type] = gym.spaces.Box(
                low=0, high=1,
                shape=(1, img_size, img_size))
    if(use_robot_obs):
        # *tcp_pos(3), *tcp_euler(1), gripper_width, gripper_action(1),
        if(task == "pickup"):
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(6,))
        else:
            if(real_world):
                # pos +  width
                obs_shape = 4
            else:
                obs_shape = 9 if oracle else 8
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(obs_shape,))
    if(affordance_cfg.gripper_cam.target_in_obs):
        obs_space_dict['detected_target_pos'] = gym.spaces.Box(
            low=-1, high=1, shape=(3,))
    if(affordance_cfg.gripper_cam.use_distance):
        obs_space_dict['target_distance'] = gym.spaces.Box(
            low=-1, high=1, shape=(1,))
    return gym.spaces.Dict(obs_space_dict)


def depth_preprocessing(frame, img_size):
    # obs is from 0-255, (img_size, img_size, 1)
    new_frame = cv2.resize(
            frame,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA)
    # (1, img_size, img_size)
    new_frame = np.expand_dims(new_frame, axis=0)
    return new_frame


def get_transforms_and_shape(transforms_cfg, in_size, out_size=None):
    apply_transforms = get_transforms(transforms_cfg, img_size=out_size)
    test_tensor = torch.zeros((3, in_size, in_size))
    test_tensor = apply_transforms(test_tensor)
    # History length
    if(test_tensor.shape == 2):
        test_tensor = test_tensor.unsqueeze(0)
    return apply_transforms, test_tensor.shape
