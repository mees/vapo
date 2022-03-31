import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vapo.agent.core.utils import get_activation_fn
from vapo.utils.utils import get_abs_path


def get_pos_shape(obs_space, key="robot_obs"):
    _obs_space_keys = list(obs_space.spaces.keys())
    if key in _obs_space_keys:
        return obs_space[key].shape[-1]
    return 0


def get_depth_network(obs_space, out_feat, activation):
    _obs_space_keys = list(obs_space.spaces.keys())
    _activation_fn = get_activation_fn(activation)
    if "depth_obs" in _obs_space_keys:
        _img_size = obs_space["depth_obs"].shape[-1]
        return CNNCommon(1, _img_size, out_feat=out_feat, activation=_activation_fn)
    return None


def get_img_network(obs_space, out_feat, activation, affordance_cfg, cam_type):
    _obs_space_keys = list(obs_space.spaces.keys())
    _activation_fn = get_activation_fn(activation)
    _channels = 0

    img_obs_key = "%s_img_obs" % cam_type
    depth_obs_key = "%s_depth_obs" % cam_type
    if img_obs_key in _obs_space_keys or depth_obs_key in _obs_space_keys:
        if img_obs_key in _obs_space_keys:
            _channels = obs_space[img_obs_key].shape[0]
            _img_size = obs_space[img_obs_key].shape[-1]

        if depth_obs_key in _obs_space_keys:
            _channels += 1
            _img_size = obs_space[depth_obs_key].shape[-1]

        # Affordance config
        aff_model_path = get_abs_path(affordance_cfg.model_path)
        use_affordance = os.path.isfile(aff_model_path) and affordance_cfg.use
        # print("Networks: Using %s cam affordance: %s" % (cam_type, use_affordance))

        # Build network
        return CNNCommon(
            _channels, _img_size, out_feat=out_feat, activation=_activation_fn, use_affordance=use_affordance
        )
    else:
        # print("No image input for %s camera" % cam_type)
        return None


def get_concat_features(aff_cfg, obs, cnn_img=None, cnn_gripper=None):
    features = []

    cam_networks = {
        "static": [cnn_img, aff_cfg.static_cam],
        "gripper": [cnn_gripper, aff_cfg.gripper_cam],
    }

    # Get features from static cam and gripper cam networks
    for cam_type, cam_dict in cam_networks.items():
        cam_net, cam_aff_cfg = cam_dict
        cnn_input = []
        if "%s_img_obs" % cam_type in obs:
            img_obs = obs["%s_img_obs" % cam_type]
            if len(img_obs.shape) == 3:
                img_obs = img_obs.unsqueeze(0)
            cnn_input.append(img_obs)

        # Concat depth
        if "%s_depth_obs" % cam_type in obs:
            img_obs = obs["%s_depth_obs" % cam_type]
            if len(img_obs.shape) == 3:
                img_obs = img_obs.unsqueeze(0)
            cnn_input.append(img_obs)

        # Concat segmentation mask
        if "%s_aff" % cam_type in obs and cam_aff_cfg.use:
            mask = obs["%s_aff" % cam_type]
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            cnn_input.append(mask)
        if len(cnn_input) > 0:
            cnn_input = torch.cat(cnn_input, 1)
            features.append(cam_net(cnn_input))
    if "robot_obs" in obs:
        features.append(obs["robot_obs"])

    if "detected_target_pos" in obs and aff_cfg.gripper_cam.target_in_obs:
        features.append(obs["detected_target_pos"])

    if "target_distance" in obs:
        features.append(obs["target_distance"])
    features = torch.cat(features, dim=-1)
    return features


# cnn common takes the function directly, not the str
class CNNCommon(nn.Module):
    def __init__(self, in_channels, input_size, out_feat, use_affordance=False, activation=F.relu):
        super(CNNCommon, self).__init__()
        w, h = self.calc_out_size(input_size, input_size, 8, 0, 4)
        w, h = self.calc_out_size(w, h, 4, 0, 2)
        w, h = self.calc_out_size(w, h, 3, 0, 1)

        # Load affordance model
        aff_channels = 0
        if use_affordance:
            aff_channels = 1  # Concatenate aff mask to inputs

        self.conv1 = nn.Conv2d(in_channels + aff_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.spatial_softmax = SpatialSoftmax(h, w)
        self.fc1 = nn.Linear(2 * 64, out_feat)  # spatial softmax output

        self._activation = activation

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self._activation(self.conv1(x))
        x = self._activation(self.conv2(x))
        x = self.spatial_softmax(self.conv3(x))
        x = self.fc1(x).squeeze()  # bs, out_feat
        return x

    def calc_out_size(self, w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2 * padding) // stride + 1
        height = (h - kernel_size + 2 * padding) // stride + 1
        return width, height


class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    # https://github.com/naruya/spatial_softmax-pytorch
    # https://github.com/cbfinn/gps/blob/82fa6cc930c4392d55d2525f6b792089f1d2ccfe/python/gps/algorithm/policy_opt/tf_model_example.py#L168
    def __init__(self, num_rows, num_cols):
        super(SpatialSoftmax, self).__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols

        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda()  # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda()  # W*H

    def forward(self, x):
        # batch, C, W*H
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = F.softmax(x, dim=2)  # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map)  # batch, C
        fp_y = torch.matmul(x, self.y_map)  # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x  # batch, C*2
