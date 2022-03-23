import torch
import torch.nn as nn
import torch.nn.functional as F
from vapo.agent.core.utils import get_activation_fn
from vapo.agent.networks.networks_common import \
     get_pos_shape, get_img_network, get_concat_features


# q function
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 activation="relu", hidden_dim=256, **kwargs):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self._activation = get_activation_fn(activation)

    def forward(self, states, actions):
        x = torch.cat((states, actions), -1)
        x = self._activation(self.fc1(x))
        x = self._activation(self.fc2(x))
        return self.q(x).squeeze()


class CNNCritic(nn.Module):
    def __init__(self, obs_space, action_dim, affordance=None,
                 hidden_dim=256, activation="relu", latent_dim=16, **kwargs):
        super(CNNCritic, self).__init__()
        _tcp_pos_shape = get_pos_shape(obs_space, "robot_obs")
        _target_pos_shape = get_pos_shape(obs_space, "detected_target_pos")
        _distance_shape = get_pos_shape(obs_space, "target_distance")
        self.cnn_img = get_img_network(
                            obs_space,
                            out_feat=latent_dim,
                            activation=activation,
                            affordance_cfg=affordance.static_cam,
                            cam_type="static")
        self.cnn_gripper = get_img_network(
                            obs_space,
                            out_feat=latent_dim,
                            activation=activation,
                            affordance_cfg=affordance.gripper_cam,
                            cam_type="gripper")
        out_feat = 0
        for net in [self.cnn_img, self.cnn_gripper]:
            if(net is not None):
                out_feat += latent_dim
        out_feat += _tcp_pos_shape + _target_pos_shape + _distance_shape
        self.out_feat = out_feat
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self._activation = activation
        self.fc0 = nn.Linear(out_feat, 32)
        self.fc1 = nn.Linear(32 + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.aff_cfg = affordance

    def forward(self, states, actions):
        features = get_concat_features(self.aff_cfg,
                                       states,
                                       self.cnn_img,
                                       self.cnn_gripper)
        x = F.elu(self.fc0(features))
        x = torch.cat((x, actions), -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.q(x).squeeze()
        return x


class CNNCriticRes(CNNCritic):
    def __init__(self, *args, **kwargs):
        super(CNNCriticRes, self).__init__(*args, **kwargs)
        out_size = self.hidden_dim + self.out_feat + self.action_dim
        self.fc0 = nn.Linear(self.out_feat + self.action_dim, self.hidden_dim)
        self.fc1 = nn.Linear(out_size, self.hidden_dim)
        self.fc2 = nn.Linear(out_size, self.hidden_dim)
        self.fc3 = nn.Linear(out_size, self.hidden_dim)
        self.q = nn.Linear(self.hidden_dim, 1)

    def forward(self, states, actions):
        features = get_concat_features(self.aff_cfg,
                                       states,
                                       self.cnn_img,
                                       self.cnn_gripper)
        features = torch.cat((features, actions), -1)
        x = F.elu(self.fc0(features))
        x = torch.cat([x, features], -1)
        x = F.elu(self.fc1(x))
        x = torch.cat([x, features], -1)
        x = F.elu(self.fc2(x))
        x = torch.cat([x, features], -1)
        x = F.elu(self.fc3(x))
        x = self.q(x).squeeze()
        return x


class CNNCriticDenseNet(CNNCritic):
    def __init__(self, *args, **kwargs):
        super(CNNCriticDenseNet, self).__init__(*args, **kwargs)
        n_layers = kwargs["n_layers"]
        self.fc_layers = []

        out_size = self.out_feat + self.action_dim
        for i in range(n_layers):
            self.fc_layers.append(nn.Linear(out_size, self.hidden_dim))
            out_size += self.hidden_dim

        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.q = nn.Linear(out_size, 1)

    def forward(self, states, actions):
        features = get_concat_features(self.aff_cfg,
                                       states,
                                       self.cnn_img,
                                       self.cnn_gripper)
        x_in = torch.cat((features, actions), -1)
        for layer in self.fc_layers:
            x_out = F.silu(layer(x_in))
            x_in = torch.cat([x_out, x_in], -1)

        x = self.q(x_in).squeeze()
        return x
