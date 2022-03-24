import importlib
import os
import pickle

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def set_init_pos(task, init_pos):
    if task == "slide":
        init_pos = [
            -1.1686195081948965,
            1.5165126497924815,
            1.7042540963745911,
            -1.6031852712241403,
            -2.5717679087567484,
            2.331416872629473,
            -1.3006358472301627,
        ]
    elif task == "drawer":
        init_pos = [
            -0.4852725866746207,
            1.0618989199760496,
            1.3903811172536515,
            -1.7446581003391255,
            -1.1359501486104144,
            1.8855365146855005,
            -1.3092771579652827,
        ]
    elif task == "banana":
        init_pos = [
            0.03740465833778156,
            1.1844912206595481,
            1.1330028132229706,
            -0.6702560563758552,
            -1.1188250499368455,
            1.6153329476732947,
            -1.7078632665627795,
        ]
    elif task == "hinge":
        init_pos = [
            -0.3803066514807313,
            0.931053115322005,
            1.1668869976984892,
            -0.8602164833917604,
            -1.4818301463768684,
            2.78299286093898,
            -1.7318962831826747,
        ]
    return init_pos


def get_nets(img_obs, obs_space, action_space, log, actor_net_str=None, critic_net_str=None):
    action_dim = action_space.shape[0]
    if img_obs:
        log.info("SAC get_nets using: %s" % str([k for k in obs_space]))
        policy = "CNNPolicy"
        critic = "CNNCritic"
    else:
        obs_space = obs_space.shape[0]
        policy = "ActorNetwork"
        critic = "CriticNetwork"
    policy = actor_net_str if actor_net_str is not None else policy
    critic = critic_net_str if critic_net_str is not None else critic

    actor_net = getattr(importlib.import_module("vapo.agent.networks.actor_network"), policy)
    critic_net = getattr(importlib.import_module("vapo.agent.networks.critic_network"), critic)
    log.info("SAC get_nets: %s, \t%s" % (policy, critic))
    return actor_net, critic_net, obs_space, action_dim


def tt(x):
    if isinstance(x, dict):
        dict_of_list = {}
        for key, val in x.items():
            dict_of_list[key] = Variable(torch.from_numpy(val).float().cuda(), requires_grad=False)
        return dict_of_list
    else:
        return Variable(torch.from_numpy(x).float().cuda(), requires_grad=False)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    soft_update(target, source, 1.0)


def get_activation_fn(non_linearity):
    if non_linearity == "elu":
        return F.elu
    elif non_linearity == "leaky_relu":
        return F.leaky_relu
    else:  # relu
        return F.relu


def read_results(file_name, folder_name="."):
    with open(os.path.join("%s/optimization_results/" % folder_name, file_name), "rb") as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])
