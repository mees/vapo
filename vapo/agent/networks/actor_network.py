import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedOneHotCategorical
from vapo.agent.core.utils import get_activation_fn
from vapo.agent.networks.networks_common import \
     get_pos_shape, get_img_network, get_concat_features


# policy
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_space,
                 activation="relu", hidden_dim=256, **kwargs):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self._activation = get_activation_fn(activation)
        self.action_high = action_space.high[0]
        self.action_low = action_space.low[0]

    def forward(self, x):
        x = self._activation(self.fc1(x))
        x = self._activation(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma

    def scale_action(self, action):
        slope = (self.action_high - self.action_low) / 2
        action = self.action_low + slope * (action + 1)
        return action

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)  # .squeeze(0)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu)
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample)
            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) - \
                torch.log((1 - action.square() + 1e-6))
            # +1e-6 to avoid no log(0)
            log_probs = log_probs.sum(-1)  # , keepdim=True)
        action = self.scale_action(action)
        return action, log_probs


class CNNPolicy(nn.Module):
    def __init__(self, obs_space, action_dim, action_space, affordance=None,
                 activation="relu", hidden_dim=256, latent_dim=16, **kwargs):
        super(CNNPolicy, self).__init__()
        self.action_high = torch.tensor(action_space.high).cuda()
        self.action_low = torch.tensor(action_space.low).cuda()
        _robot_obs_shape = get_pos_shape(obs_space, "robot_obs")
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
        out_feat += _robot_obs_shape + _target_pos_shape + _distance_shape
        self.out_feat = out_feat
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.fc0 = nn.Linear(out_feat, 32)
        self.fc1 = nn.Linear(32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Last dimension of action_dim is gripper_action
        self.mu = nn.Linear(hidden_dim, action_dim - 1)
        self.sigma = nn.Linear(hidden_dim, action_dim - 1)
        self.gripper_action = nn.Linear(hidden_dim, 2)  # open / close
        self.aff_cfg = affordance

    def forward(self, obs):
        features = get_concat_features(self.aff_cfg,
                                       obs,
                                       self.cnn_img,
                                       self.cnn_gripper)
        x = F.elu(self.fc0(features))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        gripper_action_logits = self.gripper_action(x)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma, gripper_action_logits

    def scale_action(self, action):
        slope = (self.action_high - self.action_low) / 2
        action = self.action_low + slope * (action + 1)
        return action

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma, gripper_action_logits = self.forward(curr_obs)
        log_probs, log_prob_a = None, None
        if(deterministic):
            action = torch.tanh(mu)
            if(len(gripper_action_logits.shape) == 1):
                logits = gripper_action_logits.unsqueeze(0)
            else:
                logits = gripper_action_logits
            gripper_probs = F.softmax(logits, dim=-1)
            gripper_action = torch.argmax(gripper_probs)
        else:
            dist = Normal(mu, sigma)
            gripper_dist = GumbelSoftmax(0.5, logits=gripper_action_logits)
            if(reparametrize):
                sample = dist.rsample()
                # One hot encoded
                gripper_action = gripper_dist.rsample()
                # Single action
                gripper_action = torch.argmax(gripper_action, -1)
            else:
                sample = dist.sample()
                gripper_action = gripper_dist.sample()
            action = torch.tanh(sample)

            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) -\
                torch.log((1 - action.square() + 1e-6))
            log_probs = log_probs.sum(-1)  # , keepdim=True)

            # Discrete part of the action
            log_prob_a = gripper_dist.log_prob(gripper_action)

        # Add gripper action
        # Gripper action is a integer, not a tensor, so unsqueeze to turn concat
        # Gripper action is in (0,1) -> needs to be scaled to -1 or 1
        gripper_action = gripper_action * 2 - 1
        gripper_action = gripper_action.unsqueeze(-1)
        action = torch.cat((action, gripper_action), -1)
        action = self.scale_action(action)

        # add gripper action to log_probs
        if(log_probs is not None and log_prob_a is not None):
            log_probs = log_probs + log_prob_a
        else:
            log_probs = None
        return action, log_probs


class CNNPolicyDenseNet(CNNPolicy):
    def __init__(self, *args, **kwargs):
        super(CNNPolicyDenseNet, self).__init__(*args, **kwargs)
        n_layers = kwargs["n_layers"]
        self.fc_layers = []

        out_size = self.out_feat
        for i in range(n_layers):
            self.fc_layers.append(nn.Linear(out_size, self.hidden_dim))
            out_size += self.hidden_dim

        self.fc_layers = nn.ModuleList(self.fc_layers)

        # Last dimension of action_dim is gripper_action
        self.mu = nn.Linear(out_size, self.action_dim - 1)
        self.sigma = nn.Linear(out_size, self.action_dim - 1)
        self.gripper_action = nn.Linear(out_size, 2)  # open / close

    def forward(self, obs):
        x_in = get_concat_features(self.aff_cfg,
                                   obs,
                                   self.cnn_img,
                                   self.cnn_gripper)
        for layer in self.fc_layers:
            x_out = F.silu(layer(x_in))
            x_in = torch.cat([x_out, x_in], -1)
        mu = self.mu(x_in)
        log_sigma = self.sigma(x_in)
        gripper_action_logits = self.gripper_action(x_in)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma, gripper_action_logits


class CNNPolicyReal(CNNPolicy):
    def __init__(self, *args, **kwargs):
        super(CNNPolicyReal, self).__init__(*args, **kwargs)
        n_layers = kwargs["n_layers"]
        out_size = self.out_feat
        for i in range(n_layers):
            self.fc_layers.append(nn.Linear(out_size, self.hidden_dim))
            out_size += self.hidden_dim

        self.fc_layers = nn.ModuleList(self.fc_layers)

        # Last dimension of action_dim is gripper_action
        self.mu = nn.Linear(out_size, self.action_dim)
        self.sigma = nn.Linear(out_size, self.action_dim)

    def forward(self, obs):
        x_in = get_concat_features(self.aff_cfg,
                                   obs,
                                   self.cnn_img,
                                   self.cnn_gripper)
        for layer in self.fc_layers:
            x_out = F.silu(layer(x_in))
            x_in = torch.cat([x_out, x_in], -1)
        mu = self.mu(x_in)
        log_sigma = self.sigma(x_in)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        return mu, sigma

    # return action scaled to env
    def act(self, curr_obs, deterministic=False, reparametrize=False):
        mu, sigma = self.forward(curr_obs)
        log_probs = None
        if(deterministic):
            action = torch.tanh(mu)
        else:
            dist = Normal(mu, sigma)
            if(reparametrize):
                sample = dist.rsample()
            else:
                sample = dist.sample()
            action = torch.tanh(sample)

            # For updating policy, Apendix of SAC paper
            # unsqueeze because log_probs is of dim (batch_size, action_dim)
            # but the torch.log... is (batch_size)
            log_probs = dist.log_prob(sample) -\
                torch.log((1 - action.square() + 1e-6))
            log_probs = log_probs.sum(-1)  # , keepdim=True)

        action = self.scale_action(action)

        # add gripper action to log_probs
        return action, log_probs


# https://stackoverflow.com/questions/56226133/soft-actor-critic-with-discrete-action-space
# https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/sac.py
# https://github.com/kengz/SLM-Lab/blob/master/slm_lab/lib/distribution.py
class GumbelSoftmax(RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention.
        To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)
