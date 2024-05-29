# From https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py

import torch
import torch.nn as nn
from armin_utils.utils import MLP
from armin_utils.pytorch import tensor

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


from torch.distributions import Distribution, Normal

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                tensor.zeros(self.normal_mean.size()),
                tensor.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
        
    
        
class tanh_gaussian(MLP.MLP):


    def __init__(self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3, **kwargs):
        super().__init__(hidden_sizes, input_size=obs_dim, output_size=action_dim, init_w=init_w, **kwargs)
        
        
        self.log_std = None
        self.std = std
        
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX


    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}


    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]


    def log_prob(self, obs, actions):
        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        log_prob = log_prob.sum(-1)
        return log_prob


    def forward(self, obs, reparameterize=True, deterministic=False, return_log_prob=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            
        mean = self.last_fc(h)
        
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value)