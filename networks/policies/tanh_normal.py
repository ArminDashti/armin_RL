import torch
import torch.nn as nn
from armin_utils.utils import MLP
from armin_utils.pytorch import tensor


class TanhNormalPolicy(Distribution):
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