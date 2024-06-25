"""
@author: armin
"""

import torch
import torch.nn as nn
from torch import distributions


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
        
        
# https://github.com/jakegrigsby/deep_control/blob/master/deep_control/nets.py
class beta_dist(distributions.transformed_distribution.TransformedDistribution):
    class _beta_dist_transform(distributions.transforms.Transform):
        domain = distributions.constraints.real
        codomain = distributions.constraints.interval(-1.0, 1.0)


        def __init__(self, cache_size=1):
            super().__init__(cache_size=cache_size)


        def __eq__(self, other):
            return isinstance(other, _BetaDistTransform)


        def _inverse(self, y):
            return (y.clamp(-0.99, 0.99) + 1.0) / 2.0


        def _call(self, x):
            return (2.0 * x) - 1.0


        def log_abs_det_jacobian(self, x, y):
            # return log det jacobian |dy/dx| given input and output
            return torch.Tensor([math.log(2.0)]).to(x.device)


    def __init__(self, alpha, beta):
        self.base_dist = distributions.beta.Beta(alpha, beta)
        transforms = [self._beta_dist_transform()]
        super().__init__(self.base_dist, transforms)


    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    

# https://github.com/jakegrigsby/deep_control/blob/master/deep_control/nets.py
class squashed_normal(distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
    
# https://github.com/jakegrigsby/deep_control/blob/master/deep_control/nets.py
class stochastic_actor(nn.Module):
    def __init__(
        self,
        state_space_size,
        act_space_size,
        log_std_low=-10.0,
        log_std_high=2.0,
        hidden_size=1024,
        dist_impl="pyd",
    ):
        
        super().__init__()
        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * act_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)
        self.dist_impl = dist_impl


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()
            dist = squashed_normal(mu, std)
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=1)
            dist = beta_dist(alpha, beta)
        return dist