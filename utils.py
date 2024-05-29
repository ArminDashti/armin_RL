import torch
import torch.nn as nn


# https://github.com/davidbrandfonbrener/onestep-rl/blob/main/utils.py
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        

# https://github.com/davidbrandfonbrener/onestep-rl/blob/main/policy_network.py
def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x