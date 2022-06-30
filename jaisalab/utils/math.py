import torch
import math
import numpy as np

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    #pdb.set_trace()
    return log_density.sum(1, keepdim=True)

def log_likelihood(xs, dist_info):
    means = dist_info["mean"]
    log_stds = dist_info["log_std"]
    zs = (xs - means) / np.exp(log_stds)
    return - np.sum(log_stds, axis=-1) - \
            0.5 * np.sum(np.square(zs), axis=-1) - \
            0.5 * means.shape[-1] * np.log(2 * np.pi)