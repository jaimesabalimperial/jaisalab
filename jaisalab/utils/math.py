import torch
import math
from sympy import ShapeError
from numbers import Number

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    #pdb.set_trace()
    return log_density.sum(1, keepdim=True)

def log_prob(value, mean, std):
    # compute the variance
    var = (std ** 2)
    log_scale = math.log(std) if isinstance(std, Number) else std.log()
    return -((value - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

def _std_shape_check(a, b):
    if not (a.shape[-2] == 1 and b.shape[-1] == 1 
            and a.shape[-1] == b.shape[-2]):
        raise ShapeError('Matrix multiplication should return a single value.')

def calc_mean_std(probabilities, values):
    """Calculates the mean and standard deviation of a 
    distribution of quantiles from their values and probabilities."""
    _std_shape_check(probabilities, values)
    mu = torch.matmul(probabilities, values).squeeze(-1)
    values = values.view(-1, values.shape[-2])
    probabilities = probabilities.view(-1, probabilities.shape[-1])
    var = torch.sum(torch.mul(torch.subtract(values, mu).pow(2), probabilities), -1).unsqueeze(-1)
    std = torch.sqrt(var)
    return mu, std
