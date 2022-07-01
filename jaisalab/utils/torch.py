"""
Author - Jaime Sabal
"""
import torch
import numpy as np
import pdb

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def to_device(device, *args):
    return [x.to(device) for x in args]
