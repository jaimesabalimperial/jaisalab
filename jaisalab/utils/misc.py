from datetime import datetime 
from shutil import make_archive

from dowel import tabular
import numpy as np
import torch 
import torch.nn as nn

def zip_directory(dir_path, zip_name):
    make_archive(f'{zip_name}', 'zip', dir_path)

def get_time_stamp_as_string():
    """Get the current time stamp as a string.
    
    Returns:
        date_time_str (str) : current timestemp
    """
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d-%b-%Y (%H-%M-%S)")
    return date_time_str

def to_device(device, *args):
    return [x.to(device) for x in args]

"""Functions to initialize weights of PyTorch models differently."""
def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def soft_update(local_model, target_model, tau=0.1):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def get_num_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
