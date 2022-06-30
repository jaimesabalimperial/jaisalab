import torch
from jaisalab.utils.torch import to_device

def estimate_constraint_value(costs, masks, gamma, device):
    costs, masks = to_device(torch.device('cpu'), costs, masks)
    constraint_value = torch.tensor(0)
    
    j = 1
    traj_num = 1
    for i in range(costs.size(0)):
        constraint_value = constraint_value + costs[i] * gamma**(j-1)
        
        if masks[i] == 0:
            j = 1 #reset
            traj_num = traj_num + 1
        else: 
            j = j+1
            
    constraint_value = constraint_value/traj_num
    constraint_value = to_device(device, constraint_value)
    return constraint_value[0]