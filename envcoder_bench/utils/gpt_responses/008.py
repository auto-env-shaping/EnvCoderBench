import torch
from typing import * 

    # Normalizing could be like: normalized_dof_pos = (dof_pos - pos_mean) / pos_std, where pos_mean and pos_std are predefined
    observation = torch.cat((root_states, dof_pos, dof_vel), dim=1)
    
    return observation



