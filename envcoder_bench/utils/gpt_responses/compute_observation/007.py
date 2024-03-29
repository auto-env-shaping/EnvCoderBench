import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Compute the observation which includes the root position, dof position, dof velocity, and commands
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    commands = state_dict['commands']
    
    # Concatenate observations into a single vector
    observation = torch.cat((dof_pos, dof_vel, commands), dim=1)
    return observation



