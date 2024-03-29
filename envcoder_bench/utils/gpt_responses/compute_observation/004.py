import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Combining root states, dof positions, and velocities as observation
    root_states = state_dict['root_states']
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    
    observation = torch.cat([root_states, dof_pos, dof_vel], dim=-1)
    return observation



