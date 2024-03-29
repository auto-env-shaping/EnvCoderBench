import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Combine relevant state information into a single observation vector
    root_states = state_dict["root_states"]
    dof_pos = state_dict["dof_pos"] / state_dict["dof_pos_scale"]  # Normalize
    dof_vel = state_dict["dof_vel"] / state_dict["dof_vel_scale"]  # Normalize
    commands = state_dict["commands"]
    gravity_vec = state_dict["gravity_vec"]
    
    # Concatenate to form observation vector
    observation = torch.cat([root_states, dof_pos, dof_vel, commands, gravity_vec], dim=1)
    return observation




