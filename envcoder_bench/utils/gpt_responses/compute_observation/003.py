import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example observation: concatenate root states, dof positions, dof velocities, and commands
    observation = torch.cat([
        state_dict["root_states"], 
        state_dict["dof_pos"], 
        state_dict["dof_vel"],
        state_dict["commands"]
    ], dim=1)
    return observation



