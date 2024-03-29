import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Construct observation from various parts of the state dictionary. Examples include:
    # root_states, dof_pos, dof_vel, and commands. You might want to normalize them as well.

    root_states = state_dict["root_states"]
    dof_pos = state_dict["dof_pos"]
    dof_vel = state_dict["dof_vel"]
    commands = state_dict["commands"]

    # Simple example of normalization 
    dof_pos_normalized = dof_pos / state_dict["dof_pos_scale"]
    dof_vel_normalized = dof_vel / state_dict["dof_vel_scale"]

    # Concatenate different components of the state to form a complete observation
    observation = torch.cat([root_states, dof_pos_normalized, dof_vel_normalized, commands], dim=1)
    return observation




