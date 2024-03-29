import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Observation could include the robot's velocity, joint positions, joint velocities, and external commands 
    root_states = state_dict['root_states']
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    commands = state_dict['commands']

    # Normalize joint positions and velocities as part of observation
    normalized_dof_pos = dof_pos / state_dict['dof_pos_scale']
    normalized_dof_vel = dof_vel / state_dict['dof_vel_scale']

    # Concatenate components to form the observation vector
    observation = torch.cat([root_states, normalized_dof_pos, normalized_dof_vel, commands], dim=1)
    return observation



