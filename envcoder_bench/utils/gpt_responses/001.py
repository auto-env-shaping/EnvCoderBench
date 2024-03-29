import torch
from typing import * 

    # Normalizing positions and velocities based on predefined scales
    normalized_dof_pos = dof_pos / state_dict["dof_pos_scale"]
    normalized_dof_vel = dof_vel / state_dict["dof_vel_scale"]
    observation = torch.cat([root_states, normalized_dof_pos, normalized_dof_vel], dim=-1)
    return observation



