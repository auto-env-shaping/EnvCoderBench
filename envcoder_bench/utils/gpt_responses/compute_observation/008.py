import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Concatenate features to form an observation. In a real scenario, this might include
    # normalizing or otherwise preprocessing these values.
    root_states = state_dict['root_states']
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']


