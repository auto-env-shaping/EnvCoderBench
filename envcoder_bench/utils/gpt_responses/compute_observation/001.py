import torch
from typing import * 

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example of combining various parts of state_dict into a single observation tensor
    # Here, we concatenate root states, dof positions, and velocities
    root_states = state_dict["root_states"]
    dof_pos = state_dict["dof_pos"]
    dof_vel = state_dict["dof_vel"]


