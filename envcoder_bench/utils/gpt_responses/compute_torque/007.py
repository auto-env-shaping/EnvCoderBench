import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Compute the raw torques that will be directly applied to motors based on policy output
    # Action is directly mapped to the torque, considering the action space is normalized between [-1, 1]
    torque = action * state_dict['torques']
    return torque



