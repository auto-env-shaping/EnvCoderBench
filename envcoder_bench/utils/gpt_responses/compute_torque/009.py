import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the raw torques that will be directly applied to motors based on policy output"""
    # Directly use action as torques, as the action space directly controls the torques applied
    torque = action
    return torque



