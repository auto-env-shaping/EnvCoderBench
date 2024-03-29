import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly using the action as torque
    torque = action
    return torque



