import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Directly use action as torque for simplicity
    torque = action
    return torque



