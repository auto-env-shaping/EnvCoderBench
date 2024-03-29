import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # For simplicity, assume action directly corresponds to torque
    torque = action  # This could be modified based on more complex rules involving the state

    return torque



