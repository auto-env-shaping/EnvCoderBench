import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # No transformation is applied to the actions, they directly represent torques
    torque = action
    return torque




