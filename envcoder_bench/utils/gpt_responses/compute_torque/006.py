import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Here, we can directly use actions as torques or modify them based on some policy
    # For simplicity, we assume actions are already scaled appropriately for torques
    torque = action
    return torque  




