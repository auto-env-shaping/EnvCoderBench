import torch
from typing import * 

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Directly using actions as torques for simplicity
    # Here you can add any scaling or transformation necessary before applying as torques
    torque = action 
    return torque  



