import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants for reward calculation
    speed_reward_weight = 1.0
    action_cost_weight = 0.05
    speed_temp = 0.1

    # Extract relevant states
    base_velocity = state_dict['root_states'][..., 7:10]  # Extract base linear velocity
    forward_velocity = base_velocity[..., 0]  # We're interested in the x (forward) velocity
    
    # Normalize and calculate speed reward component
    norm_forward_velocity = torch.exp(speed_temp * forward_velocity) - 1
    
    # Action cost to incentivize efficiency
    action_cost = torch.sum(action ** 2, dim=-1) * action_cost_weight
    
    # Total reward is speed reward minus cost of action
    total_reward = speed_reward_weight * norm_forward_velocity - action_cost
    return total_reward



