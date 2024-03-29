import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward function to encourage the anymal to run as fast as possible
    # Extract necessary components from the state dictionary
    root_velocity = state_dict['root_states'][:, 7:10]  # Extract linear velocity of the base
    forward_velocity = root_velocity[:, 0]  # Consider x-direction as the forward direction

    # Penalize the action magnitude slightly to encourage energy efficiency
    action_penalty = torch.sum(action**2, dim=-1)

    # Normalize forward velocity to encourage faster motion
    velocity_reward = forward_velocity
    action_penalty_weight = 0.005  # Small penalty weight for using energy
    reward = velocity_reward - action_penalty_weight * action_penalty
    
    # Introduce a temperature parameter for velocity reward transformation (exponential in this case)
    temperature_velocity = 0.1
    reward = torch.exp(reward / temperature_velocity)  # Exponential transformation for sparsification
    
    return reward



