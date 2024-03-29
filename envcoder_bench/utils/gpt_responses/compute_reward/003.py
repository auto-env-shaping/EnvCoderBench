import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Task: Make the anymal run as fast as possible, primarily judged by its x-velocity (forward speed)
    # Extract necessary components from the state dict for reward computation
    root_velocities = state_dict["root_states"][:, 7:10]  # root linear velocities
    forward_velocity = root_velocities[:, 0]  # x-velocity indicating forward speed
    
    # Components for Reward Calculation
    # Encouraging forward motion (use a temperature variable to scale this reward component)
    temp_speed = 0.1
    speed_reward = torch.exp(forward_velocity / temp_speed)
    
    # Penalize excessive action use (to encourage smooth and efficient motion)
    action_cost = torch.sum(torch.square(action), dim=-1)
    temp_action = 0.01
    action_penalty = -torch.exp(action_cost / temp_action)
    
    # Combine the components to calculate the overall reward
    reward = speed_reward + action_penalty
    
    return reward



