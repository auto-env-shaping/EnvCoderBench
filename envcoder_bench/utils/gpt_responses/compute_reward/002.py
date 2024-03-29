import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    speed_reward_temp = 0.1  # Temperature for speed reward transformation
    # Speed of the base in forward direction (assuming forward is x direction)
    lin_vel = state_dict["root_states"][:, 7:10]  # Extract linear velocity from root_states
    forward_speed = lin_vel[:, 0]  # Forward speed is x-component of linear velocity
    
    # Reward for speed, encourage the agent to run as fast as possible
    speed_reward = torch.exp(speed_reward_temp * forward_speed) - 1.0
    
    # Total reward
    reward = speed_reward
    return reward




