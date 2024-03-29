import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants
    speed_reward_temperature = 0.01

    # Extracting useful components for the reward computation
    root_vel = state_dict['root_states'][:, 7:10]  # root linear velocities
    target_speed = state_dict['commands'][:, 0]  # target forward speed

    # Compute speed difference
    speed_diff = root_vel[:, 0] - target_speed  # Considering the x-axis velocity component for forward motion
    
    # Compute the reward components
    speed_reward = torch.exp(-speed_reward_temperature * (speed_diff ** 2))
    
    # Compose the final reward
    reward = speed_reward
    
    return reward



