import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward for running speed
    root_velocity = state_dict["root_states"][..., 7:10]  # Extract the root linear velocity
    forward_speed = root_velocity[..., 0]  # Consider forward speed along the x-axis

    # Speed reward: encourage the agent to run as fast as possible
    speed_reward = forward_speed
    speed_reward_temp = 0.01  # Temperature parameter for transforming speed reward

    # Apply transformation and scaling with temperature
    transformed_speed_reward = torch.exp(speed_reward / speed_reward_temp)

    # Combine rewards
    reward = transformed_speed_reward
    return reward



