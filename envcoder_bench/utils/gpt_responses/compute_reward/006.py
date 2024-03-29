import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward for speed: We want the 'anymal' to run as fast as possible
    root_states = state_dict["root_states"]  # Size([num_envs, 13]),
    linear_velocity = root_states[:, :3]  # Extract linear velocity components
    forward_velocity = linear_velocity[:, 0]  # Assuming the first component is forward direction

    # Normalize forward velocity and use an exponential reward to encourage higher speeds
    speed_reward_temperature = 0.1
    speed_reward = torch.exp(forward_velocity / speed_reward_temperature)

    # Penalize actions to encourage energy efficiency
    action_magnitude = torch.norm(action, dim=1)
    action_penalty_temperature = 0.05
    action_penalty = torch.exp(action_magnitude / action_penalty_temperature)

    reward = speed_reward - action_penalty
    return reward




