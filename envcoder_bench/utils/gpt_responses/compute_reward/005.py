import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Parameters for reward transformation
    speed_temp = 0.1  # Temperature parameter for speed reward component
    energy_temp = 0.05  # Temperature parameter for energy usage component

    # Extract necessary components from state_dict
    root_states = state_dict['root_states']
    lin_vel = root_states[:, 7:10]  # Linear velocity of the base in root_states
    torques = state_dict['torques']
    
    # Reward for speed: We want the agent to run as fast as possible
    forward_speed = lin_vel[:, 0]  # Assuming the first component is the forward direction
    speed_reward = torch.exp(speed_temp * forward_speed)  # Exponential transformation

    # Penalty for energy usage: We want efficient motion
    energy_usage = torch.sum(torques**2, dim=1)
    energy_penalty = torch.exp(-energy_temp * energy_usage)  # Exponential transformation

    # Total reward
    reward = speed_reward + energy_penalty
    return reward



