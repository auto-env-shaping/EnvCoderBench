import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward is based on how fast the agent moves in the desired direction
    root_states = state_dict['root_states']  # Shape: [num_envs, 13]
    forward_velocity = root_states[:, 7]  # Assuming the forward direction is along the Y-axis
    lin_vel_scale = state_dict['lin_vel_scale'].item()  # Linear velocity scale
    
    # We can normalize and use the exponential of the forward velocity as a component of the reward
    forward_reward_component = torch.exp(forward_velocity / lin_vel_scale)  
    forward_temp = 0.1  # Temperature parameter for forward reward component
    
    # The overall reward combines the forward movement reward and penalizes large actions
    action_cost = torch.sum(action ** 2, dim=-1)
    action_temp = 0.01  # Temperature parameter for action cost
    
    reward = forward_reward_component / forward_temp - (action_cost * action_temp)
    return reward



