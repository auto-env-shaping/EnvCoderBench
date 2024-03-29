import torch
from typing import * 

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Compute the reward for the Anymal running task."""
    # Access relevant tensors from state dictionary
    root_states = state_dict['root_states']  # Root state of the robot
    dof_vel = state_dict['dof_vel']  # Velocity of each Degree of Freedom (DOF)
    
    # Extract linear velocity (x, y, z) from root states (index 7, 8, 9 are linear velocities)
    linear_velocity = root_states[:, 7:10]
    
    # Reward for forward movement along the x-axis
    forward_reward = linear_velocity[:, 0]
    
    # Penalize lateral movement to encourage straight running
    lateral_movement_penalty = torch.abs(linear_velocity[:, 1])
    
    # Penalize excessive rotation velocities to encourage stable running
    rotation_penalty = torch.norm(dof_vel[:, :3], dim=1)  # Assuming first 3 DOFs relate to base rotation
    
    # Parameters for adjusted rewards (temperature parameters for transformations)
    forward_temp = 0.1
    lateral_temp = 0.05
    rotation_temp = 0.05
    
    # Adjust rewards to fixed range using exponential transformation to emphasize good behaviors
    forward_reward_adjusted = torch.exp(forward_reward / forward_temp)
    lateral_penalty_adjusted = torch.exp(-lateral_movement_penalty / lateral_temp)
    rotation_penalty_adjusted = torch.exp(-rotation_penalty / rotation_temp)
    
    # Combine adjusted rewards and penalties
    total_reward = forward_reward_adjusted + lateral_penalty_adjusted + rotation_penalty_adjusted
    
    return total_reward



