
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Example reward function for making the anymal run as fast as possible
    # Reward based on forward velocity, penalize backward movement slightly less
    forward_velocity = state_dict['root_states'][:, 3]  # Assuming index 3 represents forward velocity
    vel_reward = torch.exp(0.1 * forward_velocity) - 1.0  # Reward exponential growth, temperature = 0.1
    
    # Small penalty for not moving or moving backward slightly
    backward_penalty_scale = 0.05  # Less penalization for backward
    backward_penalty = torch.where(forward_velocity < 0, torch.exp(backward_penalty_scale * forward_velocity) - 1, torch.tensor(0., device=forward_velocity.device))
    
    # Aggregate the rewards
    reward = vel_reward + backward_penalty
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly use action as torque for simplicity
    # You might want to scale your torque based on the system
    torque_scale = 1.0
    torque = action * torque_scale
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Let's assume a simple observation composed of root_states, dof_pos, and dof_vel
    # Generally, you should normalize these values based on the system's expected operational ranges
    normalized_dof_pos = state_dict['dof_pos'] / state_dict['dof_pos_scale']
    normalized_dof_vel = state_dict['dof_vel'] / state_dict['dof_vel_scale']
    
    observation = torch.cat([state_dict['root_states'], normalized_dof_pos, normalized_dof_vel], dim=1)
    return observation

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assuming the task ends if the progress_buf is filled or an episode reaches its max length
    done = state_dict['progress_buf'] >= state_dict['max_episode_length']
    return done.float()  # Ensure it's a float tensor for consistency


