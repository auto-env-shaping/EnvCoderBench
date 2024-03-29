
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    linear_speed = state_dict['root_states'][:, 3]
    
    # Reward for moving forward
    moving_forward_reward = linear_speed
    temp_speed = 0.5  # Temperature parameter for speed reward normalization
    moving_forward_reward = torch.exp(moving_forward_reward / temp_speed)
    
    # Penalty for sideways movement
    lateral_movement = state_dict['root_states'][:, 4].abs()
    temp_lateral = 0.1  # Temperature for lateral movement penalty normalization
    lateral_penalty = torch.exp(lateral_movement / temp_lateral) - 1
    
    # Total reward
    reward = moving_forward_reward - lateral_penalty
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Simply relaying action as torque for simplicity, more complex models might transform actions
    torque = action
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Flattening relevant parts of the state_dict for observation
    root_states = state_dict['root_states'].flatten(start_dim=1)
    dof_pos = state_dict['dof_pos'].flatten(start_dim=1)
    dof_vel = state_dict['dof_vel'].flatten(start_dim=1)
    observation = torch.cat([root_states, dof_pos, dof_vel], dim=1)
    return observation

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    progress = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length']
    done = torch.tensor(progress >= max_episode_length, dtype=torch.float32, device=state_dict['root_states'].device)
    return done


