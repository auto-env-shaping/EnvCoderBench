
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants for reward normalization
    speed_temp = 0.01
    action_temp = 0.001

    # Extract necessary components for reward calculation
    base_vel = state_dict['root_states'][:, 7:10] # Linear velocity of the base in x, y, z
    speed_reward = base_vel[:, 0] # We are interested in speed along the x-axis for forward movement
    
    # Penalize the use of actions to encourage efficiency
    action_penalty = torch.sum(action ** 2, dim=-1)

    # Normalize components
    norm_speed_reward = torch.exp(speed_temp * speed_reward) - 1
    norm_action_penalty = torch.exp(action_temp * action_penalty) - 1
    
    # Combine the components to form the final reward
    reward = norm_speed_reward - norm_action_penalty
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly use the action as the torque to apply without transformations
    torque = action 
    return torque  

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # For observation, we concatenate relevant parts of the state dictionary 
    # such as root states, dof position and velocities, including the gravity vector and commands
    observation = torch.cat([state_dict['root_states'], 
                             state_dict['dof_pos'],
                             state_dict['dof_vel'], 
                             state_dict['commands'],
                             state_dict['gravity_vec']], dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Compute the done signal based on the episode progress and max_episode_length
    # Check if the progress is greater than or equal to max_episode_length
    done = state_dict['progress_buf'] >= state_dict['max_episode_length'].item()
    return done
