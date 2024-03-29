
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    lin_vel_scale = state_dict['lin_vel_scale']
    ang_vel_scale = state_dict['ang_vel_scale']
    root_states = state_dict['root_states']  # Shape: (batch_size, root_state_dim)
    dof_vel = state_dict['dof_vel']  # Shape: (batch_size, dof_dim)

    # Simplified reward to encourage forward velocity along the x-axis
    # Note: root_states[..., 7] would typically represent the x-axis linear velocity of the base link in world space
    forward_velocity = root_states[:, 7]
    # Normalize forward velocity component
    forward_reward = torch.exp(forward_velocity / lin_vel_scale)

    # Penalize rotation to encourage stability
    # Note: root_states[..., 8:11] would typically represent angular velocity of the base link in world space
    angular_velocity = torch.norm(root_states[:, 8:11], dim=1)
    angular_penalty = torch.exp(-angular_velocity / ang_vel_scale)

    # Combine components with temperatures for reward shaping
    forward_temp = 0.1
    angular_temp = 0.01

    reward = torch.exp(forward_reward / forward_temp) + torch.exp(angular_penalty / angular_temp)
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    torque = action * state_dict['torques']
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Here, we concatenate relevant state components to form the observation vector
    # Adjust according to the specific needs of the environment
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    root_states = state_dict['root_states']
    # Normalizing positions and velocities
    dof_pos_scaled = dof_pos / state_dict['dof_pos_scale']
    dof_vel_scaled = dof_vel / state_dict['dof_vel_scale']

    observation = torch.cat((dof_pos_scaled, dof_vel_scaled, root_states), dim=1)
    return observation

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    episode_length = state_dict['progress_buf'] 
    max_episode_length = state_dict['max_episode_length']
    done = episode_length >= max_episode_length
    return done.float()


