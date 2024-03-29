
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward is primarily based on the forward velocity to make the agent run as fast as possible
    # Extracting linear velocities (forward direction)
    lin_vel = state_dict['root_states'][:, 7:10]  # Assuming index 7 to 9 are linear velocities in x, y, z
    forward_velocity = lin_vel[:, 0]  # Assuming x is the forward direction

    # Normalize and transform forward speed to reward
    speed_reward_temp = 0.1  # Temperature parameter for speed reward transformation
    speed_reward = torch.exp(forward_velocity / speed_reward_temp)

    # Penalize for large actions to encourage efficiency 
    action_penalty_temp = 0.01  # Temperature parameter for action penalty
    action_penalty = torch.sum(action ** 2, dim=1) * action_penalty_temp

    # Final reward is speed reward minus action penalty
    reward = speed_reward - action_penalty

    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly passing the action as the torque
    # Assuming the action is already scaled appropriately for the environment
    torque = action
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Concatenate relevant states to form the observation
    # It might include root states, dof positions, and velocities, and possibly the commands
    root_states = state_dict['root_states']
    dof_pos = state_dict['dof_pos']
    dof_vel = state_dict['dof_vel']
    commands = state_dict['commands']

    # Normalizing dof positions and velocities
    dof_pos_norm = dof_pos / state_dict['dof_pos_scale']
    dof_vel_norm = dof_vel / state_dict['dof_vel_scale']

    observation = torch.cat([root_states, dof_pos_norm, dof_vel_norm, commands], dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Episode is done if progress_buf has reached max_episode_length
    done = state_dict['progress_buf'] >= state_dict['max_episode_length']
    return done.float()  # Ensuring the output is a float tensor for consistency
