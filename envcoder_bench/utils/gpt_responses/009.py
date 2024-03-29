
import torch
from typing import Dict

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward for moving forward: positive reward for high forward velocity along the x-axis
    root_velocities = state_dict['root_states'][:, 7:10]  # root linear velocities (x, y, z)
    forward_velocity = root_velocities[:, 0]  # x-component of velocity
    speed_reward = forward_velocity

    # Punish for deviation from straight path (y-velocity and rotation)
    lateral_velocity = torch.abs(root_velocities[:, 1])
    rotational_velocity = torch.abs(state_dict['root_states'][:, 10:])  # angular velocity (x, y, z)
    deviation_penalty = -(lateral_velocity + rotational_velocity.sum(dim=1))

    # Encourage energy efficiency (minimizing action magnitude)
    action_cost = -torch.sum(torch.abs(action), dim=1)

    # Configure reward weights and temperature parameters
    speed_weight = 1.0
    deviation_weight = 0.5
    action_weight = 0.1

    # Sum up weighted components to get final reward
    reward = speed_weight * speed_reward + deviation_weight * deviation_penalty + action_weight * action_cost

    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # The action is directly used as the torque, might apply a scaling factor for real applications
    torque = action.clone()
    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Observation composed of: position, velocity, and commands
    dof_pos_scaled = state_dict['dof_pos'] / state_dict['dof_pos_scale']
    dof_vel_scaled = state_dict['dof_vel'] / state_dict['dof_vel_scale']
    commands = state_dict['commands']
    root_states = state_dict['root_states'][:, :13]  # Including up to angular velocity
    observation = torch.cat((dof_pos_scaled, dof_vel_scaled, commands, root_states), dim=1)
    return observation

def compute_done(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Consider an episode done if the max episode length is reached or if the anymal has fallen (e.g., large deviation in root z-pos)
    done_reasons = torch.zeros(state_dict['root_states'].shape[0], dtype=torch.bool, device=state_dict['root_states'].device)
    fallen = state_dict['root_states'][:, 2] < 0.3  # Assuming if the z-pos of root is below 0.3, it has fallen
    max_episode_reached = state_dict['progress_buf'] >= state_dict['max_episode_length'] - 1
    done_reasons |= fallen | max_episode_reached
    return done_reasons
