
import torch


def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Constants (temperature parameters for transformation functions)
    speed_temp = 0.1
    action_temp = 0.01
    
    # Extract necessary information from the state dictionary
    root_states = state_dict['root_states'] # Shape: [num_envs, 13]
    commands = state_dict['commands']  # Shape: [num_envs, 3] (target_velocities)
    dof_vel = state_dict['dof_vel']  # Shape: [num_envs, 12]
    
    # Reward for moving forwards - the difference in speed along the x-axis (index 0 of commands) 
    target_speed = commands[:, 0]  # Target speed (assuming forward direction is positive x)
    actual_speed = root_states[:, 0]  # Actual forward speed
    
    speed_reward = -torch.abs(target_speed - actual_speed)
    normalized_speed_reward = torch.exp(speed_temp * speed_reward)  # Exponential transformation
    
    # Penalize large actions to encourage efficiency
    action_penalty = torch.sum(action**2, dim=1)
    normalized_action_penalty = torch.exp(-action_temp * action_penalty)  # Exponential transformation
    
    # Combine rewards - Higher weight on speed reward, as the main objective is to run fast
    combined_reward = normalized_speed_reward + 0.1 * normalized_action_penalty
    
    return combined_reward


def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # For this particular task, directly use action as torques for simplicity. 
    # It assumes that the action space is designed to directly map to torque values.
    torque = action
    return torque
