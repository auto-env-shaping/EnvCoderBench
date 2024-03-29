
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Unpack necessary components from the state_dict
    root_states = state_dict['root_states']  # Robot root states (position, orientation, linear & angular velocities)
    progress_buf = state_dict['progress_buf'].float()  # Progress buffer where the current timestep of each environment is stored
    max_episode_length = state_dict['max_episode_length'].float()  # Maximum allowed episode length
    gravity_vec = state_dict['gravity_vec']  # Gravity vector to determine orientation
    
    # Calculate the forward speed of the robot
    forward_speed = root_states[..., 7]  # Assuming the index 7 corresponds to the forward linear velocity in root_states
    
    # Reward for forward speed
    forward_speed_reward = torch.exp(forward_speed / 20)  # Using exponential to encourage higher speeds, with a temperature scale of 20
    
    # Penalty for deviating from upright orientation
    # Assuming that the last 3 elements of root_states represent angular velocities along the 3 axes
    upright_penalty = torch.sum(torch.abs(root_states[..., 10:13]), dim=1)
    upright_penalty_reward = -torch.exp(upright_penalty / 5)  # Using exponential to penalize deviations, with a temperature scale of 5
    
    # Potential penalty for ending episodes early (encourage using the entire episode length)
    progress_ratio = progress_buf / max_episode_length
    episode_completion_reward = torch.exp((progress_ratio - 1.0) / 10)  # Encourage completion of episodes, with a temperature scale of 10
    
    # Combine the reward components
    reward = forward_speed_reward + upright_penalty_reward + episode_completion_reward
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly using the action tensor as the torque. Assuming the action tensor is already scaled appropriately.
    torque = action
    
    # If additional computation is needed based on the state_dict, it can be added here.
    # For example, adjusting torque based on current velocities or positions to apply corrective actions.
    # This is a simplified version assuming direct control through the action tensor.

    return torque
