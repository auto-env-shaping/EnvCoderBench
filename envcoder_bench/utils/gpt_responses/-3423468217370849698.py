
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Reward is designed to encourage speed along the forward direction (x-axis) of the robot
    
    # Extract relevant tensors from state dictionary
    root_states = state_dict['root_states']  # Shape: [num_envs, 13]
    progress_buf = state_dict['progress_buf']  # Shape: [num_envs]
    max_episode_length = state_dict['max_episode_length']  # Scalar
    
    # Extract linear velocity of the robot's base along the x-axis
    lin_vel_x = root_states[:, 7]  # Linear velocity in x is at position 7 in root_states

    # Reward for speed - encourage the agent to increase its speed in the x direction
    speed_reward = lin_vel_x
    
    # Normalize speed reward to a fixed range using exponential transformation to make it positive
    speed_reward_temperature = 0.1  # Temperature parameter for speed reward transformation
    speed_reward = torch.exp(speed_reward / speed_reward_temperature)
    
    # Penalize for episode termination to encourage staying in the task
    # Here we simply use the progress as a proxy for how long the robot stays upright and running
    progress_reward = progress_buf.float() / max_episode_length.float()
    
    # Combine rewards
    total_reward = speed_reward + progress_reward
    
    return total_reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Assume action space is directly controlling the torques applied to the robot's joints
    # In a more sophisticated setup, you might want to map action outputs to torque ranges or use a PID controller
    
    # For simplicity, we directly use the actions as torques
    torque = action
    
    return torque
