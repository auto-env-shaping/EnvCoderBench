
import torch

def compute_reward(state_dict: dict, action: torch.Tensor) -> torch.Tensor:
    # Parameters for reward calculation
    speed_reward_temp = 0.1  # temperature parameter for speed reward transformation
    
    # Extract necessary information from the state dictionary
    root_velocity = state_dict['root_states'][:, 7:10]  # Extract the linear velocity of the base
    desired_velocity = state_dict['commands'][:, 0]  # Target forward velocity from the commands
    
    # Calculate the velocity error (How much the actual velocity deviates from the desired one)
    velocity_error = torch.norm(root_velocity[:, 0] - desired_velocity, dim=-1)
    
    # Reward for speed: We want the agent to run as fast as possible towards the desired direction
    # The reward is higher when the agent's velocity is closer to the desired velocity
    speed_reward = torch.exp(-velocity_error * speed_reward_temp)
    
    # Combine the rewards
    total_reward = speed_reward
    
    return total_reward

def compute_torque(action: torch.Tensor, state_dict: dict) -> torch.Tensor:
    # In this specific task, the action directly corresponds to the torque
    # However, you might want to scale or otherwise modify the action values
    # before sending them to the motors depending on the specifics of your environment
    torque_scaling_factor = 1.0  # This might need to be adjusted based on the environment specifics
    
    torque = action * torque_scaling_factor
    
    return torque
