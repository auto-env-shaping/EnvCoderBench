
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Compute the reward based on the root linear velocity (forward speed) which is part of the root_states
    # root_states (num_envs, 13), first 3 are position, next 3 are rotation as a quaternion, and then 7:10 are linear velocity
    
    lin_vel = state_dict['root_states'][:, 7:10]
    forward_speed = lin_vel[:, 0]  # Assuming the x-direction is the forward direction
    
    # Reward for moving forward
    temp_speed = 0.01
    speed_reward = torch.exp(forward_speed / temp_speed)  # Scale and transform the forward speed into a reward
    
    # Penalize rotation to keep the agent running straight
    rotation = state_dict['root_states'][:, 3:7]  # Quaternion rotation
    # We encourage the agent to maintain upright posture. A simple proxy is minimizing the deviation of the 'w' component of the quaternion from 1.
    temp_rotation = 0.1
    rotation_penalty = torch.exp(-torch.abs(rotation[:, 0] - 1) / temp_rotation)
    
    # Penalize large actions to encourage efficiency
    temp_action = 0.05
    action_penalty = torch.exp(-torch.mean(torch.abs(action), dim=1) / temp_action)
    
    # Combine components to form final reward
    reward = speed_reward + rotation_penalty + action_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Here, we could scale the action by a factor pertaining to each joint's capability.
    # For simplicity, we'll pass the action directly as torque, assuming that the action space is designed to match torque limits
    torque = action
    
    return torque
