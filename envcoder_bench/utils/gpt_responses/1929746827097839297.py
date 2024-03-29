
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Extract velocity of the robot's base in the forward direction
    forward_velocity = state_dict['root_states'][:, 7]
    
    # Reward for moving forward: Encouraging higher speeds
    speed_reward_temperature = 0.1  # Parameter to adjust how sharp or smooth the reward increase with speed
    speed_reward = torch.exp(speed_reward_temperature * forward_velocity) - 1
    
    # Penalty for high action magnitude: Encourages energy efficiency
    action_magnitude = torch.norm(action, p=2, dim=-1)  # Calculate the norm of action across dim
    action_cost_temperature = 0.01  # This temperature parameter smoothens the penalty curve
    action_cost = torch.exp(action_cost_temperature * action_magnitude) - 1

    # Combine the rewards and penalties
    reward = speed_reward - action_cost
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # The torque can be a function of the action directly, potentially with some scaling if necessary.
    # However, in many RL environments, especially in ones involving robotics, the action output from the policy
    # can directly correspond to the desired torque. We might apply a scaling factor to ensure that the range of action
    # values corresponds to feasible torque values for the robot.
    
    # Define a scaling factor for the torque
    torque_scaling_factor = 1.0  # This can be tuned based on the specific actuator dynamics
    torque = torque_scaling_factor * action
    
    return torque
