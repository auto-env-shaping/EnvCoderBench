
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Objective: Reward the agent for running as fast as possible.
    # Extract relevant parts of the state dict
    root_states = state_dict['root_states']  # Shape: [num_envs, 13]
    lin_vel_root = root_states[..., 7:10]  # Linear velocity of the root, shape: [num_envs, 3]
    
    # Constants
    forward_velocity_weight = 5.0
    energy_cost_weight = 0.01
    temp_vel_reward = 10  # Temperature for velocity reward transformation
    temp_energy = 0.1  # Temperature for energy cost (action cost) reward transformation
    
    # Calculate forward velocity reward: we want the agent to run as fast as possible
    forward_velocity = lin_vel_root[..., 0]  # Consider only x (forward) direction
    forward_reward = torch.exp(forward_velocity_weight * forward_velocity / temp_vel_reward)
    
    # Calculate cost of action to penalize high energy expenditure
    action_cost = torch.sum(torch.square(action), dim=-1)  # Square of action values summed over all actions
    energy_cost_penalty = torch.exp(-energy_cost_weight * action_cost / temp_energy)

    # Overall reward is a balance between speed and energy efficiency
    reward = forward_reward + energy_cost_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
    # Compute the raw torques based on policy output. This could be a direct mapping for simplicity.
    # No complex torque computation is needed for the purpose of this task 
    # as the action directly maps to the desired torques for the motors.
    torque = action  # Direct mapping from action to torque

    return torque
