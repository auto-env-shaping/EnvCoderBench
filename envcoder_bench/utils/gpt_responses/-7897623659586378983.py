
def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Initialize temperature parameters for reward components transformation
    velocity_temp = 0.1
    energy_temp = 1.0
    stability_temp = 5.0
    
    # Extract necessary components from the state_dict
    root_vel = state_dict['root_states'][:, 7:10]  # Assuming index 7 to 9 are linear velocities
    
    # Reward for moving forward (positive x-direction velocity)
    forward_velocity = root_vel[:, 0]
    forward_reward = torch.exp(forward_velocity / velocity_temp)
    
    # Penalty for using too much energy (large actions)
    energy_penalty = torch.sum(action**2, dim=1)
    energy_penalty = torch.exp(-energy_penalty / energy_temp)
    
    # Stability penalty (assuming we want the z-axis rotation close to zero for straight running)
    root_orientation = state_dict['root_states'][:, 3:7]  # Assuming index 3 to 6 are quaternion orientations
    z_rotation = root_orientation[:, 2]  # Assuming z-axis rotation corresponds to the 3rd component of quaternion
    stability_penalty = torch.exp(-z_rotation**2 / stability_temp)
    
    # Combine the components to form the overall reward
    reward = forward_reward * energy_penalty * stability_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Directly use action as torque after scaling
    max_torque = 5.0  # Assuming maximum torque capacity of actuators
    torque = action * max_torque
    return torque
