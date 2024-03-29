
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Extract necessary components from the state dictionary
    root_states = state_dict['root_states']  # Shape: [num_envs, 13]
    dof_vel = state_dict['dof_vel']  # Shape: [num_envs, 12]
    progress_buf = state_dict['progress_buf']  # Shape: [num_envs]
    max_episode_length = state_dict['max_episode_length']  # Scalar

    # Constants and temperature variables for reward transformation
    forward_velocity_temp = 0.1  # Temperature for the forward velocity component
    action_cost_temp = 0.01  # Temperature for penalizing large actions
    
    # Compute forward velocity (assuming the forward direction is along one of the root state dimensions, e.g., x)
    forward_velocity = root_states[:, 0]  # Simplest assumption, modify based on actual forward direction in root_states
    # Normalize and transform forward velocity component
    forward_velocity_reward = torch.exp(forward_velocity / forward_velocity_temp)

    # Penalize large actions to encourage energy efficiency
    action_cost = torch.sum(action**2, dim=1)
    action_cost_penalty = torch.exp(-action_cost / action_cost_temp)

    # Optionally, penalize ending episodes too early to encourage longer survival
    progress_penalty = torch.tensor(progress_buf / max_episode_length, dtype=torch.float32)
    
    # Combine components to form overall reward
    # Notice that each component can be weighted differently to signify its importance
    reward = forward_velocity_reward + action_cost_penalty + progress_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Raw torques could simply be the action or a transformation of actions based on robot's dynamics or simplifications
    # For simplicity, here we assume the action directly corresponds to the desired torques without further transformations
    
    # However, if you were to implement transformations or constraints, it could involve:
    # - Clipping values to a max/min to avoid damaging the robot
    # - Scaling actions by a factor relevant to the robot's capabilities
    # - Applying filters to smooth out jerky movements etc.

    # Example (uncomment and modify as needed):
    # torque_limits = torch.tensor([100.0] * 12)  # Assuming 12 DOF with +-100Nm max torque
    # torque = torch.clamp(action, min=-torque_limits, max=torque_limits)

    torque = action  # Direct mapping for simplicity

    return torque
