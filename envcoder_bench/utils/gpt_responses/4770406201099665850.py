
import torch

def compute_reward(state_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    # Unpack necessary components from the state dictionary
    root_states = state_dict["root_states"]  # shape: [num_envs, 13]
    dof_vel = state_dict["dof_vel"]  # shape: [num_envs, 12]
    progress_buf = state_dict["progress_buf"]  # shape: [num_envs]
    max_episode_length = state_dict["max_episode_length"]  # shape: []
    contact_forces = state_dict["contact_forces"]  # shape: [num_envs, 13, 3]
    
    # Rewards for forward velocity
    forward_vel = root_states[:, 7]  # Assuming root_states contains [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    forward_reward = torch.exp(0.1 * forward_vel)  # Temperature 0.1 for exponential transformation of forward velocity

    # Penalty for using too much energy
    energy_usage = action.norm(dim=-1)
    energy_penalty = -torch.exp(0.01 * energy_usage)  # Temperature 0.01 for exponential transformation of energy usage

    # Progress reward
    progress_reward = progress_buf.float() / max_episode_length.float()

    # Contact forces penalty to discourage too much force on any single leg
    contact_force_penalty = -torch.exp(0.001 * contact_forces.norm(dim=-1).max(dim=-1).values)  # Temperature 0.001

    # Total reward
    reward = forward_reward + energy_penalty + progress_reward + contact_force_penalty
    
    return reward

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # This function assumes that the action directly corresponds to the desired torques,
    # possibly after being scaled or transformed somehow. For this simplistic implementation,
    # we will just directly return the action as the torque assuming it's pre-scaled.
    
    # torque is directly the action here, but in a more complex scenario, you might apply
    # some transformation based on the robot's state or the action's characteristics.
    torque = action
    
    return torque
