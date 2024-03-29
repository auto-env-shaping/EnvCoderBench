import torch
from typing import * 
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):

    return action  * 3.

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state_dict["dof_pos"], state_dict["dof_vel"]], dim=-1)

def compute_reward(state_dict: Dict[str, torch.Tensor]):
    
    root_states = state_dict['root_states']
    commands = state_dict['commands']

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])

    total_reward = 1/(lin_vel_error + ang_vel_error) # - torch.sum(torch.square(torques), dim=1) * 0.001

    return total_reward.detach()

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    progress = state_dict['progress_buf']
    max_episode_length = state_dict['max_episode_length']
    done = torch.tensor(progress >= max_episode_length, dtype=torch.float32, device=state_dict['root_states'].device)
    return done