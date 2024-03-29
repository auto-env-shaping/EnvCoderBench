import torch
from typing import * 
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse

def compute_torque(action: torch.Tensor, state_dict: Dict[str, torch.Tensor]):

    Kp = 85.0
    Kd = 2.0
    action_scale = 0.5

    dof_pos = state_dict['dof_pos']
    default_dof_pos = state_dict['default_dof_pos']
    dof_vel = state_dict['dof_vel']
    target_dof_pos = action * action_scale + default_dof_pos

    torque = Kp * (target_dof_pos - dof_pos) - Kd * dof_vel

    return torque

def compute_observation(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    root_states = state_dict['root_states']
    commands = state_dict['commands']
    dof_pos = state_dict['dof_pos']
    default_dof_pos = state_dict['default_dof_pos']
    dof_vel = state_dict['dof_vel']
    gravity_vec = state_dict['gravity_vec']
    lin_vel_scale = state_dict['lin_vel_scale'].float()
    ang_vel_scale = state_dict['ang_vel_scale'].float()
    dof_pos_scale = state_dict['dof_pos_scale'].float()
    dof_vel_scale = state_dict['dof_vel_scale'].float()
    # if we observe previous actions, the training tends to explode since they are torques not joint position targets
    actions = state_dict['actions'] * 0
    # back out the input to the actionspace instead
    # actions = 2.0 * (state_dict['actions'] + 2.0 * state_dict['dof_vel']) / 85.0 + state_dict['dof_pos'] - state_dict['default_dof_pos']

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs

def compute_reward(state_dict: Dict[str, torch.Tensor]):
    rew_scales = {
        "lin_vel_xy": 1.0,
        "ang_vel_z": 0.5,
        "torque": -0.000025
    }

    root_states = state_dict['root_states']
    commands = state_dict['commands']
    torques = state_dict['torques']
    contact_forces = state_dict['contact_forces']
    knee_indices = state_dict['knee_indices']
    episode_lengths = state_dict['progress_buf']
    base_index = state_dict['base_index']
    max_episode_length = state_dict['max_episode_length']

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25)  * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
    total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach()

def compute_terminal(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    base_index = state_dict['base_index']
    max_episode_length = state_dict['max_episode_length']
    contact_forces = state_dict['contact_forces']
    knee_indices = state_dict['knee_indices']
    episode_lengths = state_dict['progress_buf']

    reset = torch.norm(state_dict['contact_forces'][:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return reset