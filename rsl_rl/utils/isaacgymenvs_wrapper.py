import gym
import torch
from isaacgym import gymapi
import os
from isaacgymenvs.tasks.base.vec_task import Env
from typing import * 
from isaacgym import gymtorch 

class IsaacGymEnvsWrapper(gym.Wrapper):
    def __init__(self, env: Env, rew_scale_value=1.0, record_video=True):
        super().__init__(env)
        
        self.env = env
        self.device = self.env.device
        self.rew_scale_value = rew_scale_value
        # self.record_video = record_video
        
        self.num_obs = self.env.cfg["env"]["numObservations"]
        self.num_train_envs = self.env.num_envs
        self.episode_length_buf = torch.zeros(self.num_train_envs, dtype=torch.int, device=self.env.device, requires_grad=False)
        self.num_privileged_obs = 0
        
        if not hasattr(self, 'base_lin_vel'):
            self.base_lin_vel = torch.zeros(self.num_train_envs, 3, dtype=torch.float, device=self.env.device, requires_grad=False)
        if not hasattr(self, 'base_ang_vel'):
            self.base_ang_vel = torch.zeros(self.num_train_envs, 3, dtype=torch.float, device=self.env.device, requires_grad=False)
        if not hasattr(self, 'commands'):
            self.commands = torch.zeros(self.num_train_envs, 3, dtype=torch.float, device=self.env.device, requires_grad=False)

        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []
        
        # define function to import and compile files
        import importlib.util
        import sys

        def import_from_path(file_path):
            normalized_path = os.path.normpath(os.path.abspath(file_path))

            module_name = normalized_path.replace('.py', '')

            spec = importlib.util.spec_from_file_location(module_name, normalized_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        
        self.reward_funcs = [] 
        self.action_space_funcs = []
        self.obs_space_funcs = []
        self.terminal_condition_funcs = []
        self.init_state_dist_funcs = []
        self.goal_state_dist_funcs = []

    def l2d(self, l): 
        return {i: v for i, v in enumerate(l)}

    def register_shapers(self, shaper_funcs):

        self.reward_funcs = [shaper_func.get('compute_reward', None) for shaper_func in shaper_funcs]
        self.action_space_funcs = [shaper_func.get('compute_torque', None) for shaper_func in shaper_funcs]
        self.obs_space_funcs = [shaper_func.get('compute_observation', None) for shaper_func in shaper_funcs]
        self.terminal_condition_funcs = [shaper_func.get('compute_terminal', None) for shaper_func in shaper_funcs]

        self.num_actions = [shaper_func.get('policy_dim', None) for shaper_func in shaper_funcs]
        self.num_obs = [shaper_func.get('obs_dim', None) for shaper_func in shaper_funcs]

        # remove every None from the lists
        self.reward_funcs = self.l2d([fn for fn in self.reward_funcs if fn is not None])
        self.action_space_funcs = self.l2d([fn for fn in self.action_space_funcs if fn is not None])
        self.obs_space_funcs = self.l2d([fn for fn in self.obs_space_funcs if fn is not None])
        self.terminal_condition_funcs = self.l2d([fn for fn in self.terminal_condition_funcs if fn is not None])

        self.num_actions = [dim for dim in self.num_actions if dim is not None]
        self.num_obs = [dim for dim in self.num_obs if dim is not None]
        self.num_privileged_obs = [0 for _ in range(len(self.num_obs))]  #  THIS IS WEIRD.. 

        # check if the number of envs is divisible by the number of shapers
        assert self.num_train_envs % self.print_registration_status() == 0, "num_envs must be divisible by the number of shapers"


    def load_reward_functions(self, fns):
        self.reward_funcs = {i: fn for i, fn in enumerate(fns)}
        
    def load_action_space_functions(self, fns):
        self.action_space_funcs = {i: fn for i, fn in enumerate(fns)}

    def load_observation_space_functions(self, fns):
        self.obs_space_funcs = {i: fn for i, fn in enumerate(fns)}

    def load_terminal_condition_functions(self, fns):
        self.terminal_condition_funcs = {i: fn for i, fn in enumerate(fns)}

    def load_init_state_dist_functions(self, fns):
        self.init_state_dist_funcs = {i: fn for i, fn in enumerate(fns)}

    def load_goal_state_dist_functions(self, fns):
        self.goal_state_dist_funcs = {i: fn for i, fn in enumerate(fns)}

    def load_dimensions(self, dimensions: Dict):
        if 'action_space' in dimensions.keys():
            self.num_actions = dimensions['action_space']
        if 'observation_space' in dimensions:
            self.num_obs = dimensions['observation_space']
            self.num_privileged_obs = [0 for _ in range(len(dimensions['observation_space']))]

    def print_registration_status(self):

        print('='*50)
        print(f"Registered {len(self.reward_funcs)} reward functions")
        print(f"Registered {len(self.action_space_funcs)} action space functions")
        print(f"Registered {len(self.obs_space_funcs)} observation space functions")
        print(f"Registered {len(self.terminal_condition_funcs)} terminal condition functions")
        print(f"Registered {len(self.init_state_dist_funcs)} init state distribution functions")
        print(f"Registered {len(self.goal_state_dist_funcs)} goal state distribution functions")
        print('='*50)

        # raise if all nonzero lengths are not equal 

        # step 1. get all lengths
        lengths = [len(self.reward_funcs), len(self.action_space_funcs), len(self.obs_space_funcs), len(self.terminal_condition_funcs), len(self.init_state_dist_funcs), len(self.goal_state_dist_funcs)]

        # step 2. get all non-zero lengths
        non_zero_lengths = [l for l in lengths if l > 0]

        # step 3. check if all non-zero lengths are equal by making a set 
        if len(set(non_zero_lengths)) > 1:
            raise ValueError("All non-zero lengths should be equal")
        elif len(set(non_zero_lengths)) == 1:
            return non_zero_lengths[0]
        else:
            return 1 




    def get_parsed_state_dict(self, index, num_candidates):
        
        # go through the state dict and parse the values over index 0 
        # but only if the value is a tensor 
        # if it's a tensor, it's going to be (num_envs, *rest_of_the_shape)
        # thus, we need to parse the tensor for dim 0 (chunk by num_candidates)

        state_dict = self.env.sim_state_dict
        parsed_state_dict = {}
        for key, value in state_dict.items():
            parsed_state_dict[key] = value
            if isinstance(value, torch.Tensor):
                try:
                    if value.shape[0] == self.env.num_envs:
                        parsed_state_dict[key] = torch.chunk(value, num_candidates, dim=0)[index]  
                except:
                    pass 
                    # print(f"Error parsing state dict for key {key} with value {value} shape {value.shape}")  

        return parsed_state_dict   


    def step(self, action: List[torch.Tensor]):

        # list of actions coming from different policies 
        # these cannot be concatenated; since they are of different dimensions

        self.env.update_sim_state_dict()

        #################################
        ### STEP 0. Pre-Physics Step  ###
        #################################


        if len(self.action_space_funcs) > 0:
            torques = [] 
            num_candidates = len(self.action_space_funcs)
            for i, action_space_func in self.action_space_funcs.items():
                torque = action_space_func(action[i], self.get_parsed_state_dict(i, num_candidates))
                torques.append(torque)
                # these torques should have the same dimension
                # and can be concatenated
            torques = torch.cat(torques, dim=0)

        else:
            # this means that there is no action space function
            # thus all policies are directly outputting torques 
            torques = torch.cat(action, dim = 0)

        self.env.actions[:] = torques.clone().to(self.env.device)  # Option 1: these converted torques are being fed into self.env.actions
        # self.env.actions[:] = torch.cat([a for a in action], dim=0).to(self.env.device) # Option 2: expose the original actions instead of the torques

        # apply as a torque control 
        self.env.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

        #################################
        ### STEP 1. Physics Step      ###
        #################################

        for i in range(self.env.control_freq_inv):
            if self.env.force_render:
                self.env.render()
            self.env.gym.simulate(self.env.sim)
        
        ###################################
        ### STEP 2. Compute Observation ###
        ###################################

        dict_obs = {}

        if len(self.obs_space_funcs) > 0:
            num_candidates = len(self.obs_space_funcs)
            for i, obs_space_func in self.obs_space_funcs.items():
                obs_out = obs_space_func(self.get_parsed_state_dict(i, num_candidates))
                dict_obs[f'{i}'] = obs_out
        else: 
            obs = self.env.compute_observations()
            dict_obs['obs'] = obs

        ###################################
        ### STEP 3. Compute Reward      ###
        ###################################

        dict_rew = {} 
        dict_rew['ext'] = self.env.compute_reward() 

        if len(self.reward_funcs) > 0:
            num_candidates = len(self.reward_funcs)
            for i, reward_func in self.reward_funcs.items():
                rew_out = reward_func(self.get_parsed_state_dict(i, num_candidates))
                dict_rew[f'{i}'] = rew_out
        else: 
            dict_rew['rew'] = self.env.compute_reward()

        dict_rew = {k: v*self.rew_scale_value for k, v in dict_rew.items()}

        ###################################
        ### STEP 3. Compute Reset       ###
        ###################################

        if len(self.terminal_condition_funcs) > 0:
            dones = [] 
            num_candidates = len(self.reward_funcs)
            for i, terminal_condition_func in self.terminal_condition_funcs.items():
                done = terminal_condition_func(self.get_parsed_state_dict(i, num_candidates))
                dones.append(done.to(self.env.device))
            done = torch.cat(dones, dim=0)

        else:
            done = self.env.compute_terminal()
            
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.env.reset_idx(env_ids)

        ###################################
        ### STEP 4. MISC Operations     ###
        ###################################

        self.env.control_steps += 1
        self.env.progress_buf += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.env.timeout_buf = (self.env.progress_buf >= self.env.max_episode_length - 1) & (done != 0)
        
        info = dict() 
        info['time_outs'] = self.env.timeout_buf.to(self.env.rl_device)
        info["env_bins"] = torch.zeros(self.num_train_envs, dtype=torch.int, device=self.env.device, requires_grad=False)
        
        return dict_obs, dict_rew, done, info
    
    def get_observations(self):
        dict_obs = {}
        if len(self.obs_space_funcs) > 0:
            num_candidates = len(self.obs_space_funcs)
            for i, obs_space_func in self.obs_space_funcs.items():
                obs_out = obs_space_func(self.get_parsed_state_dict(i, num_candidates))
                dict_obs[f'{i}'] = obs_out
        else: 
            obs = self.env.compute_observations()
            dict_obs['obs'] = obs

        return dict_obs