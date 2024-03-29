import isaacgym
import os
import argparse
from collections import namedtuple
import shutil
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import hydra
from hydra.utils import to_absolute_path
from hydra.experimental import initialize, compose
from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf
import datetime
from params_proto import PrefixProto, ParamsProto
from typing import * 

ROOT_DIR = "../../isaacgymenvs"

# Options to configure runs
class RunCfg(PrefixProto, cli=False):
    task_name = "Anymal" # ["Ant", "Humanoid", "ShadowHand", ...]
    num_iterations = None # [1000, 5000, ...]
    num_envs = 3000 #* 3 # [64, 128, 256, 512, 1024, 2048, 4096]
    entropy_coef = 0.0 # [0.0, 0.01, ...]
    learning_rate = 1e-3
    sinusoidal_alpha = False
    n_policies = 1
    use_shared_layers = False
    shared_layers_mode = None
    use_shared_actor_body = False
    use_shared_critic_body = False
    use_shared_std = False
    use_shared_adaptation_module = False
    reward_group = None # ["coef_tuning", "dud_robustness"]
    
    seed = 0
    
    wandb_project_name = "eureka-dev"
    wandb_group = ""
    wandb_name = ""
    wandb_job_type = ""

def create_env(gpu_id, task_name,  **kwargs):
    with initialize(config_path=f"../../isaacgymenvs/cfg"):
        # Compose the configuration
        cfg = compose(config_name="config", overrides=[f"task={task_name}", f"headless=True"] + \
                      [f"{k}={v}" for k, v in kwargs.items() if not isinstance(v, dict)])
        
        # Update the config with nested parameters from kwargs
        def update_nested_config(config, nested_dict):
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    update_nested_config(getattr(config, key), value)
                else:
                    setattr(config, key, value)
        
        for key, value in kwargs.items():
            if isinstance(value, dict):
                update_nested_config(cfg, {key: value})
        
    cfg.wandb_group = RunCfg.wandb_group
    cfg.wandb_name = RunCfg.wandb_name

    import isaacgymenvs

    from rsl_rl.ppo_ensemble.actor_critic import AC_Args
    from rsl_rl.ppo_ensemble.ppo import PPO_Args
    from rsl_rl.ppo_ensemble import RunnerArgs
    PPO_Args.entropy_coef = RunCfg.entropy_coef
    PPO_Args.learning_rate = RunCfg.learning_rate
    RunnerArgs.save_video_interval = False
    AC_Args.use_shared_layers = RunCfg.use_shared_layers
    AC_Args.shared_layers_mode = RunCfg.shared_layers_mode
    AC_Args.use_shared_actor_body = RunCfg.use_shared_actor_body
    AC_Args.use_shared_critic_body = RunCfg.use_shared_critic_body
    AC_Args.use_shared_adaptation_module = RunCfg.use_shared_adaptation_module
    AC_Args.use_shared_std = RunCfg.use_shared_std
    config_log = {
        "AC_Args": vars(AC_Args),
        "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs),
        "RunCfg": vars(RunCfg),
    }
    
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"
    
    # cfg.seed = RunCfg.seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.train.params.config.multi_gpu = cfg.multi_gpu
    
    cfg.sim_device = f"cuda:{gpu_id}"
    cfg.rl_device = f"cuda:{gpu_id}"
    cfg.graphics_device_id = gpu_id
    
    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            if cfg.test:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0),
                    video_length=cfg.capture_video_len,
                )
            else:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0) and (step > 0),
                    video_length=cfg.capture_video_len,
                )
        return envs

    from rsl_rl.utils.isaacgymenvs_wrapper import IsaacGymEnvsWrapper
    env = create_isaacgym_env()
    env = IsaacGymEnvsWrapper(env, rew_scale_value=1.0)#, reward_group=RunCfg.reward_group)

    return env, cfg, config_log


def rollout(env, shaping_functions = None, dimensions = None, gpu_id = 0, **deps): 
        
    if shaping_functions is not None:
        env.load_reward_functions(shaping_functions['reward'])
        env.load_action_space_functions(shaping_functions['action_space'])
        env.load_observation_space_functions(shaping_functions['observation_space'])
        env.load_terminal_condition_functions(shaping_functions['terminal_condition'])
        env.load_init_state_dist_functions(shaping_functions['init_state_dist'])
        env.load_goal_state_dist_functions(shaping_functions['goal_state_dist'])
    

    num_groups = env.print_registration_status()
    print(num_groups)

    import torch
    action = torch.zeros((env.num_envs // num_groups, env.num_actions), device=env.device)  
    actions = [action] * num_groups
    from tqdm import tqdm
    env.reset()
    for i in tqdm(range(100)):
        env.step(actions)

    print(get_state_dict_summary(env))

    return env
        
def get_state_dict_summary(env):

    state_dict = env.sim_state_dict
    # get the summary of the state_dict
    # includes the key, type, and shape of the tensors
    # replace f{num_envs} to string "num_envs"

    state_dict_summary = '====== state_dict_summary ======\n'
    for key, value in state_dict.items():
        string = f'{key}: {str(value.dtype)} {str(value.shape)}\n'
        string = string.replace(f'{env.num_envs}', 'num_envs')
        state_dict_summary += string
    state_dict_summary += '=================================\n'
    return state_dict_summary