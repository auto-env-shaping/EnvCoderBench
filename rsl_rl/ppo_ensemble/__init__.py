import time
from collections import deque
import copy
import os

import torch
from torch import nn
# from ml_logger import logger
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from params_proto import PrefixProto

from .actor_critic import ActorCritic, ActorCriticModel
from .rollout_storage import RolloutStorage
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from rsl_rl.ppo_ensemble.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    save_curriculum_interval = 20
    log_freq = 10

    # load and resume
    resume = False
    resume_supercloud = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = 'ac_weights_last.pt'



class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO


        wandb.init()

        self.device = device
        self.env = env
        self.n_policies = env.print_registration_status()

        self.randint = torch.randint(0, 1000000, (1,)).item()

          
        actor_critic = ActorCritic(self.env.num_obs, # <-- ARE THESE EXPECTING LIST/DICT OR SINGLE VALUE?
                                      self.env.num_privileged_obs,  
                                      self.env.num_actions,  
                                      self.n_policies).to(self.device)

        # Load weights from checkpoint 
        if RunnerArgs.resume:
            body = wandb.restore(RunnerArgs.resume_checkpoint, run_path=RunnerArgs.resume_path)            
            actor_critic.load_state_dict(torch.load(body.name))
            print(f"Successfully loaded weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path})")
        
        self.alg = PPO(actor_critic, self.n_policies, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        num_train_envs = self.env.num_train_envs
        # print(num_train_envs, self.n_policies)
        self.alg.init_storage(num_train_envs // self.n_policies, self.num_steps_per_env, self.env.num_obs,
                              self.env.num_privileged_obs, self.env.num_actions)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()
        self.logdata = []

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        
        trigger_sync = TriggerWandbSyncHook()
        wandb.watch(self.alg.actor_critic, log=None, log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs
        
        observations = self.env.get_observations()
        observations = {n: v.to(self.device) for (n, v) in observations.items()}         
        
        self.logdata = []

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        record_log = {}
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):

            start = time.time()
            std_norm = []
            # Rollout
            for i in range(self.num_steps_per_env):
                with torch.inference_mode():
                    actions = self.alg.act(observations)
                    
                    actions = [actions[str(idx)] for idx in range(self.n_policies)]
                    ret = self.env.step(actions)
                    observations, rewards, dones, infos = ret

                    ## Logging norm of action std
                    action_stds = self.alg.actor_critic.action_std
                    std_norms = {n: v.norm(dim=-1).mean().item() for (n, v) in action_stds.items()}
                    # take mean over dictionary
                    # find number of keys in dictionary std_norms
                    std_norm.append(sum(std_norms.values()) / len(std_norms))

                    observations = {n: v.to(self.device) for (n, v) in observations.items()}
                    dones = dones.to(self.device)

                    rewards = {n: v.to(self.device) for (n, v) in rewards.items()}
                    self.alg.process_env_step({n: v[:num_train_envs] for (n, v) in rewards.items()}
                            , dones[:num_train_envs], infos)
                    if 'train/episode' in infos:
                        for (k, v) in infos['train/episode'].items():
                            if 'rew_' in k and k in record_log:
                                record_log[k] = record_log[k] + v
                            else:
                                record_log[k] = v

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

            with torch.inference_mode():
                self.alg.compute_returns(observations)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses_dict, entropy_loss = self.alg.update(it)

            stop = time.time()
            learn_time = stop - start

            for k in record_log:
                if 'rew_' in k:
                    record_log[k] = sum(record_log[k]) / len(record_log[k])
            record_log.update({
                "time_iter": learn_time,
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "entropy_loss": entropy_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss
            })
            for k, v in self.alg.learning_rate.items():
                record_log[f'lr_{k}'] = v
            record_log["kl_mean"] = self.alg.kl_mean
            record_log['std_norm'] = sum(std_norm) / len(std_norm)
            for k, v in std_norms.items():
                record_log[f'std_norm_{k}'] = v

            wandb_record_log = {}
            for k in record_log:
                if k.startswith('rew'):
                    wandb_record_log[k.replace('rew_', 'return/')] = record_log[k]
                elif 'command' in k:
                    wandb_record_log[f'commands/{k}'] = record_log[k]
                elif 'alpha' in k:
                    wandb_record_log[f'alpha/{k}'] = record_log[k]
                else:
                    wandb_record_log[f'metrics/{k}'] = record_log[k]
                
            wandb.log(wandb_record_log, step=it)
            
            rewards_log = {}
            bsz = self.env.num_envs // self.n_policies
            
            for idx in range(self.n_policies):
                rewards_log[f"rew_{idx}"] = rewards[f"{idx}"].mean()
                rewards_log[f"ext_{idx}"] = rewards["ext"][idx*bsz:(idx+1)*bsz].mean()
            wandb.log(rewards_log, step=it)

            self.logdata += [rewards_log]

            record_log = {}

            wandb.log(mean_adaptation_losses_dict, step=it)
            
            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            
            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)
            trigger_sync()

            if it % RunnerArgs.save_interval == 0:
                    print(f"Saving model at iteration {it}")

                    path = f'./tmp{self.randint}/legged_data'
                    if 'const' in path:
                        path = path + '-alpha{:.1f}'.format(self.alg.alpha)
                    for n in self.alg.actor_critic.a2c_models:
                        os.makedirs(f'{path}-{n}', exist_ok=True)

                        ac_weight_path = f'{path}-{n}/ac_weights_{it}.pt'
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weight_path)
                        wandb.save(ac_weight_path)

                        ac_weight_path = f'{path}-{n}/ac_weights_latest.pt'
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weight_path)
                        wandb.save(ac_weight_path)

                        adaptation_module_path = f'{path}-{n}/adaptation_module_{it}.jit'
                        adaptation_module = copy.deepcopy(self.alg.actor_critic.a2c_models[n].adaptation_module).to('cpu')
                        traced_script_adaptation_module = torch.jit.script(adaptation_module)
                        traced_script_adaptation_module.save(adaptation_module_path)
                        wandb.save(adaptation_module_path)

                        adaptation_module_path = f'{path}-{n}/adaptation_module_latest.jit'
                        traced_script_adaptation_module.save(adaptation_module_path)
                        wandb.save(adaptation_module_path)

                        # @TODO UNCOMMENT AND FIX TO SAVE MODELS
                        # body_path = f'{path}-{n}/body_{it}.jit'
                        # body_model = copy.deepcopy(self.alg.actor_critic.a2c_models[n].actor_body).to('cpu')
                        # traced_script_body_module = torch.jit.script(body_model)
                        # traced_script_body_module.save(body_path)
                        # wandb.save(body_path)
                        # body_path = f'{path}-{n}/body_latest.jit'
                        # traced_script_body_module.save(body_path)
                        # wandb.save(body_path)

                    
                        ac_weights_path = f"{path}-{n}/ac_weights_{it}.pt"
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weights_path)
                        ac_weights_path = f"{path}-{n}/ac_weights_latest.pt"
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weights_path)
                    
                    wandb.save(f"./tmp{self.randint}/legged_data/adaptation_module_{it}.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/body_{it}.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/ac_weights_{it}.pt")
                    wandb.save(f"./tmp{self.randint}/legged_data/adaptation_module_latest.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/body_latest.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/ac_weights_latest.pt")
                    
        self.current_learning_iteration += num_learning_iterations
        os.rmdir(f'./tmp{self.randint}')

    def get_metrics(self, latest = True):

        # get the last logdata 
        
        ext_metric = []
        latest =  self.logdata[-1]
        for idx in range(self.n_policies):
            ext_metric.append(latest[f"ext_{idx}"].item())

        best_idx = np.argmax(ext_metric)

        extra_info = dict()

        # YOU CAN ADD MORE METRICS HERE
        extra_info['ext_metric'] = ext_metric

        return best_idx, extra_info

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            wandb.log({"video": wandb.Video(video_array, fps=1 / self.env.dt)}, step=it)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
