import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from rsl_rl.ppo_ensemble import ActorCritic, ActorCriticModel
from rsl_rl.ppo_ensemble import RolloutStorage
from rsl_rl.ppo_ensemble import caches

class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    adaptation_batch_size = 64
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, n_policies=1, device='cpu'):

        self.device = device
        self.n_policies = n_policies

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)

        PPO_Args.adaptation_labels = self.actor_critic.adaptation_labels
        PPO_Args.adaptation_dims = self.actor_critic.adaptation_dims
        PPO_Args.adaptation_weights = self.actor_critic.adaptation_weights
        
        self.storage = None  # initialized later
        self.optimizer = {}
        self.adaptation_module_optimizer = {}
        self.learning_rate = {}
        self.kl_mean = {}

        for n in self.actor_critic.a2c_models:
            self.learning_rate[n] = PPO_Args.learning_rate
            self.optimizer[n] = optim.Adam(self.actor_critic.a2c_models[n].parameters(),
                                          lr=PPO_Args.learning_rate)
            self.adaptation_module_optimizer[n] = optim.Adam(self.actor_critic.a2c_models[n].parameters(),
                                          lr=PPO_Args.adaptation_module_learning_rate)

        # Initialize the kl_mean dictionary
        self.kl_mean = {}
        self.transition = RolloutStorage.Transition()        

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                    action_shape, self.n_policies, self.device)
        
    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def reinitialize_actor_critic(self, index, num_obs, num_privileged_obs, num_actions):
        # Create a new actor_critic with the updated specifications
        new_actor_critic = ActorCriticModel(
            num_privileged_obs[index] if isinstance(num_privileged_obs, list) else num_privileged_obs,
            num_obs[index] if isinstance(num_obs, list) else num_obs,
            num_actions[index] if isinstance(num_actions, list) else num_actions,
        )

        # Replace the actor_critic at the specified index with the new one
        self.actor_critic.a2c_models[str(index)] = new_actor_critic.to(self.device)

        # Update the optimizer to refer to the new parameters of the reinitialized actor_critic
        self.optimizer[str(index)] = optim.Adam(self.actor_critic.a2c_models[str(index)].parameters(), lr=PPO_Args.learning_rate)

        # Update the rollout storage dimensions for the reinitialized actor_critic
        self.storage.observations[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs,
                                                            num_obs[index], device=self.storage.device)
        self.storage.privileged_observations[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs,
                                                                        num_privileged_obs[index], device=self.storage.device)
        self.storage.actions[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs,
                                                        num_actions[index], device=self.storage.device)
        self.storage.rewards[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, 1,
                                                        device=self.storage.device)
        self.storage.values[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, 1,
                                                    device=self.storage.device)
        self.storage.returns[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, 1,
                                                        device=self.storage.device)
        self.storage.advantages[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, 1,
                                                        device=self.storage.device)
        self.storage.mu[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, num_actions[index],
                                                device=self.storage.device)
        self.storage.sigma[str(index)] = torch.zeros(self.storage.num_transitions_per_env, self.storage.num_envs, num_actions[index],
                                                    device=self.storage.device)

    def act(self, obs):
        # Compute the actions and values
        self.transition.actions = {n: v.detach() 
                                   for (n, v) in self.actor_critic.act(obs).items()}
        self.transition.values = {n: v.detach() 
                                   for (n, v) in self.actor_critic.evaluate(obs).items()}
        self.transition.actions_log_prob = {n: v.detach() 
                                            for (n, v) in self.actor_critic.get_actions_log_prob(self.transition.actions).items()}
        self.transition.action_mean = {n: v.detach() 
                                        for (n, v) in self.actor_critic.action_mean.items()}
        self.transition.action_sigma = {n: v.detach() 
                                        for (n, v) in self.actor_critic.action_std.items()}
        self.transition.observations = {str(idx): obs[str(idx)] \
                                        for idx in range(self.n_policies)}
        self.transition.critic_observations = {str(idx): obs[str(idx)] \
                                        for idx in range(self.n_policies)}
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):

        bsz = len(rewards["0"])

        self.transition.rewards = {str(idx): rewards[f"{idx}"].clone() \
                                    for idx in range(self.n_policies)
                                    }
        self.transition.dones = {str(idx): dones[idx * bsz:(idx + 1) * bsz] \
                                    for idx in range(self.n_policies)
            }
        time_outs = {str(idx): infos['time_outs'].unsqueeze(1).to(self.device)[idx * bsz:(idx + 1) * bsz] \
                        for idx in range(self.n_policies)
                        }
        values = {  str(idx): self.transition.values[str(idx)] \
                        for idx in range(self.n_policies)
        }


        if 'time_outs' in infos:
            for n in self.transition.rewards:
                # print(time_outs[n].shape, values[n].shape)
                self.transition.rewards[n] += PPO_Args.gamma * torch.squeeze(
                    values[n] * time_outs[n], 1)
                
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = {n: v.detach() 
                       for (n, v) in self.actor_critic.evaluate(last_critic_obs).items()}
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

            
    def compute_value_loss(self, value_batch, target_values_batch, returns_batch):
        if PPO_Args.use_clipped_value_loss:
            value_clipped = target_values_batch + \
                            (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                        PPO_Args.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss
    
    def compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                            1.0 + PPO_Args.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        mean_entropy_loss = 0
        
        mean_adaptation_losses = {}
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0

        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, \
            rewards_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            # current_obs_batch = torch.cat([obs_batch[idx] for idx in range(self.n_policies)])
            
            self.actor_critic.act(obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # Update each policy separately

            for name, nint in zip(self.actor_critic.a2c_models, range(self.n_policies)):
                n = str(nint)
                # KL
                if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch[n] / \
                                        old_sigma_batch[n] + 1.e-5)
                                + (torch.square(old_sigma_batch[n]) 
                                    + torch.square(old_mu_batch[n] 
                                                - mu_batch[n])) / (
                                    2.0 * torch.square(sigma_batch[n])) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        self.kl_mean[n] = kl_mean

                        if kl_mean > PPO_Args.desired_kl * 2.0:
                            self.learning_rate[name] = max(1e-5, self.learning_rate[name] / 1.5)
                        elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate[name] = min(1e-2, self.learning_rate[name] * 1.5)

                        for param_group in self.optimizer[name].param_groups:
                            param_group['lr'] = self.learning_rate[name]

            # Surrogate loss
            surrogate_loss_dict = {}
            for n in actions_log_prob_batch:
                surrogate_loss_dict[n] = self.compute_surrogate_loss(actions_log_prob_batch[n], \
                                                            old_actions_log_prob_batch[n], \
                                                            advantages_batch[n])
            surrogate_loss = sum(surrogate_loss_dict.values()) / len(surrogate_loss_dict.values())
            
            # Value function loss
            value_loss_dict = {}
            for n in value_batch:
                value_loss_dict[n] = self.compute_value_loss(value_batch[n], \
                                                            target_values_batch[n], \
                                                            returns_batch[n])
            total_value_loss = sum(value_loss_dict.values()) / len(value_loss_dict.values())
            loss = surrogate_loss + PPO_Args.value_loss_coef * total_value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            for n in self.actor_critic.a2c_models:
                self.optimizer[n].zero_grad()
            loss.backward()
                
            
                
            for n in self.actor_critic.a2c_models:
                nn.utils.clip_grad_norm_(self.actor_critic.a2c_models[str(n)].parameters(), 
                                         PPO_Args.max_grad_norm)

            for n in self.actor_critic.a2c_models:
                self.optimizer[n].step()

            mean_value_loss += total_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += -PPO_Args.entropy_coef * entropy_batch.mean().item()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps * PPO_Args.adaptation_batch_size)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        for label in PPO_Args.adaptation_labels:
            mean_adaptation_losses[label] /= (num_updates * PPO_Args.num_adaptation_module_substeps * PPO_Args.adaptation_batch_size)

        self.storage.clear()
        
        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses, mean_entropy_loss
    
