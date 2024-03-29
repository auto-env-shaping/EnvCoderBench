import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]
    
    adaptation_labels = []
    adaptation_dims = []
    adaptation_weights = []

    use_decoder = False

class ActorCriticModel(nn.Module):
    def __init__(self,
                 num_privileged_obs,
                 num_obs,
                 num_actions,
                 is_use_int=False,
                 output_tuple=False,
                 **kwargs):
        super().__init__()

        self.activation = get_activation(AC_Args.activation)
        self.is_use_int = is_use_int
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        
        self.adaptation_module = self.build_adaptation_layers()

        # Policy
        self.actor_body_pre, self.actor_body_post = self.build_policy_layers(num_actions)

        # Value function
        self.critic_body_pre, self.critic_body_post = self.build_critic_layers()
        
        self.critic_body_enrg = None
        self.output_tuple = output_tuple
        if is_use_int:
            self.critic_body_int = self.build_critic_layers()
        
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        
    def build_adaptation_layers(self):
        
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(self.activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                            AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(self.activation)
        adaptation_module = nn.Sequential(*adaptation_module_layers)
        return adaptation_module

    def build_policy_layers(self, num_actions):

        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(self.activation)
        
        actor_body_pre = nn.Identity()
        actor_body_post = nn.Sequential(*actor_layers)
            
        return actor_body_pre, actor_body_post

    def build_critic_layers(self):

        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(self.activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(self.activation)
        critic_body_pre = nn.Identity()
        critic_body_post = nn.Sequential(*critic_layers)

        return critic_body_pre, critic_body_post
    
    def preprocess(self, obs, adaptation_module_input=None, privileged_observations=None):
        if privileged_observations is None:
            latent = self.adaptation_module(adaptation_module_input)
            pre_latent_critic = None
        else:
            latent = privileged_observations
            pre_latent_critic = self.critic_body_pre(torch.cat((obs, latent), dim=-1))
        
        pre_latent = self.actor_body_pre(torch.cat((obs, latent), dim=-1))
        return pre_latent, pre_latent_critic

    def update_distribution(self, post_latent):
        mean = self.actor_body_post(post_latent)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def get_current_actions_log_prob(self, post_latent):
        mean = self.actor_body_post(post_latent)
        dis = Normal(mean, mean * 0. + self.std)
        return dis.log_prob(dis.sample()).sum(dim=-1)

    def act(self, post_latent, **kwargs):
        self.update_distribution(post_latent)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, post_latent, policy_info={}):
        return self.act_teacher(post_latent, policy_info=policy_info)

    def act_inference(self, post_latent, policy_info={}):
        return self.act_student(post_latent, policy_info=policy_info)

    def act_student(self, post_latent, policy_info={}):
        actions_mean = self.actor_body_post(post_latent)
        # policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, post_latent, privileged_info, policy_info={}):
        actions_mean = self.actor_body_post(post_latent)
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, post_latent_critic, **kwargs):
        value = self.critic_body_post(post_latent_critic)
        return value

    def get_student_latent(self, obs):
        return self.adaptation_module(obs)
    

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_actions,
                 n_policies=1,
                 ptrn_path=None,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        self.n_policies=n_policies

        self.adaptation_labels = AC_Args.adaptation_labels
        self.adaptation_dims = AC_Args.adaptation_dims
        self.adaptation_weights = AC_Args.adaptation_weights

        self.black_list = [] 

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        super().__init__()

        self.a2c_models = nn.ModuleDict({str(idx): ActorCriticModel(num_privileged_obs[idx] if isinstance(num_privileged_obs, list) else num_privileged_obs,
                                                            num_obs[idx] if isinstance(num_obs, list) else num_obs,
                                                            num_actions[idx] if isinstance(num_actions, list) else num_actions) for idx in range(self.n_policies)})
        print(f"Adaptation Module: {self.a2c_models['0'].adaptation_module}")
        print(f"Actor Pre: {self.a2c_models['0'].actor_body_pre}")
        print(f"Actor Post: {self.a2c_models['0'].actor_body_post}")
        print(f"Critic Pre: {self.a2c_models['0'].critic_body_pre}")
        print(f"Critic Post: {self.a2c_models['0'].critic_body_post}")

        self.distributions = {}
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return {n: self.distributions[n].mean for n in self.distributions}       
     
    @property
    def action_std(self):
        return {n: self.distributions[n].stddev for n in self.distributions}
      
    @property
    def entropy(self):
        entropies = {n: self.distributions[n].entropy().sum(dim=-1) for n in self.distributions}
        return sum(entropies.values())
       
    def update_distribution(self, obs):
        for idx in range(self.n_policies):
            try:
                latent, critic_latent = self.a2c_models[str(idx)].preprocess(obs[f'{idx}'], obs[f'{idx}'])
                self.a2c_models[str(idx)].update_distribution(latent)
                self.distributions[str(idx)] = self.a2c_models[str(idx)].distribution
            except:
                print('Error in updating distribution {}'.format(idx))
                pass 
        
    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        sampled_actions = {n: self.distributions[n].sample() for n in self.distributions}
        return sampled_actions

    def get_actions_log_prob(self, actions: dict):
        log_probs = {n: self.distributions[n].log_prob(actions[n]).sum(dim=-1) for n in self.distributions}
        return log_probs

    def act_expert(self, obs, policy_info={}):
        return self.act_teacher(obs)

    def act_inference(self, obs, policy_info={}):
        return self.act_student(obs, policy_info=policy_info)

    def act_student(self, obs, policy_infos={}):
        policy_infos["latents"] = {}
        actions_means = {}
      
        for idx in range(self.n_policies):
            latent, critic_latent = self.a2c_models[str(idx)].preprocess(obs[f'{idx}'], obs[f'{idx}'])
          
            policy_info = {}
            actions_means[idx] = self.a2c_models[str(idx)].act_student(latent, policy_info)
            policy_infos["latents"][str(idx)] = policy_info["latents"]

        return actions_means

    def act_teacher(self, obs, privileged_info, policy_infos={}):
        policy_infos["latents"] = {}
        actions_means = {}
        
        for idx in range(self.n_policies):
            latent, critic_latent = self.a2c_models[str(idx)].preprocess(obs[f'{idx}'], obs[f'{idx}'])
            policy_info = {}
            actions_means[idx] = self.a2c_models[str(idx)].act_teacher(latent, policy_info)
            policy_info["latents"][str(idx)] = privileged_info.detach().cpu().numpy() 
        
        return actions_means

    def evaluate(self, obs, **kwargs):
        values = {}
        
        for idx in range(self.n_policies):
            # latent, critic_latent = self.a2c_models[str(idx)].preprocess(obs[f'{idx}'], obs[f'{idx}'])
            try:
                values[str(idx)] = self.a2c_models[str(idx)].evaluate(obs[f'{idx}'])
            except:
                print('Error in evaluating value function {}'.format(idx))
                pass 
        return values

    def get_student_latent(self, obs):
        return self.adaptation_module(obs)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
