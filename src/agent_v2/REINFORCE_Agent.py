from typing import Tuple
from agent.basic_agent import Basic_Agent
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from agent.utils import *
from VectorEnv.great_para_env import ParallelEnv

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.logprobs[:]
        del self.rewards[:]


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class REINFORCE_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.max_grad_norm = self.config.max_grad_norm
        self.device = self.config.device
        
        # figure out the actor network
        # self.model = None
        assert hasattr(self, 'model')

        # figure out the optimizer
        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(
            [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # figure out the lr schedule
        assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)

        # move to device
        self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir,'checkpoint'+str(self.cur_checkpoint),self)
        self.cur_checkpoint += 1

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def train_episode(self, 
                      envs, 
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        
        memory = Memory()

        # params for training
        gamma = self.gamma
        
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            entropy = []

            action, log_lh, entro_p  = self.model(state)
            

            memory.logprobs.append(log_lh)
            
            entropy.append(entro_p.detach().cpu())

            # state transient
            state, rewards, is_end, info = env.step(action)
            memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
            _R += rewards
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        
        # begin update
        logprobs = torch.stack(memory.logprobs).view(-1).to(self.device)
        Reward = []
        reward_reversed = memory.rewards[::-1]
        R = torch.zeros(len(envs))
        for r in range(len(reward_reversed)):
            R = R * gamma + reward_reversed[r]
            Reward.append(R)
        # clip the target:
        Reward = torch.stack(Reward[::-1], 0)
        Reward = Reward.view(-1).to(self.device)
        loss = - torch.mean(logprobs * Reward)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
        self.optimizer.step()
        
        memory.clear_memory()
        
        self.learning_time += 1
        if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
            save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1
            
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, }
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        
        return is_train_ended, return_info
    
    def rollout_episode(self, 
                        env,
                        seed=None,
                        required_info={}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                except:
                    state = [state]
                action = self.model(state)
                action = action.cpu().numpy().squeeze()
                state, reward, is_done = env.step(action)
                R += reward
            results = {'return': R}
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                              num_cpus: Optional[Union[int, None]]=1,
                              num_gpus: int=0,
                              required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        R = torch.zeros(len(env))
        entropy = []
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action, log_lh, entro_p  = self.model(state)
            
            entropy.append(entro_p.detach().cpu())

            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        results = {'return': R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results

