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


class DDQN_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.memory_size = self.config.memory_size
        self.batch_size = self.config.batch_size
        self.warm_up_size = self.config.warm_up_size
        self.target_update_interval = self.config.target_update_interval
        self.device = self.config.device
        
        # figure out the actor network
        # self.model = None
        assert hasattr(self, 'model')
        
        self.target_model = copy.deepcopy(self.model)
        
        # figure out the optimizer
        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(
            [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # figure out the lr schedule
        assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()
        
        self.replay_buffer = ReplayBuffer(self.memory_size)
        
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
        
    def get_action(self, state, epsilon_greedy=False):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.n_act, size=len(state))
        else:
            action = int(torch.argmax(Q_list, -1).detach().cpu().numpy())
        return action

    def train_episode(self, 
                      envs, 
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={'cost': 'cost',
                                     }):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        
        # params for training
        gamma = self.gamma
        
        state = env.reset()
        try:
            state = torch.FloatTensor(state)
        except:
            pass
        
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)
                        
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            # store info
            for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.FloatTensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)
            
            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
                pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device), self.n_act)  # [batch_size, n_act]
                
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
                target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * self.target_model(batch_next_obs.to(self.device)).max(1).detach()
                
                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                self.optimizer.step()

                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1
                    
                if self.learning_time % self.target_update_interval == 0:
                    for target_parma, parma in zip(self.target_model.parameters(), self.model.parameters()):
                        target_parma.data.copy_(parma.data)

                if self.learning_time >= self.config.max_learning_step:
                    return_info = {'return': _R, 'learn_steps': self.learning_time, }
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info
        
            
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
                action = self.get_action(state)[0]
                state, reward, is_done,info = env.step(action)
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
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.get_action(state)
            

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

