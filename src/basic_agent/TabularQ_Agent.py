import math
from typing import Optional, Union, Literal

from VectorEnv.great_para_env import ParallelEnv
from basic_agent.basic_agent import Basic_Agent
from basic_agent.utils import *


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


class TabularQ_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.n_state = self.config.n_state
        self.epsilon = self.config.epsilon
        self.lr_model = self.config.lr_model

        self.q_table = torch.zeros(self.n_state, self.n_act)
        
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
        
    # def get_action(self, state, epsilon_greedy=False):
    #     Q_list = self.q_table(state)
    #     if epsilon_greedy and np.random.rand() < self.epsilon:
    #         action = np.random.randint(low=0, high=self.n_act, size=len(state))
    #     else:
    #         action = torch.argmax(Q_list, -1).numpy()
    #     return action
    
    
    def get_action(self, state, epsilon_greedy=False):
        Q_list = torch.stack([self.q_table[st] for st in state ])
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.n_act, size=len(state))
        else:
            action = torch.argmax(Q_list, -1).numpy()
        return action

    def train_episode(self, 
                      envs, 
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={'normalizer': 'normalizer',
                                     'gbest':'gbest'
                                     }):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        
        # params for training
        gamma = self.gamma
        
        state = env.reset()
        state = torch.FloatTensor(state)
        
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)
                        
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            
            # error = reward + gamma * self.q_table[next_state].max() - self.q_table[state][action]
            
            error = [reward[i] + gamma * self.q_table[next_state[i]].max() - self.q_table[state[i]][action[i]]\
                for i in range(len(state)) ]
            
            for i in range(len(state)):
                self.q_table[state[i]][action[i]] += self.lr_model * error[i]
            
            # store info
            state = torch.FloatTensor(next_state)
            
            self.learning_time += 1
            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

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
                      required_info={'normalizer': 'normalizer',
                                     'gbest':'gbest'
                                     }):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                action = self.get_action([state])[0]
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
                      required_info={'normalizer': 'normalizer',
                                     'gbest':'gbest'
                                     }):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        
        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(torch.FloatTensor(state))
            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            state = torch.FloatTensor(state)
        results = {'return': R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results

