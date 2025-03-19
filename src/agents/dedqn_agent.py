import numpy as np
import torch
from .DQN_Agent import DQN_Agent
from agent.networks import MLP
from agent.utils import *
import math

class DEDQN_Agent(DQN_Agent):
    def __init__(self, config):
        
        self.config = config
        self.config.state_size = 4
        self.config.n_act = 3
        self.config.mlp_config = [{'in': config.state_size, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.epsilon = 0.1
        self.config.gamma = 0.8
        self.config.memory_size = 100
        self.config.batch_size = 64
        self.config.warm_up_size = config.batch_size

        self.config.device = config.device
        self.model = MLP(self.config.mlp_config).to(self.config.device)
        # origin DEDDQN doesn't have clip 
        self.config.max_grad_norm = math.inf
        self.config.optimizer = 'AdamW'
        # origin code does not have lr_scheduler
        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'
        
        # self.__optimizer = torch.optim.AdamW(self.__dqn.parameters(), lr=config.lr)
        # self.__criterion = torch.nn.MSELoss()
        # self.__n_act = config.n_act
        # self.__epsilon = config.epsilon
        # self.__gamma = config.gamma
        # self.__replay_buffer = ReplayBuffer(config.memory_size)
        # self.__warm_up_size = config.warm_up_size
        # self.__batch_size = config.batch_size
        # self.__max_learning_step = config.max_learning_step
        # self.__global_ls = 0

        # self.__cur_checkpoint=0
        super().__init__(self.config)
        # save init agent
        # if self.__cur_checkpoint==0:
        #     save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
        #     self.__cur_checkpoint+=1

    # def __get_action(self, state, options=None):
    #     state = torch.Tensor(state).to(self.__device)
    #     action = None
    #     Q_list = self.__dqn(state)
    #     if options['epsilon_greedy'] and np.random.rand() < self.__epsilon:
    #         action = np.random.randint(low=0, high=self.__n_act)
    #     if action is None:
    #         action = int(torch.argmax(Q_list).detach().cpu().numpy())
    #     Q = Q_list[action].detach().cpu().numpy()
    #     return action, Q

    # def train_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     R = 0
    #     while not done:
    #         action, _ = self.__get_action(state, {'epsilon_greedy': True})
    #         next_state, reward, done = env.step(action)
    #         R += reward
    #         self.__replay_buffer.append((state, action, reward, next_state, done))
    #         # backward propagation
    #         if len(self.__replay_buffer) >= self.__warm_up_size:
    #             batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.__replay_buffer.sample(self.__batch_size)
    #             pred_Vs = self.__dqn(batch_obs.to(self.__device))  # [batch_size, n_act]
    #             action_onehot = torch.nn.functional.one_hot(batch_action.to(self.__device), self.__n_act)  # [batch_size, n_act]
    #             predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
    #             target_Q = batch_reward.to(self.__device) + (1 - batch_done.to(self.__device)) * self.__gamma * self.__dqn(batch_next_obs.to(self.__device)).max(1)[0]
    #             self.__optimizer.zero_grad()
    #             loss = self.__criterion(predict_Q, target_Q)
    #             loss.backward()
    #             self.__optimizer.step()
    #             self.__global_ls += 1

    #             if self.__global_ls >= (self.__config.save_interval * self.__cur_checkpoint):
    #                 save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
    #                 self.__cur_checkpoint+=1

    #             if self.__global_ls >= self.__max_learning_step:
    #                 break
    #         state = next_state
    #     return self.__global_ls >= self.__max_learning_step, {'normalizer': env.optimizer.cost[0],
    #                                                           'gbest': env.optimizer.cost[-1],
    #                                                           'return': R,
    #                                                           'learn_steps': self.__global_ls}

    # def rollout_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     R=0
    #     while not done:
    #         action, Q = self.__get_action(state, {'epsilon_greedy': False})
    #         next_state, reward, done = env.step(action)
    #         R+=reward
    #         state = next_state
    #     return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
