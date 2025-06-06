import copy
import math
from typing import Optional, Union, Literal

import torch.nn.functional as F
from typing import Optional, Union, Literal, List
from ..environment.parallelenv.parallelenv import ParallelEnv
from .basic_agent import Basic_Agent
from .utils import *
import torch
import numpy as np

class DDQN_Agent(Basic_Agent):
    """
    # Introduction
    The `DDQN_Agent` class implements a Double Deep Q-Network (DDQN) agent for reinforcement learning. This agent leverages experience replay, target networks, and epsilon-greedy exploration to learn optimal policies in a given environment.
    # Original paper
    "[**Deep Reinforcement Learning with Double Q-learning**](https://arxiv.org/abs/1509.06461)."Proceedings of the AAAI Conference on Artificial Intelligence, 2016
    # Official Implementation
    None
    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    - `networks` (dict): A dictionary of neural networks used by the agent, with keys as network names (e.g., 'actor', 'critic') and values as the corresponding network instances.
    - `learning_rates` (float): Learning rate for the optimizer.
    # Attributes
    - `gamma` (float): Discount factor for future rewards.
    - `n_act` (int): Number of possible actions in the environment.
    - `epsilon` (float): Epsilon value for epsilon-greedy exploration.
    - `max_grad_norm` (float): Maximum gradient norm for gradient clipping.
    - `memory_size` (int): Size of the replay buffer.
    - `batch_size` (int): Batch size for training.
    - `warm_up_size` (int): Minimum number of experiences required in the replay buffer before training starts.
    - `target_update_interval` (int): Interval for updating the target network.
    - `device` (str): Device to run the computations on (e.g., 'cpu' or 'cuda').
    - `replay_buffer` (ReplayBuffer): Replay buffer for storing experiences.
    - `network` (list): List of network names used by the agent.
    - `optimizer` (torch.optim.Optimizer): Optimizer for training the networks.
    - `criterion` (torch.nn.Module): Loss function used for training.
    - `learning_time` (int): Counter for the number of training steps.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.
    # Methods
    - `__init__(config, networks, learning_rates)`: Initializes the DDQN agent with the given configuration, networks, and learning rates.
    - `set_network(networks, learning_rates)`: Sets up the networks, optimizer, and loss function for the agent.
    - `get_step()`: Returns the current training step.
    - `update_setting(config)`: Updates the agent's configuration settings.
    - `get_action(state, epsilon_greedy=False)`: Selects an action based on the current state and exploration strategy.
    - `train_episode(envs, seeds, para_mode, compute_resource, tb_logger, required_info)`: Trains the agent for one episode in a parallelized environment.
    - `rollout_episode(env, seed, required_info)`: Executes a single episode in the environment without training.
    - `rollout_batch_episode(envs, seeds, para_mode, compute_resource, required_info)`: Executes multiple episodes in parallel environments without training.
    - `log_to_tb_train(tb_logger, mini_step, grad_norms, loss, Return, Reward, predict_Q, target_Q, extra_info)`: Logs training metrics to TensorBoard.
    # Returns
    - `train_episode`: A tuple containing:
        - `is_train_ended` (bool): Whether the training has reached the maximum learning steps.
        - `return_info` (dict): Dictionary containing training metrics such as return, loss, and environment-specific information.
    - `rollout_episode`: A dictionary containing episode results such as return, cost, and metadata.
    - `rollout_batch_episode`: A dictionary containing batch episode results such as return, cost, and environment-specific information.
    # Raises
    - `AssertionError`: If required network attributes (e.g., `model`) are not set or if the optimizer/criterion specified in the configuration is invalid.
    - `ValueError`: If the length of the `learning_rates` list does not match the number of networks provided.
    """
    def __init__(self, config, networks: dict, learning_rates: float):
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

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.set_network(networks, learning_rates)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def set_network(self, networks: dict, learning_rates: float):
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)  # Assign each network in the dictionary to the class instance
        self.network = Network_name

        assert hasattr(self, 'model')  # Ensure that 'model' is set as an attribute of the class
        self.target_model = copy.deepcopy(self.model)  # Create a deep copy of the model as the target model

        # If the learning rates are a single value, expand it to match the number of networks
        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates] * len(networks)  # Expand to match the number of networks
        elif len(learning_rates) != len(networks):
            raise ValueError("The length of the learning rates list must match the number of networks!")  # Check if the learning rates list is valid

        all_params = []  # List to store parameters for optimizer
        for id, network_name in enumerate(networks):
            network = getattr(self, network_name)  # Get the network from the class instance
            all_params.append({'params': network.parameters(), 'lr': learning_rates[id]})  # Append network parameters and their corresponding learning rates

        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(all_params)

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()

        for network_name in networks:
            getattr(self, network_name).to(self.device)

    def get_step(self):
        return self.learning_time

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def get_action(self, state, epsilon_greedy = False):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(low = 0, high = self.n_act, size = len(state))
        else:
            action = torch.argmax(Q_list, -1).detach().cpu().numpy()
        return action

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      # num_cpus: Optional[Union[int, None]] = 1,
                      # num_gpus: int = 0,
                      compute_resource = {},
                      tb_logger = None,
                      required_info = {}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus = num_cpus, num_gpus = num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.Tensor(state)
        except:
            pass

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state = state, epsilon_greedy = True)

            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            _reward.append(torch.Tensor(reward))
            # store info
            # convert next_state into tensor
            try:
                next_state = torch.Tensor(next_state).to(self.device)
            except:
                pass
            for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = next_state
            except:
                state = copy.deepcopy(next_state)

            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
                pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device), self.n_act)  # [batch_size, n_act]

                _avg_predict_Q = (pred_Vs * action_onehot).mean(0)  # [n_act]
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]

                target_output = self.target_model(batch_next_obs.to(self.device))
                _avg_target_Q = batch_reward.to(self.device)[:, None] + (1 - batch_done.to(self.device))[:, None] * gamma * target_output
                target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * target_output.max(1)[0].detach()
                _avg_target_Q = _avg_target_Q.mean(0)  # [n_act]

                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                self.optimizer.step()

                _loss.append(loss.item())
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time % self.target_update_interval == 0:
                    for target_parma, parma in zip(self.target_model.parameters(), self.model.parameters()):
                        target_parma.data.copy_(parma.data)

                if not self.config.no_tb:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         loss,
                                         _R, _reward,
                                         _avg_predict_Q, _avg_target_Q)

                if self.learning_time >= self.config.max_learning_step:
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.Tensor(state).unsqueeze(0).to(self.device)
                except:
                    state = [state]
                action = self.get_action(state)[0]
                state, reward, is_done, info = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}
            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                metadata = {'X': meta_X, 'Cost': meta_Cost}
                results['metadata'] = metadata
            for key in required_info.keys():
                results[key] = env.get_env_attr(required_info[key])
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds = None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              # num_cpus: Optional[Union[int, None]] = 1,
                              # num_gpus: int = 0,
                              compute_resource = {},
                              required_info = {}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)

        env.seed(seeds)
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
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
            R += torch.Tensor(rewards).squeeze()
            # store info
            try:
                state = torch.Tensor(state).to(self.device)
            except:
                pass
        _Rs = R.detach().numpy().tolist()
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
        for key in required_info:
            results[key] = getattr(env, key)
        return results

    # todo add metric
    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return, Reward,
                        predict_Q, target_Q,
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # learning rate
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)
        #
        # # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)

        # loss
        tb_logger.add_scalar('loss', loss.item(), mini_step)

        # Q
        for id, (p_q, t_q) in enumerate(zip(predict_Q, target_Q)):
            tb_logger.add_scalar(f"Predict_Q/action_{id}", p_q.item(), mini_step)
            tb_logger.add_scalar(f"Target_Q/action_{id}", t_q.item(), mini_step)

        # train metric
        avg_reward = torch.stack(Reward).mean().item()
        max_reward = torch.stack(Reward).max().item()
        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
        tb_logger.add_scalar('train/max_reward', max_reward, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)