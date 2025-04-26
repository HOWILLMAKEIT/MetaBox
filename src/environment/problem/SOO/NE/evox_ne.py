import sys
import subprocess
import numpy as np
import time
import torch
import torch.nn as nn
from evox.problems.neuroevolution.brax import BraxProblem
from evox.utils import ParamsAndVector

from ....problem.basic_problem import Basic_Problem


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_num):
        super(MLP,self).__init__()
        self.networks = nn.ModuleList()
        # self.in_layer = nn.Sequential(nn.Linear(state_dim,32),nn.Tanh())
        self.networks.append(nn.Sequential(nn.Linear(state_dim,32),nn.Tanh()))
        # self.hidden_layers = []
        for _ in range(hidden_layer_num):
            self.networks.append(nn.Sequential(nn.Linear(32,32),nn.Tanh()))
        # self.out_layer = nn.Linear(32,action_dim)
        self.networks.append(nn.Linear(32,action_dim))
    def forward(self, state):
        # h = self.in_layer(state)
        for layer in self.networks:
            state = layer(state)
        return torch.tanh(state)


envs = {
    'ant': {'state_dim':27, 'action_dim':8,}, # https://github.com/google/brax/blob/main/brax/envs/ant.py
    'halfcheetah': {'state_dim':18, 'action_dim':6,}, # https://github.com/google/brax/blob/main/brax/envs/half_cheetah.py
    'hopper': {'state_dim':11, 'action_dim':3,}, # https://github.com/google/brax/blob/main/brax/envs/hopper.py
    'humanoid':{'state_dim':376, 'action_dim':17,}, # https://github.com/google/brax/blob/main/brax/envs/humanoid.py
    'humanoidstandup':{'state_dim':376, 'action_dim':17,}, # https://github.com/google/brax/blob/main/brax/envs/humanoidstandup.py
    'inverted_pendulum':{'state_dim':4, 'action_dim':1,}, # https://github.com/google/brax/blob/main/brax/envs/inverted_pendulum.py
    'inverted_double_pendulum':{'state_dim':8, 'action_dim':1,}, # https://github.com/google/brax/blob/main/brax/envs/inverted_double_pendulum.py
    'pusher':{'state_dim':23, 'action_dim':7,}, # https://github.com/google/brax/blob/main/brax/envs/pusher.py
    'reacher':{'state_dim':11, 'action_dim':2,}, # https://github.com/google/brax/blob/main/brax/envs/reacher.py
    'swimmer':{'state_dim':8, 'action_dim':2,}, # https://github.com/google/brax/blob/main/brax/envs/swimmer.py
    'walker2d':{'state_dim':17, 'action_dim':6,}, # https://github.com/google/brax/blob/main/brax/envs/ant.py
}

model_depth = [
    0,
    1,
    2,
    3,
    4,
    5
]

class NE_Problem(Basic_Problem):
    """
    # Introduction
    `NE_Problem` sets up a neural network-based optimization problem for a given Brax environment. It initializes the environment, neural network model, and evaluation mechanism, and provides a function to evaluate batches of neural network parameters.
    # Args:
    - env_name (str): The name of the Brax environment to solve.
    - model_depth (int): The number of layers (depth) for the neural network policy.
    - seed (int): Random seed for reproducibility.
    # Attributes:
    - env_state_dim (int): Dimension of the environment's state space.
    - env_action_dim (int): Dimension of the environment's action space.
    - nn_model (MLP): The neural network policy model.
    - dim (int): Total number of parameters in the neural network.
    - ub (float): Upper bound for parameter values.
    - lb (float): Lower bound for parameter values.
    - pop_size (int): Population size for evolutionary algorithms.
    - adapter (ParamsAndVector): Adapter for converting between parameter vectors and model parameters.
    - evaluator (BraxProblem): Evaluator for running policy rollouts in the environment.
    # Methods:
    ## func(x)
    Evaluates a batch of neural network parameter vectors by running them in the environment and returning their rewards.
    ### Args:
    - x (np.ndarray): Batch of neural network parameters with shape (batch_size, num_params).
    ### Returns:
    - torch.Tensor: Rewards for each parameter vector in the batch.
    ### Raises:
    - AssertionError: If the input parameter dimension does not match the expected dimension.
    """
    
    def __init__(self,env_name,model_depth, seed):
        self.env_state_dim = envs[env_name]['state_dim']
        self.env_action_dim = envs[env_name]['action_dim']
        self.nn_model = MLP(self.env_state_dim, self.env_action_dim, model_depth)
        self.dim = sum(p.numel() for p in self.nn_model.parameters())
        self.ub = 5.
        self.lb = -5.
        self.pop_size = 500
        self.adapter = ParamsAndVector(dummy_model= self.nn_model) 
        self.evaluator = BraxProblem(
            policy=self.nn_model,
            env_name=env_name,
            max_episode_length=200, #todo: 10,3,5,1 should be indicated in config.py and loaded here
            num_episodes=10,
            pop_size=self.pop_size,
            seed=seed,
            reduce_fn=torch.mean,
        )

    def func(self,x): # x is a batch of neural network parameters: bs * num_params, type: numpy.array
        # x_cuda = torch.from_numpy(x).double().to(torch.get_default_device())
        # x_cuda = torch.from_numpy(x)
        # print(1)
        torch.set_default_device("cuda")
        assert x.shape[-1] == self.dim, "solution dimension not equal to problem dimension!"
        x = torch.tensor(x, device=torch.get_default_device()).float()
        pop_size = x.shape[0]
        if x.shape[0] < self.pop_size:
            x = torch.concat([x, torch.zeros(self.pop_size - pop_size, self.dim)], 0)
        nn_population = self.adapter.batched_to_params(x)
        # for key in nn_population.keys():
        #     print(nn_population[key].shape)
        rewards = self.evaluator.evaluate(nn_population)
        torch.set_default_device("cpu")
        return rewards[:pop_size]


