from typing import Any

from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

# torch
class B2OPT_Optimizer(Learnable_Optimizer):
    """
    # Introduction
      B2Opt: Learning to Optimize Black-box Optimization with Little Budget.
    # Original paper
    "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).
    # Official Implementation
    [B2Opt](https://github.com/ninja-wm/B2Opt)
    # Args:
    - config (object): Configuration object containing parameters such as `maxFEs`, `log_interval`, 
      `device`, and `full_meta_data`.
    # Attributes:
    - NP (int): Population size, default is 100.
    - MaxFEs (int): Maximum number of function evaluations allowed.
    - ems (int): Number of evaluations per step.
    - fes (int): Current number of function evaluations.
    - cost (list): List of best costs logged at intervals.
    - log_index (int): Index for logging progress.
    - log_interval (int): Interval for logging optimization progress.
    - ems_index (int): Index for tracking optimization steps.
    - population (torch.Tensor): Current population of solutions.
    - c_cost (torch.Tensor): Current costs of the population.
    - gbest_val (float): Global best value found so far.
    - init_gbest (torch.Tensor): Initial global best value.
    - meta_X (list): List of population states (if `full_meta_data` is enabled).
    - meta_Cost (list): List of cost states (if `full_meta_data` is enabled).
    # Methods:
    - `__str__() -> str`: Returns the string representation of the optimizer.
    - `get_costs(position, problem) -> torch.Tensor`: Evaluates the cost of a given position 
      in the problem space.
    - `init_population(problem) -> torch.Tensor`: Initializes the population and evaluates 
      their costs.
    - `get_state() -> torch.Tensor`: Returns the current state of the optimizer (costs).
    - `update(action, problem) -> Tuple[torch.Tensor, float, bool, dict]`: Updates the population 
      based on the provided action, evaluates new costs, and returns the next state, reward, 
      termination flag, and additional info.
    # Returns:
    - The optimizer tracks the progress of optimization and provides the current state, 
      reward, and termination status after each update.
    # Notes:
    - The optimizer assumes that the problem object has attributes `dim`, `ub`, `lb`, and 
      a method `eval(position)` for evaluating solutions.
    - The `action` parameter in the `update` method is expected to be a policy network 
      that generates new candidate solutions.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.NP = 100

        self.MaxFEs = config.maxFEs
        self.ems = (self.MaxFEs + self.NP - 1) // self.NP - 1

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        self.ems_index = 0

    def __str__(self):
        return "B2OPT_Optimizer"

    def get_costs(self, position, problem):
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        return cost

    def __sort(self):
        _, index = torch.sort(self.c_cost)
        self.population = self.population[index]
        self.c_cost = self.c_cost[index]

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for an optimization problem and computes initial costs.
        # Args:
        - problem (object): An object representing the optimization problem. It must have the following attributes:
            - `dim` (int): Dimensionality of the problem.
            - `lb` (torch.Tensor): Lower bounds of the search space.
            - `ub` (torch.Tensor): Upper bounds of the search space.
        # Returns:
        - dict: The initial state of the optimizer, including population, costs, and other metadata.
        # Attributes:
        - `self.rng_torch` (torch.Generator): Random number generator for the specified device.
        - `self.fes` (int): Function evaluation counter, initialized to the population size.
        - `self.population` (torch.Tensor): The initialized population within the problem's bounds.
        - `self.c_cost` (torch.Tensor): Costs of the initial population.
        - `self.gbest_val` (float): The best cost value in the initial population.
        - `self.init_gbest` (torch.Tensor): The best cost value as a tensor.
        - `self.cost` (list): A list to track the best cost values over iterations.
        - `self.log_index` (int): Index for logging purposes.
        - `self.meta_X` (list, optional): Metadata for population positions, if `full_meta_data` is enabled.
        - `self.meta_Cost` (list, optional): Metadata for population costs, if `full_meta_data` is enabled.
        # Notes:
        - The method uses PyTorch for tensor operations and supports GPU acceleration if configured.
        - The population is initialized uniformly within the bounds defined by `problem.lb` and `problem.ub`.
        - Metadata logging is optional and controlled by the `full_meta_data` configuration.
        """
        
        dim = problem.dim
        self.rng_torch = self.rng_cpu
        if self.config.device != "cpu":
            self.rng_torch = self.rng_gpu

        self.fes = 0
        self.population = (problem.ub - problem.lb) * torch.rand((self.NP, dim), generator = self.rng_torch, device = self.config.device, dtype = torch.float64) + problem.lb
        self.c_cost = self.get_costs(position = self.population, problem = problem)

        self.fes += self.NP

        self.ems_index = 0 # opt ob pointer

        self.gbest_val = torch.min(self.c_cost).detach().cpu().numpy()

        self.init_gbest = torch.min(self.c_cost).detach().cpu()

        self.cost = [self.gbest_val]
        self.log_index = 1

        self.__sort()

        if self.config.full_meta_data:
            self.meta_X = [self.population.detach().cpu().numpy()]
            self.meta_Cost = [self.c_cost.detach().cpu().numpy()]

        return self.get_state()
    def get_state(self):
        Y = self.c_cost
        return Y

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the optimizer based on the given action and problem, and calculates the reward, next state, and termination condition.
        # Args:
        - action (callable): A policy network function that takes the current population, costs, and EMS index as input and returns updated positions.
        - problem (object): The optimization problem instance containing problem-specific details.
        # Returns:
        - tuple: A tuple containing:
            - next_state (torch.Tensor): The updated state of the optimizer.
            - reward (float): The reward calculated based on the improvement in the global best value.
            - is_end (bool): A flag indicating whether the optimization process has reached its termination condition.
            - info (dict): Additional information (currently empty).
        # Notes:
        - The method updates the population and costs based on the optimization process.
        - It calculates the reward as the relative improvement in the global best value compared to the initial best value.
        - The termination condition is determined by the maximum number of function evaluations (`MaxFEs`).
        - If `full_meta_data` is enabled in the configuration, the population and costs are logged for each step.
        - The global best value (`gbest_val`) is updated and logged at specified intervals.
        """

        # 这里的action 是policy 网络
        pre_gbest = torch.min(self.c_cost.detach().cpu())


        v = action(self.population[None, :].clone().detach(), self.c_cost[None, :].clone().detach(), self.ems_index)[0] # off
        self.ems_index += 1

        new_cost = self.get_costs(position = v, problem = problem)
        self.fes += self.NP

        old_population = self.population.clone().detach()
        old_c_cost = self.c_cost.clone().detach()
        optim = new_cost.detach() < old_c_cost

        old_population[optim] = v[optim]
        old_c_cost[optim] = new_cost[optim]
        self.population = old_population
        self.c_cost = old_c_cost

        new_gbest_val = torch.min(self.c_cost).detach().cpu()

        reward = (pre_gbest - new_gbest_val) / self.init_gbest

        new_gbest_val = new_gbest_val.numpy()

        self.gbest_val = np.minimum(self.gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        self.__sort()

        if self.config.full_meta_data:
            self.meta_X.append(self.population.detach().cpu().numpy())
            self.meta_Cost.append(self.c_cost.detach().cpu().numpy())

        next_state = self.get_state()

        if self.fes >= self.log_interval * self.log_index:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info


