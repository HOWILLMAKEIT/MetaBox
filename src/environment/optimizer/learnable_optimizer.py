"""
This is a basic class for learnable backbone optimizer.
Your own backbone optimizer should inherit from this class and have the following methods:
    1. __init__(self, config) : to initialize the backbone optimizer.
    2. init_population(self, problem) : to initialize the population, calculate costs using problem.eval()
       and record some information such as pbest and gbest if needed. It's expected to return a state for
       agent to make decisions.
    3. update(self, action, problem) : to update the population or one individual in population as you wish
       using the action given by agent, calculate new costs using problem.eval() and update some records
       if needed. It's expected to return a tuple of [next_state, reward, is_done] for agent to learn.
"""
from typing import Any, Tuple
from ..problem.basic_problem import Basic_Problem
import numpy as np
import torch
import time

class Learnable_Optimizer:
    """
    # Introduction
    Abstract superclass for learnable backbone optimizers, providing a template for population initialization, update mechanisms, and random seed management for reproducible optimization experiments.
    # Args:
    - config (Any): Configuration object containing optimizer settings, including device information.
    # Attributes:
    - __config (Any): Stores the configuration object.
    - rng_seed (Optional[int]): The random seed used for reproducibility.
    - rng (np.random.RandomState): Numpy random number generator initialized with the seed.
    - rng_cpu (torch.Generator): PyTorch CPU random number generator initialized with the seed.
    - rng_gpu (Optional[torch.Generator]): PyTorch GPU random number generator, if applicable.
    # Methods:
    - init_population(problem: Basic_Problem) -> Any: Abstract method to initialize the population for the optimization problem.
    - update(action: Any, problem: Basic_Problem) -> Tuple[Any]: Abstract method to update the optimizer state based on the given action and problem.
    - seed(seed: Optional[int] = None): Sets the random seed for numpy and PyTorch generators for reproducibility.
    # Raises:
    - NotImplementedError: If `init_population` or `update` methods are not implemented in a subclass.
    """
    
    """
    Abstract super class for learnable backbone optimizers.
    """
    
    def __init__(self, config):
        self.__config = config
        self.rng_seed = None

    def init_population(self,
                        problem: Basic_Problem) -> Any:
        """
        # Introduction
        Initializes the population for the optimization process based on the provided problem definition.
        # Args:
        - problem (Basic_Problem): An instance representing the optimization problem, containing information such as the search space, constraints, and objective function.
        # Returns:
        - Any: The initialized population, with the specific type and structure depending on the implementation and problem requirements.
        # Raises:
        - NotImplementedError: This method must be implemented by subclasses; calling this base method will always raise this exception.
        """
        raise NotImplementedError

    def update(self,
               action: Any,
               problem: Basic_Problem) -> Tuple[Any]:
        """
            # Introduction
            Updates the optimizer's internal state based on the provided action and problem instance.
            # Args:
            - action (Any): The action to be applied for updating the optimizer.
            - problem (Basic_Problem): The problem instance containing relevant information for the update.
            # Returns:
            - Tuple[Any]: A tuple containing the results of the update operation.
            # Raises:
            - NotImplementedError: This method must be implemented by subclasses.
            
            """
        raise NotImplementedError

    def seed(self, seed = None):
        rng_seed = int(time.time()) if seed is None else seed

        self.rng_seed = rng_seed

        self.rng = np.random.RandomState(rng_seed)

        self.rng_cpu = torch.Generator().manual_seed(rng_seed)

        self.rng_gpu = None
        if self.__config.device != 'cpu':
            self.rng_gpu = torch.Generator(device = self.__config.device).manual_seed(rng_seed)
        # GPU: torch.rand(4, generator = rng_gpu, device = 'self.__config.device')
        # CPU: torch.rand(4, generator = rng_cpu)
