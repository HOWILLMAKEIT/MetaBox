"""
This is the basic optimizer class. All traditional optimizers should inherit from this class.
Your own traditional should have the following functions:
    1. __init__(self, config) : to initialize the optimizer
    2. run_episode(self, problem) : to run the optimizer for an episode
"""
from ..problem.basic_problem import Basic_Problem
import numpy as np
import torch
import time
class Basic_Optimizer:
    """
    Basic_Optimizer serves as an abstract base class for optimization algorithms. It provides common functionality such as random seed management for reproducibility across NumPy and PyTorch (CPU and GPU). Subclasses should implement the `run_episode` method to define the specific optimization process for a given problem.
    - config (object): Configuration object containing optimizer settings, including the target device (e.g., 'cpu' or 'cuda').
    # Attributes:
    - rng_seed (int or None): The random seed used for reproducibility.
    - rng (np.random.RandomState): NumPy random number generator initialized with the seed.
    - rng_cpu (torch.Generator): PyTorch CPU random number generator initialized with the seed.
    - rng_gpu (torch.Generator or None): PyTorch GPU random number generator initialized with the seed if device is not 'cpu'.
    # Methods:
    - seed(seed=None): Sets the random seed for NumPy and PyTorch (CPU/GPU) generators.
    - run_episode(problem): Abstract method to execute a single optimization episode; must be implemented by subclasses.
    - NotImplementedError: If `run_episode` is called directly on the base class.
    """
    def __init__(self, config):
        self.__config = config
        self.rng_seed = None

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

    def run_episode(self, problem: Basic_Problem):
        """
        # Introduction
        Executes a single episode of the optimization process for the given problem instance.
        # Args:
        - problem (Basic_Problem): An instance of the optimization problem to be solved.
        # Returns:
        - None
        # Raises:
        - NotImplementedError: This method must be implemented by subclasses.
        """
        
        raise NotImplementedError
