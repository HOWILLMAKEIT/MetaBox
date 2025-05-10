from ...environment.optimizer.basic_optimizer import Basic_Optimizer
import cma
import numpy as np
import time
import warnings
import math


# please refer:https://pypop.readthedocs.io/en/latest/applications.html
# this .py display pypop7-SHADE
class CMAES(Basic_Optimizer):
    """
    # Introduction
    A novel evolutionary optimization strategy based on the derandomized evolution strategy with covariance matrix adaptation. This is accomplished by efficientlyincorporating the available information from a large population, thus significantly re-ducing the number of generations needed to adapt the covariance matrix.
    # Original paper
    "[**Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)**](https://ieeexplore.ieee.org/abstract/document/6790790/)." Evolutionary Computation 11.1 (2003): 1-18.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the CMA-ES optimizer with the provided configuration settings.
        # Args:
        - config (object): config object from src/config.py.
            - The Attributes needed for the CMAES are the following:
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                - __FEs (int): Counter for the number of function evaluations.Default is 0.
                - __config (object): Stores the config object from src/config.py.
                - NP(int): Set the population size in config to 50.
        # Attributes:
        - __config (object): Configuration object containing algorithm parameters.     
        """
        
        super().__init__(config)
        config.NP = 50
        self.__config = config

        self.log_interval = config.log_interval
        self.n_logpoint = config.n_logpoint
        self.full_meta_data = config.full_meta_data
        self.__FEs = 0

    def __str__(self):
        """
        Returns a string representation of the CMAES class.
        # Returns:
        - str: The string "CMAES".
        """
        return "CMAES"

    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm on a given problem instance. Tracks the optimization process, logs costs at specified intervals, and optionally collects meta-data about the search trajectory.
        Use the cma package to implement the CMA-ES algorithm.
        # Args:
        - problem (object): An optimization problem instance that must provide the following attributes and methods:
            - `dim` (int): Dimensionality of the problem.
            - `lb` (np.ndarray): Lower bounds for the variables.
            - `ub` (np.ndarray): Upper bounds for the variables.
            - `eval(x)` (callable): Function to evaluate the objective at point `x`.
            - `optimum` (float or None): Known optimum value of the problem (if available).
        # Returns:
        - dict: A dictionary containing:
            - `'cost'` (list of float): The best cost found at each logging interval.
            - `'fes'` (int): The total number of function evaluations performed.
            - `'metadata'` (dict, optional): If `self.full_meta_data` is True, includes:
                - `'X'` (list of np.ndarray): The population positions at each iteration (in original problem scale).
                - `'Cost'` (list of np.ndarray): The corresponding costs for each population.
        """
        
        cost = []
        self.meta_X = []
        self.meta_Cost = []

        def problem_eval(x):

            x = np.clip(x, 0, 1)
            x = x * (problem.ub - problem.lb) + problem.lb

            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cma.evolution_strategy._CMASolutionDict = cma.evolution_strategy._CMASolutionDict_empty
            es = cma.CMAEvolutionStrategy(np.ones(problem.dim), 0.3,
                                          {'popsize': self.__config.NP,
                                           'bounds': [0, 1],
                                           'maxfevals': self.__config.maxFEs, 'tolfun': 1e-20, 'tolfunhist': 0})
        done = False
        X_batch = es.ask()  # initial population
        y = problem_eval(X_batch)
        self.__FEs += self.__config.NP
        if self.full_meta_data:
            self.meta_X.append(np.array(X_batch.copy()) * (problem.ub - problem.lb) + problem.lb)
            self.meta_Cost.append(np.array(y.copy()))
        index = 1
        cost.append(np.min(y).copy())

        while not done:
            es.tell(X_batch, y)
            X_batch = es.ask()
            y = problem_eval(X_batch)
            self.__FEs += self.__config.NP
            if self.full_meta_data:
                self.meta_X.append(np.array(X_batch.copy()) * (problem.ub - problem.lb) + problem.lb)
                self.meta_Cost.append(np.array(y.copy()))
            gbest = np.min(y)

            if self.__FEs >= index * self.log_interval:
                index += 1
                cost.append(gbest)

            if problem.optimum is None:
                done = self.__FEs >= self.__config.maxFEs
            else:
                done = self.__FEs >= self.__config.maxFEs

            if done:
                if len(cost) >= self.__config.n_logpoint + 1:
                    cost[-1] = gbest
                else:
                    while len(cost) < self.__config.n_logpoint + 1:
                        cost.append(gbest)
                break

        results = {'cost': cost, 'fes': es.result[3]}

        if self.full_meta_data:
            metadata = {'X': self.meta_X, 'Cost': self.meta_Cost}
            results['metadata'] = metadata
        return results

