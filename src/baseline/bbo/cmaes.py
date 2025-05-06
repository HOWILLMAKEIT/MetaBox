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
    # Official Implementation
    None
    # Args:
    - config (object): Configuration object containing optimizer parameters such as population size (`NP`), 
      logging interval (`log_interval`), number of log points (`n_logpoint`), maximum function evaluations (`maxFEs`), 
      and whether to collect full meta data (`full_meta_data`).
    # Methods:
    - __str__(): Returns the string representation of the optimizer.
    - run_episode(problem): Runs a single optimization episode on the given problem instance.
    # run_episode Args:
    - problem (object): An optimization problem instance with attributes `dim` (dimension), `ub` (upper bound), 
      `lb` (lower bound), `optimum` (optional, known optimum value), and an `eval(x)` method for fitness evaluation.
    # run_episode Returns:
    - dict: A dictionary containing:
        - 'cost' (list): The best cost found at each logging interval.
        - 'fes' (int): The total number of function evaluations performed.
        - 'metadata' (dict, optional): If `full_meta_data` is True, contains:
            - 'X' (list): List of candidate solutions at each logging interval.
            - 'Cost' (list): List of corresponding costs at each logging interval.
    # Raises:
    - Any exceptions raised by the underlying `cma` library or the problem's `eval` method.
    # Notes:
    - The optimizer automatically rescales candidate solutions to the problem's bounds.
    - Logging and metadata collection are controlled by the configuration object.
    """
    
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50
        self.__config = config

        self.log_interval = config.log_interval
        self.n_logpoint = config.n_logpoint
        self.full_meta_data = config.full_meta_data
        self.__FEs = 0

    def __str__(self):
        return "CMAES"

    def run_episode(self, problem):
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

