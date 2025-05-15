# from pypop7.optimizers.bo import *
# from pypop7.optimizers.cc import *
# from pypop7.optimizers.cem import *
# from pypop7.optimizers.cem import *
# from pypop7.optimizers.core import *
from pypop7.optimizers.de import SHADE as PYPOP7SHADE
# from pypop7.optimizers.ds import *
# from pypop7.optimizers.eda import *
# from pypop7.optimizers.es import *
# from pypop7.optimizers.ga import *
# from pypop7.optimizers.nes import *
# from pypop7.optimizers.pso import *
# from pypop7.optimizers.rs import *
# from pypop7.optimizers.sa import *
from ...environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
import time

# please refer:https://pypop.readthedocs.io/en/latest/applications.html
# this .py display pypop7-SHADE
class SHADE(Basic_Optimizer):
    """
    # Introduction
    A parameter adaptation technique for DE which uses a historical memory of successful control parameter settings to guide the selection of future control parameter values.
    # Original paper
    "[**Success-history based parameter adaptation for differential evolution**](https://ieeexplore.ieee.org/abstract/document/6557555/)." 2013 IEEE Congress on Evolutionary Computation. IEEE, 2013.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the class with the provided configuration, sets up population size, logging intervals, and metadata options.
        # Args:
        - config (object): 
            - The Attributes needed for the SHADE in config are the following:
                - log_interval (int): Interval at which logs are recorded. Default is maxFEs // n_logpoint.
                - n_logpoint (int): Number of log points for tracking progress. Default is 50.
                - full_meta_data (bool): Flag indicating whether to store complete solution history. Default is False.
                - maxFEs (int): Maximum number of function evaluations allowed. Default directly depends on the type of the problem.
        # Attributes:
        - NP (int):Set the population size in config object. Default is 50.
        - __config (object): Stores the configuration object internally.
        """
        
        super().__init__(config)
        config.NP = 50
        self.__config = config

        self.log_interval = config.log_interval
        self.n_logpoint = config.n_logpoint
        self.full_meta_data = config.full_meta_data

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the SHADE algorithm class.
        # Returns:
        - str: The string "SHADE".
        """
        
        return "SHADE"

    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode using the SHADE algorithm on the provided problem instance. Tracks the best solution found at specified logging intervals and optionally collects full meta-data for each iteration.
        # Args:
        - problem: An object representing the optimization problem. Must have the following attributes and methods:
            - `dim` (int): Dimensionality of the problem.
            - `lb` (array-like): Lower boundary of the search space.
            - `ub` (array-like): Upper boundary of the search space.
            - `optimum` (float or None): Known optimum value of the problem, or None if unknown.
            - `eval(x)` (callable): Function to evaluate the fitness of a solution `x`.
            - `__str__()` (callable): Returns a string representation of the problem.
        # Returns:
        - dict: A dictionary containing:
            - `'cost'` (list): Best fitness values found at each logging interval.
            - `'fes'` (int): Total number of function evaluations performed.
            - `'metadata'` (dict, optional): If `self.full_meta_data` is True, includes:
                - `'X'` (list): List of solution populations at each iteration.
                - `'Cost'` (list): List of fitness values for each population at each iteration.
        # Raises:
        - None directly, but may propagate exceptions from the problem's `eval` method or from the SHADE optimizer.
        """
        
        cost = []
        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness

        cost_fn = {'fitness_function': problem_eval,
                   'ndim_problem': problem.dim,
                   'lower_boundary': problem.lb,
                   'upper_boundary': problem.ub,
                   'problem_name': problem.__str__(),
                   }
        options = {'max_function_evaluations': self.__config.maxFEs,
                   'n_individuals': self.__config.NP,
                   'n_parents': self.__config.NP,
                   'seed_rng': self.rng_seed}

        opt = PYPOP7SHADE(cost_fn, options)

        opt.start_time = time.time()
        x, y, a = opt.initialize()

        gbest = np.min(y)
        cost.append(gbest)
        index = 1

        if self.full_meta_data:
            self.meta_X = [x.copy()]
            self.meta_Cost = [y.copy()]

        while not opt._check_terminations():
            x, y, a = opt.iterate(x, y, a)
            gbest = np.min(y)
            if opt.n_function_evaluations >= index * self.log_interval:
                index += 1
                cost.append(gbest)

            if self.full_meta_data:
                self.meta_X.append(x.copy())
                self.meta_Cost.append(y.copy())

        if len(cost) >= self.n_logpoint + 1:
            cost[-1] = gbest
        else:
            while len(cost) < self.n_logpoint + 1:
                cost.append(gbest)
        results = {'cost': cost, 'fes': opt.n_function_evaluations}

        if self.full_meta_data:
            metadata = {'X': self.meta_X, 'Cost': self.meta_Cost}
            results['metadata'] = metadata
        return results


if __name__ == '__main__':

    class config:
        def __init__(self):
            self.log_interval = 50
            self.n_logpoint = 20
            self.full_meta_data = True
            self.maxFEs = 2000
            self.NP = 50
            self.dim = 10
            self.device = "cpu"

    from environment.problem.SOO.COCO_BBOB.bbob_dataset import BBOB_Dataset
    problem_set = BBOB_Dataset.get_datasets('bbob', 10, 5)[0]
    problem = problem_set[5]

    problem.ub = 5 * np.ones(10)
    problem.lb = -5 * np.ones(10)

    configs = config()

    Pypop7_SHADE = PYPOP7(configs)
    Pypop7_SHADE.seed(7)
    problem.reset()
    Pypop7_SHADE.run_episode(problem)







