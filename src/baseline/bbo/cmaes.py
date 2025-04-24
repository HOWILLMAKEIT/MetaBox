from environment.optimizer.basic_optimizer import Basic_Optimizer
import cma
import numpy as np
import time
import warnings
import math

class CMAES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50
        self.__config = config

        self.log_interval = config.log_interval
        self.n_logpoint = config.n_logpoint
        self.full_meta_data = config.full_meta_data

    def __str__(self):
        return "CMAES"

    def run_episode(self, problem):
        cost = []
        self.meta_X = []
        self.meta_Cost = []
        index = 1

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cma.evolution_strategy._CMASolutionDict = cma.evolution_strategy._CMASolutionDict_empty
            es = cma.CMAEvolutionStrategy(np.ones(problem.dim), (problem.ub - problem.lb) * 0.3,
                                          {'popsize':self.__config.NP,
                                           'bounds': [problem.lb, problem.ub],
                                           'maxfevals': self.__config.maxFEs, 'tolfun': 1e-20, 'tolfunhist': 0})

        while not es.stop():
            X_batch = es.ask()
            y = problem_eval(X_batch)
            if self.full_meta_data:
                self.meta_X.append(np.array(X_batch.copy()))
                self.meta_Cost.append(np.array(y.copy()))
            gbest = np.min(y)
            es.tell(X_batch, y)
            if es.result[3] >= index * self.log_interval:
                index += 1
                cost.append(gbest)

            if len(cost) >= self.n_logpoint + 1:
                cost[-1] = gbest
            else:
                while len(cost) < self.n_logpoint + 1:
                    cost.append(gbest)

        results = {'cost': cost, 'fes': es.result[3]}

        if self.full_meta_data:
            metadata = {'X': self.meta_X, 'Cost': self.meta_Cost}
            results['metadata'] = metadata
        return results

