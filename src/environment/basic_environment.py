from typing import Any
from problem.basic_problem import Basic_Problem
from optimizer.learnable_optimizer import Learnable_Optimizer
import gym

class PBO_Env(gym.Env):
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: Learnable_Optimizer,
                 ):
        super(PBO_Env, self).__init__()
        self.problem = problem
        self.optimizer = optimizer
        self.normalizer = None
        self.gbest = None

    def reset(self):
        self.problem.reset()
        reset_ = self.optimizer.init_population(self.problem)
        self.normalizer = self.optimizer.gbest_val
        self.gbest = self.optimizer.gbest_val
        return reset_

    def step(self, action: Any):
        update_ = self.optimizer.update(action, self.problem)
        self.gbest = self.optimizer.gbest_val
        return update_
