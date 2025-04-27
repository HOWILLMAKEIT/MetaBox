import numpy as np
import torch
from .learnable_optimizer import Learnable_Optimizer

def scale(x,lb,ub):
    x=torch.sigmoid(x)
    x=lb+(ub-lb)*x
    return x

def np_scale(x,lb,ub):
    x=1/(1 + np.exp(-x))
    x=lb+(ub-lb)*x
    return x

class OPRO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pop_size = 10
        self.fes = None
        self.best = None
        self.old_value_pairs = None
        self.cost = []
    def __str__(self):
        return "OPRO_Optimizer"

    def init_population(self, problem):
        problem.reset()
        self.old_value_pairs = []
        self.fes=0
        self.best=None
    #     init population here and return the best popsize pop when step
        self.population = self.rng.random.uniform(
            low=problem.lb, high=problem.ub, size=(self.pop_size, problem.dim)
        )
        y = problem.func(self.population)
        # find the best theta and y
        best_theta = self.population[np.argmin(y)]
        self.best = np.min(y)
        self.cost.append(self.best)
        for i in range(self.pop_size):
            self.old_value_pairs.append((self.population[i], y[i]))

        if self.config.full_meta_data:
            self.meta_X = [self.population.copy()]
            self.meta_Cost = [y.copy()]

        return self.old_value_pairs

    def get_old_value_pairs(self):
        return self.old_value_pairs
    def update(self,action,problem):
        new_thetas = action

        if len(new_thetas) > 0:
            new_thetas = np.stack(new_thetas)
            # evaluate the new theta
            new_y = problem.eval(new_thetas)
            # update the best theta and y
            cur_best_theta = new_thetas[np.argmin(new_y)]
            cur_best_y = np.min(new_y)
            if cur_best_y < self.best or self.best is None:
                best_theta = cur_best_theta
                self.best = cur_best_y
            # add to old_value_pairs_set
            for i in range(len(new_thetas)):
                self.old_value_pairs.append((new_thetas[i], new_y[i]))

            if self.config.full_meta_data:
                gen_meta_cost = []
                gen_meta_X = []
                for i in range(len(new_thetas)):
                    gen_meta_cost.append(new_y[i])
                    gen_meta_X.append(new_thetas[i])
                self.meta_Cost.append(np.array(gen_meta_cost).copy())
                self.meta_X.append(np.array(gen_meta_X).copy())

        self.cost.append(self.best)
        self.fes += len(new_thetas)

        info = {}

        return self.old_value_pairs, 0, False, info