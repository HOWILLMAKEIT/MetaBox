from typing import Any

from environment.optimizer.learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

# torch
class GLHF_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.NP = 100

        self.MaxFEs = config.maxFEs

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "GLHF_Optimizer"

    def get_costs(self, position, problem):
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        return cost, torch.cat([cost.unsqueeze(1), position], dim = 1)

    def init_population(self, problem):
        dim = problem.dim
        self.rng_torch = self.rng_cpu
        if self.config.device != "cpu":
            self.rng_torch = self.rng_gpu

        self.fes = 0
        self.population = (problem.ub - problem.lb) * torch.rand((self.NP, dim), generator = self.rng_torch, device = self.config.device, dtype = torch.float64) + problem.lb
        self.c_cost, _ = self.get_costs(position = self.population, problem = problem)

        self.fes += self.NP

        self.gbest_val = torch.min(self.c_cost).detach().cpu().numpy()

        self.init_gbest = torch.min(self.c_cost).detach().cpu()

        self.cost = [self.gbest_val]
        self.log_index = 1

        if self.config.full_meta_data:
            self.meta_X = [self.population.detach().cpu().numpy()]
            self.meta_Cost = [self.c_cost.detach().cpu().numpy()]

        self.cost_fn = lambda position: self.get_costs(position, problem)

        return self.get_state()
    def get_state(self):
        X = self.population
        Y = self.c_cost.unsqueeze(1)
        return torch.cat([Y, X], dim = 1)

    def update(self, action, problem):
        # 这里的action 是policy 网络
        pre_gbest = self.c_cost.detach().cpu()
        batch_pop = self.get_state()[None, :]
        new_batch_pop, _, _ = action(batch_pop, self.cost_fn)

        new_cost = new_batch_pop[0, :, 0]
        new_population = new_batch_pop[0, :, 1:]

        self.fes += self.NP

        self.population = new_population
        self.c_cost = new_cost

        new_gbest_val = torch.min(new_cost).detach().cpu()

        reward = (pre_gbest - new_gbest_val) / self.init_gbest

        new_gbest_val = new_gbest_val.numpy()

        self.gbest_val = np.min(self.gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs or self.gbest_val <= 1e-8

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
                self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info


