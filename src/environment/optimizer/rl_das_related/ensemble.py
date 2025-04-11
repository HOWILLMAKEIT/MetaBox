import time

import gym
from gym import spaces
import numpy as np
from .optimizer import *
from .utils import *
from .Population import *
from .cec_test_func import *
import warnings
import copy
import collections


class RL_DAS_optimizer(gym.Env):
    def __init__(self, config):
        self.MaxFEs = config.maxFEs
        self.period = 2500
        self.max_step = self.MaxFEs // self.period
        self.sample_times = 2
        self.n_dim_obs = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_dim_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.optimizers))
        self.final_obs = None
        self.terminal_error = 1e-8

    def init_population(self, problem):
        self.dim = problem.dim
        self.problem = problem

        optimizers = ['NL_SHADE_RSP', 'MadDE',  'JDE21']
        self.optimizers = []
        for optimizer in optimizers:
            self.optimizers.append(eval(optimizer)(self.dim))
        self.best_history = [[] for _ in range(len(optimizers))]
        self.worst_history = [[] for _ in range(len(optimizers))]

        self.population = Population(self.dim)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)

    def local_sample(self):
        samples = []
        costs = []
        min_len = 1e9
        sample_size = self.population.NP
        for i in range(self.sample_times):
            sample, _, _ = self.optimizers[np.random.randint(len(self.optimizers))].step(copy.deepcopy(self.population),
                                                                                         self.problem,
                                                                                         self.FEs,
                                                                                         self.FEs + sample_size,
                                                                                         self.MaxFEs)
            samples.append(sample)
            cost = sample.cost
            costs.append(cost)
            min_len = min(min_len, cost.shape[0])
        self.FEs += sample_size * self.sample_times
        if self.FEs >= self.MaxFEs:
            self.done = True
        for i in range(self.sample_times):
            costs[i] = costs[i][:min_len]
        return np.array(samples), np.array(costs)

    # observed env state
    def observe(self):
        # =======================================================================
        samples, sample_costs = self.local_sample()
        feature = self.population.get_feature(self.problem,
                                              sample_costs,
                                              self.cost_scale_factor,
                                              self.FEs / self.MaxFEs)
        # =======================================================================
        best_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        worst_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        move = np.zeros((len(self.optimizers) * 2, self.dim)).tolist()
        for i in range(len(self.optimizers)):
            if len(self.best_history[i]) > 0:
                move[i*2] = np.mean(self.best_history[i], 0).tolist()
                move[i * 2 + 1] = np.mean(self.worst_history[i], 0).tolist()
                best_move[i] = np.mean(self.best_history[i], 0).tolist()
                worst_move[i] = np.mean(self.worst_history[i], 0).tolist()
        move.insert(0, feature)
        return move

    def seed(self, seed=None):
        np.random.seed(seed)

    # initialize env
    def reset(self):
        self.problem.reset()
        self.population = Population(self.dim)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.Fevs = np.array([])
        self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
        return self.observe()

    def update(self, action, problem):
        warnings.filterwarnings("ignore")
        if not self.done:
            act = action

            last_cost = self.population.gbest
            pre_best = self.population.gbest_solution
            pre_worst = self.population.group[np.argmax(self.population.cost)]
            period = self.period
            start = time.time()
            end = self.FEs + self.period
            while self.FEs < end and self.FEs < self.MaxFEs and self.population.gbest > self.terminal_error:                    
                optimizer = self.optimizers[act]
                FEs_end = self.FEs + period

                self.population, self.FEs = optimizer.step(self.population,
                                                            problem,
                                                            self.FEs,
                                                            FEs_end,
                                                            self.MaxFEs,
                                                            )
            end = time.time()
            pos_best = self.population.gbest_solution
            pos_worst = self.population.group[np.argmax(self.population.cost)]
            self.best_history[act].append((pos_best - pre_best) / 200)
            self.worst_history[act].append((pos_worst - pre_worst) / 200)
            self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
            reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)

            observe = self.observe()
            self.final_obs = observe
            return observe, reward, self.done, {} # next state, reward, is done
        else:
            return self.final_obs, -1, self.done, {} # next state, reward, is done
