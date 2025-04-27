import numpy as np
from collections import deque
from .learnable_optimizer import Learnable_Optimizer
from typing import Union, Iterable
import copy


class DEDDQN_Optimizer(Learnable_Optimizer):
    
    """
    # Introduction
    DE-DDQN is an adaptive operator selection method based on Double Deep Q-Learning (DDQN), a Deep Reinforcement Learning method, to control the mutation strategies of Differential Evolution (DE).
    # Original paper
    "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).
    # Official Implementation
    [DE-DDQN](https://github.com/mudita11/DE-DDQN)
    # Args:
    - config (object): A configuration object containing the following attributes:
        - F (float): Mutation factor for DE.
        - Cr (float): Crossover probability for DE.
        - NP (int): Population size.
        - gen_max (int): Maximum number of generations for DE.
        - W (int): Window size for storing recent operation metrics.
        - maxFEs (int): Maximum number of function evaluations.
        - dim (int): Dimensionality of the problem.
        - dim_max (int): Maximum dimensionality of the problem.
        - log_interval (int): Interval for logging optimization progress.
        - full_meta_data (bool): Whether to store full metadata during optimization.
        - n_logpoint (int): Number of log points for cost tracking.
    # Methods:
    - `__str__() -> str`: Returns the string representation of the optimizer.
    - `init_population(problem) -> np.ndarray`: Initializes the population and returns the initial state.
    - `update(action, problem) -> Tuple[np.ndarray, float, bool, dict]`: Updates the optimizer based on the selected action 
      and returns the next state, reward, termination status, and additional information.
    # Attributes:
    - `fes (int)`: Tracks the number of function evaluations performed.
    - `cost (list)`: Stores the best cost values at each logging interval.
    - `log_index (int)`: Tracks the current logging index.
    - `meta_X (list)`: Stores the population at each step (if `full_meta_data` is enabled).
    - `meta_Cost (list)`: Stores the cost values at each step (if `full_meta_data` is enabled).
    # Raises:
    - ValueError: If an invalid action is provided during the `update` method.
    # Notes:
    - The optimizer uses four mutation strategies: 'rand/1', 'rand/2', 'rand-to-best/2', and 'cur-to-rand/1'.
    - The optimizer maintains detailed metrics for each mutation strategy to adaptively select the most effective one.
    - The `__get_state` method extracts features from the current population and optimization progress to represent the state 
      for the reinforcement learning agent.
    """
    
    def __init__(self, config):
        super().__init__(config)
        config.F = 0.5
        config.Cr = 1.0
        config.NP = 100
        config.gen_max = 10
        config.W = 50
        self.__config = config

        self.__F = config.F
        self.__Cr = config.Cr
        self.__NP = config.NP
        self.__maxFEs = config.maxFEs
        self.__gen_max = config.gen_max
        self.__W = config.W
        self.__dim = config.dim
        self.__dim_max = config.dim
        # records
        self.__gen = None  # record current generation
        self.__pointer = None  # the index of the individual to be updated
        self.__stagcount = None  # stagnation counter
        self.__X = None  # population
        self.__cost = None
        self.__X_gbest = None
        self.__c_gbest = None
        self.__c_gworst = None
        self.__X_prebest = None
        self.__c_prebest = None
        self.__OM = None
        self.__N_succ = None
        self.__N_tot = None
        self.__OM_W = None
        self.__r = None  # random indexes used for generating states and mutation
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "DEDDQN_Optimizer"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for an optimization problem, setting up the initial solutions, their costs, and various tracking variables.
        # Args:
        - problem (object): An instance of the optimization problem, which must provide the following attributes and methods:
            - `ub` (array-like): Upper bounds of the problem's search space.
            - `lb` (array-like): Lower bounds of the problem's search space.
            - `optimum` (float or None): The known optimal value of the problem, if available.
            - `eval(X)` (callable): A method to evaluate the cost of a given population `X`.
        # Returns:
        - dict: The initial state of the optimizer, including population, costs, and other relevant metadata.
        # Attributes Initialized:
        - `self.__X` (ndarray): The initialized population within the search space bounds.
        - `self.__cost` (ndarray): The evaluated costs of the population.
        - `self.fes` (int): The number of function evaluations performed.
        - `self.__gen` (int): The current generation counter.
        - `self.__pointer` (int): A pointer for tracking operations.
        - `self.__stagcount` (int): A counter for stagnation detection.
        - `self.__X_gbest` (ndarray): The best solution found so far.
        - `self.__c_gbest` (float): The cost of the best solution found so far.
        - `self.__c_gworst` (float): The cost of the worst solution in the population.
        - `self.__X_prebest` (ndarray): The best solution from the previous generation.
        - `self.__c_prebest` (float): The cost of the best solution from the previous generation.
        - `self.__OM` (list): A list of deques for tracking operator metadata.
        - `self.__N_succ` (list): A list of deques for tracking successful operations.
        - `self.__N_tot` (list): A list of deques for tracking total operations.
        - `self.__OM_W` (list): A list for tracking operator weights.
        - `self.log_index` (int): An index for logging purposes.
        - `self.cost` (list): A list to store the best cost at each generation.
        - `self.meta_X` (list, optional): A list to store the population at each generation (if `full_meta_data` is enabled).
        - `self.meta_Cost` (list, optional): A list to store the costs at each generation (if `full_meta_data` is enabled).
        # Notes:
        - This method assumes that the `problem` object provides the necessary attributes and methods for population initialization and evaluation.
        - If `problem.optimum` is provided, the costs are adjusted relative to the optimum.
        """
        
        # population initialization
        self.__X = self.rng.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb
        if problem.optimum is None:
            self.__cost = problem.eval(self.__X)
        else:
            self.__cost = problem.eval(self.__X) - problem.optimum
        # reset records
        self.fes = self.__NP
        self.__gen = 0
        self.__pointer = 0
        self.__stagcount = 0
        self.__X_gbest = self.__X[np.argmin(self.__cost)]
        self.__c_gbest = np.min(self.__cost)
        self.__c_gworst = np.max(self.__cost)
        self.__X_prebest = self.__X[np.argmin(self.__cost)]
        self.__c_prebest = np.min(self.__cost)
        self.__OM = [[], [], [], []]
        self.__N_succ = [[], [], [], []]
        self.__N_tot = []
        self.__OM_W = []
        for op in range(4):
            self.__N_tot.append(deque(maxlen = self.__gen_max))
            for m in range(4):
                self.__OM[op].append(deque(maxlen = self.__gen_max))
                self.__N_succ[op].append(deque(maxlen = self.__gen_max))
        self.log_index = 1
        self.cost = [self.__c_gbest]

        if self.__config.full_meta_data:
            self.meta_X = [self.__X.copy()]
            self.meta_Cost = [self.__cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        return self.__get_state(problem)

    def __get_state(self, problem):
        """
        # Introduction
        Generates a feature vector representing the current state of the optimization problem.
        The features are derived from various properties of the optimization process, including
        population diversity, fitness values, and historical operator performance.
        # Args:
        - problem (object): The optimization problem instance containing bounds and other relevant data.
        # Returns:
        - numpy.ndarray: A 1D array of 99 features representing the current state of the optimization process.
        # Notes:
        - The feature vector includes normalized fitness values, distances between solutions, 
          operator success rates, and other statistical measures.
        - The method uses internal attributes such as population positions, fitness values, 
          and operator performance metrics to compute the features.
        """
        
        max_dist = np.linalg.norm(np.array([problem.ub - problem.lb]).repeat(self.__dim), 2)
        features = np.zeros(99)
        features[0] = (self.__cost[self.__pointer] - self.__c_gbest) / (self.__c_gworst - self.__c_gbest)
        features[1] = (np.mean(self.__cost) - self.__c_gbest) / (self.__c_gworst - self.__c_gbest)
        features[2] = np.std(self.__cost) / ((self.__c_gworst - self.__c_gbest) / 2)
        features[3] = (self.__maxFEs - self.fes) / self.__maxFEs
        features[4] = self.__dim / self.__dim_max
        features[5] = self.__stagcount / self.__maxFEs
        self.__r = self.rng.randint(0, self.__NP, 5)
        for j in range(0, 5):  # features[6] ~ features[10]
            features[j + 6] = np.linalg.norm(self.__X[self.__pointer] - self.__X[self.__r[j]], 2) / max_dist
        features[11] = np.linalg.norm(self.__X[self.__pointer] - self.__X_prebest, 2) / max_dist
        for j in range(0, 5):  # features[12] ~ features[16]
            features[j + 12] = (self.__cost[self.__pointer] - self.__cost[self.__r[j]]) / (self.__c_gworst - self.__c_gbest)
        features[17] = (self.__cost[self.__pointer] - self.__c_prebest) / (self.__c_gworst - self.__c_gbest)
        features[18] = np.linalg.norm(self.__X[self.__pointer] - self.__X_gbest, 2) / max_dist
        i = 19
        for op in range(4):
            for m in range(4):
                for g in range(min(self.__gen_max, self.__gen)):
                    if self.__N_tot[op][g] > 0:
                        features[i] += self.__N_succ[op][m][g] / self.__N_tot[op][g]  # features[19] ~ features[34]
                i = i + 1
        for op in range(4):
            sum_N_tot = 0
            for g in range(min(self.__gen_max, self.__gen)):
                sum_N_tot += self.__N_tot[op][g]
            for m in range(4):
                for g in range(min(self.__gen_max, self.__gen)):
                    for k in range(self.__N_succ[op][m][g]):
                        features[i] += self.__OM[op][m][g][k]
                if sum_N_tot > 0:
                    features[i] = features[i] / sum_N_tot  # features[35] ~ features[50]
                i = i + 1
        if self.__gen >= 2:
            for op in range(4):
                for m in range(4):
                    # features[51] ~ features[66]
                    if self.__N_tot[op][0] - self.__N_tot[op][1] != 0 and self.__N_succ[op][m][0] > 0 and self.__N_succ[op][m][1] > 0:
                        features[i] = (np.max(self.__OM[op][m][0]) - np.max(self.__OM[op][m][1])) / (np.max(self.__OM[op][m][1]) * np.abs(self.__N_tot[op][0] - self.__N_tot[op][1]))
                    i = i + 1
        else:
            i = 67
        for op in range(4):
            for m in range(4):
                for g in range(min(self.__gen_max, self.__gen)):
                    if self.__N_succ[op][m][g] > 0:
                        features[i] += np.max(self.__OM[op][m][g])  # features[67] ~ features[82]
                i = i + 1
        for w in range(min(self.__W, len(self.__OM_W))):
            for m in range(4):
                features[i + self.__OM_W[w][0] * 4 + m] += self.__OM_W[w][m + 1]  # features[83] ~ features[98]
        return features

    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's state based on the selected action and the problem instance.
        This function implements the core logic of the DE-based optimizer, including mutation, 
        crossover, selection, and reward computation.
        # Args:
        - action (int): The action index representing the mutation strategy to use. 
          Valid values are:
            - 0: 'rand/1'
            - 1: 'rand/2'
            - 2: 'rand-to-best/2'
            - 3: 'cur-to-rand/1'
        - problem (Problem): The optimization problem instance containing the objective 
          function, bounds, and other problem-specific details.
        # Returns:
        - next_state (np.ndarray): The next state of the optimizer after applying the action.
        - reward (float): The reward obtained from the action, calculated as the improvement 
          in cost.
        - is_done (bool): A flag indicating whether the optimization process has reached 
          its termination condition.
        - info (dict): Additional information about the current state of the optimizer.
        # Raises:
        - ValueError: If the provided `action` is not a valid mutation strategy index.
        """

        if self.__pointer == 0:
            # update prebest
            self.__X_prebest = self.__X_gbest
            self.__c_prebest = self.__c_prebest
            # update gen
            self.__gen = self.__gen + 1
            for op in range(4):
                self.__N_tot[op].appendleft(0)
                for m in range(4):
                    self.__OM[op][m].appendleft(list())
                    self.__N_succ[op][m].appendleft(0)
        # mutation  ['rand/1', 'rand/2', 'rand-to-best/2', 'cur-to-rand/1']
        if action == 0:
            donor = rand_1_single(self.__X, self.__F, self.__pointer, self.__r, rng = self.rng)
        elif action == 1:
            donor = rand_2_single(self.__X, self.__F, self.__pointer, self.__r, rng = self.rng)
        elif action == 2:
            donor = rand_to_best_2_single(self.__X, self.__X_gbest, self.__F, self.__pointer, self.__r, rng = self.rng)
        elif action == 3:
            donor = cur_to_rand_1_single(self.__X, self.__F, self.__pointer, self.__r, rng = self.rng)
        else:
            raise ValueError('Action error')
        # BC
        donor = clipping(donor, problem.lb, problem.ub)
        # crossover
        trial = binomial(self.__X[self.__pointer], donor, self.__Cr, self.rng)
        # get the cost of the trial vector
        if problem.optimum is None:
            trial_cost = problem.eval(trial)
        else:
            trial_cost = problem.eval(trial) - problem.optimum
        self.fes += 1
        # compute reward
        reward = max(self.__cost[self.__pointer] - trial_cost, 0)
        # update records OM, N_succ, N_tot, OM_W
        self.__N_tot[action][0] += 1
        om = np.zeros(4)
        om[0] = self.__cost[self.__pointer] - trial_cost
        om[1] = self.__c_prebest - trial_cost
        om[2] = self.__c_gbest - trial_cost
        om[3] = np.median(self.__cost) - trial_cost
        for m in range(4):
            if om[m] > 0:
                self.__N_succ[action][m][0] += 1
                self.__OM[action][m][0].append(om[m])
        # update OM_W
        if len(self.__OM_W) >= self.__W:
            found = False
            for i in range(len(self.__OM_W)):
                if self.__OM_W[i][0] == action:
                    found = True
                    del self.__OM_W[i]
                    break
            if not found:
                del self.__OM_W[np.argmax(np.array(self.__OM_W)[:, 5])]
        self.__OM_W.append([action, om[0], om[1], om[2], om[3], trial_cost])
        # update stagcount
        if trial_cost >= self.__c_gbest:
            self.__stagcount += 1
        # selection
        if trial_cost <= self.__cost[self.__pointer]:  # better than its parent
            self.__cost[self.__pointer] = trial_cost
            self.__X[self.__pointer] = trial
            # update gbest, cbest
            if trial_cost <= self.__c_gbest:  # better than the global best
                self.__c_gbest = trial_cost
                self.__X_gbest = trial
        # update gworst
        if trial_cost > self.__c_gworst:
            self.__c_gworst = trial_cost

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__X[self.__pointer].copy())
            self.meta_tmp_cost.append(self.__cost[self.__pointer][j].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.__NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()

        self.__pointer = (self.__pointer + 1) % self.__NP

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__c_gbest)

        if problem.optimum is None:
            is_done = (self.fes >= self.__maxFEs)
        else:
            is_done = self.fes >= self.__maxFEs
        # get next state
        next_state = self.__get_state(problem)

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__c_gbest
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__c_gbest)

        info = {}
        return next_state, reward, is_done, info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)

def binomial(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float], rng) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    jrand = rng.randint(dim, size=NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(rng.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u

def generate_random_int_single(NP: int, cols: int, pointer: int, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a random array of integers within a specified range, ensuring that a given pointer value is not included in the result.
    # Args:
    - NP (int): The upper bound (exclusive) for the random integers.
    - cols (int): The number of random integers to generate.
    - pointer (int): The integer value that must not appear in the generated array.
    - rng (np.random.RandomState, optional): A random number generator instance. Defaults to None.
    # Returns:
    - np.ndarray: An array of randomly generated integers of length `cols`, excluding the `pointer` value.
    """
    
    r = rng.randint(low=0, high=NP, size=cols)
    while pointer in r:
        r = rng.randint(low=0, high=NP, size=cols)
    return r

def rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Implements the "rand/1" mutation strategy commonly used in Differential Evolution (DE) optimization algorithms.
    This function generates a new candidate solution by combining three randomly selected vectors from the population.
    # Args:
    - x (np.ndarray): The population of candidate solutions, where each row represents an individual solution.
    - F (float): The scaling factor used to control the amplification of the differential variation.
    - pointer (int): The index of the current candidate solution in the population.
    - r (np.ndarray, optional): An array of three unique random indices used to select individuals from the population. 
      If `None`, the indices will be generated automatically.
    - rng (np.random.RandomState, optional): A random number generator for reproducibility. If `None`, the default RNG is used.
    # Returns:
    - np.ndarray: A new candidate solution generated by applying the "rand/1" mutation strategy.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer,rng=rng)
    return x[r[0]] + F * (x[r[1]] - x[r[2]])

def rand_2_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a new vector based on the DE/rand/2 mutation strategy used in Differential Evolution (DE) algorithms.
    This method combines elements from multiple vectors in the population to create a trial vector.
    # Args:
    - x (np.ndarray): The population array where each row represents an individual vector.
    - F (float): The scaling factor used to control the amplification of the differential variation.
    - pointer (int): The index of the current target vector in the population.
    - r (np.ndarray, optional): An array of indices used to select vectors from the population. If `None`, random indices are generated.
    - rng (np.random.RandomState, optional): A random number generator instance for reproducibility. If `None`, the default RNG is used.
    # Returns:
    - np.ndarray: A new vector generated by applying the DE/rand/2 mutation strategy.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer, rng=rng)
    return x[r[0]] + F * (x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])

def rand_to_best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a new candidate solution vector using the "rand-to-best/2" mutation strategy, commonly used in Differential Evolution algorithms.
    # Args:
    - x (np.ndarray): Population array of candidate solutions.
    - best (np.ndarray): The current best solution vector.
    - F (float): Differential weight, a scaling factor for the mutation.
    - pointer (int): Index of the target vector in the population.
    - r (np.ndarray, optional): Array of 5 unique random indices for mutation. If None, they are generated automatically.
    - rng (np.random.RandomState, optional): Random number generator for reproducibility.
    # Returns:
    - np.ndarray: The mutated candidate solution vector.
    # Raises:
    - ValueError: If the input arrays have incompatible shapes or if insufficient unique indices are available for mutation.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer, rng=rng)
    return x[r[0]] + F * (best - x[r[0]] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])

def cur_to_rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer, rng=rng)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])
