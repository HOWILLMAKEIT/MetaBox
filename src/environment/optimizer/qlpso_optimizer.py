from .learnable_optimizer import Learnable_Optimizer
import numpy as np
from typing import Union, Iterable

# a function for optimizer to calculate reward
def cal_reward(f_old, f_new, d_old, d_new):
    if f_new < f_old and d_new > d_old:
        return 2
    if f_new < f_old and d_new <= d_old:
        return 1
    if f_new >= f_old and d_new > d_old:
        return 0
    if f_new >= f_old and d_new <= d_old:
        return -2


class QLPSO_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    QLPSO is a problem-free PSO which integrates a reinforcement learning method.
    # Original paper
    "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications (2020).
    # Official Implementation
    None
    # Args:
    - config (object): Configuration object containing hyperparameters such as population size (`NP`), acceleration coefficients (`C`, `W`), problem dimensionality (`dim`), maximum function evaluations (`maxFEs`), logging interval (`log_interval`), and meta-data options (`full_meta_data`, `n_logpoint`).
    # Methods:
    - __init__(self, config): Initializes the optimizer with the given configuration and sets up internal state.
    - __cal_diversity(self): Computes the diversity of the current population.
    - __cal_velocity(self, action): Calculates the velocity update for the current solution based on the selected action and neighborhood.
    - init_population(self, problem): Initializes the population within the problem bounds and evaluates initial costs.
    - update(self, action, problem): Updates the current solution using the specified action, applies boundary control, evaluates the new solution, updates personal/global bests, logs progress, and returns the next state, reward, done flag, and info dictionary.
    # Returns:
    - Various methods return updated states, rewards, done flags, and logging information as appropriate for reinforcement learning and optimization workflows.
    # Raises:
    - No explicit exceptions are raised, but underlying operations (e.g., array indexing, problem evaluation) may raise exceptions if misconfigured.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the QLPSO optimizer with the provided configuration, setting up hyperparameters and internal state variables required for optimization.
        # Args:
        - config (object): Configuration object containing optimizer parameters such as population size, acceleration coefficients, inertia weight, problem dimensionality, maximum function evaluations, and logging interval.
        # Attributes:
        - __C (float): Acceleration coefficient.
        - __W (float): Inertia weight.
        - __NP (int): Population size.
        - __dim (int): Dimensionality of the problem.
        - __maxFEs (int): Maximum number of function evaluations.
        - __solution_pointer (int): Index indicating which solution receives the action.
        - __population (np.ndarray or None): Population of candidate solutions.
        - __pbest (np.ndarray or None): Personal best solutions.
        - __velocity (np.ndarray or None): Velocities of particles.
        - __cost (np.ndarray or None): Costs of current solutions.
        - __gbest_cost (float or None): Global best cost found so far.
        - __diversity (float or None): Diversity measure of the population.
        - __state (object or None): Internal state for optimizer.
        - fes (int or None): Current number of function evaluations.
        - cost (list or None): List of costs maintained by the optimizer.
        - log_index (int or None): Index for logging.
        - log_interval (int): Interval for logging progress.
        """
        
        super().__init__(config)
        # define hyperparameters that backbone optimizer needs
        config.NP = 30
        config.C = 1.49618
        config.W = 0.729844
        self.__config = config

        self.__C = config.C
        self.__W = config.W
        self.__NP = config.NP
        self.__dim = config.dim
        self.__maxFEs = config.maxFEs
        self.__solution_pointer = 0  # indicate which solution receive the action
        self.__population = None
        self.__pbest = None
        self.__velocity = None
        self.__cost = None
        self.__gbest_cost = None
        self.__diversity = None
        self.__state = None
        self.fes = None
        self.cost = None  # a list of costs that need to be maintained by EVERY backbone optimizers
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_diversity(self):
        return np.mean(np.sqrt(np.sum(np.square(self.__population - np.mean(self.__population,0)),1)))

    def __cal_velocity(self, action):
        i = self.__solution_pointer
        x = self.__population[i]
        v = self.__velocity[i]
        k = 0
        # calculate neighbour indexes
        if action == 0:
            k=4
        if action == 1:
            k=8
        if action == 2:
            k=16
        if action == 3:
            k=30

        nbest = None
        nbest_cost = np.inf
        for j in range(-k//2,k//2+1):
            if self.__cost[(i+j) % self.__population.shape[0]] < nbest_cost:
                nbest_cost = self.__cost[(i+j) % self.__population.shape[0]]
                nbest = self.__population[(i+j) % self.__population.shape[0]]
        return self.__W * v \
               + self.__C * self.rng.rand() * (nbest - x) \
               + self.__C * self.rng.rand() * (self.__pbest[i] - x)

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the QLPSO optimizer based on the provided optimization problem.
        # Args:
        - problem: An object representing the optimization problem, which must have attributes `ub` (upper bounds), `lb` (lower bounds), `optimum` (optional known optimum), and an `eval` method for evaluating solutions.
        # Returns:
        - int: The state value of the solution pointer after initialization.
        # Side Effects:
        - Initializes or updates the following instance attributes:
            - `__population`: The randomly generated initial population within the problem bounds.
            - `__pbest`: The personal best positions, initialized to the population.
            - `__velocity`: The initial velocities, set to zero.
            - `__diversity`: The diversity of the population.
            - `__cost`: The evaluated cost of the population (optionally offset by the problem optimum).
            - `__gbest_cost`: The best cost found in the initial population.
            - `fes`: The function evaluation count.
            - `log_index`: The logging index.
            - `cost`: The history of global best costs.
            - `__state`: The state of each individual in the population.
            - `meta_X`, `meta_Cost`: (If `full_meta_data` is enabled) Lists storing the population and cost history.
        # Notes:
        - If the problem's optimum is provided, the cost is offset by this value.
        - If `full_meta_data` is enabled in the configuration, additional metadata is stored for analysis.
        """
        
        self.__population = self.rng.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__pbest = self.__population.copy()
        self.__velocity = np.zeros(shape=(self.__NP, self.__dim))
        self.__diversity = self.__cal_diversity()
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest_cost = self.__cost.min().copy()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = self.rng.randint(low=0, high=4, size=self.__NP)
        if self.__config.full_meta_data:
            self.meta_X = [self.__population.copy()]
            self.meta_Cost = [self.__cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        return self.__state[self.__solution_pointer]

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the optimizer by applying the given action to the current solution, evaluating the new solution, updating rewards, and managing logging and metadata.
        # Args:
        - action (Any): The action to be applied to the current solution, typically representing a velocity or direction in the search space.
        - problem (object): The optimization problem instance, which must provide `lb`, `ub`, `eval()`, and `optimum` attributes/methods.
        # Returns:
        - tuple: A tuple containing:
            - state (Any): The updated state after applying the action.
            - reward (float): The calculated reward based on the change in cost and diversity.
            - is_done (bool): Whether the optimization episode should be terminated.
            - info (dict): Additional information (currently empty).
        # Notes:
        - The method updates internal population, velocity, cost, diversity, and logging information.
        - Handles boundary control and personal/global best updates.
        - Supports optional metadata logging if configured.
        - Ensures the cost log is filled up to the required number of log points at the end of an episode.
        """
        
        self.__velocity[self.__solution_pointer] = self.__cal_velocity(action)
        self.__population[self.__solution_pointer] += self.__velocity[self.__solution_pointer]
        # Boundary control
        self.__population[self.__solution_pointer] = clipping(self.__population[self.__solution_pointer], problem.lb, problem.ub)
        # calculate reward's data
        f_old = self.__cost[self.__solution_pointer]
        if problem.optimum is None:
            f_new = problem.eval(self.__population[self.__solution_pointer])
        else:
            f_new = problem.eval(self.__population[self.__solution_pointer]) - problem.optimum
        self.fes += 1
        d_old = self.__diversity
        d_new = self.__cal_diversity()
        reward = cal_reward(f_old,f_new,d_old,d_new)
        # update population information
        self.__cost[self.__solution_pointer] = f_new
        self.__diversity = d_new
        self.__gbest_cost = min(self.__gbest_cost, self.__cost.min().copy())
        if f_new < f_old:
            self.__pbest[self.__solution_pointer] = self.__population[self.__solution_pointer] #record pbest position
        self.__state[self.__solution_pointer] = action

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__population[self.__solution_pointer].copy())
            self.meta_tmp_cost.append(self.__cost[self.__solution_pointer].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.__NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()

        self.__solution_pointer = (self.__solution_pointer + 1) % self.__NP

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)
        # if an episode should be terminated
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = self.fes >= self.__maxFEs

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__gbest_cost)
                
        info = {}
        return self.__state[self.__solution_pointer], reward, is_done , info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)
