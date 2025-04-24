import numpy as np
from .learnable_optimizer import Learnable_Optimizer
from typing import Union, Iterable

class RLPSO_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    RLPSO develops a reinforcement learning strategy to enhance PSO in convergence by replacing the uniformly distributed random number in the updating function with a random number generated from a selected normal distribution.
    # Original paper
    "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization (2022).Intelligence. (2021).
    # Official Implementation
    None
    # Args:
    - config (object): Configuration object containing optimizer parameters such as dimension (`dim`), inertia weight decay (`w_decay`), acceleration coefficient (`c`), population size (`NP`), maximum function evaluations (`maxFEs`), logging interval (`log_interval`), and meta-data logging flag (`full_meta_data`).
    # Attributes:
    - __config (object): Stores the configuration object.
    - __dim (int): Dimensionality of the problem.
    - __w_decay (bool): Flag indicating whether to use inertia weight decay.
    - __w (float): Current inertia weight.
    - __c (float): Acceleration coefficient.
    - __NP (int): Number of particles in the swarm.
    - __max_fes (int): Maximum number of function evaluations.
    - fes (int or None): Current number of function evaluations.
    - cost (list or None): List of global best costs at each logging interval.
    - log_index (int or None): Current logging index.
    - log_interval (int): Interval for logging global best cost.
    - meta_X (list): List of particle positions for meta-data logging (if enabled).
    - meta_Cost (list): List of particle costs for meta-data logging (if enabled).
    # Methods:
    - __init__(self, config): Initializes the optimizer with the given configuration.
    - __str__(self): Returns the string representation of the optimizer.
    - init_population(self, problem): Initializes the particle swarm population and returns the initial state.
    - __get_state(self, index): Returns the concatenated state vector for the given particle index.
    - __get_costs(self, problem, position): Evaluates the cost(s) of the given position(s) using the problem's evaluation function.
    - update(self, action, problem): Updates the state of the optimizer using the provided action and returns the new state, reward, done flag, and additional info.
    # Usage:
    Instantiate with a configuration object and use `init_population` to initialize. Call `update` iteratively with actions to perform optimization steps.
    # Raises:
    - AttributeError: If required configuration attributes are missing.
    - ValueError: If input dimensions do not match the problem specification.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        config.w_decay = True
        config.c = 2.05
        config.NP = 100
        self.__config = config

        self.__dim = config.dim
        self.__w_decay = config.w_decay
        if self.__w_decay:
            self.__w = 0.9
        else:
            self.__w = 0.729
        self.__c = config.c
        
        self.__NP = config.NP

        self.__max_fes = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "RLPSO_Optimizer"

    # initialize PSO environment
    def init_population(self, problem):
        """
        # Introduction
        Initializes the particle population for the RLPSO (Reinforcement Learning Particle Swarm Optimization) algorithm, setting up positions, velocities, and tracking variables for optimization.
        # Args:
        - problem (object): An object representing the optimization problem, which must provide lower and upper bounds (`lb`, `ub`) for the search space.
        # Returns:
        - object: The initial state of the optimizer, as returned by `self.__get_state(self.__cur_index)`.
        # Side Effects:
        - Initializes and updates internal attributes such as particle positions, velocities, personal and global bests, and logging structures.
        - Resets function evaluation count and other tracking variables.
        - Optionally stores meta-data if configured.
        # Notes:
        - Assumes that `self.rng` is a random number generator and that `self.__get_costs` and `self.__get_state` are defined elsewhere in the class.
        - The method is intended to be called at the start of the optimization process.
        """
        
        rand_pos = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.__NP, self.__dim))
        self.fes = 0
        self.__max_velocity=0.1*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity, high=self.__max_velocity, size=(self.__NP, self.__dim))

        c_cost = self.__get_costs(problem,rand_pos) # ps

        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]
        self.__max_cost = np.max(c_cost)

        
        self.__particles={'current_position': rand_pos.copy(),  # ?ps, dim
                          'c_cost': c_cost.copy(),  # ?ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ?ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'gbest_index': gbest_index  # 1
                          }
        if self.__w_decay:
            self.__w = 0.9
            
        self.__cur_index = 0
        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]

        if self.__config.full_meta_data:
            self.meta_X = [rand_pos.copy()]
            self.meta_Cost = [c_cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        return self.__get_state(self.__cur_index)

    def __get_state(self, index):
        return np.concatenate((self.__particles['gbest_position'], self.__particles['current_position'][index]), axis=-1)

    # calculate costs of solutions
    def __get_costs(self, problem, position):
        if len(position.shape) == 2:
            self.fes += position.shape[0]
        else:
            self.fes += 1
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the RL-PSO (Reinforcement Learning Particle Swarm Optimization) optimizer for a single particle based on the provided action and problem definition. This includes updating velocity, position, personal best, and global best, as well as calculating rewards and logging progress.
        # Args:
        - action (np.ndarray or float): The action to be taken, typically representing a random factor for velocity update.
        - problem (object): The optimization problem instance, which must provide lower and upper bounds (`lb`, `ub`), and optionally an optimum value.
        # Returns:
        - state (np.ndarray): The updated state representation for the next step.
        - reward (float): The reward signal computed from the improvement in cost.
        - is_done (bool): Whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty, reserved for future use).
        # Notes:
        - The method linearly decreases the inertia coefficient if enabled.
        - Velocity and position are updated according to the PSO update rules, with velocity clipping and position boundary handling.
        - Updates personal and global bests if improvements are found.
        - Logs global best values at specified intervals.
        - Handles full meta-data logging if configured.
        - Ensures the cost log is filled up to the required number of log points upon completion.
        """
        
        is_done = False

        # record the gbest_val in the begining
        self.__pre_gbest = self.__particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.__w_decay:
            self.__w -= 0.5 / (self.__max_fes / self.__NP)

        # generate two set of random val for pso velocity update
        rand1 = self.rng.rand()
        rand2 = np.squeeze(action)
       
        j = self.__cur_index
        v = self.__particles['velocity'][j]
        x = self.__particles['current_position'][j]
        pbest_pos = self.__particles['pbest_position'][j]
        gbest_pos = self.__particles['gbest_position']
        pre_cost = self.__particles['c_cost'][j]

        # update velocity
        new_velocity = self.__w*v+self.__c*rand1*(pbest_pos-x)+self.__c*rand2*(gbest_pos-x)

        # clip the velocity if exceeding the boarder
        new_velocity = np.clip(new_velocity, -self.__max_velocity, self.__max_velocity)
        
        # update position
        new_x = x+new_velocity

        # print("velocity.shape = ",new_velocity.shape)
        new_x = clipping(new_x, problem.lb, problem.ub)

        # update population
        self.__particles['current_position'][j] = new_x
        self.__particles['velocity'][j] = new_velocity

        # calculate the new costs
        new_cost = self.__get_costs(problem, new_x)
        self.__particles['c_cost'][j] = new_cost
        
        # update pbest
        if new_cost < self.__particles['pbest'][j]:
            self.__particles['pbest'][j] = new_cost
            self.__particles['pbest_position'][j] = new_x
        # update gbest
        if new_cost < self.__particles['gbest_val']:
            self.__particles['gbest_val'] = new_cost
            self.__particles['gbest_position'] = new_x
            self.__particles['gbest_index'] = j

        # see if the end condition is satisfied
        if problem.optimum is None:
            is_done = self.fes >= self.__max_fes
        else:
            is_done = self.fes >= self.__max_fes

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        reward = (pre_cost-new_cost)/(self.__max_cost-self.__particles['gbest_val'])

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__particles['current_position'][j].copy())
            self.meta_tmp_cost.append(self.__particles['c_cost'][j].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.__NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()


        self.__cur_index = (self.__cur_index+1) % self.__NP

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__particles['gbest_val']
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__particles['gbest_val'])
                
        info = {}
        
        return self.__get_state(self.__cur_index), reward, is_done, info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)
