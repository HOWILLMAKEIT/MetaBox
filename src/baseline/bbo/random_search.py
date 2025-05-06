import numpy as np
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class Random_search(Basic_Optimizer):
    """
    # Introduction
    Random_search is an implementation of a basic random search optimization algorithm, inheriting from Basic_Optimizer. It generates random candidate solutions within the problem bounds and tracks the best solution found so far. The optimizer supports logging of progress and optional collection of full meta-data for analysis.
    
    # Orighinal paper:
     
    # Official Implementation:
    
    # Args:
    - config (object): Configuration object containing the following attributes:
        - maxFEs (int): Maximum number of function evaluations allowed.
        - n_logpoint (int): Number of log points for recording progress.
        - log_interval (int): Interval of function evaluations between logs.
        - full_meta_data (bool): Whether to collect and store full meta-data during optimization.
    # Methods:
    - __init__(self, config): Initializes the optimizer with the given configuration.
    - __str__(self): Returns the string representation of the optimizer.
    - __reset(self, problem): Resets the optimizer state for a new optimization run.
    - __random_population(self, problem, init): Generates a random population and evaluates their costs.
    - run_episode(self, problem): Runs a single optimization episode on the given problem.
    # Returns (from run_episode):
    - dict: A dictionary containing:
        - 'cost' (list): The best cost found at each log point.
        - 'fes' (int): The total number of function evaluations performed.
        - 'metadata' (dict, optional): If `full_meta_data` is True, includes:
            - 'X' (list): List of candidate solutions evaluated.
            - 'Cost' (list): List of corresponding costs for each candidate solution.
    # Raises:
    - None explicitly, but may propagate exceptions from problem evaluation or configuration errors.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.__fes=0
        self.log_index=None
        self.cost=None
        self.__max_fes=config.maxFEs
        self.__NP=100
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
    
    def __str__(self):
        return 'Random_search'
    
    def __reset(self,problem):
        self.__fes=0
        self.cost=[]
        self.__random_population(problem,init=True)
        self.cost.append(self.gbest)
        self.log_index=1
    
    def __random_population(self,problem,init):
        rand_pos=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.__NP, problem.dim))
        if problem.optimum is None:
            cost=problem.eval(rand_pos)
        else:
            cost=problem.eval(rand_pos)-problem.optimum
            
        if self.full_meta_data:
            self.meta_Cost.append(cost.copy())
            self.meta_X.append(rand_pos.copy())
        self.__fes+=self.__NP
        if init:
            self.gbest=np.min(cost)
        else:
            if self.gbest>np.min(cost):
                self.gbest=np.min(cost)

    def run_episode(self, problem):
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        problem.reset()
        self.__reset(problem)
        is_done = False
        while not is_done:
            self.__random_population(problem,init=False)
            while self.__fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest)

            if problem.optimum is None:
                is_done = self.__fes>=self.__max_fes
            else:
                is_done = self.__fes>=self.__max_fes

            if is_done:
                if len(self.cost) >= self.__n_logpoint + 1:
                    self.cost[-1] = self.gbest
                else:
                    while len(self.cost) < self.__n_logpoint + 1:
                        self.cost.append(self.gbest)
                break
                
        results = {'cost': self.cost, 'fes': self.__fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
