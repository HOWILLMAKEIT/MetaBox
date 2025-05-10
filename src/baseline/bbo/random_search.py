import numpy as np
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class Random_search(Basic_Optimizer):
    """
    # Introduction
    Random_search is an implementation of a basic random search optimization algorithm, inheriting from Basic_Optimizer. It generates random candidate solutions within the problem bounds and tracks the best solution found so far. The optimizer supports logging of progress and optional collection of full meta-data for analysis.
    """
    
    def __init__(self, config):
        """
        Initializes the random search optimizer with the config object constructed in src/config.py.
        # Args:
        - config (object): 
            - The Attributes needed for the Random_search in config are the following:
                - maxFEs (int): Maximum number of function evaluations allowed.Defaullt value depends on the type of the problem.
                - n_logpoint (int): Number of log points for tracking progress. Default is 50.
                - log_interval (int): Interval at which logs are recorded.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is None.
        # Attributes:
        - __fes (int): Counter for the number of function evaluations performed.Default is 0.
        - log_index (int or None): Index for logging, initialized as None.Default is None.
        - cost (any): Placeholder for cost value, initialized as None.Default is None.
        - __NP (int): Population size, set to 100.
        """
        
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
        """
        Returns a string representation of the Random Search optimizer.
        # Returns:
        - str: The name of the optimizer, 'Random_search'.
        """
        
        return 'Random_search'
    
    def __reset(self,problem):
        """
        # Introduction
        Resets the internal state of the random search optimizer for a new optimization run on the given problem.
        # Args:
        - problem: The optimization problem instance to initialize the population for.
        # Effects:
        - Resets the function evaluation counter.
        - Clears the cost history.
        - Initializes a new random population.
        - Appends the initial global best solution to the cost history.
        - Sets the log index to 1.
        """
        
        self.__fes=0
        self.cost=[]
        self.__random_population(problem,init=True)
        self.cost.append(self.gbest)
        self.log_index=1
    
    def __random_population(self,problem,init):
        """
        # Introduction
        Generates a random population of candidate solutions within the problem's bounds, evaluates their costs, and updates the global best solution.
        # Args:
        - problem (object): The optimization problem instance, expected to have attributes `lb` (lower bounds), `ub` (upper bounds), `dim` (dimension), `optimum` (optional optimum value), and an `eval` method for evaluating solutions.
        - init (bool): Indicates whether this is the initial population generation. If True, initializes the global best; otherwise, updates it if a better solution is found.
        # Side Effects:
        - Updates `self.meta_Cost` and `self.meta_X` if `self.full_meta_data` is True.
        - Increments `self.__fes` by the population size (`self.__NP`).
        - Updates `self.gbest` with the minimum cost found in the current population.
        """
        
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
        """
        # Introduction
        Executes a single optimization episode using random search on the provided problem instance. Tracks the best solution found, logs progress at specified intervals, and optionally collects metadata for analysis.
        # Args:
        - problem: An object representing the optimization problem. 
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list): The logged best costs at each interval.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `self.full_meta_data` is True, includes:
                - 'X' (list of np.ndarray): The population positions at each logging interval.
                - 'Cost' (list of float): The fitness values of the population at each logging interval.
        # Notes:
        - The function ensures that the cost log is filled up to the required number of log points.
        - If `full_meta_data` is True, additional metadata about the search process is included in the results.
        """
        
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
