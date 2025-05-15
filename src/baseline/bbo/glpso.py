import numpy as np
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class GLPSO(Basic_Optimizer):
    """
    # Introduction
    The PSO algorithm is hybridized with genetic evolution mechanisms. In this approach, genetic operators—specifically crossover, mutation, and selection—are incorporated into the PSO framework to construct promising exemplars and enhance the search performance.
    # Original paper
    "[**Genetic learning particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/7271066/)." IEEE Transactions on Cybernetics 46.10 (2015): 2277-2290.
    # Official Implementation
    [GLPSO](http://www.ai.sysu.edu.cn/GYJ/glpso/c_co)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the GLPSO (Global Learning Particle Swarm Optimization) algorithm with the specified configuration parameters.
        # Args:
        - config (object): 
            - The Attributes needed for the MOEAD optimizer in config are the following:
            - maxFEs (int): Maximum number of function evaluations allowed. Default value depends on the type of the problem.
            - n_logpoint (int): Number of log points for tracking progress. Default is 50.
            - log_interval (int): Interval at which logs are recorded. Default is maxFEs // n_logpoint.
            - full_meta_data (bool): Flag indicating whether to store complete solution history. Default is False.
        # Attributes:
        - __pm (float): Mutation probability, default is 0.01.
        - __NP (int): Population size, default is 100.
        - __nsel (int): Number of selected individuals, default is 10.
        - __w (float): Inertia weight, default is 0.7298.
        - __c1 (float): Cognitive coefficient, default is 1.49618.
        - __sg (int): Number of subgroups, default is 7.
        - __rho (float): Learning rate, default is 0.2.
        - config (object): Stores the configuration object.
        - __fes (int): Function evaluation counter, initialized to 0.
        - __exemplar_stag (np.ndarray): Array to track stagnation of exemplars, initialized to zeros of length `__NP`.
        """
        
        super().__init__(config)
        self.__pm=0.01
        self.__NP=100
        self.__nsel=10
        self.__w=0.7298
        self.__c1=1.49618
        self.__sg=7
        self.__rho=0.2
        self.config=config
        
        self.__fes=0
        self.__exemplar_stag=np.zeros(self.__NP)
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        """
        Returns the string representation of the GLPSO class.
        # Returns:
        - str: The string "GLPSO", representing the class name.
        """
        
        return "GLPSO"
    
    def __exemplar_crossover(self, problem):
        """
        # Introduction
        Performs the exemplar crossover operation for the GLPSO (Global Learning Particle Swarm Optimization) algorithm, generating new exemplars for each particle based on their personal bests, global best, and randomly selected peers.
        # Args:
        - problem (object): The problem object representing the optimization problem.
        # Modifies:
        - self.__new_exemplar (np.ndarray): Updates the array with new exemplars for each particle, determined by either a random peer's personal best or a uniform crossover between the particle's personal best and the global best.
        # Details:
        - For each particle and each dimension, selects a random peer and compares their personal best value.
        - If the random peer's personal best is better, adopts their position as the exemplar.
        - Otherwise, performs a uniform crossover between the particle's personal best and the global best position.
        """
        
        rand_index=self.rng.randint(low=0,high=self.__NP,size=(self.__NP,problem.dim))
        xs=self.__particles['pbest_position']
        rand_par=xs[rand_index,np.arange(problem.dim)[None,:]]
        rand_pbest_val=self.__particles['pbest'][rand_index]
        filter=rand_pbest_val<self.__particles['pbest'][:,None]
        r=self.rng.rand(self.__NP,problem.dim)
        uniform_crossover=r*self.__particles['pbest_position']+(1-r)*self.__particles['gbest_position'][None,:]
        self.__new_exemplar=np.where(filter,rand_par,uniform_crossover)

    def __exemplar_mutation(self, problem):
        """
        # Introduction
        Performs exemplar mutation on the population by probabilistically replacing elements with random values within the search bounds.
        # Args:
        - problem: The problem object representing the optimization problem.
        # Modifies:
        - self.__new_exemplar (np.ndarray): Updates the exemplar population by mutating elements with probability `self.__pm` using random values within `[self.__lb, self.__ub]`.
        # Notes:
        - Uses the instance's random number generator (`self.rng`) for reproducibility.
        - Mutation is applied independently to each element of the population matrix.
        """
        
        rand_pos=self.rng.uniform(low=self.__lb,high=self.__ub,size=(self.__NP,problem.dim))
        self.__new_exemplar=np.where(self.rng.rand(self.__NP,problem.dim)<self.__pm,rand_pos,self.__new_exemplar)
    
    def __exemplar_selection(self,problem,init=False):
        """
        # Introduction
        Selects and updates exemplars based on their costs for a given optimization problem, supporting both initialization and iterative improvement phases.
        # Args:
        - problem: The problem object representing the optimization problem.
        - init (bool): If True, initializes the exemplars with new candidates; otherwise, performs selection and updates based on cost comparison.
        # Updates:
        - self.__exemplar: The current set of exemplars, updated if new candidates have lower cost.
        - self.__exemplar_cost: The costs associated with the current exemplars.
        - self.__exemplar_stag: Stagnation counters for each exemplar, reset if improved.
        - self.__found_best: Tracks the best cost found so far across all exemplars.
        # Notes:
        - Uses numpy operations for efficient batch updates.
        - Assumes that self.__get_costs, self.__new_exemplar, self.__exemplar, self.__exemplar_cost, self.__exemplar_stag, and self.__found_best are defined as class attributes.
        """
        
        new_exemplar_cost=self.__get_costs(problem,self.__new_exemplar)
        if init:
            self.__exemplar=self.__new_exemplar
            self.__exemplar_cost=new_exemplar_cost
        else:
            suv_filter=new_exemplar_cost<self.__exemplar_cost
            self.__exemplar=np.where(suv_filter[:,None],self.__new_exemplar,self.__exemplar)
            self.__exemplar_stag=np.where(suv_filter,np.zeros_like(self.__exemplar_stag),self.__exemplar_stag+1)
            self.__exemplar_cost=np.where(suv_filter,new_exemplar_cost,self.__exemplar_cost)
        
        min_exemplar_cost=np.min(self.__exemplar_cost)
        
        self.__found_best=np.where(min_exemplar_cost<self.__found_best,min_exemplar_cost,self.__found_best)

    def __exemplar_tour_selection(self):
        """
        # Introduction
        Selects exemplar tours for each particle by randomly sampling a subset of exemplars and choosing the one with the minimum cost.
        # Returns:
        - np.ndarray: An array of selected exemplar tours, one for each particle in the population.
        """
        
        rand_index=self.rng.randint(low=0,high=self.__NP,size=(self.__NP,self.__nsel))
        rand_exemplar=self.__exemplar[rand_index]
        rand_exemplar_cost=self.__exemplar_cost[rand_index]
        min_exemplar_index=np.argmin(rand_exemplar_cost,axis=-1)  # bs, ps
        selected_exemplar=rand_exemplar[range(self.__NP),min_exemplar_index]
        return selected_exemplar
    
    def __exemplar_update(self,problem,init):
        """
        # Introduction
        Updates the exemplar solutions in the population by performing crossover, mutation, and selection operations. Additionally, applies a tour selection mechanism to exemplars that have stagnated beyond a specified threshold.
        # Args:
        - problem: The optimization problem instance containing evaluation and constraint information.
        - init: Initialization data or state required for the selection process.
        # Returns:
        - None
        """
        
        self.__exemplar_crossover(problem)
        self.__exemplar_mutation(problem)
        self.__exemplar_selection(problem,init)
        
        filter=self.__exemplar_stag>self.__sg
        if np.any(filter):
            self.__exemplar=np.where(filter[:,None],self.__exemplar_tour_selection(),self.__exemplar)
    
    def run_episode(self,problem):
        """
        # Introduction
        Executes a single optimization episode for the given problem using the GLPSO algorithm. Initializes the population, iteratively updates particle positions, and collects metadata if enabled.
        # Args:
        - problem: The problem object representing the optimization problem. 
        # Returns:
        - dict: A dictionary containing the results of the optimization episode.Containing:
            - 'cost' (list): The best fitness value found at each logging interval.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `self.full_meta_data` is True, includes:
                - 'X' (list of np.ndarray): The population positions at each logging interval.
                - 'Cost' (list of float): The fitness values of the population at each logging interval.
        
        # Notes:
        - The method resets and initializes the population at the start of each episode.
        - Metadata collection is optional and controlled by the `self.full_meta_data` attribute.
        """
        
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        self.__init_population(problem)
        is_done=False
        while not is_done:
            is_done,info=self.__update(problem)
            # print('gbest:{}'.format(self.__particles['gbest_val']))
        results = info
        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
    

    def __init_population(self,problem):
        """
        # Introduction
        Initializes the particle population for the GLPSO (Global Learning Particle Swarm Optimization) algorithm, setting up positions, velocities, and tracking variables for optimization.
        # Args:
        - problem (object): The problem object representing the optimization problem. 
        # Side Effects:
        - Initializes and sets internal attributes such as upper/lower bounds, function evaluation counter, exemplar cost, particle positions, velocities, costs, and best solutions.
        - Updates the internal log and cost tracking variables.
        # Notes:
        - This method is intended for internal use and assumes that the random number generator (`self.rng`) and population size (`self.__NP`) are already defined.
        - Calls internal methods for cost evaluation and exemplar update.
        """
        
        
        self.__ub=problem.ub
        self.__lb=problem.lb
        self.__fes=0
        self.__exemplar_cost=1e+10
        
        rand_pos=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.__NP,problem.dim))
        self.__max_velocity=self.__rho*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity,high=self.__max_velocity,size=(self.__NP,problem.dim))
        c_cost = self.__get_costs(problem,rand_pos) # ps
        
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]
        self.__max_cost=np.min(c_cost)
        # print("rand_pos.shape:{}".format(rand_pos.shape))

        self.__particles={'current_position': rand_pos.copy(), #  ps, dim
                        'c_cost': c_cost.copy(), #  ps
                        'pbest_position': rand_pos.copy(), # ps, dim
                        'pbest': c_cost.copy(), #  ps
                        'gbest_position':gbest_position.copy(), # dim
                        'gbest_val':gbest_val,  # 1
                        'velocity': rand_vel.copy(), # ps,dim
                        'gbest_index':gbest_index # 1
                        }

        self.__found_best=self.__particles['gbest_val'].copy()
        self.__exemplar_update(problem,init=True)

        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]

    def __get_costs(self,problem,position):
        """
        # Introduction
        Computes the cost(s) of a given position or set of positions for a specified optimization problem, optionally adjusting by the known optimum. Also records meta-data if enabled.
        # Args:
        - problem: The problem object representing the optimization problem.
        - position (np.ndarray): The position(s) in the search space for which the cost is to be evaluated.
        # Returns:
        - cost (float or np.ndarray): The computed cost(s) for the given position(s), optionally shifted by the problem's optimum if available.
        # Side Effects:
        - Increments the function evaluation counter (`self.__fes`) by the number of positions evaluated.
        - If `self.full_meta_data` is True, appends the computed cost(s) and position(s) to `self.meta_Cost` and `self.meta_X`, respectively.
        """
        
        ps=position.shape[0]
        self.__fes+=ps
        if problem.optimum is None:
            cost=problem.eval(position)
        else:
            cost=problem.eval(position)-problem.optimum
        if self.full_meta_data:
            self.meta_Cost.append(cost.copy())
            self.meta_X.append(position.copy())
        return cost

    def __update(self,problem):
        """
        # Introduction
        Updates the state of the particle swarm in the GLPSO (Global Learning Particle Swarm Optimization) algorithm for a single iteration, including velocity and position updates, personal and global best tracking, and logging of optimization progress.
        # Args:
        - problem (object): The problem object representing the optimization problem.
        # Returns:
        - is_end (bool): Indicates whether the optimization process should terminate based on the number of function evaluations or other stopping criteria.
        - dict: A dictionary containing:
            - 'cost' (list): The history of global best costs at logged intervals.
            - 'fes' (int): The current number of function evaluations.
        # Notes:
        - This method updates particle positions and velocities according to PSO rules, applies boundary constraints, updates personal and global bests, and manages logging of optimization progress.
        - The method assumes that the class maintains internal state for particles, random number generator, logging, and function evaluation counters.
        """
        
        is_end=False
        
        rand=self.rng.rand(self.__NP,problem.dim)
        new_velocity=self.__w*self.__particles['velocity']+self.__c1*rand*(self.__exemplar-self.__particles['current_position'])
        new_velocity=np.clip(new_velocity,-self.__max_velocity,self.__max_velocity)

        new_position=self.__particles['current_position']+new_velocity

        new_velocity=np.where(new_position>problem.ub,new_velocity*-0.5,new_velocity)
        new_velocity=np.where(new_position<problem.lb,new_velocity*-0.5,new_velocity)
        new_position=np.clip(new_position,problem.lb,problem.ub)

        new_cost=self.__get_costs(problem,new_position)

        filters = new_cost < self.__particles['pbest']
        # new_cbest_val,new_cbest_index=torch.min(new_cost,dim=1)
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        filters_best_val=new_cbest_val<self.__particles['gbest_val']
        # update particles
        new_particles = {'current_position': new_position, # bs, ps, dim
                            'c_cost': new_cost, # bs, ps
                            'pbest_position': np.where(np.expand_dims(filters,axis=-1),
                                                        new_position,
                                                        self.__particles['pbest_position']),
                            'pbest': np.where(filters,
                                                new_cost,
                                                self.__particles['pbest']),
                            'velocity': new_velocity,
                            'gbest_val':np.where(filters_best_val,
                                                    new_cbest_val,
                                                    self.__particles['gbest_val']),
                            'gbest_position':np.where(np.expand_dims(filters_best_val,axis=-1),
                                                        new_position[new_cbest_index],
                                                        self.__particles['gbest_position']),
                            'gbest_index':np.where(filters_best_val,new_cbest_index,self.__particles['gbest_index'])
                            }
        
        self.__particles=new_particles

        if self.__fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        self.__exemplar_update(problem,init=False)

        self.__found_best=np.where(self.__particles['gbest_val']<self.__found_best,self.__particles['gbest_val'],self.__found_best)

        if problem.optimum is None:
            is_end = self.__fes >= self.config.maxFEs
        else:
            is_end = self.__fes >= self.config.maxFEs
        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.__particles['gbest_val']
            else:
                while len(self.cost) < self.config.n_logpoint + 1:
                    self.cost.append(self.__particles['gbest_val'])
        return is_end, {'cost': self.cost, 'fes': self.__fes}
