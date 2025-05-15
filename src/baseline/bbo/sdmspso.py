import torch
import numpy as np
from scipy.optimize import minimize
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class SDMSPSO(Basic_Optimizer):
    """
    # Introduction
    The sDMS-PSO is a self-adaptive dynamic multi-swarm particle swarm optimizer that incorporates parameter adaptation, cooperative coevolution among multiple swarms, and a quasi-Newton local search to enhance convergence speed and optimization performance.
    # Original paper
    "[**A self-adaptive dynamic particle swarm optimizer**](https://ieeexplore.ieee.org/abstract/document/7257290/)." 2015 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2015.
    """
    def __init__(self,config):
        """
        # Introduction
        Initializes the SDMS-PSO (Sub-swarm Dynamic Multi-Swarm Particle Swarm Optimization) algorithm with the provided configuration, setting up algorithm parameters, population structure, and tracking variables.
        # Args:
        - config (object): 
            - The Attributes needed for the SDMSPSO optimizer in config are the following:
                - maxFEs (int): Maximum number of function evaluations allowed.Default directly depends on the type of the problem.
                - n_logpoint (int): Number of log points for tracking progress. Default is 50.
                - log_interval (int): Interval at which logs are recorded. Default is maxFEs // n_logpoint.
                - full_meta_data (bool): Flag indicating whether to store complete solution history. Default is False.
                - seed (int): Random seed for reproducibility. Used for initializing positions and velocities.
        # Attributes:
        - __w (float): Inertia weight parameter. Default is 0.729.
        - __c1 (float): Cognitive learning factor. Default is 1.49445.
        - __c2 (float): Social learning factor. Default is 1.49445.
        - __m (int): Sub-swarm size. Default is 3.
        - __R (int): Regrouping interval. Default is 10.
        - __LP (int): Learning period. Default is 10.
        - __LA (int): Length of archives. Default is 8.
        - __L (int): Local refining period. Default is 100.
        - __L_FEs (int): Max fitness evaluations using in the local search. Default is 200.
        - __NP (int): Total population size. Default is 99.
        - __n_swarm (int): Number of sub-swarms (NP/m). Default is 33.
        - __w_decay (bool): Whether to use weight decay strategy. Default is True.
    
        """
        
        super().__init__(config)
        self.__w,self.__c1,self.__c2=0.729,1.49445,1.49445
        self.__m,self.__R,self.__LP,self.__LA,self.__L,self.__L_FEs=3,10,10,8,100,200
        self.__NP=99
        self.__w_decay=True
        self.config = config

        self.__max_fes = config.maxFEs

        self.__n_swarm=self.__NP//self.__m
        self.__fes=0
        
        self.__group_index=np.zeros(self.__NP,dtype=np.int8)
        self.__per_no_improve=np.zeros(self.__NP)
        self.__lbest_no_improve=np.zeros(self.__n_swarm)
        
        assert self.__NP%self.__m==0, 'population cannot be update averagely'
        for sub_swarm in range(self.__n_swarm):
            if sub_swarm!=self.__n_swarm-1:
                self.__group_index[sub_swarm*self.__m:(sub_swarm+1)*self.__m]=sub_swarm
            else:
                self.__group_index[sub_swarm*self.__m:]=sub_swarm
        
        self.__parameter_set=[]
        self.__success_num=np.zeros((self.__n_swarm))
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        """
        # Introduction
        Returns the string representation of the SDMSPSO class.
        # Returns:
        - str: The string 'SDMSPSO', representing the class name.
        """
        
        return 'SDMSPSO'
    
    def __get_costs(self,problem,position):
        """
        # Introduction
        Computes the cost(s) for a given set of positions using the provided problem's evaluation function, optionally adjusting by the problem's optimum. Also stores metadata if enabled.
        # Args:
        - problem: The problem object.
        - position (np.ndarray): An array of candidate solutions (positions) to be evaluated.
        # Returns:
        - np.ndarray or float: The computed cost(s) for the provided positions.
        # Side Effects:
        - Increments the function evaluation counter (`self.__fes`) by the number of positions evaluated.
        - If `self.full_meta_data` is True, appends the computed costs and positions to `self.meta_Cost` and `self.meta_X`, respectively.
        """
        
        ps=position.shape[0]
        self.__fes+=ps
        if problem.optimum is None:
            cost=problem.eval(position)
        else:
            cost=problem.eval(position) - problem.optimum
        if self.full_meta_data:
            self.meta_Cost.append(cost.copy())
            self.meta_X.append(position.copy())
        
        return cost

    def __initilize(self,problem):
        """
        # Introduction
        Initializes the particle swarm for the optimization problem, setting up positions, velocities, and best values.
        # Args:
        - problem: The problem object.
        # Side Effects:
        - Initializes and sets the following instance attributes:
            - `__dim`: Dimension of the problem.
            - `__max_velocity`: Maximum velocity for particles.Decided by the problem's bounds.
            - `__particles`: Dictionary containing particle positions, velocities, and best values.
            - `__max_cost`: Minimum cost found during initialization.
        # Notes:
        - Uses a random number generator (`self.rng`) to initialize particle positions and velocities within the specified bounds.
        - Computes initial costs and identifies global and personal bests for the swarm.
        """
        
        self.__dim = problem.dim
        rand_pos=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.__NP,self.__dim))
        self.__max_velocity=0.1*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity,high=self.__max_velocity,size=(self.__NP,self.__dim))

        c_cost = self.__get_costs(problem,rand_pos) # ps

        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]
        self.__max_cost=np.min(c_cost)

        self.__particles={'current_position': rand_pos.copy(), #  ps, dim
                          'c_cost': c_cost.copy(), #  ps
                          'pbest_position': rand_pos.copy(), # ps, dim
                          'pbest': c_cost.copy(), #  ps
                          'gbest_position':gbest_position.copy(), # dim
                          'gbest_val':gbest_val,  # 1
                          'velocity': rand_vel.copy(), # ps,dim
                          }
        self.__particles['lbest_cost']=np.zeros(self.__n_swarm)
        self.__particles['lbest_position']=np.zeros((self.__n_swarm,self.__dim))
        
    def __reset(self,problem):
        """
        # Introduction
        Resets the internal state and parameters of the optimizer for a new optimization run on the given problem instance.
        # Args:
        - problem: The problem object to be optimized.
        # Attributes:
        - __dim (int): Problem dimension, obtained from problem.dim.
        - __w (float): Inertia weight parameter, reset to 0.9 if __w_decay is True, otherwise maintains default value of 0.729.
        - __gen (int): Generation counter, reset to 0.
        - __fes (int): Function evaluation counter, reset to 0.
        - __per_no_improve (numpy.ndarray): Counter for number of iterations without improvement for each particle, reset to 0.
        - __lbest_no_improve (numpy.ndarray): Counter for number of iterations without improvement for each subswarm's local best, reset to 0.
        - __learning_period (bool): Learning period flag, reset to True, indicating a new learning phase.
        - __parameter_set (list): Parameter set storage, reset to empty list.
        - __success_num (numpy.ndarray): Counter for successful updates for each subswarm, reset to zeros.
        - __cur_mode (str): Current operating mode, set to 'ls' (local search mode).
        - __fes_eval (numpy.ndarray): Copy of function evaluation counter, reset to 0, used for specific calculations.
        - log_index (int): Logging index, reset to 1, used to control logging frequency.
        - cost (list): Storage for best cost values during optimization, initialized with current global best value.
        # Returns:
        - None
        # Side Effects:
        - Resets or reinitializes various internal attributes such as generation count, function evaluation counters, learning period flags, parameter sets, and success counters.
        - Reinitializes the swarm, regrouping particles and updating local bests.
        - Resets logging and cost tracking variables.
        """
        
        self.__dim = problem.dim
        if self.__w_decay:
            self.__w=0.9
        self.__gen=0
        self.__fes-=self.__fes
        self.__per_no_improve-=self.__per_no_improve
        self.__lbest_no_improve-=self.__lbest_no_improve
        self.__learning_period=True
        self.__parameter_set=[]
        self.__success_num=np.zeros((self.__n_swarm))
        self.__cur_mode='ls'
        self.__initilize(problem)
        self.__random_regroup()
        self.__update_lbest(init=True)
        self.__fes_eval=np.zeros_like(self.__fes)
        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]
        return None

    def __random_regroup(self):
        """
        # Introduction
        Randomly shuffles the order of particles in the swarm and resets improvement counters.
        # Attributes:
        - Rearranges the following particle parameters according to the random permutation:
            - current_position: Current positions of all particles are rearranged
            - c_cost: Current cost values are reordered to match the new particle arrangement
            - pbest_position: Personal best positions are maintained but reassigned to different sub-swarms
            - pbest: Personal best costs are reordered accordingly
            - velocity: Particle velocities are preserved but reassigned
            - __per_no_improve: Personal improvement counters follow their respective particles
        """
        
        regroup_index=torch.randperm(n=self.__NP)
        self.__lbest_no_improve-=self.__lbest_no_improve
        self.__regroup_index=regroup_index
        self.__particles['current_position']=self.__particles['current_position'][regroup_index] # bs, ps, dim
        self.__particles['c_cost']= self.__particles['c_cost'][regroup_index] # bs, ps
        self.__particles['pbest_position']=self.__particles['pbest_position'][regroup_index] # bs, ps, dim
        self.__particles['pbest']= self.__particles['pbest'][regroup_index] # bs, ps
        self.__particles['velocity']=self.__particles['velocity'][regroup_index]
        self.__per_no_improve=self.__per_no_improve[regroup_index]
        
    def __update_lbest(self,init=False):
        """
        # Introduction
        Updates the local best (lbest) positions and costs for each swarm in a particle swarm optimization (PSO) algorithm. Handles both initialization and iterative update of lbest values, tracking improvements and updating relevant counters.
        # Args:
        - init (bool, optional): If True, performs initialization of lbest values using the current personal bests (pbest) of the particles. If False, updates lbest values based on new pbest values and tracks improvements. Default is False.
        # Updates:
        - self.__particles['lbest_cost']: The best cost found in each swarm.
        - self.__particles['lbest_position']: The position corresponding to the best cost in each swarm.
        - self.__lbest_index: The index of the lbest particle in each swarm.
        - self.__success_num: The number of times a new lbest is found for each swarm.
        - self.__lbest_no_improve: The number of iterations since the last improvement for each swarm.
        # Notes:
        - Assumes that self.__particles['pbest'] and self.__particles['pbest_position'] are properly shaped and populated.
        - Uses numpy operations for efficient batch updates.
        """
        
        if init:
            grouped_pbest=self.__particles['pbest'].reshape(self.__n_swarm,self.__m)
            grouped_pbest_pos=self.__particles['pbest_position'].reshape(self.__n_swarm,self.__m,self.__dim)

            self.__particles['lbest_cost']=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            self.__lbest_index=index+np.arange(self.__n_swarm)*self.__m   # n_swarm,
            self.__particles['lbest_position']=grouped_pbest_pos[range(self.__n_swarm),index]
            
        else:
            grouped_pbest=self.__particles['pbest'].reshape(self.__n_swarm,self.__m)
            grouped_pbest_pos=self.__particles['pbest_position'].reshape(self.__n_swarm,self.__m,self.__dim)
            lbest_cur=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            
            lbest_pos_cur=grouped_pbest_pos[range(self.__n_swarm),index]
            filter_lbest=lbest_cur<self.__particles['lbest_cost']
            self.__lbest_index=np.where(filter_lbest,index+np.arange(self.__n_swarm)*self.__m,self.__lbest_index)

            # update success_num
            
            success=np.sum(grouped_pbest<self.__particles['lbest_cost'][:,None],axis=-1)
            
            self.__success_num+=success
            
            self.__particles['lbest_cost']=np.where(filter_lbest,lbest_cur,self.__particles['lbest_cost'])
            self.__particles['lbest_position']=np.where(filter_lbest[:,None],lbest_pos_cur,self.__particles['lbest_position'])
            self.__lbest_no_improve=np.where(filter_lbest,np.zeros_like(self.__lbest_no_improve),self.__lbest_no_improve+1)

    def __get_iwt(self):
        """
        # Introduction
        Computes and updates the inertia weight (`__iwt`) for the swarm based on the current state of the parameter set and success count.
        # Description
        The method determines the inertia weight for each particle in the swarm. If the number of parameters in the set is less than a threshold (`__LA`) or the sum of successful updates is less than or equal to another threshold (`__LP`), the inertia weight is set to a random value in the range [0.4, 0.9) for each particle. Otherwise, it is sampled from a normal distribution centered at the median of the parameter set with a standard deviation of 0.1.
        # Args
        None
        # Returns
        None
        # Side Effects
        - Updates the `__iwt` attribute with a numpy array of inertia weights for the swarm.
        # Notes
        - Relies on the attributes: `__parameter_set`, `__LA`, `__success_num`, `__LP`, `rng`, and `__n_swarm`.
        """
        
        if len(self.__parameter_set)<self.__LA or np.sum(self.__success_num)<=self.__LP:
            self.__iwt=0.5*self.rng.rand(self.__n_swarm)+0.4

        else:
            self.__iwt=self.rng.normal(loc=np.median(self.__parameter_set),scale=0.1,size=(self.__n_swarm,))

    def __update(self,problem):
        """
        # Introduction
        Updates the state of the particle swarm in the PSO (Particle Swarm Optimization) algorithm for a given optimization problem. This includes updating particle velocities, positions, personal bests, local/global bests, and logging progress.
        # Args:
        - problem: An object representing the optimization problem, which must provide lower and upper bounds (`lb`, `ub`) and a cost evaluation method.
        # Updates:
        - Particle velocities and positions based on the current mode (`ls` for local best, `gs` for global best).
        - Personal best positions and costs for each particle.
        - Global best position and value across all particles.
        - Local best positions and costs if in local search mode.
        - Internal logging of global best cost at specified intervals.
        # Notes:
        - Uses random coefficients and inertia weights for velocity updates.
        - Applies boundary constraints to particle positions.
        - Assumes internal state variables such as `self.__particles`, `self.__c1`, `self.__c2`, `self.__w`, etc., are properly initialized.
        - Logging is performed at intervals defined by `self.log_interval`.
        """
        
        rand1=self.rng.rand(self.__NP,1)
        rand2=self.rng.rand(self.__NP,1)
        c1=self.__c1
        c2=self.__c2
        v_pbest=rand1*(self.__particles['pbest_position']-self.__particles['current_position'])
        if self.__cur_mode=='ls':
            v_lbest=rand2*(self.__particles['lbest_position'][self.__group_index]-self.__particles['current_position'])
            self.__get_iwt()
            new_velocity=self.__iwt[self.__group_index][:,None]*self.__particles['velocity']+c1*v_pbest+c2*v_lbest
        elif self.__cur_mode=='gs':
            v_gbest=rand2*(self.__particles['gbest_position'][None,:]-self.__particles['current_position'])
            new_velocity=self.__w*self.__particles['velocity']+c1*v_pbest+c2*v_gbest
        new_velocity=np.clip(new_velocity,-self.__max_velocity,self.__max_velocity)
        raw_position=self.__particles['current_position']+new_velocity
        new_position = np.clip(raw_position,problem.lb,problem.ub)
        new_cost=self.__get_costs(problem,new_position)
        filters = new_cost < self.__particles['pbest']
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
                            'lbest_position':self.__particles['lbest_position'],
                            'lbest_cost':self.__particles['lbest_cost']
                            }
        self.__particles=new_particles

        if self.__fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        if self.__cur_mode=='ls':
            self.__update_lbest()
        
    def __update_parameter_set(self):
        """
        # Introduction
        Updates the internal parameter set by selecting the parameter corresponding to the highest success count.
        Maintains the parameter set size within the limit specified by `self.__LA`.
        # Args:
        None
        # Modifies:
        - self.__parameter_set (list): Updates the list by appending the most successful parameter and removing the oldest if the size limit is reached.
        # Returns:
        None
        """
        
        max_success_index=np.argmax(self.__success_num)
        if len(self.__parameter_set)<self.__LA:
            self.__parameter_set.append(self.__iwt[max_success_index])
        else:
            del self.__parameter_set[0]
            self.__parameter_set.append(self.__iwt[max_success_index])

    def __quasi_Newton(self):
        """
        # Introduction
        Applies a quasi-Newton (BFGS) local refinement to the best-performing particles in the swarm to improve their positions and costs.
        # Args:
        None
        # Modifies:
        - Updates `self.__particles['lbest_position']` and `self.__particles['lbest_cost']` for selected particles if a better solution is found.
        - Updates `self.__particles['pbest_position']` and `self.__particles['pbest']` for corresponding personal bests.
        - Increments `self.__fes` by the number of function evaluations used in the local optimization.
        # Notes:
        - Selects the top quarter of particles with the lowest local best cost for refinement.
        - Uses the BFGS algorithm with a maximum of 9 iterations for local optimization.
        - Only updates positions and costs if the local optimization finds a better solution.
        """
        
        sorted_index=np.argsort(self.__particles['lbest_cost'])
        refine_index=sorted_index[:int(self.__n_swarm//4)]
        refine_pos=self.__particles['lbest_position'][refine_index]
        for i in range(refine_pos.shape[0]):
            res=minimize(self.__problem.eval,refine_pos[i],method='BFGS',options={'maxiter':9})
            self.__fes+=res.nfev
            if self.__particles['lbest_cost'][refine_index[i]]>res.fun:
                self.__particles['lbest_position'][refine_index[i]]=res.x
                self.__particles['lbest_cost'][refine_index[i]]=res.fun
                # uodate pbest
                self.__particles['pbest_position'][self.__lbest_index[refine_index[i]]]=res.x
                self.__particles['pbest'][self.__lbest_index[refine_index[i]]]=res.fun

    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode using a metaheuristic algorithm, managing the optimization process, updating parameters, and collecting results and optional metadata.
        # Args:
        - problem (object): The optimization problem instance, which should provide an evaluation interface and may have an `optimum` attribute.
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list): The cost values logged during the episode.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `self.full_meta_data` is True, includes:
                - 'X' (list): The meta-level solution data collected during the run.
                - 'Cost' (list): The meta-level cost data collected during the run.
        # Notes:
        - The method manages both local search and global search phases, parameter updates, and logging.
        - If the problem's optimum is not specified, the episode ends when the maximum number of function evaluations is reached.
        - Ensures the cost log is filled up to the configured number of log points.
        """
        
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        self.__reset(problem)
        while self.__fes<self.__max_fes:
            while self.__fes<0.95*self.__max_fes:
                self.__cur_mode='ls'
                self.__gen+=1
                if self.__w_decay:
                    self.__w-=0.5/(self.__max_fes/self.__NP)

                # learning period:
                self.__success_num-=self.__success_num
                for j in range(self.__LP):
                    self.__update(problem)
                self.__update_parameter_set()
                if self.__gen%self.__L==0:
                    self.__quasi_Newton()

                if self.__gen%self.__R==0:
                    self.__random_regroup()
                    self.__update_lbest(init=True)

            while self.__fes<self.__max_fes:
                self.__cur_mode='gs'
                self.__update(problem)

            if problem.optimum is None:
                done = self.__fes>=self.__max_fes
            else:
                done = self.__fes>=self.__max_fes
            if done:
                if len(self.cost) >= self.config.n_logpoint + 1:
                    self.cost[-1] = self.__particles['gbest_val']
                else:
                    while len(self.cost) < self.config.n_logpoint + 1:
                        self.cost.append(self.__particles['gbest_val'])
                break

        results = {'cost': self.cost, 'fes': self.__fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
    
