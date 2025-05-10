import numpy as np
from deap import base
from deap import creator
from deap import tools
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class PSO(Basic_Optimizer):
    """
    # Introduction
    Particle Swarm Optimization (PSO) optimizer implementation for black-box optimization problems.  
    This class inherits from `Basic_Optimizer` and utilizes the DEAP library to perform PSO, maintaining a population of particles that iteratively update their positions and velocities to search for the global optimum.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the PSO (Particle Swarm Optimization) optimizer with the config object constructed in config.py, setting default hyperparameters and preparing internal state.
        # Args:
        - config (object): Configuration object containing PSO parameters and metadata.
            - The Attributes needed for the PSO are the following:
                - log_interval (int): Interval at which logs are recorded.
                - n_logpoint (int): Number of log points to record. Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data. Default is False.
                - maxFEs (int): Maximum number of function evaluations allowed. Default value depends on the type of the problem.
                - phi1 (float): Cognitive coefficient for particle velocity update. Default is 2.0.
                - phi2 (float): Social coefficient for particle velocity update. Default is 2.0.
                - population_size (int): Size of the particle population. Default is 50.
        # Attributes:
        - __config (object): Stores the configuration object.
        - __toolbox (object or None): Placeholder for the optimization toolbox, initialized as None.
        - __creator (object or None): Placeholder for the creator utility, initialized as None.
        # Notes:
        - Sets default values for `phi1`, `phi2`, and `population_size` in the configuration.
        """
        
        super().__init__(config)
        config.phi1 = 2.
        config.phi2 = 2.
        config.population_size = 50

        self.__config = config
        self.__toolbox = None
        self.__creator = None
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        """
        Returns a string representation of the PSO (Particle Swarm Optimization) class.
        # Returns:
            str: The string "PSO", representing the class name.
        """
        
        return "PSO"
    def run_episode(self, problem):
        """
        # Introduction
        Executes a single episode of Particle Swarm Optimization (PSO) on the given optimization problem, tracking the best solution found and optionally collecting meta-data about the optimization process.
        # Args:
        - problem (object): An object representing the optimization problem to solve. Must have attributes `lb` (lower bounds), `ub` (upper bounds), `dim` (dimension), `eval` (evaluation function), and optionally `optimum` (known optimum value).
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list of float): The best fitness value found at each logging interval.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `self.full_meta_data` is True, includes:
                - 'X' (list of np.ndarray): The population positions at each logging interval.
                - 'Cost' (list of float): The fitness values of the population at each logging interval.
        # Raises:
        - AttributeError: If required attributes are missing from the `problem` object.
        - Exception: For errors during the optimization process, such as invalid configuration or evaluation failures.
        """
        
        self.rng_gpu = None
        self.rng_cpu = None
        self.rng = None
        np.random.seed(self.rng_seed)

        if self.__toolbox is None:
            self.__toolbox = base.Toolbox()
            self.__creator = creator

        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.__creator.create("Particle", np.ndarray, fitness=creator.Fitnessmin, speed=list, smin=None, smax=None, best=None)

        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        def generate(size, pmin, pmax, smin, smax):
            part = self.__creator.Particle(np.random.uniform(pmin, pmax, size))
            part.speed = np.random.uniform(smin, smax, size)
            part.smin = smin
            part.smax = smax
            return part

        def updateParticle(part, best, phi1, phi2, pmin, pmax):
            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            v_u1 = u1 * (part.best - part)
            v_u2 = u2 * (best - part)
            part.speed += v_u1 + v_u2
            for i, speed in enumerate(part.speed):
                smin = part.smin if np.isscalar(part.smin) else part.smin[i]
                smax = part.smax if np.isscalar(part.smax) else part.smax[i]

                if speed < smin:
                    part.speed[i] = smin
                elif speed > smax:
                    part.speed[i] = smax
            part += part.speed
            for i, value in enumerate(part):
                pm = pmin if np.isscalar(pmin) else pmin[i]
                pma = pmax if np.isscalar(pmax) else pmax[i]

                if value < pm:
                    part[i] = pm
                elif value > pma:
                    part[i] = pma
            return part

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        pmax = problem.ub
        pmin = problem.lb
        smax = 0.5 * problem.ub
        smin = -smax

        self.__toolbox.register("particle", generate, size=problem.dim, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.particle)
        self.__toolbox.register("update", updateParticle, phi1=self.__config.phi1, phi2=self.__config.phi2, pmin=pmin, pmax=pmax)
        self.__toolbox.register("evaluate", problem_eval)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # init population
        best = None  # gbest particle
        pop = self.__toolbox.population(n=self.__config.population_size)
        for part in pop:
            part.fitness.values = self.__toolbox.evaluate(part)
            part.best = self.__creator.Particle(part)
            part.best.fitness.values = part.fitness.values
            if best is None or part.fitness.values[0] < best.fitness.values[0]:
                best = self.__creator.Particle(part)
                best.fitness.values = part.fitness.values
        
        if self.full_meta_data:
            self.meta_X.append(np.array([ind.copy() for ind in pop]))
            self.meta_Cost.append(np.array([ind.fitness.values[0] for ind in pop]))
        
        
        fes = self.__config.population_size

        log_index = 1
        cost = [best.fitness.values[0]]

        done = False
        while not done:
            for part in pop:
                self.__toolbox.update(part, best)
                part.fitness.values = self.__toolbox.evaluate(part)
                if part.fitness.values[0] < part.best.fitness.values[0]:  # update pbest
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if part.fitness.values[0] < best.fitness.values[0]:  # update gbest
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                fes += 1
                if fes >= log_index * self.log_interval:
                    log_index += 1
                    cost.append(best.fitness.values[0])

                if problem.optimum is None:
                    done = fes >= self.__config.maxFEs
                else:
                    done = fes >= self.__config.maxFEs 

                if done:
                    if len(cost) >= self.__config.n_logpoint + 1:
                        cost[-1] = best.fitness.values[0]
                    else:
                        while len(cost) < self.__config.n_logpoint + 1:
                            cost.append(best.fitness.values[0])
                    break
            if self.full_meta_data:
                self.meta_X.append(np.array([ind.copy() for ind in pop]))
                self.meta_Cost.append(np.array([ind.fitness.values[0] for ind in pop]))
                
        results = {'cost': cost, 'fes': fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
