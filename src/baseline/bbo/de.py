import numpy as np
from deap import base
from deap import creator
from deap import tools
from ...environment.optimizer.basic_optimizer import Basic_Optimizer

class DE(Basic_Optimizer):
    """
    # Introduction
    Differential Evolution (DE) optimizer implementation based on the DEAP framework.  
    This class inherits from `Basic_Optimizer` and provides an evolutionary algorithm for global optimization, supporting logging and optional metadata collection.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the Differential Evolution (DE) optimizer with the config object from src/config.py.
        # Args:
        - config (object): config object from src/config.py containing algorithm parameters and metadata options.
            - The Attributes needed for the DE are the following:
                - log_interval (int): Interval at which logs are recorded.
                - n_logpoint (int): Number of log points to record. Needed in run_episode(). Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                - __FEs (int): Counter for the number of function evaluations. Needed in run_episode(). Default is 0.
                - NP (int): Set the Population size in the config to 50.
                - F (float):Set the Differential parameter in the config to 0.5.
                - Cr(float): Set the Crossover parameter in the config to 0.5.
        # Attributes:
        - __config (object): Configuration object containing algorithm parameters.
        - __toolbox (object): DEAP toolbox for evolutionary algorithms.
        - __creator (object): DEAP creator for defining custom types.
                        
        # Side Effects:
        - Modifies `config` by setting `NP`, `F`, and `Cr` attributes.
        - Initializes internal attributes for further use in the class.
        """
        
        super().__init__(config)
        config.NP = 50
        config.F = 0.5
        config.Cr = 0.5

        self.__config = config

        self.__toolbox = None
        self.__creator = None

        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "DE" representing the object.
        """
        
        return "DE"


    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode using a Differential Evolution (DE) algorithm on the provided problem instance. The method initializes the population, evaluates individuals, applies mutation and crossover, and tracks the best solution found. Optionally, it collects metadata for each generation.
        # Args:
        - problem: An object representing the optimization problem. It must have the following attributes:
            - `lb` (float or array-like): Lower bound(s) of the search space.
            - `ub` (float or array-like): Upper bound(s) of the search space.
            - `dim` (int): Dimensionality of the problem.
            - `eval(x)` (callable): Function to evaluate a solution `x`.
            - `optimum` (float or None): Known optimum value (optional).
        # Returns:
        - dict: A dictionary containing:
            - `cost` (list of float): Best fitness value found at each logging interval.
            - `fes` (int): Total number of function evaluations performed.
            - `metadata` (dict, optional): If `self.full_meta_data` is True, includes:
                - `X` (list of np.ndarray): Population snapshots at each generation.
                - `Cost` (list of np.ndarray): Fitness values of the population at each generation.
        # Raises:
        - AttributeError: If required attributes are missing from the `problem` object.
        - Exception: Propagates exceptions raised during evaluation or population initialization.
        """
        
        self.rng_gpu = None
        self.rng_cpu = None
        self.rng = None
        np.random.seed(self.rng_seed)

        if self.__toolbox is None:
            self.__toolbox = base.Toolbox()
            self.__creator = creator

        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.__creator.create("Individual", list, fitness=creator.Fitnessmin)

        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        self.__toolbox.register("evaluate", problem_eval)
        self.__toolbox.register("select", tools.selTournament, tournsize=3)
        self.__toolbox.register("attr_float", np.random.uniform, problem.lb, problem.ub)

        if np.isscalar(problem.lb):
            self.__toolbox.register("individual", tools.initRepeat, creator.Individual, self.__toolbox.attr_float, n=problem.dim)
        else:
            self.__toolbox.register("individual", tools.initIterate, creator.Individual, self.__toolbox.attr_float)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        hof = tools.HallOfFame(1)

        pop = self.__toolbox.population(n=self.__config.NP)
        fitnesses = self.__toolbox.map(self.__toolbox.evaluate, pop)

        self.__FEs = self.__config.NP
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        if self.full_meta_data:
            self.meta_X.append(np.array([ind.copy() for ind in pop]))
            self.meta_Cost.append(np.array([ind.fitness.values[0] for ind in pop]))
        hof.update(pop)

        log_index = 1
        self.cost = [hof[0].fitness.values[0]]

        done = False
        while not done:
            for k, agent in enumerate(pop):
                a, b, c = self.__toolbox.select(pop, 3)
                y = self.__toolbox.clone(agent)
                # mutate & crossover
                index = np.random.randint(0, problem.dim, 1)[0]
                for i, value in enumerate(agent):
                    if np.random.rand() < self.__config.Cr or i == index:
                        y[i] = a[i] + self.__config.F * (b[i] - c[i])
                        lb = problem.lb if np.isscalar(problem.lb) else problem.lb[i]
                        ub = problem.ub if np.isscalar(problem.ub) else problem.ub[i]
                        y[i] = np.clip(y[i], lb, ub)
                y.fitness.values = self.__toolbox.evaluate(y)
                # selection
                if y.fitness.values[0] < agent.fitness.values[0]:
                    pop[k] = y

                hof.update(pop)
                self.__FEs += 1

                if self.__FEs >= log_index * self.log_interval:
                    log_index += 1
                    self.cost.append(hof[0].fitness.values[0])

                if problem.optimum is None:
                    done = self.__FEs >= self.__config.maxFEs
                else:
                    done = self.__FEs >= self.__config.maxFEs

                if done:
                    if len(self.cost) >= self.__config.n_logpoint + 1:
                        self.cost[-1] = hof[0].fitness.values[0]
                    else:
                        while len(self.cost) < self.__config.n_logpoint + 1:
                            self.cost.append(hof[0].fitness.values[0])
                    break

            if self.full_meta_data:
                self.meta_X.append(np.array([ind.copy() for ind in pop]))
                self.meta_Cost.append(np.array([ind.fitness.values[0] for ind in pop]))
    
        results = {'cost': self.cost, 'fes': self.__FEs}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results
