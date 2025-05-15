import numpy as np
import copy
import scipy.stats as stats
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


def test(x):
    return np.sum(np.isnan(x))


class NLSHADELBC(Basic_Optimizer):
    """
    # Introduction
    Non-Linear population size reduction Success-History Adaptive Differential Evolution with Linear Bias Change.It combines selective pressure, biased parameter adaptation with linear bias change, current-to-pbest strategy, resampling of solutions as bound constraint handling techniques, as well as the non-linear population size reduction.
    # Original paper
    "[**NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization**](https://ieeexplore.ieee.org/abstract/document/9870295/)." 2022 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2022.
    """
    def __init__(self, config):
        """
        Initializes the NLSHADELBC optimizer with the given configuration.
        # Args:
        - config (object): 
            - The Attributes needed for the NLSHADELBC in config are the following:
                - maxFEs (int): Maximum number of function evaluations allowed. Default directly depends on the type of the problem.
                - n_logpoint (int): Number of log points for tracking progress. Default is 50.
                - log_interval (int): Interval at which logs are recorded. Default is maxFEs // n_logpoint.
                - full_meta_data (bool): Flag indicating whether to store complete solution history. Default is False.
        # Attributes:
        - __pb (float): Rate of best individuals in mutation.
        - __pa (float): Rate of selecting individual from archive.
        - __m (float): Parameter for mutation.
        - __p_iniF (float): Initial value for mutation factor.
        - __p_iniCr (float): Initial value for crossover rate.
        - __p_fin (float): Final value for parameter p.
        - __Nmin (int): Lower bound of population size.
        - __archive (np.ndarray): Archive of replaced individuals.
        - __k (int): Index for updating elements in MF and MCr.
        - __MaxFEs (int): Maximum number of function evaluations.
        - __FEs (int): Current number of function evaluations.
        - gbest (float): Best global fitness value found.
        """
        
        super(NLSHADELBC, self).__init__(config)
        self.__pb = 0.4  # rate of best individuals in mutation
        self.__pa = 0.5  # rate of selecting individual from archive
        self.__m = 1.5
        self.__p_iniF = 3.5
        self.__p_iniCr = 1.0
        self.__p_fin = 1.5
        self.__Nmin = 4  # the lowerbound of population size

        self.__archive = np.array([])  # the archive(collection of replaced individuals)
        self.__k = 0  # the index of updating element in MF and MCr
        self.__MaxFEs = config.maxFEs
        self.__FEs = 0
        self.gbest = 1e15
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data

    def __str__(self):
        """
        Returns the string representation of the NLSHADELBC class.
        # Returns:
        - str: The string 'NLSHADELBC', representing the class name.
        """
        
        return 'NLSHADELBC'

    def __evaluate(self, problem, u):
        """
        # Introduction
        Evaluates the cost of a solution vector `u` for a given optimization problem, optionally adjusting by the problem's optimum and storing metadata if enabled.
        # Args:
        - problem: An object representing the optimization problem.
        - u (array-like): The solution vector to be evaluated.
        # Returns:
        - cost (float): The (possibly shifted) evaluation of the solution vector.
        # Side Effects:
        - If `self.full_meta_data` is True, appends the cost and solution vector to `self.meta_Cost` and `self.meta_X`, respectively.
        """
        if problem.optimum is None:
            cost = problem.eval(u)
        else:
            cost = problem.eval(u) - problem.optimum
        if self.full_meta_data:
            self.meta_Cost.append(cost.copy())
            self.meta_X.append(u.copy())
        return cost

    # Binomial crossover
    def __Binomial(self, x, v, cr):
        """
        # Introduction
        Performs binomial crossover operation used in evolutionary algorithms, particularly in Differential Evolution (DE). This method generates trial vectors by mixing parent vectors (`x`) and donor vectors (`v`) based on a crossover rate (`cr`).
        # Args:
        - x (np.ndarray): Parent population array of shape (NP, dim), where NP is the population size and dim is the dimensionality.
        - v (np.ndarray): Donor population array of the same shape as `x`.
        - cr (np.ndarray): Crossover rate array of shape (NP,), specifying the probability of crossover for each individual.
        # Returns:
        - np.ndarray: Trial population array of shape (NP, dim) after binomial crossover.
        """
        
        NP, dim = x.shape
        jrand = self.rng.randint(dim, size=NP)
        u = np.where(self.rng.rand(NP, dim) < cr.repeat(dim).reshape(NP, dim), v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    # Exponential crossover
    def __Exponential(self, x, v, cr):
        """
        # Introduction
        Performs the exponential crossover operation used in differential evolution algorithms. This method generates trial vectors by combining parent vectors (`x`) and donor vectors (`v`) based on a crossover rate (`cr`), following an exponential (contiguous) crossover scheme.
        # Args:
        - x (np.ndarray): The current population matrix of shape (NP, dim), where NP is the population size and dim is the dimensionality.
        - v (np.ndarray): The donor (mutant) population matrix of shape (NP, dim).
        - cr (np.ndarray): The crossover rate for each individual, of shape (NP, 1) or (NP,).
        # Returns:
        - np.ndarray: The trial population matrix after exponential crossover, of shape (NP, dim).
        # Notes:
        - This method ensures that for each individual, a contiguous subset of variables is inherited from the donor vector, starting from a randomly chosen position.
        """
        
        NP, dim = x.shape
        Crs = cr.repeat(dim).reshape(NP, dim)
        u = copy.deepcopy(x)
        L = self.rng.randint(dim, size=(NP, 1)).repeat(dim).reshape(NP, dim)
        R = np.ones(NP) * dim
        rvs = self.rng.rand(NP, dim)
        i = np.arange(dim).repeat(NP).reshape(dim, NP).transpose()
        rvs[rvs > Crs] = np.inf
        rvs[i <= L] = -np.inf
        k = np.where(rvs == np.inf)
        ki = np.stack(k).transpose()
        if ki.shape[0] > 0:
            k_ = np.concatenate((ki, ki[None, -1] + 1), 0)
            _k = np.concatenate((ki[None, 0] - 1, ki), 0)
            ind = ki[(k_[:, 0] != _k[:, 0]).reshape(-1, 1).repeat(2).reshape(-1, 2)[:-1]].reshape(-1, 2).transpose()
            R[ind[0]] = ind[1]

        R = R.repeat(dim).reshape(NP, dim)
        u[(i >= L) * (i < R)] = v[(i >= L) * (i < R)]
        return u

    # update pa according to cost changes
    def __update_Pa(self, fa, fp, na, NP):
        """
        # Introduction
        Updates the internal probability parameter `__pa` based on the provided counts of accepted and rejected solutions.
        # Args:
        - fa (float): The number of successful (accepted) solutions.
        - fp (float): The number of unsuccessful (rejected) solutions.
        - na (int): The number of accepted solutions in the current population.
        - NP (int): The total population size.
        # Returns:
        - None: This method updates the internal state (`__pa`) of the object.
        # Notes:
        - If there are no accepted solutions (`na == 0`) or no successful solutions (`fa == 0`), `__pa` is set to 0.5.
        - The updated `__pa` is constrained to the range [0.1, 0.9].
        """
        
        if na == 0 or fa == 0:
            self.__pa = 0.5
            return
        self.__pa = (fa / (na + 1e-15)) / ((fa / (na + 1e-15)) + (fp / (NP - na + 1e-15)))
        self.__pa = np.minimum(0.9, np.maximum(self.__pa, 0.1))

    def __mean_wL_Cr(self, df, s):
        """
        # Introduction
        Computes a weighted mean of the input array `s` using weights derived from `df` and a dynamically calculated exponent `pg`.
        The calculation is based on the current and maximum function evaluations, as well as initial and final parameter values.
        # Args:
        - df (np.ndarray): Array of weights or differences used for normalization.
        - s (np.ndarray): Array of values to be exponentiated and averaged.
        # Returns:
        - float: The weighted mean value computed using the specified formula. Returns 0.9 if the sum of `df` is zero.
        # Notes:
        - The method uses internal attributes: `__MaxFEs`, `__FEs`, `__p_iniCr`, `__p_fin`, and `__m`.
        """
        
        if np.sum(df) > 0.:
            w = df / np.sum(df)
            pg = (self.__MaxFEs - self.__FEs) * (self.__p_iniCr - self.__p_fin) / self.__MaxFEs + self.__p_fin
            res = np.sum(w * (s ** pg)) / np.sum(w * (s ** (pg - self.__m)))
            return res
        else:
            return 0.9

    def __mean_wL_F(self, df, s):
        """
        # Introduction
        Computes a weighted mean of the input array `s` using weights derived from the array `df` and a dynamically calculated exponent `pg`. This function is typically used in optimization algorithms to aggregate solutions based on their fitness or other criteria.
        # Args:
        - df (np.ndarray): Array of weights or fitness differences for the population.
        - s (np.ndarray): Array of solution values or candidate solutions.
        # Returns:
        - float: The weighted mean of `s` raised to the power `pg`, normalized by the weighted mean of `s` raised to the power `pg - self.__m`. Returns 0.5 if the sum of `df` is zero.
        # Notes:
        - The exponent `pg` is dynamically computed based on the current number of function evaluations (`self.__FEs`), the maximum allowed evaluations (`self.__MaxFEs`), and parameters `self.__p_iniF` and `self.__p_fin`.
        - If the sum of `df` is zero, a default value of 0.5 is returned.
        """
        
        if np.sum(df) > 0.:
            w = df / np.sum(df)
            pg = (self.__MaxFEs - self.__FEs) * (self.__p_iniF - self.__p_fin) / self.__MaxFEs + self.__p_fin
            return np.sum(w * (s ** pg)) / np.sum(w * (s ** (pg - self.__m)))
        else:
            return 0.5

    def __update_M_F_Cr(self, SF, SCr, df):
        """
        # Introduction
        Updates the memory arrays for mutation factor (F) and crossover rate (Cr) based on successful parameter values and their corresponding fitness differences.
        # Args:
        - SF (np.ndarray): Array of successful mutation factors from the current generation.
        - SCr (np.ndarray): Array of successful crossover rates from the current generation.
        - df (np.ndarray): Array of fitness differences associated with the successful parameters.
        # Side Effects:
        - Updates the internal memory arrays `self.__MF` and `self.__MCr` at the current index `self.__k`.
        - Advances the memory index `self.__k` in a circular manner.
        # Notes:
        - If there are no successful parameters (`SF.shape[0] == 0`), default values are assigned to the memory arrays.
        """
        
        if SF.shape[0] > 0:
            mean_wL = self.__mean_wL_F(df, SF)
            self.__MF[self.__k] = mean_wL
            mean_wL = self.__mean_wL_Cr(df, SCr)
            self.__MCr[self.__k] = mean_wL
            self.__k = (self.__k + 1) % self.__MF.shape[0]
        else:
            self.__MF[self.__k] = 0.5
            self.__MCr[self.__k] = 0.9

    def __choose_F_Cr(self):
        """
        # Introduction
        Generates crossover rate (Cr) and scaling factor (F) values for a population in an evolutionary algorithm, using normal and Cauchy distributions with adaptive memory.
        # Args:
        None
        # Returns:
        - tuple:
            - C_r (np.ndarray): Array of crossover rates for the population, clipped to [0, 1].
            - F (np.ndarray): Array of scaling factors for the population, adjusted to be at least 0 and at most 1.
        # Notes:
        - Crossover rates are sampled from a normal distribution centered at memory values (`self.__MCr`).
        - Scaling factors are sampled from a Cauchy distribution centered at memory values (`self.__MF`), with negative values reflected.
        """
        
        # generate Cr can be done simutaneously
        gs = self.__NP
        ind_r = self.rng.randint(0, self.__H, size=gs)  # index
        C_r = np.minimum(1, np.maximum(0, self.rng.normal(loc=self.__MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.__MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    def __sort(self):
        """
        # Introduction
        Sorts the population and corresponding cost arrays in ascending order of cost.
        # Args:
        None
        # Returns:
        None
        # Side Effects:
        - Updates `self.__cost` and `self.__population` so that both are sorted according to the ascending order of `self.__cost`.
        """
        
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __update_archive(self, old_id):
        """
        # Introduction
        Updates the archive of solutions by either appending a new individual from the population or replacing an existing one at random.
        # Args:
        - old_id (int): The index of the individual in the current population to be added to or used to update the archive.
        # Modifies:
        - self.__archive (np.ndarray): The archive of solutions, which is either expanded or updated in place.
        # Notes:
        - If the archive has not reached its maximum size (`self.__NA`), the individual is appended.
        - If the archive is full, a random entry is replaced with the new individual.
        """
        
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.__population[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[self.rng.randint(self.__archive.shape[0])] = self.__population[old_id]

    def __NLPSR(self):
        """
        # Introduction
        Adjusts the population and archive sizes dynamically based on the current number of function evaluations in the optimization process. This method is typically used in evolutionary algorithms to balance exploration and exploitation by resizing the population and archive as the search progresses.
        # Args:
        None
        # Modifies:
        - self.__NP (int): Updates the current population size.
        - self.__population (np.ndarray): Truncates the population array to the new size.
        - self.__cost (np.ndarray): Truncates the cost array to the new size.
        - self.__NA (int): Updates the current archive size.
        - self.__archive (np.ndarray): Truncates the archive array to the new size.
        # Raises:
        None
        """
        
        self.__sort()
        N = np.round(self.__Nmax + (self.__Nmin - self.__Nmax) * np.power(self.__FEs / self.__MaxFEs,
                                                                          1 - self.__FEs / self.__MaxFEs))
        A = int(max(N, self.__Nmin))
        N = int(N)
        if N < self.__NP:
            self.__NP = N
            self.__population = self.__population[:N]
            self.__cost = self.__cost[:N]
        if A < self.__archive.shape[0]:
            self.__NA = A
            self.__archive = self.__archive[:A]

    def __init_population(self, problem):
        """
        # Introduction
        Initializes the population and related parameters for the NL-SHADE-LBC evolutionary algorithm based on the given optimization problem.
        # Args:
        - problem (object): The problem object.
        # Side Effects:
        - Sets up the initial population, cost values, memory arrays for DE parameters, archive, and other internal state variables required for the algorithm's execution.
        # Notes:
        - The population size, memory size, and other parameters are determined as multiples of the problem's dimensionality.
        - The initial population is randomly generated within the provided bounds.
        - The best cost found so far is stored and logged.
        """
        
        self.__dim = problem.dim
        self.__Nmax = 23 * problem.dim
        self.__H = 20 * problem.dim
        self.__NP = 23 * problem.dim
        self.__MF = np.ones(self.__H) * 0.5  # the set of step length of DE
        self.__MCr = np.ones(self.__H) * 0.9  # the set of crossover rate of DE
        self.__population = self.rng.rand(self.__NP, problem.dim) * (problem.ub - problem.lb) + problem.lb
        self.__cost = self.__evaluate(problem, self.__population)
        self.__FEs = self.__NP
        self.__archive = np.array([])
        self.__MF = np.ones(self.__H) * 0.5
        self.__MCr = np.ones(self.__H) * 0.9
        self.__k = 0
        self.gbest = np.min(self.__cost)
        self.log_index = 1
        self.cost = [self.gbest]
        self.__NA = self.__NP  # the size of archive(collection of replaced individuals)
        self.__H = 20 * self.__dim

    # step method for ensemble, optimize population for a few times
    def __update(self,
                 problem,  # the problem instance
                 ):
        """
        # Introduction
        Performs one iteration of the NL-SHADE-LBC algorithm, updating the population, archive, and adaptive parameters based on current solutions and their fitness. Handles mutation, crossover, selection, and parameter adaptation for the evolutionary process.
        # Args:
        - problem (object): The problem object.
        # Returns:
        - None
        # Side Effects:
        - Updates internal state variables such as population, archive, fitness values, adaptive parameters, and logging information.
        # Notes:
        - This method is intended for internal use within the NL-SHADE-LBC optimizer class.
        - The method assumes that the population and archive are properly initialized.
        """
        if self.__NA < self.__archive.shape[0]:
            self.__archive = self.__archive[:self.__NA]
        self.__pa = 0.5
        NP = self.__NP
        # check record point lest missing it
        self.__sort()
        # select crossover rate and step length
        Cr, F = self.__choose_F_Cr()
        Cr = np.sort(Cr)
        # initialize some record values
        fa = 0  # sum of cost improvement using archive
        fp = 0  # sum of cost improvement without archive
        df = np.array([])  # record of cost improvement of each individual
        pr = np.exp(-(np.arange(
            NP) + 1) / NP)  # calculate the rate of individuals at different positions being selected in others' mutation
        pr /= np.sum(pr)
        na = 0  # the number of archive usage
        # randomly select a crossover method for the population
        pb_upper = int(np.maximum(2, NP * self.__pb))  # the range of pbest selection
        pbs = self.rng.randint(pb_upper, size=NP)  # select pbest for all individual
        count = 0
        duplicate = np.where(pbs == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 1:
            pbs[duplicate] = self.rng.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(pbs == np.arange(NP))[0]
            count += 1
        xpb = self.__population[pbs]
        r1 = self.rng.randint(NP, size=NP)
        count = 0
        duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
            count += 1
        x1 = self.__population[r1]
        rvs = self.rng.rand(NP)
        r2_pop = np.where(rvs >= self.__pa)[0]  # the indices of mutation with population
        r2_arc = np.where(rvs < self.__pa)[0]  # the indices of mutation with archive
        use_arc = np.zeros(NP, dtype=bool)  # a record for archive usage, used in parameter updating
        use_arc[r2_arc] = 1
        if self.__archive.shape[0] < 25:  # if the archive is empty, indices above are canceled
            r2_pop = np.arange(NP)
            r2_arc = np.array([], dtype=np.int32)
        r2 = self.rng.choice(np.arange(NP), size=r2_pop.shape[0], p=pr)
        count = 0
        duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.choice(np.arange(NP), size=duplicate.shape[0], p=pr)
            duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
            count += 1
        x2 = np.zeros((NP, self.__dim))
        # scatter indiv from population and archive into x2
        if r2_pop.shape[0] > 0:
            x2[r2_pop] = self.__population[r2]
        if r2_arc.shape[0] > 0:
            x2[r2_arc] = self.__archive[
                self.rng.randint(np.minimum(self.__archive.shape[0], self.__NA), size=r2_arc.shape[0])]
        Fs = F.repeat(self.__dim).reshape(NP, self.__dim)  # adjust shape for batch processing
        vs = self.__population + Fs * (xpb - self.__population) + Fs * (x1 - x2)
        # crossover rate for Binomial crossover has a different way for calculating
        Crb = np.zeros(NP)
        if self.__FEs + NP > self.__MaxFEs // 2:
            tmp_id = min(NP, self.__FEs + NP - self.__MaxFEs // 2)
            Crb[-tmp_id:] = 2 * ((self.__FEs + np.arange(tmp_id) + NP - tmp_id) / self.__MaxFEs - 0.5)

        usB = self.__Binomial(self.__population, vs, Crb)
        usE = self.__Exponential(self.__population, vs, Cr)
        us = usB
        CrossExponential = self.rng.rand(NP) > 0.5
        CrossExponential = CrossExponential.repeat(self.__dim).reshape(NP, self.__dim)
        us[CrossExponential] = usE[CrossExponential]
        # reinitialize values exceed valid range
        # us = us * ((-100 <= us) * (us <= 100)) + ((us > 100) + (us < -100)) * (self.rng.rand(NP, dim) * 200 - 100)
        us = np.where(us < problem.lb, (self.__population + problem.lb) / 2, us)
        us = np.where(us > problem.ub, (self.__population + problem.ub) / 2, us)

        cost = self.__evaluate(problem, us)
        optim = np.where(cost < self.__cost)[0]  # the indices of indiv whose costs are better than parent
        for i in range(optim.shape[0]):
            self.__update_archive(i)
        SF = F[optim]
        SCr = Cr[optim]
        df = (self.__cost[optim] - cost[optim]) / (self.__cost[optim] + 1e-9)
        arc_usage = use_arc[optim]
        fp = np.sum(df[arc_usage])
        fa = np.sum(df[np.array(1 - arc_usage, dtype=bool)])
        na = np.sum(arc_usage)
        self.__population[optim] = us[optim]
        self.__cost[optim] = cost[optim]

        if np.min(cost) < self.gbest:
            self.gbest = np.min(cost)

        self.__FEs += NP
        # adaptively adjust parameters
        self.__pb = 0.2 + 0.1 * (self.__FEs / self.__MaxFEs)
        self.__NLPSR()
        self.__update_M_F_Cr(SF, SCr, df)
        self.__update_Pa(fa, fp, na, NP)

        if self.__FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest)

        # 取消早停
        # if problem.optimum is None:
        #     return False
        # else:
        #     return self.gbest <= 1e-8

    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode for the given problem, updating the population and tracking the best solution found.
        # Args:
        - problem (object): The problem object.
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list): The history of best costs found during the episode.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `full_meta_data` is enabled, includes:
                - 'X' (list): The history of population states.
                - 'Cost' (list): The history of population costs.
        # Notes:
        - The method initializes the population, iteratively updates it until the maximum number of function evaluations is reached, and logs the best solution.
        - If `full_meta_data` is True, additional metadata about the optimization process is included in the results.
        """
        
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        self.__init_population(problem)

        while self.__FEs < self.__MaxFEs:
            self.__update(problem)

        if len(self.cost) >= self.__n_logpoint + 1:
            self.cost[-1] = self.gbest
        else:
            self.cost.append(self.gbest)
        results = {'cost': self.cost, 'fes': self.__FEs}

        if self.full_meta_data:
            metadata = {'X': self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        # 与agent一致，去除return，加上metadata
        return results

