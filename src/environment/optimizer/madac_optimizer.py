import copy
import functools
import numpy as np
import math
import sys
import tianshou

from operator import itemgetter
from scipy.spatial.distance import cdist
from .learnable_optimizer import Learnable_Optimizer

EPSILON = sys.float_info.epsilon

POSITIVE_INFINITY = float("inf")
EPSILON = sys.float_info.epsilon


class PlatypusError(Exception):
    pass


def safe_extend(lst, items):
    # 如果 items 是 None，跳过处理
    if items is None:
        return
    # 判断是否是 list 或 ndarray
    if isinstance(items, (list, np.ndarray)):
        # 获取维度
        shape = np.shape(items)
        if len(shape) > 1:  # 多维的：用 extend
            lst.extend(items)
        else:  # 一维的：用 append
            lst.append(items)
    else:
        # 不是列表/数组，默认 append
        lst.append(items)


class MADAC_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        # Problem Related
        self.n_ref_points = 1000
        # # MDP Related
        self.reward_type = 0
        self.n_agents = 4
        self.early_stop = False
        # # MOEA/D Algorithm Related
        self.population_size = 100
        self.moead_neighborhood_maxsize = 30
        self.moead_delta = 0.8
        self.moead_eta = 2
        self.adaptive_open = True
        self.max_fes = config.maxFEs

    def init_population(self, problem):
        # problem related
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_var = problem.n_var
        # polulation related
        self.weights = self.get_weights(self.n_obj)
        self.neighborhoods = self.get_neighborhoods()
        if self.population_size != len(self.weights):
            self.population_size = len(self.weights)
        self.population = self.rng.uniform(low = problem.lb, high = problem.ub, size = (self.population_size, problem.n_var))
        self.population_obj = problem.eval(self.population)
        # eval budget
        self.done = False
        self.fes = len(self.population)
        self.moead_generation = 0
        self.episode_limit = self.max_fes // self.population_size
        # reference point
        self.archive_maximum = np.max(self.population_obj, axis = 0)
        self.archive_minimum = np.min(self.population_obj, axis = 0)
        self.ideal_point = copy.deepcopy(self.archive_minimum)
        self.problem_ref_points = self.problem.get_ref_set(
            n_ref_points = self.n_ref_points)
        # others
        self.init_adaptive_weights()
        self.init_static()

        self.operators = Operators(self.rng)
        # indicators
        self.initial_igd = self.get_igd()
        self.last_igd = self.initial_igd
        self.best_igd = self.initial_igd
        self.initial_hv = self.get_hv()
        self.last_hv = self.initial_hv
        self.best_hv = self.initial_hv
        self.metadata = {'X': [], 'cost': []}
        self.update_information()
        return self.get_state()

    def init_adaptive_weights(self):
        self.EP = []
        self.EP_obj = []
        self.EP_MaxSize = int(self.population_size * 1.5)
        self.rate_update_weight = 0.05  # rate_update_weight * N = nus
        self.nus = int(
            self.rate_update_weight * self.population_size)  # maximal number of subproblems needed to be adjusted
        # adaptive iteration interval, Units are population
        self.wag = int(self.episode_limit * 0.05)
        self.adaptive_cooling_time = self.wag
        self.adaptive_end = int(self.max_fes * 0.9)

    def init_static(self):
        self.last_bonus = 0
        # The number of iterations without promotion, maximum 10 is stag_count_max
        self.stag_count = 0
        self.stag_count_max = self.episode_limit / 10

        self.hv_his = []
        self.hv_last5 = tianshou.utils.MovAvg(size = 5)

        self.nds_ratio_his = []
        self.nds_ratio_last5 = tianshou.utils.MovAvg(size = 5)

        self.ava_dist_his = []
        self.ava_dist_last5 = tianshou.utils.MovAvg(size = 5)

        self.hv_running = tianshou.utils.RunningMeanStd()
        self.nds_ratio_running = tianshou.utils.RunningMeanStd()
        self.ava_dist_running = tianshou.utils.RunningMeanStd()

        self.reward_his = []
        self.obs_his = []
        self.igd_his = []

    def get_neighborhoods(self):
        neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(len(self.weights)):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])
        return neighborhoods

    def get_weights(self, n_obj):
        weights = None
        if n_obj == 2:
            weights = self.normal_boundary_weights(n_obj, 99, 0)
        elif n_obj == 3:
            weights = self.normal_boundary_weights(n_obj, 13, 0)
        elif n_obj == 5:
            weights = self.normal_boundary_weights(n_obj, 5, 0)
        elif n_obj == 7:
            weights = self.normal_boundary_weights(n_obj, 3, 2)
        elif n_obj == 8:
            weights = self.normal_boundary_weights(n_obj, 3, 1)
        elif n_obj == 10:
            weights = self.normal_boundary_weights(n_obj, 2, 2)
        else:
            weights = self.random_weights(n_obj, self.population_size)

        # if n_obj == 3:
        #     weights = self.normal_boundary_weights(n_obj, 13, 0)
        # elif n_obj == 6:
        #     weights = self.normal_boundary_weights(n_obj, 4, 1)
        # elif n_obj == 8:
        #     weights = self.normal_boundary_weights(n_obj, 3, 2)
        # else:
        #     weights = self.normal_boundary_weights(n_obj, 2, 3)

        return weights

    def get_igd(self):
        igd_calculator = InvertedGenerationalDistance(reference_set = self.problem_ref_points)
        igd_value = igd_calculator.calculate(self.population_obj)
        self.igd_his.append(igd_value)
        return igd_value

    def get_hv(self, n_samples = 1e5):
        if self.problem.n_obj <= 3:
            hv_fast = False
        else:
            hv_fast = True
        if not hv_fast:
            # Calculate the exact hv value
            hyp = Hypervolume(minimum = [0 for _ in range(
                self.n_obj)], maximum = self.archive_maximum)
            hv_value = hyp.calculate(np.array(self.population_obj))
        else:
            # Estimate the hv value by Monte Carlo

            popobj = copy.deepcopy(self.population_obj)
            optimum = self.problem_ref_points
            fmin = np.clip(np.min(popobj, axis = 0), np.min(popobj), 0)
            fmax = np.max(optimum, axis = 0)

            popobj = (popobj - np.tile(fmin, (self.population_size, 1))) / (
                np.tile(1.1 * (fmax - fmin), (self.population_size, 1)))
            index = np.all(popobj < 1, 1).tolist()
            popobj = popobj[index]
            if popobj.shape[0] <= 1:
                hv_value = 0
                self.hv_his.append(hv_value)
                self.hv_last5.add(hv_value)
                self.hv_running.update(np.array([hv_value]))
                return hv_value
            assert np.max(popobj) < 1
            hv_maximum = np.ones([self.n_obj])
            hv_minimum = np.min(popobj, axis = 0)
            n_samples_hv = int(n_samples)
            samples = np.zeros([n_samples_hv, self.n_obj])
            for i in range(self.n_obj):
                samples[:, i] = self.rng.uniform(
                    hv_minimum[i], hv_maximum[i], n_samples_hv)
            for i in range(popobj.shape[0]):
                domi = np.ones([samples.shape[0]], dtype = bool)
                m = 0
                while m < self.n_obj and any(domi):
                    domi = np.logical_and(domi, popobj[i, m] <= samples[:, m])
                    m += 1
                save_id = np.logical_not(domi)
                samples = samples[save_id, :]
            hv_value = np.prod(hv_maximum - hv_minimum) * (
                    1 - samples.shape[0] / n_samples_hv)
        self.hv_his.append(hv_value)
        self.hv_last5.add(hv_value)
        self.hv_running.update(np.array([hv_value]))
        return hv_value

    def get_reward(self, value):
        """
        use the value to get reward
        value(default is igd), the smaller the better
        :return: reward based on current igd and historical igd
        """
        reward = 0
        if self.reward_type == 0:
            if value < self.best_igd:
                bonus = (self.initial_igd - value) / self.initial_igd
                reward = (self.last_bonus + bonus) * (bonus - self.last_bonus)
            reward *= 100
        elif self.reward_type == 1:
            reward = max(self.last_igd - value, 0)
        elif self.reward_type == 2:
            if value < self.best_igd:
                reward = 10
            elif value < self.last_igd:
                reward = 1
        elif self.reward_type == 3:
            reward = max((self.last_igd - value) / value, 0)
        else:
            raise ValueError("Invaild Reward Type.")
        return reward

    def get_state(self):
        """
        # Introduction
        Constructs and returns the current state observation for all agents in the optimizer environment. The state is represented as a 22-dimensional feature vector containing normalized problem parameters, progress indicators, and various statistical metrics related to the optimization process.
        # Returns:
        - list of numpy.ndarray: A list containing the same 22-dimensional observation vector for each agent (`self.n_agents`), where each vector encodes the current environment state.
        # Observation Vector Details:
        - obs_[0]: Inverse of the number of objectives.
        - obs_[1]: Inverse of the number of variables.
        - obs_[2]: Normalized current generation count.
        - obs_[3]: Normalized stagnation count.
        - obs_[4]: Current hypervolume metric.
        - obs_[5]: Ratio of non-dominated solutions.
        - obs_[6]: Average distance metric.
        - obs_[7]: Recent change in hypervolume.
        - obs_[8]: Recent change in non-dominated solution ratio.
        - obs_[9]: Recent change in average distance.
        - obs_[10]: Mean of the last 5 hypervolume values.
        - obs_[11]: Mean of the last 5 non-dominated solution ratios.
        - obs_[12]: Mean of the last 5 average distances.
        - obs_[13]: Standard deviation of the last 5 hypervolume values.
        - obs_[14]: Standard deviation of the last 5 non-dominated solution ratios.
        - obs_[15]: Standard deviation of the last 5 average distances.
        - obs_[16]: Running mean of hypervolume.
        - obs_[17]: Running mean of non-dominated solution ratio.
        - obs_[18]: Running mean of average distance.
        - obs_[19]: Running variance of hypervolume.
        - obs_[20]: Running variance of non-dominated solution ratio.
        - obs_[21]: Running variance of average distance.
        """
        
        obs_ = np.zeros(22)
        obs_[0] = 1 / self.problem.n_obj
        obs_[1] = 1 / self.problem.n_var
        obs_[2] = (self.moead_generation) / self.episode_limit
        obs_[3] = self.stag_count / self.stag_count_max
        obs_[4] = self.last_hv
        obs_[5] = self.get_ratio_nondom_sol()
        obs_[6] = self.get_average_dist()
        obs_[7] = self.get_pre_k_change(1, self.hv_his)
        obs_[8] = self.get_pre_k_change(1, self.nds_ratio_his)
        obs_[9] = self.get_pre_k_change(1, self.ava_dist_his)
        obs_[10] = self.hv_last5.mean()
        obs_[11] = self.nds_ratio_last5.mean()
        obs_[12] = self.ava_dist_last5.mean()
        obs_[13] = self.hv_last5.std()
        obs_[14] = self.nds_ratio_last5.std()
        obs_[15] = self.ava_dist_last5.std()
        obs_[16] = self.hv_running.mean
        obs_[17] = self.nds_ratio_running.mean
        obs_[18] = self.ava_dist_running.mean
        obs_[19] = self.hv_running.var
        obs_[20] = self.nds_ratio_running.var
        obs_[21] = self.ava_dist_running.var
        return [obs_] * self.n_agents

    def get_action(self, action_idx, action):
        """
        # Introduction
        Maps an action index and action value to a specific agent parameter value based on predefined lists.
        # Args:
        - action_idx (int): The index indicating which agent parameter to select.  
            - 0: Neighbor size  
            - 1: Operator strategy  
            - 2: Probability constant  
            - 3: Weight
        - action (int): The index within the selected parameter list to retrieve the value.
        # Returns:
        - int | float | str: The value from the corresponding agent parameter list based on the provided indices.
        # Raises:
        - IndexError: If `action` is out of range for the selected parameter list.
        """
        
        neighborsize_agent = [15, 20, 25, 30]
        os_agent = ['DE1', 'DE2', 'DE3', 'DE4']
        pc_agent = [0.4, 0.5, 0.6, 0.7]
        weight_agent = [0, 1]
        if action_idx == 0:
            return neighborsize_agent[action]
        if action_idx == 1:
            return os_agent[action]
        if action_idx == 2:
            return pc_agent[action]
        if action_idx == 3:
            return weight_agent[action]

    def update(self, action, problem):
        """
        # Introduction
        Performs a single update step in the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) optimization process. This includes solution generation, selection, and adaptive weight adjustment.
        # Args:
        - action (list): A list of parameters controlling the update step, including neighborhood size, operator type, operator parameter, and weight adjustment flag.
        - problem (object): The optimization problem instance, providing evaluation and problem-specific methods.
        # Returns:
        - tuple: A tuple containing:
            - obs (object): The updated state observation.
            - rewards (list): A list of reward values for each agent.
            - done (bool): Whether the optimization process has reached its stopping condition.
            - info (dict): Additional information, including best and last IGD (Inverted Generational Distance) values.
        # Raises:
        - Exception: If the weight adjustment flag in `action[3]` is greater than 1.
        """
        
        """
        one step update in moea/d
        inclue solution generation and solution selection
        @param action: neighboor size; operator type; operator parameter
        :return:
        """

        self.moead_neighborhood_size = self.get_action(0, action[0])
        self.os = self.get_action(1, action[1])
        self.pc = self.get_action(2, action[2])
        self.weight_adjust = self.get_action(3, action[3])

        if self.adaptive_open is False:
            action[3] = 0

        subproblems = self.moead_get_subproblems()
        self.offspring_list = []
        self.offspring_obj_list = []
        for index in subproblems:
            mating_indices = self.moead_get_mating_indices(index)
            mating_population = [self.population[i] for i in mating_indices]
            if index in mating_indices:
                mating_indices.remove(index)

            parents = [self.population[index]] + \
                      [self.population[i] for i in
                       self.rng.choice(mating_indices, 6, replace = False)]
            offspring = getattr(self.operators, self.os)(problem, parents, self.pc)
            offspring_obj = problem.eval(offspring)
            self.fes += len(offspring)

            self.offspring_list.extend(offspring)
            self.offspring_obj_list.extend(offspring_obj)
            for child, child_obj in zip(offspring, offspring_obj):
                self.moead_update_ideal(child_obj)
                self.moead_update_solution(child, child_obj, mating_indices)  # selection

        if self.adaptive_open:
            self.update_ep()
        if action[3] > 1:
            raise Exception("action[3] > 1.")
        if action[3] == 1 and self.adaptive_cooling_time <= 0:
            self.adaptive_cooling_time = self.wag
            self.update_weight()
        self.adaptive_cooling_time -= 1
        self.moead_generation += 1
        if len(self.population_obj) != self.population_size:
            print("problem:", self.problem, "n", self.n_obj, "d", self.n_var)
            print("population_obj:", self.population_obj)
            print("population_size:", self.population_size)
            print("the length of population_obj", len(self.population_obj))
            print("the length of population", len(self.population))
            raise Exception("population_obj is not equal to population_size.")

        value = self.get_igd()
        reward = self.get_reward(value)
        self.update_igd(value)
        self.last_hv = self.get_hv()
        if self.last_hv > self.best_hv:
            self.best_hv = self.last_hv
        self.update_information()
        self.obs = self.get_state()
        # if stop, then return the information
        if self.fes >= self.max_fes:
            self.done = True
            print("fes:{},best_igd:{}".format(self.fes, self.best_igd))

        else:
            self.done = False

        info = {"best_igd": self.best_igd, "last_igd": self.last_igd}
        # print(
        #     "generation:{},fes:{},reward:{},last_igd{}".format(self.moead_generation, self.fes,reward,self.last_value))
        return self.obs, [reward] * self.n_agents, self.done, info

    def update_information(self):
        index = self.find_non_dominated_indices(self.population_obj)
        self.cost = [copy.deepcopy(self.population_obj[i]) for i in index]  # parato front
        self.metadata['X'].append(copy.deepcopy(self.population))
        self.metadata['cost'].append(copy.deepcopy(self.population_obj))

    def update_ep(self):
        """
        # Introduction
        Updates the current evolutionary population (EP) by incorporating offspring, filtering non-dominated solutions, and maintaining the population size using a crowding distance-based selection.
        # Args:
        None
        # Returns:
        None
        # Details:
        - Extends the current population (`EP` and `EP_obj`) with offspring solutions.
        - Filters the population to retain only non-dominated solutions.
        - If the population exceeds the maximum allowed size (`EP_MaxSize`), removes overcrowded solutions based on a crowding distance metric to maintain diversity.
        """
        
        """Update the current evolutional population EP
        """
        self.EP.extend(copy.deepcopy(self.offspring_list))
        self.EP_obj.extend(copy.deepcopy(self.offspring_obj_list))

        indices = self.find_non_dominated_indices(self.EP_obj)
        self.EP = [self.EP[i] for i in indices]
        self.EP_obj = [self.EP_obj[i] for i in indices]

        l = len(self.EP_obj)
        if l <= self.EP_MaxSize:
            return
        # Delete the overcrowded solutions in EP
        dist = cdist(
            [self.EP_obj[i] for i in range(l)],
            [self.EP_obj[i] for i in range(l)]
        )
        for i in range(l):
            dist[i][i] = np.inf
        dist.sort(axis = 1)
        # find max self.EP_MaxSize item
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis = 1)
        idx = np.argpartition(sub_dist, - self.EP_MaxSize)[-self.EP_MaxSize:]
        self.EP = list((itemgetter(*idx)(self.EP)))
        self.EP_obj = list((itemgetter(*idx)(self.EP_obj)))

    def update_weight(self):

        # Delete the overcrowded subproblems
        l_ep = len(self.EP)
        nus = min(l_ep, self.nus)
        dist = cdist(
            [self.population_obj[i] for i in range(
                self.population_size)],
            [self.population_obj[i] for i in range(
                self.population_size)]
        )
        for i in range(self.population_size):
            dist[i][i] = np.inf
        dist.sort(axis = 1)
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis = 1)
        idx = np.argpartition(
            sub_dist, -(self.population_size - nus))[-(self.population_size - nus):]
        self.population = list((itemgetter(*idx)(self.population)))
        self.population_obj = list((itemgetter(*idx)(self.population_obj)))
        self.weights = list((itemgetter(*idx)(self.weights)))
        # Add new subproblems
        l_p = len(self.population)
        dist = cdist(
            [self.EP_obj[i] for i in range(l_ep)],
            [self.population_obj[i] for i in range(l_p)]
        )  # shape = (l_ep, l_p)
        dist.sort(axis = 1)
        sub_dist = np.prod(dist[:, 0:self.n_obj], axis = 1)
        idx = np.argpartition(sub_dist, -nus)[-nus:]
        # 这两个可能都是一维列表
        # add_EP = list((itemgetter(*idx)(self.EP)))
        # add_EP_obj = list((itemgetter(*idx)(self.EP_obj)))
        if len(idx) == 1:
            add_EP = [self.EP[idx[0]]]
            add_EP_obj = [self.EP_obj[idx[0]]]
        else:
            add_EP = list(itemgetter(*idx)(self.EP))
            add_EP_obj = list(itemgetter(*idx)(self.EP_obj))
        add_weights = []
        for e in add_EP_obj:
            ans = np.asarray(e) - np.asarray(self.ideal_point)
            ans[ans < EPSILON] = 1
            ans = 1 / ans
            ans[ans == np.inf] = 1  # when f = z
            add_weights.append((ans / np.sum(ans)).tolist())

        self.population.extend(add_EP)
        self.population_obj.extend(add_EP_obj)
        self.weights.extend(add_weights)
        # safe_extend(self.population,add_EP)
        # safe_extend(self.population_obj,add_EP_obj)
        # safe_extend(self.weights,add_weights)
        # Update the neighbor
        self.neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(self.population_size):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            self.neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])

    def update_igd(self, value):
        self.igd_his.append(value)
        if value < self.best_igd:
            self.stag_count = 0
            self.best_igd = value
        else:
            self.stag_count += 1
        self.last_igd = value

    def moead_update_ideal(self, solution_obj):
        for i in range(solution_obj.shape[-1]):
            self.ideal_point[i] = min(
                self.ideal_point[i], solution_obj[i])

    def moead_calculate_fitness(self, solution_obj, weights):
        return chebyshev(solution_obj, self.ideal_point, weights)

    def moead_update_solution(self, solution, solution_obj, mating_indices):
        """
        # Introduction
        Updates the MOEA/D population by potentially replacing individuals in the mating neighborhood with a given solution if it improves the scalarized fitness value. Ensures that the number of replacements does not exceed a predefined threshold.
        # Args:
        - solution (Any): The candidate solution to potentially insert into the population.
        - solution_obj (Any): The objective values associated with the candidate solution.
        - mating_indices (List[int]): Indices of the population members considered for replacement.
        # Returns:
        - None
        # Notes:
        - The method shuffles the mating indices, evaluates each candidate in the neighborhood, and replaces it with the new solution if the new solution has a better scalarized fitness value according to the corresponding weight vector.
        - The number of replacements is limited by `self.moead_eta`.
        """
        """
        repair solution, make constraint satisfiable
        :param solution:
        :param mating_indices:
        :return:
        """

        c = 0
        self.rng.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            candidate_obj = self.population_obj[i]
            weights = self.weights[i]
            replace = False
            if self.moead_calculate_fitness(solution_obj, weights) < self.moead_calculate_fitness(candidate_obj,
                                                                                                  weights):
                replace = True

            if replace:
                self.population[i] = copy.deepcopy(solution)
                self.population_obj[i] = copy.deepcopy(solution_obj)
                c = c + 1

            if c >= self.moead_eta:
                break

    @staticmethod
    def moead_sort_weights(base, weights):
        """
        # Introduction
        Sorts a list of weight vectors by their Euclidean distance to a given base weight vector, returning the indices of the weights in ascending order of distance.
        # Args:
        - base (list[float]): The reference weight vector to which distances are computed.
        - weights (list[list[float]]): A list of weight vectors to be sorted by proximity to the base.
        # Returns:
        - list[int]: A list of indices representing the order of weights sorted by increasing distance to the base vector.
        # Notes:
        - The function uses Euclidean distance as the metric for sorting.
        """
        
        """Returns the index of weights nearest to the base weight."""

        def compare(weight1, weight2):
            dist1 = math.sqrt(
                sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(
                sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(
            enumerate(weights), key = functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def moead_get_subproblems(self):
        """
        # Introduction
        Determines the order of subproblems to be searched in the MOEA/D optimization process. 
        If utility-based updating is enabled, the method follows the utility-based MOEA/D search; 
        otherwise, it uses the original MOEA/D specification.
        # Returns:
        - List[int]: A shuffled list of indices representing the subproblems to be searched in the current iteration.
        """
        
        """
        Determines the subproblems to search.
        If :code:`utility_update` has been set, then this method follows the
        utility-based moea/D search.
        Otherwise, it follows the original moea/D specification.
        """
        
        indices = list(range(self.population_size))
        self.rng.shuffle(indices)
        return indices

    def moead_get_mating_indices(self, index):
        """
        # Introduction
        Determines the mating indices for the MOEA/D algorithm based on a probabilistic selection between the neighborhood and the entire population.
        # Args:
        - index (int): The index of the current individual in the population for which mating indices are to be determined.
        # Returns:
        - list[int]: A list of indices representing the selected mating pool, either from the individual's neighborhood or the entire population.
        # Notes:
        - With probability `moead_delta`, the method returns the indices of the individual's neighborhood up to `moead_neighborhood_size`.
        - Otherwise, it returns the indices of the entire population.
        """
        
        
        """Determines the mating indices.

        Returns the population members that are considered during mating.  With
        probability :code:`delta`, the neighborhood is returned.  Otherwise,
        the entire population is returned.
        """
        if self.rng.uniform(0.0, 1.0) <= self.moead_delta:
            return self.neighborhoods[index][:self.moead_neighborhood_size]
        else:
            return list(range(self.population_size))

    def find_non_dominated_indices(self, population_list):
        """
        此函数用于找出种群中的支配解
        :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
        :return: 支配解的列表
        """
        # 将列表转换为 numpy 数组
        population = np.array(population_list)
        n_solutions = population.shape[0]
        is_dominated = np.zeros(n_solutions, dtype = bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # 检查是否存在解 j 支配解 i
                    if np.all(population[j] <= population[i]) and np.any(population[j] < population[i]):
                        is_dominated[i] = True
                        break

        # 找出非支配解的索引
        non_dominated_indices = np.where(~is_dominated)[0]
        return non_dominated_indices

    def get_ratio_nondom_sol(self):
        """
        # Introduction
        Calculates the ratio of non-dominated solutions in the current population and updates historical tracking metrics.
        # Returns:
        - float: The ratio of non-dominated solutions to the total number of solutions in the population.
        # Side Effects:
        - Appends the computed ratio to `self.nds_ratio_his`.
        - Adds the ratio to `self.nds_ratio_last5`.
        - Updates the running average in `self.nds_ratio_running` with the new ratio value.
        """
        
        count = len(self.find_non_dominated_indices(self.population_obj))
        ratio_value = count / len(self.population_obj)
        self.nds_ratio_his.append(ratio_value)
        self.nds_ratio_last5.add(ratio_value)
        self.nds_ratio_running.update(np.array([ratio_value]))
        return ratio_value

    def get_average_dist(self):
        """
        # Introduction
        Calculates the normalized average pairwise distance between individuals in the population based on their objective values.
        # Args:
        None
        # Returns:
        - float: The normalized average distance between all pairs of individuals in the population.
        # Raises:
        - SystemExit: If the computed average distance is NaN, prints diagnostic information and exits the program.
        """
        
        total_distance = cdist(
            [self.population_obj[i] for i in range(
                self.population_size)],
            [self.population_obj[i] for i in range(self.population_size)])
        if np.max(total_distance) == 0:
            ava_dist = 0
        else:
            ava_dist = np.mean(total_distance) / np.max(total_distance)
        if (np.isnan(ava_dist)):
            for i in range(self.population_size):
                print(self.population[i].objectives)
            print("total_distance:", total_distance)
            print("ava_dist is nan")
            sys.exit(0)
        self.ava_dist_his.append(ava_dist)
        self.ava_dist_last5.add(ava_dist)
        self.ava_dist_running.update(np.array([ava_dist]))
        return ava_dist

    def get_pre_k_change(self, k, value_his):
        """
        # Introduction
        Calculates the change in value over the last `k` generations from a history of values.
        # Args:
        - k (int): The number of generations to look back.
        - value_his (list[float]): A list containing the historical values.
        # Returns:
        - float: The difference between the most recent value and the value `k` generations ago. Returns 0 if there are not enough generations.
        # Raises:
        - IndexError: If `value_his` does not contain enough elements for the calculation when `self.moead_generation >= k`.
        """
        
        if self.moead_generation >= k:
            return value_his[-1] - value_his[-(k + 1)]
        else:
            return 0

    def close(self):
        self.reset()

    def normal_boundary_weights(self, nobjs, divisions_outer, divisions_inner = 0):
        """Returns weights generated by the normal boundary method.

        The weights produced by this method are uniformly distributed on the
        hyperplane intersecting

            [(1, 0, ..., 0), (0, 1, ..., 0), ..., (0, 0, ..., 1)].

        Parameters
        ----------
        nobjs : int
            The number of objectives.
        divisions_outer : int
            The number of divisions along the outer set of weights.
        divisions_inner : int (optional)
            The number of divisions along the inner set of weights.
        """

        def generate_recursive(weights, weight, left, total, index):
            if index == nobjs - 1:
                weight[index] = float(left) / float(total)
                weights.append(copy.copy(weight))
            else:
                for i in range(left + 1):
                    weight[index] = float(i) / float(total)
                    generate_recursive(weights, weight, left - i, total, index + 1)

        def generate_weights(divisions):
            weights = []
            generate_recursive(weights, [0.0] * nobjs, divisions, divisions, 0)
            return weights

        weights = generate_weights(divisions_outer)

        if divisions_inner > 0:
            inner_weights = generate_weights(divisions_inner)

            for i in range(len(inner_weights)):
                weight = inner_weights[i]

                for j in range(len(weight)):
                    weight[j] = (1.0 / nobjs + weight[j]) / 2.0

                weights.append(weight)

        return weights

    def random_weights(self, nobjs, population_size):
        """Returns a set of randomly-generated but uniformly distributed weights.

        Simply producing N randomly-generated weights does not necessarily produce
        uniformly-distributed weights.  To help produce more uniformly-distributed
        weights, this method picks weights from a large collection of randomly-
        generated weights such that the distances between weights is maximized.

        Parameters
        ----------
        nobjs : int
            The number of objectives.
        population_size : int
            The number of weights to generate.
        """

        weights = []

        if nobjs == 2:
            weights = [[1, 0], [0, 1]]
            weights.extend([(i / (population_size - 1.0), 1.0 - i / (population_size - 1.0)) for i in range(1, population_size - 1)])
        else:
            # generate candidate weights
            candidate_weights = []

            for i in range(population_size * 50):
                random_values = [np.random.uniform(0.0, 1.0) for _ in range(nobjs)]
                candidate_weights.append([x / sum(random_values) for x in random_values])

            # add weights for the corners
            for i in range(nobjs):
                weights.append([0] * i + [1] + [0] * (nobjs - i - 1))

            # iteratively fill in the remaining weights by finding the candidate
            # weight with the largest distance from the assigned weights
            while len(weights) < population_size:
                max_index = -1
                max_distance = -POSITIVE_INFINITY

                for i in range(len(candidate_weights)):
                    distance = POSITIVE_INFINITY

                    for j in range(len(weights)):
                        temp = math.sqrt(sum([math.pow(candidate_weights[i][k] - weights[j][k], 2.0) for k in range(nobjs)]))
                        distance = min(distance, temp)

                    if distance > max_distance:
                        max_index = i
                        max_distance = distance

                weights.append(candidate_weights[max_index])
                del candidate_weights[max_index]

        return weights


class Indicator(object):
    # __metaclass = ABCMeta

    def __init__(self):
        super(Indicator, self).__init__()

    def __call__(self, set):
        return self.calculate(set)

    def calculate(self, set):
        raise NotImplementedError("method not implemented")


class Hypervolume(Indicator):
    # 只适用于最小化问题

    def __init__(self, reference_set = None, minimum = None, maximum = None):
        super(Hypervolume, self).__init__()
        if reference_set is not None:
            if minimum is not None or maximum is not None:
                raise ValueError("minimum and maximum must not be specified if reference_set is defined")
            self.minimum, self.maximum = normalize(reference_set)
        else:
            if minimum is None or maximum is None:
                raise ValueError("minimum and maximum must be specified when no reference_set is defined")
            self.minimum, self.maximum = minimum, maximum

    def invert(self, solution_normalized_obj: np.ndarray):
        for i in range(solution_normalized_obj.shape[1]):
            solution_normalized_obj[:, i] = 1.0 - np.clip(solution_normalized_obj[:, i], 0.0, 1.0)
        return solution_normalized_obj

    def dominates(self, solution1_obj, solution2_obj, nobjs):
        better = False
        worse = False

        for i in range(nobjs):
            if solution1_obj[i] > solution2_obj[i]:
                better = True
            else:
                worse = True
                break
        return not worse and better

    def swap(self, solutions_obj, i, j):
        solutions_obj[[i, j]] = solutions_obj[[j, i]]
        return solutions_obj

    def filter_nondominated(self, solutions_obj, nsols, nobjs):
        i = 0
        n = nsols
        while i < n:
            j = i + 1
            while j < n:
                if self.dominates(solutions_obj[i], solutions_obj[j], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, j, n)
                elif self.dominates(solutions_obj[j], solutions_obj[i], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, i, n)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1
        return n

    def surface_unchanged_to(self, solutions_normalized_obj, nsols, obj):
        return np.min(solutions_normalized_obj[:nsols, obj])

    def reduce_set(self, solutions, nsols, obj, threshold):
        i = 0
        n = nsols
        while i < n:
            if solutions[i, obj] <= threshold:
                n -= 1
                solutions = self.swap(solutions, i, n)
            else:
                i += 1
        return n

    def calc_internal(self, solutions_obj: np.ndarray, nsols, nobjs):
        volume = 0.0
        distance = 0.0
        n = nsols

        while n > 0:
            nnondom = self.filter_nondominated(solutions_obj, n, nobjs - 1)

            if nobjs < 3:
                temp_volume = solutions_obj[0][0]
            else:
                temp_volume = self.calc_internal(solutions_obj, nnondom, nobjs - 1)

            temp_distance = self.surface_unchanged_to(solutions_obj, n, nobjs - 1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_set(solutions_obj, n, nobjs - 1, distance)

        return volume

    def calculate(self, solutions_obj: np.ndarray):

        # 对可行解进行归一化
        solutions_normalized_obj = normalize(solutions_obj, self.minimum, self.maximum)

        # 筛选出所有目标值都小于等于 1.0 的解
        valid_mask = np.all(solutions_normalized_obj <= 1.0, axis = 1)
        valid_feasible = solutions_normalized_obj[valid_mask]

        if valid_feasible.size == 0:
            return 0.0

        # 对可行解进行反转操作
        inverted_feasible = self.invert(valid_feasible)

        # 计算超体积
        nobjs = inverted_feasible.shape[1]
        return self.calc_internal(inverted_feasible, len(inverted_feasible), nobjs)


class InvertedGenerationalDistance(Indicator):
    def __init__(self, reference_set):
        super(InvertedGenerationalDistance, self).__init__()
        self.reference_set = reference_set

    def calculate(self, set):
        return sum([distance_to_nearest(s, set) for s in self.reference_set]) / len(self.reference_set)


def distance_to_nearest(solution_obj, set):
    if len(set) == 0:
        return POSITIVE_INFINITY

    return min([euclidean_dist(solution_obj, s) for s in set])


def euclidean_dist(x, y):
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        print("x:", x)
        print("y:", y)
        raise TypeError("x and y must be lists or tuples.")

    return math.sqrt(sum([math.pow(x[i] - y[i], 2.0) for i in range(len(x))]))


def normalize(solutions_obj: np.ndarray, minimum: np.ndarray = None, maximum: np.ndarray = None) -> np.ndarray:
    """Normalizes the solution objectives.

    Normalizes the objectives of each solution within the minimum and maximum
    bounds.  If the minimum and maximum bounds are not provided, then the
    bounds are computed based on the bounds of the solutions.

    Parameters
    ----------
    solutions_obj : numpy.ndarray
        The solutions to be normalized. It should be a 2D numpy array.
    minimum : numpy.ndarray
        The minimum values used to normalize the objectives.
    maximum : numpy.ndarray
        The maximum values used to normalize the objectives.

    Returns
    -------
    numpy.ndarray
        The normalized solutions.
    """
    # 如果输入数组为空，直接返回空数组
    if len(solutions_obj) == 0:
        return solutions_obj

    # 获取目标的数量
    n_obj = solutions_obj.shape[1]

    # 如果 minimum 或 maximum 未提供，则计算它们
    if minimum is None or maximum is None:
        if minimum is None:
            minimum = np.min(solutions_obj, axis = 0)
        if maximum is None:
            maximum = np.max(solutions_obj, axis = 0)

    # 检查是否有目标的范围为空
    if np.any(maximum - minimum < EPSILON):
        raise ValueError("objective with empty range")

    # 进行归一化操作
    solutions_normalized_obj = (solutions_obj - minimum) / (maximum - minimum)

    return solutions_normalized_obj


class Operators:
    def __init__(self, rng):
        self.rng = rng

    def DE1(self, problem, parents, step_size = 0.5, crossover_rate = 1.0):
        """arity = 3"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.randint(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[1][j] - parents[2][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])

    DE1.arity = 3

    def DE2(self, problem, parents, step_size = 0.5, crossover_rate = 1.0):
        """arity = 5"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.randint(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[1][j] - parents[2][j]) + step_size * (parents[3][j] - parents[4][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])

    DE2.arity = 5

    def DE3(self, problem, parents, step_size = 0.5, crossover_rate = 1.0):
        """arity = 6"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.randint(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[0][j] - parents[1][j]) \
                    + step_size * (parents[2][j] - parents[3][j]) \
                    + step_size * (parents[4][j] - parents[5][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])

    DE3.arity = 6

    def DE4(self, problem, parents, step_size = 0.5, crossover_rate = 1.0):
        """arity = 4"""
        result = copy.deepcopy(parents[0])
        jrand = self.rng.randint(problem.n_var)

        for j in range(problem.n_var):
            if self.rng.uniform() <= crossover_rate or j == jrand:
                y = parents[0][j] + step_size * (parents[0][j] - parents[1][j]) + step_size * (parents[2][j] - parents[3][j])
                y = np.clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])

    DE4.arity = 4


def chebyshev(solution_obj, ideal_point, weights, min_weight = 0.0001):
    """Chebyshev (Tchebycheff) fitness of a solution with multiple objectives.

    This function is designed to only work with minimized objectives.

    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point : list of float
        The ideal point.
    weights : list of float
        The weights.
    min_weight : float
        The minimum weight allowed.
    """
    objs = solution_obj
    n_obj = objs.shape[-1]
    return max([max(weights[i], min_weight) * (objs[i] - ideal_point[i]) for i in range(n_obj)])


if __name__ == "__main__":
    optimizer = MADAC_Optimizer(1)
    weights = optimizer.get_weights(2)
    print("weights:", len(weights))
    weights = optimizer.get_weights(3)
    print("weights:", len(weights))
    weights = optimizer.get_weights(5)
    print("weights:", len(weights))
    weights = optimizer.get_weights(7)
    print("weights:", len(weights))
    weights = optimizer.get_weights(8)
    print("weights:", len(weights))
    weights = optimizer.get_weights(10)
    print("weights:", len(weights))
    # dtlz2 = DTLZ2()
    # optimizer.init_population(dtlz2)
    # optimizer.get_state()
    # for i in range(100):
    #     first_three = np.random.randint(0, 4, 3)
    #     last_one = np.random.randint(0, 2, 1)
    #     action = np.concatenate((first_three, last_one))
    #     optimizer.step(action, dtlz2)