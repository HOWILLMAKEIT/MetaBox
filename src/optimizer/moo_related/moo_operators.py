import random
import sys
import copy
import numpy as np

EPSILON = sys.float_info.epsilon


class SBX:

    def __init__(self, probability=0.7, distribution_index=20.0):
        self.probability = probability
        self.distribution_index = distribution_index
        self.arity = 2

    def evolve(self, problem, parents):

        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])

        if random.uniform(0.0, 1.0) <= self.probability:
            nvars = problem.n_var

            for i in range(nvars):
                if random.uniform(0.0, 1.0) <= 0.5:
                    x1 = float(child1[i])
                    x2 = float(child2[i])
                    lb = problem.lb[i]
                    ub = problem.ub[i]
                    x1, x2 = self.sbx_crossover(x1, x2, lb, ub)
                    child1[i] = x1
                    child2[i] = x2

        return np.array([child1, child2])

    def sbx_crossover(self, x1, x2, lb, ub):
        dx = x2 - x1

        if dx > EPSILON:
            if x2 > x1:
                y2 = x2
                y1 = x1
            else:
                y2 = x1
                y1 = x2

            beta = 1.0 / (1.0 + (2.0 * (y1 - lb) / (y2 - y1)))
            alpha = 2.0 - pow(beta, self.distribution_index + 1.0)
            rand = random.uniform(0.0, 1.0)

            if rand <= 1.0 / alpha:
                alpha = alpha * rand
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0))
            else:
                alpha = alpha * rand;
                alpha = 1.0 / (2.0 - alpha)
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0))

            x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            beta = 1.0 / (1.0 + (2.0 * (ub - y2) / (y2 - y1)));
            alpha = 2.0 - pow(beta, self.distribution_index + 1.0);

            if rand <= 1.0 / alpha:
                alpha = alpha * rand
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0));
            else:
                alpha = alpha * rand
                alpha = 1.0 / (2.0 - alpha)
                betaq = pow(alpha, 1.0 / (self.distribution_index + 1.0));

            x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

            # randomly swap the values
            if bool(random.getrandbits(1)):
                x1, x2 = x2, x1

            x1 = clip(x1, lb, ub)
            x2 = clip(x2, lb, ub)

        return x1, x2


class DE1:
    """DE/rand/1"""

    def __init__(self, step_size=0.5, crossover_rate=1):
        self.arity = 3
        self.crossover_rate = crossover_rate
        self.step_size = step_size

    def evolve(self, problem, parents, ):
        result = copy.deepcopy(parents[0])
        jrand = random.randrange(problem.n_var)

        for j in range(problem.n_var):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v0 = float(parents[0][j])
                v1 = float(parents[1][j])
                v2 = float(parents[2][j])
                y = v0 + self.step_size * (v1 - v2)
                y = clip(y, problem.lb[j], problem.ub[j])
                result[j] = y
        return np.array([result])


class DE2:
    """DE/rand/2/bin"""

    def __init__(self, step_size=0.5, crossover_rate=1):
        self.arity = 5
        self.crossover_rate = crossover_rate
        self.step_size = step_size

    def evolve(self, problem, parents):
        result = copy.deepcopy(parents[0])
        jrand = random.randrange(problem.n_var)

        for j in range(problem.n_var):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v0 = float(parents[0][j])
                v1 = float(parents[1][j])
                v2 = float(parents[2][j])
                v3 = float(parents[3][j])
                v4 = float(parents[4][j])
                y = v0 + self.step_size * (v1 - v2) + self.step_size * (v3 - v4)
                y = clip(y, problem.lb[j], problem.ub[j])

                result[j] = y
        return np.array([result])


class DE3:
    """DE/current/2"""

    def __init__(self, step_size=0.5, crossover_rate=1):
        self.arity = 6
        self.crossover_rate = crossover_rate
        self.step_size = step_size

    def evolve(self, problem,parents):
        result = copy.deepcopy(parents[0])
        jrand = random.randrange(problem.n_var)
        for j in range(problem.n_var):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v0 = float(parents[0][j])
                v1 = float(parents[1][j])
                v2 = float(parents[2][j])
                v3 = float(parents[3][j])
                v4 = float(parents[4][j])
                v5 = float(parents[5][j])
                y = v0 + self.step_size * (v0 - v1) + self.step_size * (v2 - v3) + self.step_size * (v4 - v5);
                y = clip(y, problem.lb[j], problem.ub[j])

                result[j] = y

        return np.array([result])


class DE4:
    """DE/current to rand1"""

    def __init__(self, step_size=0.5, crossover_rate=1):
        self.arity = 4
        self.crossover_rate = crossover_rate
        self.step_size = step_size

    def evolve(self, problem,parents):
        result = copy.deepcopy(parents[0])
        jrand = random.randrange(problem.n_var)
        for j in range(problem.n_var):
            if random.uniform(0.0, 1.0) <= self.crossover_rate or j == jrand:
                v0 = float(parents[0][j])
                v1 = float(parents[1][j])
                v2 = float(parents[2][j])
                v3 = float(parents[3][j])
                y = v0 + self.step_size * (v0 - v1) + self.step_size * (v2 - v3);
                y = clip(y, problem.lb[j], problem.ub[j])

                result[j] = y

        return np.array([result])


class PM:
    def __init__(self, probability=1, distribution_index=20.0):
        self.probability = probability
        self.distribution_index = distribution_index

    def evolve(self, problem, parent):
        child = copy.deepcopy(parent)
        probability = self.probability
        probability /= float(problem.n_var)
        for i in range(problem.n_var):
            if random.uniform(0.0, 1.0) <= probability:
                child[i] = self.pm_mutation(child[i], problem.lb[i], problem.ub[i])
        return np.array([child])

    def pm_mutation(self, x, lb, ub):
        u = random.uniform(0, 1)
        dx = ub - lb

        if u < 0.5:
            bl = (x - lb) / dx
            b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, self.distribution_index + 1.0)
            delta = pow(b, 1.0 / (self.distribution_index + 1.0)) - 1.0
        else:
            bu = (ub - x) / dx
            b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, self.distribution_index + 1.0)
            delta = 1.0 - pow(b, 1.0 / (self.distribution_index + 1.0))

        x = x + delta * dx
        x = clip(x, lb, ub)

        return x


def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def chebyshev(solution_obj, ideal_point, weights, min_weight=0.0001):
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
