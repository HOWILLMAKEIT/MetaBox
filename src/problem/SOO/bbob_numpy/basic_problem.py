import numpy as np
import time

class Basic_Problem:
    """
    Abstract super class for problems and applications.
    """

    def reset(self):
        self.T1=0

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

    def func(self, x):
        raise NotImplementedError


class BBOB_Basic_Problem(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError

    def sr_func(self, x, Os, Mr):  # shift and rotate
        y = x[:, :Os.shape[-1]] - Os
        return np.matmul(Mr, y.transpose()).transpose()

    def rotate_gen(self, dim):  # Generate a rotate matrix
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            mat = np.eye(dim)
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat[n - 1:, n - 1:] = Hx
            H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H

    def osc_transform(self, x):
        """
        Implementing the oscillating transformation on objective values or/and decision values.

        :param x: If x represents objective values, x is a 1-D array in shape [NP] if problem is single objective,
                or a 2-D array in shape [NP, number_of_objectives] if multi-objective.
                If x represents decision values, x is a 2-D array in shape [NP, dim].
        :return: The array after transformation in the shape of x.
        """
        y = x.copy()
        idx = (x > 0.)
        y[idx] = np.log(x[idx]) / 0.1
        y[idx] = np.exp(y[idx] + 0.49 * (np.sin(y[idx]) + np.sin(0.79 * y[idx]))) ** 0.1
        idx = (x < 0.)
        y[idx] = np.log(-x[idx]) / 0.1
        y[idx] = -np.exp(y[idx] + 0.49 * (np.sin(0.55 * y[idx]) + np.sin(0.31 * y[idx]))) ** 0.1
        return y

    def asy_transform(self, x, beta):
        """
        Implementing the asymmetric transformation on decision values.

        :param x: Decision values in shape [NP, dim].
        :param beta: beta factor.
        :return: The array after transformation in the shape of x.
        """
        NP, dim = x.shape
        idx = (x > 0.)
        y = x.copy()
        y[idx] = y[idx] ** (
                    1. + beta * np.linspace(0, 1, dim).reshape(1, -1).repeat(repeats=NP, axis=0)[idx] * np.sqrt(y[idx]))
        return y

    def pen_func(self, x, ub):
        """
        Implementing the penalty function on decision values.

        :param x: Decision values in shape [NP, dim].
        :param ub: the upper-bound as a scalar.
        :return: Penalty values in shape [NP].
        """
        return np.sum(np.maximum(0., np.abs(x) - ub) ** 2, axis=-1)


class NoisyProblem:
    def noisy(self, ftrue):
        raise NotImplementedError

    def eval(self, x):
        ftrue = super().eval(x)
        return self.noisy(ftrue)

    def boundaryHandling(self, x):
        return 100. * self.pen_func(x, self.ub)


class GaussNoisyProblem(NoisyProblem):
    """
    Attribute 'gause_beta' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * np.exp(self.gauss_beta * np.random.randn(*ftrue_unbiased.shape))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class UniformNoisyProblem(NoisyProblem):
    """
    Attributes 'uniform_alpha' and 'uniform_beta' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * (np.random.rand(*ftrue_unbiased.shape) ** self.uniform_beta) * \
                          np.maximum(1., (1e9 / (ftrue_unbiased + 1e-99)) ** (
                                      self.uniform_alpha * (0.49 + 1. / self.dim) * np.random.rand(
                                  *ftrue_unbiased.shape)))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class CauchyNoisyProblem(NoisyProblem):
    """
    Attributes 'cauchy_alpha' and 'cauchy_p' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased + self.cauchy_alpha * np.maximum(0.,
                                                                          1e3 + (np.random.rand(
                                                                              *ftrue_unbiased.shape) < self.cauchy_p) * np.random.randn(
                                                                              *ftrue_unbiased.shape) / (np.abs(
                                                                              np.random.randn(
                                                                                  *ftrue_unbiased.shape)) + 1e-199))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)

