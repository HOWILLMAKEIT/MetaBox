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

