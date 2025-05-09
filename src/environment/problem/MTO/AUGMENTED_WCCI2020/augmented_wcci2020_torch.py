from ....problem.basic_problem import Basic_Problem_Torch
import numpy as np
import torch
import time 

class AUGMENTED_WCCI2020_Torch_Problem(Basic_Problem_Torch):
    """
    # AUGMENTED_WCCI2020_Torch_Problem
      A Pytorch-based implementation of base class for defining basic functions in AUGMENTED WCCI2020 Multitask Optimization(MTO) benchmark problems.
    # Introduction
      Augmented WCCI2020 proposes 127 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
    None
    # License
    None
    # Problem Suite Composition
      The Augmented WCCI2020 problem suite contains a total of 127 benchmark problems, with each problem consisting of multiple different basic functions with unique transformations(shifts and rotations).
      The number of basic functions can be specified according to the user's requirements. Defaults to 10.
      These 127 benchmark problems are composed based on all combinations of the seven basic functions as Shpere, Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass and Schwefel.
      For each benchmark problem, the basic functions in the correspondent combination are selected randomly and added with unique transformations(shifts and rotations) until the number of basic functions is reached.
    # Args:
    - `dim` (int): The dimensionality of the problem.
    - `shift` (Union[list, numpy.ndarray, torch.Tensor]): The shift vector applied to the input space.
    - `rotate` (Union[list, numpy.ndarray, torch.Tensor]): The rotation matrix applied to the input space.
    - `bias` (float): A bias value added to the objective function.
    # Attributes:
    - `T1` (float): A timer attribute used to measure evaluation time in milliseconds.
    - `dim` (int): The dimensionality of the problem.
    - `shift` (torch.Tensor): The shift vector applied to the input space.
    - `rotate` (torch.Tensor): The rotation matrix applied to the input space.
    - `bias` (float): A bias value added to the objective function.
    - `lb` (float): The lower bound of the problem's search space.
    - `ub` (float): The upper bound of the problem's search space.
    - `FES` (int): The function evaluation count.
    - `opt` (torch.Tensor): The optimal solution in the search space.
    - `optimum` (float): The objective function value at the optimal solution.
    # Methods:
    - `get_optimal() -> torch.Tensor`: Returns the optimal solution in the search space.
    - `func(x: torch.Tensor) -> torch.Tensor`: Abstract method to define the objective function. Must be implemented in subclasses.
    - `decode(x: torch.Tensor) -> torch.Tensor`: Decodes a solution from the constrained space [0,1] to the original search space.
    - `sr_func(x: torch.Tensor, shift: torch.Tensor, rotate: torch.Tensor) -> torch.Tensor`: Applies shift and rotation transformations to the input.
    - `eval(x: torch.Tensor) -> torch.Tensor`: Evaluates the objective function for a given solution or population. Supports both individual and population evaluations.
    # Raises:
    - `NotImplementedError`: Raised if the `func` method is not implemented in a subclass.
    """
    def __init__(self, dim, shift, rotate, bias):
        self.T1 = 0
        self.dim = dim
        self.shift = shift if not isinstance(shift, torch.Tensor) else torch.tensor(shift, dtype=torch.float64)
        self.rotate = rotate if not isinstance(shift, torch.Tensor) else torch.tensor(shift, dtype=torch.float64)
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
    
    def decode(self, x):
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        y = x - shift
        return torch.matmul(rotate, y.transpose(0,1)).transpose(0,1)
    
    def eval(self, x):
        """
        A specific version of func() with adaptation to evaluate both individual and population in MTO.
        """
        start=time.perf_counter()
        x = self.decode(x)  # the solution in MTO is constrained in a unified space [0,1]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            x = x[:self.dim]
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            x = x[:, :self.dim]
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            x = x[:,:,:self.dim]
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

class Sphere_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -100
    UB = 100
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2, -1)
    
    def __str__(self):
        return 'S'

class Ackley_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        sum1 = -0.2 * torch.sqrt(torch.sum(z ** 2, -1) / self.dim)
        sum2 = torch.sum(torch.cos(2 * torch.pi * z), -1) / self.dim
        return torch.round(torch.e + 20 - 20 * torch.exp(sum1) - torch.exp(sum2), decimals = 15) + self.bias
    
    def __str__(self):
        return 'A'
    
class Griewank_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -100
    UB = 100
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        s = torch.sum(z ** 2, -1)
        p = torch.ones(x.shape[0])
        for i in range(self.dim):
            p *= torch.cos(z[:, i] / torch.sqrt(torch.tensor(1 + i)))
        return 1 + s / 4000 - p + self.bias
    
    def __str__(self):
        return 'G'


class Rastrigin_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2 - 10 * torch.cos(2 * torch.pi * z) + 10, -1) + self.bias
    
    def __str__(self):
        return 'R'
    
class Rosenbrock_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return torch.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias
    
    def __str__(self):
        return 'Ro'

class Weierstrass_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -0.5
    UB = 0.5
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -0.5
        self.ub = 0.5

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a, b, k_max = torch.tensor(0.5), torch.tensor(3.0), torch.tensor(20)
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += torch.sum(torch.pow(a, k) * torch.cos(2 * torch.pi * torch.pow(b, k) * (z + 0.5)), -1)
            sum2 += torch.pow(a, k) * torch.cos(2 * torch.pi * torch.pow(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias
    
    def __str__(self):
        return 'W'
     
class Schwefel_Torch(AUGMENTED_WCCI2020_Torch_Problem):
    LB = -500
    UB = 500
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -500
        self.ub = 500

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        z = torch.clip(z, min=self.lb, max=self.ub)
        g = z * torch.sin(torch.sqrt(torch.abs(z)))
        return b * self.dim - torch.sum(g,-1)
    
    def __str__(self):
        return 'Sc'