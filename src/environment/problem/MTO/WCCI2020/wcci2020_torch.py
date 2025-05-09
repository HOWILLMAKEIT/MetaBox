from ....problem.basic_problem import Basic_Problem_Torch
import numpy as np
import time 
import torch

class WCCI2020_Torch_Problem(Basic_Problem_Torch):
    """
    # WCCI2020_Torch_Problem
      A Pytorch-based implementation of base class for defining basic functions in WCCI2020 Multitask Optimization(MTO) benchmark problems.
    # Introduction
      WCCI2020 proposes 10 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
      [WCCI2020](http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html)
    # License
    None
    # Problem Suite Composition
      The WCCI2020 problem suite contains a total of 10 benchmark problems, each consisting of 50 different basic functions with unique transformations(shifts and rotations).
      For each benchmark problem, fifty basic functions are added sequentially and cyclically to constitute the problem.
      These ten benchmark problems are classified according to the specific combination of different types of basic functions:
        P1: Shpere
        P2: Rosenbrock
        P3: Rastrigin
        P4: Shpere, Rosenbrock, Ackley
        P5: Rastrigin, Griewank, Weierstrass
        P6: Rosenbrock, Griewank, Schwefel
        P7: Rastrigin, Ackley, Weierstrass
        P8: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass
        P9: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
        P10:Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
    # Args:
    - `dim` (int): Dimensionality of the problem.
    - `shift` (Union[list, torch.Tensor]): Shift vector for the problem. Can be a list or a PyTorch tensor.
    - `rotate` (Union[list, torch.Tensor]): Rotation matrix for the problem. Can be a list or a PyTorch tensor.
    - `bias` (float): Bias value added to the objective function.
    # Attributes:
    - `T1` (float): Accumulated time in milliseconds for evaluations.
    - `dim` (int): Dimensionality of the problem.
    - `shift` (torch.Tensor): Shift vector converted to a PyTorch tensor.
    - `rotate` (torch.Tensor): Rotation matrix converted to a PyTorch tensor.
    - `bias` (float): Bias value for the problem.
    - `lb` (float): Lower bound of the search space.
    - `ub` (float): Upper bound of the search space.
    - `FES` (int): Function evaluation count.
    - `opt` (torch.Tensor): Optimal solution (shift vector).
    - `optimum` (float): Optimal objective value.
    # Methods:
    - `get_optimal() -> torch.Tensor`: Returns the optimal solution (shift vector).
    - `func(x: torch.Tensor) -> torch.Tensor`: Abstract method to define the objective function. Must be implemented in subclasses.
    - `decode(x: torch.Tensor) -> torch.Tensor`: Decodes a solution from the normalized space [0,1] to the original search space.
    - `sr_func(x: torch.Tensor, shift: torch.Tensor, rotate: torch.Tensor) -> torch.Tensor`: Applies shift and rotation transformations to the input.
    - `eval(x: Union[torch.Tensor, list]) -> torch.Tensor`: Evaluates the objective function for an individual or a population. Handles decoding and timing.
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

class Sphere_Torch(WCCI2020_Torch_Problem):
    def __init__(self, dim, shift, rotate, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2, -1)
    
    def __str__(self):
        return 'S'

class Ackley_Torch(WCCI2020_Torch_Problem):
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
    

class Griewank_Torch(WCCI2020_Torch_Problem):
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

class Rastrigin_Torch(WCCI2020_Torch_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2 - 10 * torch.cos(2 * torch.pi * z) + 10, -1) + self.bias
    
    def __str__(self):
        return 'R'
    
class Rosenbrock_Torch(WCCI2020_Torch_Problem):
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

class Weierstrass_Torch(WCCI2020_Torch_Problem):
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
    
class Schwefel_Torch(WCCI2020_Torch_Problem):
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