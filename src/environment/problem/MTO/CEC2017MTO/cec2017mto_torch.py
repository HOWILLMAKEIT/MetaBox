from ....problem.basic_problem import Basic_Problem_Torch
import numpy as np
import time 
import torch

class CEC2017_Torch_Problem(Basic_Problem_Torch):
    """
    # CEC2017_Torch_Problem
      A Pytorch-based implementation of base class for defining basic functions in CEC2017 Multitask Optimization(MTO) benchmark problems.
    # Introduction
      CEC2017MTO proposes 9 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
       "[Evolutionary Multitasking for Single-objective Continuous Optimization: Benchmark Problems, Performance Metric, and Bseline Results](https://arxiv.org/pdf/1706.03470)."
    # Official Implementation
      [CEC2017MTO](http://www.bdsc.site/websites/MTO/index.html)
    # License
    None
    # Problem Suite Composition
      The CEC2017MTO problem suite contains a total of 9 benchmark problems, each consisting of two basic functions.
      These nine benchmark problems are classified according to the degree of intersection and the inter-task similarity between the two constitutive functions:
        P1. Complete intersection and high similarity(CI+HS)
        P2. Complete intersection and medium similarity(CI+MS)  
        P3. Complete intersection and low similarity(CI+LS)
        P4. Partial intersection and high similarity(PI+HS)
        P5. Partial intersection and medium similarity(PI+MS)  
        P6. Partial intersection and low similarity(PI+LS)
        P7. No intersection and high similarity(NI+HS)
        P8. No intersection and medium similarity(NI+MS) 
        P9. No intersection and low similarity(NI+LS)
    # Args:
    - `dim` (int): Dimensionality of the problem.
    - `shift` (list or None): Shift vector applied to the problem. If `None`, no shift is applied.
    - `rotate` (list or None): Rotation matrix applied to the problem. If `None`, no rotation is applied.
    - `bias` (float): Bias value added to the problem's objective function.
    # Attributes:
    - `T1` (float): Accumulated time (in milliseconds) spent evaluating the problem.
    - `dim` (int): Dimensionality of the problem.
    - `shift` (torch.Tensor or None): Shift vector as a PyTorch tensor.
    - `rotate` (torch.Tensor or None): Rotation matrix as a PyTorch tensor.
    - `bias` (float): Bias value for the problem.
    - `lb` (float): Lower bound of the problem's search space.
    - `ub` (float): Upper bound of the problem's search space.
    - `FES` (int): Function evaluation count.
    - `opt` (torch.Tensor): Optimal solution for the problem.
    - `optimum` (float): Optimal objective value for the problem.
    # Methods:
    - `get_optimal() -> torch.Tensor`: Returns the optimal solution for the problem.
    - `func(x: torch.Tensor) -> torch.Tensor`: Abstract method to compute the objective function value(s) for input `x`. Must be implemented in subclasses.
    - `decode(x: torch.Tensor) -> torch.Tensor`: Decodes a solution from the normalized space [0, 1] to the problem's actual search space.
    - `sr_func(x: torch.Tensor, shift: torch.Tensor, rotate: torch.Tensor) -> torch.Tensor`: Applies shift and rotation transformations to the input `x`.
    - `eval(x: torch.Tensor) -> torch.Tensor`: Evaluates the objective function for a single solution or a population of solutions. Supports inputs of different dimensions (1D, 2D, or 3D).
    # Raises:
    - `NotImplementedError`: Raised if the `func` method is not implemented in a subclass.
    """

    def __init__(self, dim, shift, rotate, bias):
        self.T1 = 0
        self.dim = dim
        self.shift = shift if shift is None else torch.tensor(shift, dtype=torch.float64)
        self.rotate = rotate if rotate is None else torch.tensor(rotate, dtype=torch.float64)
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0

        self.opt = self.shift if self.shift is not None else torch.zeros(size=(self.dim,))
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
    
    def decode(self, x):
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        if shift is not None: 
            y = x - shift
        else:
            y = x
        
        if rotate is not None:
            z = torch.matmul(rotate, y.transpose(0,1)).transpose(0,1)
        else:
            z = y 
        return z
    
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

class Sphere_Torch(CEC2017_Torch_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2, -1)
    
    def __str__(self):
        return 'Sphere'

class Ackley_Torch(CEC2017_Torch_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        sum1 = -0.2 * torch.sqrt(torch.sum(z ** 2, -1) / self.dim)
        sum2 = torch.sum(torch.cos(2 * torch.pi * z), -1) / self.dim
        return torch.round(torch.e + 20 - 20 * torch.exp(sum1) - torch.exp(sum2), decimals = 15) + self.bias
    
    def __str__(self):
        return 'Ackley'
    

class Griewank_Torch(CEC2017_Torch_Problem):
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
        return 'Griewank'

class Rastrigin_Torch(CEC2017_Torch_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2 - 10 * torch.cos(2 * torch.pi * z) + 10, -1) + self.bias
    
    def __str__(self):
        return 'Rastrigin'
    
class Rosenbrock_Torch(CEC2017_Torch_Problem):
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
        return 'Rosenbrock'


class Weierstrass_Torch(CEC2017_Torch_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
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
        return 'Weierstrass'
    
class Schwefel_Torch(CEC2017_Torch_Problem):
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
        return 'Schwefel'