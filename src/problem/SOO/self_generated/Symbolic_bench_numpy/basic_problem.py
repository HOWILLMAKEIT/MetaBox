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

class GP_problem(Basic_Problem):
    def __init__(self,execute,problemID,lb,ub,dim):
        self.problem = execute
        self.lb = lb
        self.ub = ub
        self.optimum = None
        self.opt = None
        self.problemID = problemID
        self.dim = dim
        self.T1 = 0
        self.FES = 0
        
    def func(self,x):
        return self.problem(x)
    
    def __call__(self, x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x)
    
    def get_optimal(self):
        return self.opt
    
    def __str__(self):
        return f'GP_Problem_{self.problemID}'
    
    def __name__(self):
        return f'GP_Problem_{self.problemID}'