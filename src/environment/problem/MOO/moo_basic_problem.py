import numpy as np
import torch as th
import time
from problem.basic_problem import Basic_Problem



    
    
class Moo_Basic_Problem(Basic_Problem):
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 lb=None,
                 ub=None,
                 vtype=None,
                 vars=None,
                 **kwargs):

        """
        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        lb : np.array, float, int
            Lower bounds for the variables. if integer all lower bounds are equal.

        ub : np.array, float, int
            Upper bounds for the variable. if integer all upper bounds are equal.

        vtype : type
            The variable type. So far, just used as a type hint.

        """

        # number of variable
        self.n_var = n_var
        # number of objectives
        self.n_obj = n_obj
        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.lb, self.ub = lb, ub
        # the variable type (only as a type hint at this point)
        self.vtype = vtype
        #if it is a problem with an actual number of variables - make sure lb and ub are numpy arrays
        self.dict = kwargs
        if n_var > 0:
            if self.lb is not None:
                if not isinstance(self.lb, np.ndarray) and not isinstance(self.lb,th.Tensor):
                    self.lb = np.ones(n_var) * lb
                self.lb = self.lb.astype(float)

            if self.ub is not None and not isinstance(self.lb,th.Tensor):
                if not isinstance(self.ub, np.ndarray):
                    self.ub = np.ones(n_var) * ub
                self.ub = self.ub.astype(float)





    

    