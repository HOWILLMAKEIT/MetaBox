import numpy as np
import torch as th
import time



    
    
class Basic_Problem:
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 lb=None,
                 ub=None,
                 vtype=None,
                 vars=None,
                 requires_kwargs=False,
                 replace_nan_values_by=None,
                 exclude_from_serialization=None,
                 callback=None,
                 strict=True,
                 **kwargs):

        """

        Parameters
        ----------
        n_var : int
            Number of Variables

        n_obj : int
            Number of Objectives

        n_ieq_constr : int
            Number of Inequality Constraints

        n_eq_constr : int
            Number of Equality Constraints

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

        # number of inequality constraints
        self.n_ieq_constr = n_ieq_constr if "n_constr" not in kwargs else max(n_ieq_constr, kwargs["n_constr"])

        # number of equality constraints
        self.n_eq_constr = n_eq_constr

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.lb, self.ub = lb, ub

        # a callback function to be called after every evaluation
        self.callback = callback

        # if the variables are provided in their explicit form
        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)

            if self.lb is None:
                self.lb = {name: var.lb if hasattr(var, "lb") else None for name, var in vars.items()}
            if self.ub is None:
                self.ub = {name: var.ub if hasattr(var, "ub") else None for name, var in vars.items()}

        # the variable type (only as a type hint at this point)
        self.vtype = vtype

        # whether evaluation requires kwargs (passing them can cause overhead in parallelization)
        self.requires_kwargs = requires_kwargs

        # whether the shapes are checked strictly
        self.strict = strict

        # if it is a problem with an actual number of variables - make sure lb and ub are numpy arrays
        # if n_var > 0:
        #
        #     if self.lb is not None:
        #         if not isinstance(self.lb, np.ndarray) and not isinstance(self.lb,th.Tensor):
        #             self.lb = np.ones(n_var) * lb
        #         self.lb = self.lb.astype(float)
        #
        #     if self.ub is not None and not isinstance(self.lb,th.Tensor):
        #         if not isinstance(self.ub, np.ndarray):
        #             self.ub = np.ones(n_var) * ub
        #         self.ub = self.ub.astype(float)

        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

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





    

    