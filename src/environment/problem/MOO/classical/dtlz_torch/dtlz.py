import numpy as np
import torch as th
import math
import time
from problem.basic_problem import Basic_Problem
import geatpy as ea

class DTLZ_Problem(Basic_Problem):
    def __init__(self,
                 n_var=-1,
                 n_obj=1,
                 lb=None,
                 ub=None,
                 vtype=None,
                 vars=None,
                 callback=None,
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

        # type of the variable to be evaluated
        self.data = dict(**kwargs)

        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.lb, self.ub = lb, ub

        # a callback function to be called after every evaluation
        self.callback = callback

        # the variable type (only as a type hint at this point)
        self.vtype = vtype
        #if it is a problem with an actual number of variables - make sure lb and ub are numpy arrays
        if n_var > 0:
            if self.lb is not None:
                if not isinstance(self.lb, np.ndarray) and not isinstance(self.lb,th.Tensor):
                    self.lb = np.ones(n_var) * lb
                self.lb = self.lb.astype(float)

            if self.ub is not None and not isinstance(self.lb,th.Tensor):
                if not isinstance(self.ub, np.ndarray):
                    self.ub = np.ones(n_var) * ub
                self.ub = self.ub.astype(float)

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

class DTLZ(DTLZ_Problem):
    def __init__(self, n_var, n_obj, k=None, **kwargs):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, lb=0, ub=1, vtype=float, **kwargs)

    def g1(self, X_M):
        return 100 * (self.k + th.sum(th.square(X_M - 0.5) - th.cos(20 * math.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return th.sum(th.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= th.prod(th.cos(th.pow(X_[:, :X_.shape[1] - i], alpha) * math.pi / 2.0), axis=1)
            if i > 0:
                _f *= th.sin(th.pow(X_[:, X_.shape[1] - i], alpha) * math.pi / 2.0)

            f.append(_f)

        f = th.column_stack(f)
        return f

class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= th.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return th.column_stack(f)

    def eval(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g)
        return out

    def get_ref_set(self,n_ref_points=1000): # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / 2
        return referenceObjV

class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def eval(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out= self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000): # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV

class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def eval(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000): # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV

class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)
        self.alpha = alpha
        self.d = d


    def eval(self, x,  *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out =  self.obj_func(X_, g, alpha=self.alpha)
        return out

    def get_ref_set(self,n_ref_points=1000): # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        uniformPoint, ans = ea.crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV

class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def eval(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = th.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out
    def get_ref_set(self,n_ref_points=1000):
        # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        P = th.vstack([th.linspace(0, 1, N), th.linspace(1, 0, N)]).T
        P = P / th.tile(th.sqrt(th.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = th.hstack([P[:, th.zeros(self.n_obj - 2, dtype=th.long)], P])
        referenceObjV = P / th.sqrt(th.tensor(2, dtype=th.float32)) ** th.tile(th.hstack([th.tensor(self.n_obj - 2), th.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV

class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def eval(self, x, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = th.sum(th.pow(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = th.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out

    def get_ref_set(self,n_ref_points = 1000):
        N = n_ref_points  #
        P = th.vstack([th.linspace(0, 1, N), th.linspace(1, 0, N)]).T
        P = P / th.tile(th.sqrt(th.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = th.hstack([P[:, th.zeros(self.n_obj - 2, dtype=th.long)], P])
        referenceObjV = P / th.sqrt(th.tensor(2,dtype=th.float32)) ** th.tile(th.hstack([th.tensor(self.n_obj - 2), th.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV

class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def eval(self, x,*args, **kwargs):
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = th.column_stack(f)

        g = 1 + 9 / self.k * th.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - th.sum(f / (1 + g[:, None]) * (1 + th.sin(3 * math.pi * f)), axis=1)

        out = th.column_stack([f, (1 + g) * h])
        return out
    def get_ref_set(self,n_ref_points = 1000):
        # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points  # 欲生成10000个全局帕累托最优解
        # 参数a,b,c为求解方程得到，详见DTLZ7的参考文献
        a = 0.2514118360889171
        b = 0.6316265307000614
        c = 0.8594008566447239
        Vars, Sizes = ea.crtgp(self.n_obj - 1, N)  # 生成单位超空间内均匀的网格点集
        Vars = th.tensor(Vars)
        middle = 0.5
        left = Vars <= middle
        right = Vars > middle
        maxs_Left = th.max(Vars[left])
        if maxs_Left > 0:
            Vars[left] = Vars[left] / maxs_Left * a
        Vars[right] = (Vars[right] - middle) / (th.max(Vars[right]) - middle) * (c - b) + b
        P = th.hstack([Vars, (2 * self.n_obj - th.sum(Vars * (1 + th.sin(3 * math.pi * Vars)), 1, keepdims=True))])
        referenceObjV = P
        return referenceObjV
 
