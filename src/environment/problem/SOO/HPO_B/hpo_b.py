import numpy as np
import xgboost as xgb
import pickle
import os, time
from ....problem.basic_problem import Basic_Problem


class HPOB_Problem(Basic_Problem):
    """
    # Introduction
    Represents a single-objective optimization problem using a surrogate model (e.g., XGBoost) for function evaluations, with optional normalization and tracking of the global best solution.
    # Args:
    - bst_surrogate: The trained surrogate model used to predict objective values (e.g., an XGBoost model).
    - dim (int): Dimensionality of the input search space.
    - y_min (float or None): Minimum value for normalization. If None, normalization uses the minimum of the predicted values.
    - y_max (float or None): Maximum value for normalization. If None, normalization uses the maximum of the predicted values.
    - lb (array-like): Lower bounds of the search space.
    - ub (array-like): Upper bounds of the search space.
    - normalized (bool, optional): Whether to normalize the predicted objective values. Defaults to False.
    # Attributes:
    - bst_surrogate: The surrogate model for predictions.
    - y_min: Minimum value for normalization.
    - y_max: Maximum value for normalization.
    - dim: Dimensionality of the problem.
    - gbest: Current global best value found.
    - normalized: Indicates if normalization is applied.
    - collect_gbest: List of global best values collected during optimization.
    - lb: Lower bounds of the search space.
    - ub: Upper bounds of the search space.
    - optimum: Stores the optimum solution found (if any).
    # Methods:
    - func(position): Evaluates the surrogate model at the given position, applies normalization if enabled, updates and records the global best value, and returns the (possibly normalized) cost.
    - normalize(y): Normalizes the input value(s) `y` based on provided or computed min/max values if normalization is enabled; otherwise, returns `y` unchanged.
    # Returns:
    - func(position): Returns the (possibly normalized) cost at the given position.
    """
    
    def __init__(self,bst_surrogate,dim,y_min,y_max,lb,ub,normalized=False) -> None:
        self.bst_surrogate=bst_surrogate
        self.y_min=y_min
        self.y_max=y_max
        self.dim=dim
        self.gbest=1e+10
        self.normalized = normalized
        self.collect_gbest=[]
        self.lb = lb
        self.ub = ub
        self.optimum = None
    def func(self,position):
        x_q = xgb.DMatrix(position.reshape(-1,self.dim))
        new_y = self.bst_surrogate.predict(x_q)
        cost=-self.normalize(new_y)
        self.gbest=np.minimum(self.gbest,cost)
        self.collect_gbest.append(self.gbest)
        return cost

    def normalize(self, y):
        if self.normalized:
            if self.y_min is None:
                return (y-np.min(y))/(np.max(y)-np.min(y))
            else:
                return np.clip((y-self.y_min)/(self.y_max-self.y_min),0,1)
        return y
