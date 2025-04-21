import numpy as np
import time
import torch

class Basic_Problem:
    """
    Abstract super class for problems and applications.
    """

    def reset(self):
        """
        # Introduction
        Resets the environment state by initializing or clearing relevant attributes.The basic solution is to set T1 to 0.
        # Args:
        None
        # Returns:
        None
        # Raises:
        None
        """
        
        self.T1=0

    def eval(self, x):
        """
        # Introduction
        Evaluates the objective function for either a single individual or a population, adapting the input as needed. Also measures and accumulates the evaluation time.
        # Args:
        - x (array-like or np.ndarray): Input vector(s) representing either a single individual (1D array) or a population (2D array).
        # Returns:
        - float or np.ndarray: The evaluated result(s) from the objective function. Returns a single value for an individual or an array for a population.
        # Raises:
        - None explicitly, but may raise exceptions from the underlying `func` or if input shapes are incompatible.
        """
        
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

class Basic_Problem_Torch(Basic_Problem):
    """
    Abstract super class for problems and applications.
    """

    def reset(self):
        """
        # Introduction
        Resets the environment state by initializing or clearing relevant attributes.
        # Args:
        None
        # Returns:
        None
        """
        
        self.T1 = 0

    def eval(self, x):
        """
        # Introduction
        Evaluates the objective function for a given individual or population, handling input adaptation and timing the evaluation process.
        # Args:
        - x (torch.Tensor or array-like): Input data representing either a single individual (1D) or a population (2D or higher). If not a torch.Tensor, it will be converted.
        # Returns:
        - torch.Tensor: The evaluated result(s) from the objective function, matching the input shape.
        # Notes:
        - Automatically adapts input to the correct shape and dtype (`torch.float64`).
        - Measures and accumulates the evaluation time in milliseconds in `self.T1`.
        - Temporarily sets the default torch device to match the input tensor's device during evaluation.
        """
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        
        torch.set_default_device(x.device)
        start = time.perf_counter()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            y = self.func(x.reshape(1, -1))[0]
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
        elif x.ndim == 2:  # x is a whole population
            y = self.func(x)
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
        else:
            y = self.func(x.reshape(-1, x.shape[-1]))
            end = time.perf_counter()
            self.T1 += (end - start) * 1000
        torch.set_default_device("cpu")
        return y

    def func(self, x):
        """
        # Introduction
        Abstract method to be implemented by subclasses, defining the specific evaluation function for the problem.
        # Args:
        - x: Input parameter to be processed by the function. The type and purpose should be defined in the subclass implementation.
        # Returns:
        - Any: The result of processing `x`. The return type should be specified in the subclass.
        # Raises:
        - NotImplementedError: Always raised to indicate that this method must be implemented by a subclass.
        """
        
        raise NotImplementedError
