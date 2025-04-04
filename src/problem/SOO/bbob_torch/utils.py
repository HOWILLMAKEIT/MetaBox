import torch
import numpy as np

def sr_func(x, Os, Mr):   #shift and rotate
    y = x[:, :Os.shape[-1]] - Os
    return torch.matmul(Mr, y.t()).t()


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return torch.tensor(H, dtype=torch.float64)

def osc_transform(x):
    """
    Implementing the oscillating transformation on objective values or/and decision values.

    :param x: If x represents objective values, x is a 1-D array in shape [NP] if problem is single objective,
              or a 2-D array in shape [NP, number_of_objectives] if multi-objective.
              If x represents decision values, x is a 2-D array in shape [NP, dim].
    :return: The array after transformation in the shape of x.
    """
    y = x.clone()
    idx = (x > 0.)
    y[idx] = torch.log(x[idx]) / 0.1
    y[idx] = torch.exp(y[idx] + 0.49 * (torch.sin(y[idx]) + torch.sin(0.79 * y[idx]))) ** 0.1
    idx = (x < 0.)
    y[idx] = torch.log(-x[idx]) / 0.1
    y[idx] = -torch.exp(y[idx] + 0.49 * (torch.sin(0.55 * y[idx]) + torch.sin(0.31 * y[idx]))) ** 0.1
    return y


def asy_transform(x, beta):
    """
    Implementing the asymmetric transformation on decision values.

    :param x: Decision values in shape [NP, dim].
    :param beta: beta factor.
    :return: The array after transformation in the shape of x.
    """
    NP, dim = x.shape
    idx = (x > 0.)
    y = x.clone()
    y[idx] = y[idx] ** (
                1. + beta * torch.linspace(0, 1, dim, dtype=torch.float64).reshape(1, -1).repeat(repeats=(NP, 1))[idx] * torch.sqrt(y[idx]))
    return y


def pen_func(x, ub):
    """
    Implementing the penalty function on decision values.

    :param x: Decision values in shape [NP, dim].
    :param ub: the upper-bound as a scalar.
    :return: Penalty values in shape [NP].
    """
    return torch.sum(torch.maximum(torch.tensor(0., dtype=torch.float64), torch.abs(x) - ub) ** 2, dim=-1)
