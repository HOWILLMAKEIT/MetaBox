from typing import Any
import torch
from torch import nn
from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

def vector2nn(x,net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.Tensor(x[ptr: ptr+num_of_params])
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net

class Policy(nn.Module):
    """
    # Introduction
    The `Policy` class implements a neural network-based policy for evolutionary optimization, utilizing attention mechanisms for selection and adaptation of candidate solutions. It is designed to operate on populations of solutions, transforming fitness values and controlling mutation step sizes in an adaptive manner.
    # Args:
    - pop_size (int): The size of the population.
    - mu (float, optional): Mean for weight initialization. Default is 0.
    - sigma (float, optional): Standard deviation for weight initialization. Default is 1.0.
    - DK (int, optional): Dimensionality of the key/query/value vectors in attention layers. Default is 16.
    - device (torch.device or str, optional): Device on which tensors are allocated (e.g., 'cpu' or 'cuda'). Default is None.
    # Attributes:
    - pop_size (int): Population size.
    - mu (float): Mean for weight initialization.
    - sigma (float): Standard deviation for weight initialization.
    - DF (int): Dimensionality of fitness features (fixed at 2).
    - D_sigma (int): Dimensionality of sigma features (fixed at 1).
    - DK (int): Dimensionality of key/query/value vectors.
    - device (torch.device or str): Device for computation.
    - W_QP, W_KC, W_VC, W_QS, W_KS, W_QM, W_KM, W_VM, W_sigma (nn.Linear): Linear layers for attention mechanisms.
    # Methods:
    - _init_weights(mu, sigma): Initializes the weights and biases of all linear layers with a normal distribution.
    - trans_F(f): Transforms fitness values into standardized z-scores and scaled ranks.
    - adaptation(fitness, sigma): Computes adaptive mutation step sizes using attention over fitness and sigma features.
    - selection(fitness_c, fitness_p): Performs selection among candidate and parent fitness values using attention, returning a one-hot selection mask.
    # Returns:
    - adaptation: Returns updated sigma values for each individual in the population.
    - selection: Returns a one-hot encoded tensor indicating selected individuals.
    # Notes:
    This class is intended for use in evolutionary algorithms where neural attention mechanisms are leveraged for adaptive selection and mutation. It requires PyTorch and is designed to be compatible with GPU acceleration.
    """
    
    def __init__(self, pop_size, mu = 0, sigma = 1.0, DK = 16, device = None):
        super(Policy, self).__init__()
        self.pop_size = pop_size
        self.mu = mu
        self.sigma = sigma
        self.DF = 2
        self.D_sigma = 1
        self.DK = DK
        self.device = device

        # Linear layers with bias
        self.W_QP = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48
        self.W_KC = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48
        self.W_VC = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48

        self.W_QS = nn.Linear(DK, DK, bias = True) # 256 + 16 = 272
        self.W_KS = nn.Linear(self.DF, DK, bias = True) # 32 + 16 = 48

        self.W_QM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64
        self.W_KM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64
        self.W_VM = nn.Linear(self.DF + self.D_sigma, DK, bias = True) # 48 + 16 = 64

        self.W_sigma = nn.Linear(DK, self.D_sigma, bias = True) # 16 + 1 = 17

        # Apply custom initialization
        self._init_weights(self.mu, self.sigma)

    def _init_weights(self, mu, sigma):
        """
        # Introduction
        Initializes the weights and biases of specific neural network layers using a normal distribution.
        # Args:
        - mu (float): The mean value for the normal distribution used to initialize weights and biases.
        - sigma (float): The standard deviation for the normal distribution used to initialize weights and biases.
        # Details:
        Iterates over a predefined list of layer attributes (`self.W_QP`, `self.W_KC`, `self.W_VC`, `self.W_QS`, `self.W_KS`, `self.W_QM`, `self.W_KM`, `self.W_VM`, `self.W_sigma`) and applies normal initialization to both their `weight` and `bias` parameters.
        """
        
        for layer in [
            self.W_QP, self.W_KC, self.W_VC,
            self.W_QS, self.W_KS,
            self.W_QM, self.W_KM, self.W_VM,
            self.W_sigma
        ]:
            nn.init.normal_(layer.weight, mean = mu, std = sigma)
            nn.init.normal_(layer.bias, mean = mu, std = sigma)

    def trans_F(self, f):
        """
        # Introduction
        Transforms the input tensor by computing its z-score normalization and scaled rank, returning both as a stacked tensor.
        # Args:
        - f (torch.Tensor): A 1D tensor of numerical values to be transformed.
        # Returns:
        - torch.Tensor: A 2D tensor of shape [N, 2], where the first column contains the z-score normalized values and the second column contains the scaled ranks.
        # Notes:
        - The z-score is computed as (f - mean) / (std + 1e-8) for numerical stability.
        - The scaled rank is computed such that it is centered around zero.
        """
        
        z_score = (f - f.mean()) / (f.std() + 1e-8)
        ranks = torch.argsort(torch.argsort(-1 * z_score))
        scaled_rank = ranks / (len(ranks) - 1) - 0.5

        return torch.stack([z_score, scaled_rank], dim=1) # [NP, 2]

    def adaptation(self, fitness, sigma):
        """
        # Introduction
        Adapts the mutation step size (`sigma`) for an evolutionary algorithm using an attention-based neural network mechanism.
        # Args:
        - fitness (array-like): The fitness values of the population, shape [NP].
        - sigma (array-like): The mutation step sizes for the population, shape [NP].
        # Returns:
        - torch.Tensor: The adapted mutation step sizes, shape [NP], after applying the learned adaptation.
        # Notes:
        - This method converts the input arrays to PyTorch tensors and processes them on the configured device.
        - It uses neural network layers (`W_KM`, `W_QM`, `W_VM`, `W_sigma`) and an attention mechanism to compute the adaptation.
        - The adaptation is applied multiplicatively to the original `sigma`.
        """
        
        # 先变 torch
        fitness = torch.Tensor(fitness).to(self.device)
        sigma = torch.Tensor(sigma).to(self.device)

        F_P = self.trans_F(fitness)
        F_M = torch.cat([F_P, sigma.unsqueeze(1)], dim = 1) # [NP, 3]

        K_M = self.W_KM(F_M) # [NP, DK]
        Q_M = self.W_QM(F_M) # [NP, DK]
        V_M = self.W_VM(F_M) # [NP, DK]

        scale = Q_M.shape[-1] ** 0.5
        A_M = torch.softmax(torch.matmul(Q_M, K_M.T) / scale, dim = 1)
        A_M = torch.matmul(A_M, V_M) # [NP, DK]

        delta_sigma = torch.exp(0.5 * self.W_sigma(A_M))[:, 0] # [NP]

        return delta_sigma * sigma

    def selection(self, fitness_c, fitness_p):
        """
        # Introduction
        Performs a selection operation using attention mechanisms on child and parent fitness values, producing a one-hot encoded selection matrix.
        # Args:
        - fitness_c (array-like): Fitness values of the child population.
        - fitness_p (array-like): Fitness values of the parent population.
        # Returns:
        - torch.Tensor: A one-hot encoded tensor of shape [E, NP + 1], representing the selection outcome for each entity.
        # Notes:
        - The method applies linear transformations and attention mechanisms to compute selection probabilities.
        - The last column in the output corresponds to a special selection (e.g., "no selection" or "new individual").
        """
        
        # 先变 torch
        fitness_c = torch.Tensor(fitness_c).to(self.device)
        fitness_p = torch.Tensor(fitness_p).to(self.device)

        F_C = self.trans_F(fitness_c)
        F_P = self.trans_F(fitness_p)

        K_C = self.W_KC(F_C) # [N, DK]
        Q_P = self.W_QP(F_P) # [E, DK]
        V_C = self.W_VC(F_C) # [N, DK]

        scale = Q_P.shape[-1] ** 0.5
        A_S = torch.softmax(torch.matmul(Q_P, K_C.T) / scale, dim = 1)
        A_S = torch.matmul(A_S, V_C)

        Q_S = self.W_QS(A_S) # [E, DK]
        K_S = self.W_KS(F_C) # [N, DK]

        M_S = torch.matmul(Q_S, K_S.T) / scale
        # 创建一个全是 1 的列，shape [NP, 1]
        ones_column = torch.ones(M_S.size(0), 1, device = M_S.device)

        # 拼接到 attn_scores 的最后一列
        M_S = torch.cat((M_S, ones_column), dim = 1)  # [E, NP+1]
        M_S = torch.softmax(M_S, dim = 1) # [E, NP + 1]

        idx = torch.distributions.Categorical(probs = M_S).sample() # [E]

        S = torch.nn.functional.one_hot(idx, num_classes = M_S.size(1)).float() # [E, NP + 1]
        return S



class LGA_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    **L**earned **G**enetic **A**lgorithm parametrizes selection and mutation rate adaptation as cross- and self-attention modules and use MetaBBO to evolve their parameters on a set of diverse optimization tasks.
    # Original paper
    "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. (2023).
    # Official Implementation
    [LGA](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/lga.py)

    # Args:
    - config (object): Configuration object containing optimizer parameters such as maximum function evaluations (`maxFEs`), logging interval (`log_interval`), device specification (`device`), and metadata options (`full_meta_data`, `n_logpoint`).
    # Attributes:
    - NP (int): Population size.
    - MaxFEs (int): Maximum number of function evaluations allowed.
    - policy (Policy): Neural network-based policy for adaptation and selection.
    - fes (int): Current number of function evaluations.
    - cost (list): Log of best costs found during optimization.
    - log_index (int): Current logging index.
    - log_interval (int): Interval for logging progress.
    - population (np.ndarray): Current population of candidate solutions.
    - sigma (np.ndarray): Mutation step sizes for each individual.
    - c_cost (np.ndarray): Current costs for each individual in the population.
    - fitness (np.ndarray): Fitness values for the population.
    - gbest_val (float): Best cost found so far.
    - init_gbest (float): Initial best cost.
    - meta_X (list): (Optional) History of populations for metadata logging.
    - meta_Cost (list): (Optional) History of costs for metadata logging.
    # Methods:
    - __str__(): Returns the string representation of the optimizer.
    - get_costs(position, problem): Evaluates the cost of given positions for a problem.
    - get_state(): Returns the current fitness state.
    - softmax(x): Computes the softmax of input array `x`.
    - init_population(problem): Initializes the population and related attributes for a given problem.
    - update(action, problem): Performs one or more optimization steps using the provided action (policy network and skip step), updates the population, and logs progress.
    # Returns (from update):
    - fitness (np.ndarray): Updated fitness values after the optimization step(s).
    - reward (float): Relative improvement in best cost during the update.
    - is_end (bool): Whether the optimization process has reached its end condition.
    - info (dict): Additional information (currently empty).
    # Raises:
    - None explicitly, but may raise exceptions from underlying numpy or neural network operations if inputs are invalid.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.NP = 16

        self.MaxFEs = config.maxFEs

        self.policy = Policy(self.NP, 0, 1, 16, self.config.device)

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        return "LGA_Optimizer"

    def get_costs(self, position, problem):
        """
        # Introduction
        Calculates the cost of a given position for an optimization problem, optionally adjusting by the known optimum.
        # Args:
        - position (numpy.ndarray): A 2D array of shape (NP, dim) representing the population to be evaluated, where NP is the number of individuals (population size) and dim is the problem dimension. Each row represents an individual's position in the search space.
        - problem (object): The optimization problem instance, which must have `eval(position)` and `optimum` attributes.
        # Returns:
        - float: The cost associated with the given position. If the problem's optimum is known, returns the difference between the evaluated cost and the optimum; otherwise, returns the evaluated cost.
        # Raises:
        - AttributeError: If the `problem` object does not have the required `eval` method or `optimum` attribute.
        """
        
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        return cost

    def get_state(self):
        """
        # Introduction
        Retrieves the current fitness value representing the state of the optimizer.
        # Returns:
        - float: The current fitness value of the optimizer.
        """
        
        return self.fitness

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis = 0) + 1e-8)

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the optimizer based on the given problem definition.
        # Args:
        - problem: An object representing the optimization problem, expected to have attributes `dim` (int), `ub` (upper bounds), and `lb` (lower bounds).
        # Returns:
        - None
        # Side Effects:
        - Initializes or updates the following instance attributes:
            - self.fes: Function evaluation counter.
            - self.population: The initial population matrix.
            - self.sigma: Standard deviation array for the population.
            - self.c_cost: Costs of the initial population.
            - self.fitness: Fitness values for the population.
            - self.gbest_val: Best cost value found so far.
            - self.cost: List of best cost values per generation.
            - self.log_index: Logging index for tracking progress.
            - self.init_gbest: Initial best cost value.
            - self.meta_X, self.meta_Cost: (If enabled) Metadata for population and costs.
        # Notes:
        - Uses a random number generator (`self.rng`) to initialize the population.
        - Assumes the existence of `get_costs` and `softmax` methods.
        - If `self.config.full_meta_data` is True, stores additional metadata for analysis.
        """
        
        self.fes = 0
        dim = problem.dim

        self.population = (problem.ub - problem.lb) * self.rng.rand(self.NP, dim) + problem.lb
        self.sigma = np.ones(self.NP) * 0.2

        self.c_cost = self.get_costs(self.population, problem)

        self.fitness = (1 - self.softmax(self.c_cost)) / (self.NP - 1)

        self.fes += self.NP

        self.gbest_val = np.min(self.c_cost)

        self.cost = [self.gbest_val]
        self.log_index = 1

        self.init_gbest = self.gbest_val

        if self.config.full_meta_data:
            self.meta_X = [self.population.copy()]
            self.meta_Cost = [self.c_cost.copy()]

        return None

    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's state based on the provided action and problem definition. This method performs one or more optimization steps, updating the population, fitness, and other internal variables according to the current policy and the results of the optimization process.
        # Args:
        - action (dict): A dictionary containing the current policy network ('net') and optional 'skip_step' parameter. The policy network is used for adaptation and selection during the optimization process.
        - problem (object): An object representing the optimization problem, which must have attributes such as `dim` (problem dimensionality), `lb` (lower bounds), `ub` (upper bounds), and optionally `optimum`.
        # Returns:
        - tuple:
            - fitness (np.ndarray): The updated fitness values of the population after the optimization step(s).
            - improvement (float): The relative improvement in the best cost value from the initial to the current generation.
            - is_end (bool): A flag indicating whether the optimization process has reached its termination condition.
            - info (dict): Additional information about the optimization process (currently an empty dictionary).
        # Notes:
        - The method supports early stopping based on the number of function evaluations (`MaxFEs`), the minimum cost achieved, or a specified number of steps (`skip_step`).
        - The optimizer maintains logs of the best cost values and, if configured, full meta-data about the population and costs at each step.
        - The policy network is updated using the provided action, and is used for both mutation adaptation and selection.
        """
        
        # action 是 网络

        self.policy = vector2nn(action['net'], self.policy).to(self.config.device)

        skip_step = action['skip_step']

        step = 0
        is_end = False
        init_y = None
        dim = problem.dim

        while not is_end:
            indices = self.rng.choice(np.arange(self.NP), size = self.NP, replace = True, p = self.fitness)

            population = self.population[indices]

            sigma = self.sigma[indices]
            fitness = self.fitness[indices]
            c_cost = self.c_cost[indices]

            # cal MRA

            sigma_C = self.policy.adaptation(fitness, sigma).detach().cpu().numpy()  # [NP]
            sigma_C_dim = np.tile(sigma_C[:, None], (1, dim))

            # mutate
            child_population = population + sigma_C_dim * self.rng.randn(self.NP, dim)  # [NP, dim]

            child_population = np.clip(child_population, problem.lb, problem.ub)


            child_c_cost = self.get_costs(child_population, problem)
            self.fes += self.NP

            child_fitness = (1 - self.softmax(child_c_cost)) / (self.NP - 1)  # [NP]

            S = self.policy.selection(child_fitness, fitness).detach().cpu().numpy()  # [E, NP + 1]

            self.population = S[:, :self.NP] @ child_population + np.diag(S[:, -1]) @ population

            self.sigma = S[:, :self.NP] @ sigma_C + np.diag(S[:, -1]) @ sigma

            self.fitness = S[:, :self.NP] @ child_fitness + np.diag(S[:, -1]) @ fitness

            self.fitness = self.softmax(self.fitness)

            self.c_cost = S[:, :self.NP] @ child_c_cost + np.diag(S[:, -1]) @ c_cost

            if step == 0:
                init_y = np.min(self.c_cost)

            step += 1

            self.gbest_val = np.minimum(self.gbest_val, np.min(self.c_cost))

            if problem.optimum is None:
                is_end = (self.fes >= self.MaxFEs)
            else:
                is_end = (self.fes >= self.MaxFEs or np.min(self.c_cost) <= 1e-8)

            if skip_step is not None:
                is_end = is_end or step >= skip_step

            if self.fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest_val)

            if self.config.full_meta_data:
                self.meta_X.append(self.population.copy())
                self.meta_Cost.append(self.c_cost.copy())

            if is_end:
                if len(self.cost) >= self.config.n_logpoint + 1:
                    self.cost[-1] = self.gbest_val
                else:
                    self.cost.append(self.gbest_val)
        info = {}
        return self.fitness, (init_y - self.gbest_val) / init_y, is_end, info






















    # ?????????????
        dim = problem.dim
        # sample
        indices = self.rng.choice(np.arange(self.NP), size = self.NP, replace = True, p = self.fitness)

        population = self.population[indices]

        sigma = self.sigma[indices]
        fitness = self.fitness[indices]
        c_cost = self.c_cost[indices]

        # cal MRA

        sigma_C = action.adaptation(fitness, sigma).detach().cpu().numpy() # [NP]
        sigma_C_dim = np.tile(sigma_C[:, None], (1, dim))

        # mutate
        child_population = population + sigma_C_dim * self.rng.randn((self.NP, dim)) # [NP, dim]

        child_c_cost = self.get_costs(child_population, problem)
        self.fes += self.NP

        child_fitness = (1 - self.softmax(child_c_cost)) / (self.NP - 1) # [NP]

        S = action.selection(child_fitness, fitness).detach().cpu().numpy() # [E, NP + 1]

        self.population = S[:, :self.NP] @ child_population + np.diag(S[:, -1]) @ population

        self.sigma = S[:, :self.NP] @ sigma_C + np.diag(S[:, -1]) @ sigma

        self.fitness = S[:, :self.NP] @ child_fitness + np.diag(S[:, -1]) @ fitness

        self.c_cost = S[:, :self.NP] @ child_c_cost + np.diag(S[:, -1]) @ c_cost

        new_gbest_val = np.min(self.c_cost)

        reward = (pre_gbest_val - new_gbest_val) / self.init_gbest

        self.gbest_val = np.minimum(pre_gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        if self.config.full_meta_data:
            self.meta_X.append(self.population.copy())
            self.meta_Cost.append(self.c_cost.copy())

        next_state = self.get_state()

        if self.fes >= self.log_interval * self.log_index:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info


