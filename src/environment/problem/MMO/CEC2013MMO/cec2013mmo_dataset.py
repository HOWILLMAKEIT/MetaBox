from torch.utils.data import Dataset
import numpy as np
import torch
from .cec2013mmo_numpy import *
from .cec2013mmo_torch import *
import math

class CEC2013MMO_Dataset(Dataset):
    """
    # Introduction
    CEC2013 MMO benchmark puts together 20 multimodal problems (including several identical functions with different dimension sizes), with different characteristics, for evaluating niching algorithms.
    # Original Paper
    "[Benchmark Functions for CEC’2013 Special Session and Competition on Niching Methods for Multimodal Function Optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/NichingMultimodalOptimization/CEC2015_NichingMethods_TechnicalReport.pdf)"
    # Official Implementation
    [CEC2013MMO](https://github.com/mikeagn/CEC2013)
    # License
    Simplified BSD License
    # Problem Suite Composition
    The CEC2013 MMO problem suite contains 20 optimization problems, each with specific characteristics such as dimensionality, bounds, and multimodal properties. These problems are categorized into different difficulty levels (`easy`, `difficult`, and `all`) and can be used for benchmarking optimization algorithms.
    # Args:
    - `data` (list): A list of problem instances to be included in the dataset.
    - `batch_size` (int, optional): The size of each batch for data retrieval. Defaults to 1.
    # Attributes:
    - `data` (list): The list of problem instances in the dataset.
    - `maxdim` (int): The maximum dimensionality among all problem instances in the dataset.
    - `batch_size` (int): The size of each batch for data retrieval.
    - `N` (int): The total number of problem instances in the dataset.
    - `ptr` (list): A list of indices representing the start of each batch.
    - `index` (numpy.ndarray): An array of indices used for shuffling and accessing data.
    # Methods:
    - `get_datasets(version='numpy', train_batch_size=1, test_batch_size=1, difficulty=None, user_train_list=None, user_test_list=None, instance_seed=3849)`: 
        A static method to generate training and testing datasets based on the specified difficulty or user-defined problem lists.
    - `__getitem__(item)`: Retrieves a batch of problem instances based on the given batch index.
    - `__len__()`: Returns the total number of problem instances in the dataset.
    - `__add__(other)`: Combines two datasets into a single dataset.
    - `shuffle()`: Shuffles the dataset to randomize the order of problem instances.
    # Raises:
    - `ValueError`: Raised in `get_datasets` if neither `difficulty` nor `user_train_list` and `user_test_list` are provided, or if an invalid `difficulty` value is specified.
    """

    def __init__(self, data, batch_size = 1):
        super().__init__()
        self.data = data
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(version = 'numpy',
                    train_batch_size=1,
                    test_batch_size=1,
                    difficulty = None,
                    user_train_list = None,
                    user_test_list = None,
                    instance_seed=3849):

        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')

        problem_id_list = [i for i in range(1, 21)]
        functions = [1, 2, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 11, 12, 11, 12, 11, 12, 12]
        fopt = [-200.0, -1.0, -1.0, -200.0, -1.031628453489877, -186.7309088310239, -1.0, -2709.093505572820, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rho = [0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.2, 0.5, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, ]
        nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
        maxfes = [50000, 50000, 50000, 50000, 50000, 200000, 200000, 400000, 400000, 200000, 200000, 200000, 200000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, ]
        dimensions = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

        if instance_seed > 0:
            np.random.seed(instance_seed)

        if difficulty == 'all':
            train_set_pids = [ids for ids in range(1, 21)]
            test_set_pids = [ids for ids in range(1, 21)]
        elif user_train_list is None and user_test_list is None and difficulty is not None:
            if difficulty == 'easy':
                train_set_pids = [8, 9, 13, 14, 15, 16, 17, 18, 19, 20]
                test_set_pids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
            elif difficulty == 'difficult':
                train_set_pids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                test_set_pids = [8, 9, 13, 14, 15, 16, 17, 18, 19, 20]

        else:
            if user_train_list is not None and user_test_list is not None:
                train_set_pids = user_train_list
                test_set_pids = user_test_list
            elif user_train_list is not None:
                train_set_pids = user_train_list
                test_set_pids = [item for item in range(1, 21) if item not in train_set_pids]
            elif user_test_list is not None:
                test_set_pids = user_test_list
                train_set_pids = [item for item in range(1, 21) if item not in test_set_pids]

        train_set = []
        test_set = []
        for id in problem_id_list:
            # lbound
            if id == 1 or id == 2 or id == 3:
                lb = 0
            elif id == 4:
                lb = -6
            elif id == 5:
                lb = -1.1
            elif id == 6 or id == 8:
                lb = -10
            elif id == 7 or id == 9:
                lb = 0.25
            elif id == 10:
                lb = 0
            elif id > 10:
                lb = -5.0
            # ubound
            if id == 1:
                ub = 30
            elif id == 2 or id == 3:
                ub = 1
            elif id == 4:
                ub = 6
            elif id == 5:
                ub = 1.1
            elif id == 6 or id == 8:
                ub = 10
            elif id == 7 or id == 9:
                ub = 10
            elif id == 10:
                ub = 1
            elif id > 10:
                ub = 5.0

            if version == 'numpy':
                instance = eval(f'F{functions[id - 1]}')(dim = dimensions[id - 1], lb = lb, ub = ub, fopt = fopt[id - 1], rho = rho[id - 1], nopt = nopt[id - 1], maxfes = maxfes[id - 1])
            else:
                instance = eval(f'F{functions[id - 1]}_Torch')(dim = dimensions[id - 1], lb = lb, ub = ub, fopt = fopt[id - 1], rho = rho[id - 1], nopt = nopt[id - 1], maxfes = maxfes[id - 1])

            if id in train_set_pids:
                train_set.append(instance)
            if id in test_set_pids:
                test_set.append(instance)
        return CEC2013MMO_Dataset(train_set, train_batch_size), CEC2013MMO_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: "CEC2013MMO_Dataset"):
        return CEC2013MMO_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
