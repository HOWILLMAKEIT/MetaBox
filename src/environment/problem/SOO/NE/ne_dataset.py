from torch.utils.data import Dataset
import sys
import subprocess
import numpy as np
from .evox_ne import *


class NE_Dataset(Dataset):
    """
    # Args:
    - data (list): A list of NE_Problem instances or similar data objects.
    - batch_size (int, optional): Number of samples per batch. Defaults to 1.
    # Attributes:
    - data (list): The dataset containing NE_Problem instances.
    - batch_size (int): The batch size for data loading.
    - N (int): Total number of data samples.
    - ptr (list): List of starting indices for each batch.
    - index (np.ndarray): Array of indices for shuffling and sampling.
    - maxdim (int): Maximum dimension among all data items.
    # Methods:
    - get_datasets(train_batch_size=1, test_batch_size=1, difficulty='easy', user_train_list=None, user_test_list=None, instance_seed=3849):
        Static method to generate training and testing NE_Dataset instances based on difficulty or user-defined splits.
    - __getitem__(item):
        Returns a batch of data corresponding to the batch index.
    - __len__():
        Returns the total number of data samples.
    - __add__(other):
        Concatenates two NE_Dataset instances.
    - shuffle():
        Randomly permutes the order of data indices for shuffling.
    # Returns:
    - NE_Dataset: An instance of the NE_Dataset class.
    # Raises:
    - AssertionError: If the provided difficulty is not one of ['all', 'easy', 'difficult', 'user-define'].
    - NotImplementedError: If user-defined lists are not provided for 'user-define' difficulty.
    """
    
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     user_train_list = None,
                     user_test_list = None,
                     instance_seed=3849):
        assert difficulty in ['all','easy','difficult','user-define']
        train_set = []
        test_set = []
        if difficulty == 'all':
            for env in envs.keys():
                for depth in model_depth:
                    train_set.append(NE_Problem(env, depth, instance_seed))
                    test_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
            
        elif difficulty == 'easy':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        test_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        train_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
        
        elif difficulty == 'difficult':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        train_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        test_set.append(NE_Problem(env, depth, instance_seed))
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)
        
        elif difficulty == 'user-define':
            for env in envs.keys():
                for depth in model_depth:
                    if user_train_list is not None and user_test_list is not None:
                        if f'{env}-{depth}' in user_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        if f'{env}-{depth}' in user_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif user_train_list is not None:
                        if f'{env}-{depth}' in user_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif user_test_list is not None:
                        if f'{env}-{depth}' in user_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        raise NotImplementedError
                
            return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'NE_Dataset'):
        return NE_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)

