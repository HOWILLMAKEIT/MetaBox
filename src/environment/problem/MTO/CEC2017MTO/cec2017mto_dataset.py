from .cec2017mto_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .cec2017mto_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch,Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
import scipy.io as sio
import importlib.resources as pkg_resources

def mat2np(path):
    with path.open('rb') as f:
        data = sio.loadmat(f)
    return data


class CEC2017MTO_Tasks():
    def __init__(self, tasks):
        self.tasks = tasks
        self.T1 = None
        self.dim = 0
    
    def reset(self):
        for _ in range(len(self.tasks)):
            self.dim = max(self.dim, self.tasks[_].dim)

        for _ in range(len(self.tasks)):
            self.tasks[_].reset()
        self.T1 = 0
    
    def __str__(self):
        name = ''
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            name += task.__str__()
        return name

    def update_T1(self):
        eval_time = 0
        for _ in range(len(self.tasks)):
            eval_time += self.tasks[_].T1
        self.T1 = eval_time

class CEC2017MTO_Dataset(Dataset):
    """
    # Introduction
      CEC2017MTO proposes 9 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
      "[Evolutionary Multitasking for Single-objective Continuous Optimization: Benchmark Problems, Performance Metric, and Bseline Results](https://arxiv.org/pdf/1706.03470)."
    # Official Implementation
      [CEC2017MTO](http://www.bdsc.site/websites/MTO/index.html)
    # License
    None
    # Problem Suite Composition
      The CEC2017MTO problem suite contains a total of 9 benchmark problems, each consisting of two basic functions.
      These nine benchmark problems are classified according to the degree of intersection and the inter-task similarity between the two constitutive functions:
        P1. Complete intersection and high similarity(CI+HS)
        P2. Complete intersection and medium similarity(CI+MS)  
        P3. Complete intersection and low similarity(CI+LS)
        P4. Partial intersection and high similarity(PI+HS)
        P5. Partial intersection and medium similarity(PI+MS)  
        P6. Partial intersection and low similarity(PI+LS)
        P7. No intersection and high similarity(NI+HS)
        P8. No intersection and medium similarity(NI+MS) 
        P9. No intersection and low similarity(NI+LS)
    # Args:
    - `data` (list): A list of task datasets, where each dataset contains multiple tasks.
    - `batch_size` (int, optional): The size of each batch for data retrieval. Defaults to 1.
    # Attributes:
    - `data` (list): The dataset containing tasks for the CEC2017MTO problem suite.
    - `batch_size` (int): The size of each batch for data retrieval.
    - `maxdim` (int): The maximum dimensionality across all tasks in the dataset.
    - `N` (int): The total number of datasets in the collection.
    - `ptr` (list): A list of indices for batching the dataset.
    - `index` (numpy.ndarray): An array of shuffled indices for dataset access.
    # Methods:
    - `__getitem__(item)`: Retrieves a batch of data based on the specified index.
    - `__len__()`: Returns the total number of datasets in the collection.
    - `__add__(other)`: Combines the current dataset with another `CEC2017MTO_Dataset` instance.
    - `shuffle()`: Randomly shuffles the dataset indices.
    - `get_datasets(version, train_batch_size, test_batch_size, difficulty, user_train_list, user_test_list)`: 
        Static method to generate training and testing datasets based on the specified difficulty level or user-defined task lists.
    # Raises:
    - `ValueError`: Raised in the following cases:
        - If `difficulty`, `user_train_list`, and `user_test_list` are all `None`.
        - If an invalid `difficulty` value is provided.
    """

    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.maxdim = 0
        for data_lis in self.data:
            for item in data_lis.tasks:
                self.maxdim = max(self.maxdim, item.dim)
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res
    
    def __len__(self):
        return self.N

    
    def __add__(self, other: 'CEC2017MTO_Dataset'):
        return CEC2017MTO_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


    @staticmethod
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        
        folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
        func_id = [i for i in range(0, 9)]
        if difficulty == 'easy':
            train_id = [0, 1, 2, 3, 4, 5]
            test_id = [6, 7, 8]
        elif difficulty == 'difficult':
            train_id = [6, 7, 8]
            test_id = [0, 1, 2, 3, 4, 5]
        elif difficulty == 'all':
            train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            test_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif difficulty is None:
            train_id = user_train_list
            test_id = user_test_list
        
        train_set = []
        test_set = []
        for task_ID in func_id:
            Tasks = []
            if task_ID == 0:
                file_name = 'CI_H.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Griewank(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Griewank_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 1:
                file_name = 'CI_M.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 2:
                file_name = 'CI_L.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', None, 'Rotation_Task1',None]
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel(50)
                else :
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel_Torch(50)
                Tasks = [task1, task2]

            if task_ID == 3:
                file_name = 'PI_H.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
                    task2 = Sphere(50, data[keys[1]])
                else:
                    task1 = Rastrigin_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Sphere_Torch(50, data[keys[1]])
                Tasks = [task1, task2]

            if task_ID == 4:
                file_name = 'PI_M.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1',None, 'Rotation_Task1', None]
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Rosenbrock(50)
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rosenbrock_Torch(50)
                Tasks = [task1, task2]

            if task_ID == 5:
                file_name = 'PI_L.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass(25, data[keys[1]], data[keys[3]])
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass_Torch(25, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 6:
                file_name = 'NI_H.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = [None, 'GO_Task2', None, 'Rotation_Task2']
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Rosenbrock(50)
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Rosenbrock_Torch(50)
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]
            
            if task_ID == 7:
                file_name = 'NI_M.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Griewank(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Griewank_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 8:
                file_name = 'NI_L.mat'
                file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                keys = ['GO_Task1',None, 'Rotation_Task1',None]
                data = mat2np(file_path)
                if version == 'numpy':
                    task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel(50)
                else:
                    task1 = Rastrigin_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel_Torch(50)
                Tasks = [task1, task2]
            
            if task_ID in train_id:
                train_set.append(CEC2017MTO_Tasks(Tasks))
            if task_ID in test_id:
                test_set.append(CEC2017MTO_Tasks(Tasks))

        return CEC2017MTO_Dataset(train_set, train_batch_size), CEC2017MTO_Dataset(test_set, test_batch_size)
