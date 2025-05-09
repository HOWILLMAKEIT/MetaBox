from .wcci2020_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .wcci2020_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch, Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
import importlib.resources as pkg_resources

class WCCI2020MTO_Tasks():
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

class WCCI2020_Dataset(Dataset):
    """
    # Introduction
      WCCI2020 proposes 10 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
      [WCCI2020](http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html)
    # License
    None
    # Problem Suite Composition
      The WCCI2020 problem suite contains a total of 10 benchmark problems, each consisting of 50 different basic functions with unique transformations(shifts and rotations).
      For each benchmark problem, fifty basic functions are added sequentially and cyclically to constitute the problem.
      These ten benchmark problems are classified according to the specific combination of different types of basic functions:
        P1: Shpere
        P2: Rosenbrock
        P3: Rastrigin
        P4: Shpere, Rosenbrock, Ackley
        P5: Rastrigin, Griewank, Weierstrass
        P6: Rosenbrock, Griewank, Schwefel
        P7: Rastrigin, Ackley, Weierstrass
        P8: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass
        P9: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
        P10:Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
    # Args:
    - `data` (list): A list of task datasets, where each dataset contains multiple tasks.
    - `batch_size` (int, optional): The size of each batch when retrieving data. Defaults to 1.
    # Attributes:
    - `data` (list): The dataset containing tasks.
    - `batch_size` (int): The size of each batch.
    - `maxdim` (int): The maximum dimensionality across all tasks in the dataset.
    - `N` (int): The total number of task datasets.
    - `ptr` (list): A list of indices for batching.
    - `index` (numpy.ndarray): An array of indices for shuffling and accessing data.
    # Methods:
    - `__getitem__(item)`: Retrieves a batch of tasks based on the given index.
    - `__len__()`: Returns the total number of task datasets.
    - `__add__(other)`: Combines the current dataset with another `WCCI2020_Dataset` instance.
    - `shuffle()`: Shuffles the dataset indices to randomize the order of tasks.
    - `get_datasets(version, train_batch_size, test_batch_size, difficulty, user_train_list, user_test_list)`: A static method to generate training and testing datasets based on the specified difficulty or user-defined task lists.
    # Raises:
    - `ValueError`: Raised in the `get_datasets` method if:
        - Neither `difficulty` nor `user_train_list` and `user_test_list` are provided.
        - An invalid `difficulty` value is specified.
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

    
    def __add__(self, other: 'WCCI2020_Dataset'):
        return WCCI2020_Dataset(self.data + other.data, self.batch_size)

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

        func_id = [i for i in range(0, 10)]
        if difficulty == 'easy':
            train_id = [0, 1, 2, 3, 4, 5]
            test_id = [6, 7, 8, 9]
        elif difficulty == 'difficult':
            train_id = [6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5]
        elif difficulty == 'all':
            train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif difficulty is None:
            train_id = user_train_list
            test_id = user_test_list

        train_set = []
        test_set = []
        for task_ID in func_id:
            dim = 50
            task_size = 50
            choice_functions = []
            if task_ID == 0:
                choice_functions = [1]
            if task_ID == 1:
                choice_functions = [2]
            if task_ID == 2:
                choice_functions = [4]
            if task_ID == 3:
                choice_functions = [1,2,3]
            if task_ID == 4:
                choice_functions = [4,5,6]
            if task_ID == 5:
                choice_functions = [2,5,7]
            if task_ID == 6:
                choice_functions = [3,4,6]
            if task_ID == 7:
                choice_functions = [2,3,4,5,6]
            if task_ID == 8:
                choice_functions = [2,3,4,5,6,7]
            if task_ID == 9:
                choice_functions = [3,4,5,6,7]

            Tasks = []
            for task_id in range(1, task_size+1):
                id = (task_id-1) % len(choice_functions)
                func_id = choice_functions[id]

                folder_package = f"metaevobox.environment.problem.MTO.WCCI2020.datafile.benchmark_{task_ID + 1}"
                shift_file_path = pkg_resources.files(folder_package).joinpath(f'bias_{task_id}')
                rotate_file_path = pkg_resources.files(folder_package).joinpath(f'matrix_{task_id}')
                # folder_dir = os.path.join(os.path.dirname(__file__),'datafile',f'benchmark_{task_ID+1}')
                # shift_file = os.path.join(folder_dir, f'bias_{task_id}')
                # rotate_file = os.path.join(folder_dir, f'matrix_{task_id}')

                with shift_file_path.open('r') as f:
                    shift = np.loadtxt(f)
                with rotate_file_path.open('r') as f:
                    rotate_matrix = np.loadtxt(f)

                if func_id == 1:
                    if version == 'numpy': 
                        task = Sphere(dim,shift, rotate_matrix)
                    else:
                        task = Sphere_Torch(dim,shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 2:
                    if version == 'numpy': 
                        task = Rosenbrock(dim, shift, rotate_matrix)
                    else:
                        task = Rosenbrock_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 3:
                    if version == 'numpy':
                         task = Ackley(dim, shift, rotate_matrix)
                    else:
                        task = Ackley_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 4:
                    if version == 'numpy':
                        task = Rastrigin(dim, shift, rotate_matrix)
                    else:
                        task = Rastrigin_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 5:
                    if version == 'numpy':
                        task = Griewank(dim, shift, rotate_matrix)
                    else:
                        task = Griewank_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 6:
                    if version == 'numpy':
                        task = Weierstrass(dim, shift, rotate_matrix)
                    else:
                        task = Weierstrass_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 7:
                    if version == 'numpy':
                        task = Schwefel(dim, shift, rotate_matrix)
                    else:
                        task = Schwefel_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)
            
            if task_ID in train_id:
                train_set.append(WCCI2020MTO_Tasks(Tasks))
            if task_ID in test_id:
                test_set.append(WCCI2020MTO_Tasks(Tasks))

        return WCCI2020_Dataset(train_set, train_batch_size), WCCI2020_Dataset(test_set, test_batch_size)
