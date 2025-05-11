# Develop your Problem Set

```{important}
The dataset needs to be constructed in **two versions: torch and numpy**, \
to facilitate the use of different optimizers. \ 
The following is an example of the numpy version
```

1️⃣ Inheritance and Initialization
```python
from metaevobox.environment.problem.basic_problem import Basic_Problem
# torch version : from metaevobox.environment.problem.basic_problem import Basic_Problem_Torch
class MyProblem(Basic_Problem):
# torch version : class MyProblem(Basic_Problem_Torch)
    def __init__(self):
        # Init parameters
        self.opt = None
        self.optimum = 0.0

    def get_optimial(self)
        return self.opt

    def func(self, x):
        raise NotImplementedError
```

2️⃣ Instantiation Function
```python
class MyFunction(MyProblem):
    def __init__(self):
        super().__init__()

        self.opt = # Specific setting

    def __str__(self):
        return 'MyFunction'
    
    def func(self, x):
        # Specific Operation
        return result
```

2️⃣ Instantiation Dataset
```python
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, batch_size=1):
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
        func_id = [i for i in range(1, # The number of functions)]
        train_set = []
        test_set = []

        if difficulty == 'easy':
            train_id = [# Specific setting]
            for id in func_id:
                if version == 'numpy':
                    instance = eval(f'F{id}')()
                else:
                    instance = eval(f'F{id}_Torch')()
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)

         # difficulty == 'difficult' is the same
         elif difficulty == 'all':
             for id in func_id:
                 if version == 'numpy':
                     instance = eval(f'F{id}')()
                 else:
                     instance = eval(f'F{id}_Torch')()
                 train_set.append(instance)
                 test_set.append(instance)
         elif difficulty is None:
             train_id = user_train_list
             test_id = user_test_list
             for id in func_id:
                 if version == 'numpy':
                     instance = eval(f'F{id}')()
                 else:
                     instance = eval(f'F{id}_Torch')()
                 if id in train_id:
                     train_set.append(instance)
                 elif id in test_id:
                     test_set.append(instance)

        return MyDataset(train_set, train_batch_size), MyDataset(test_set, test_batch_size)
    
    # get a batch of data
    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    # get the number of data
    def __len__(self):
        return self.N

    def __add__(self, other: 'MyDataset'):
        return MyDataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
```
