from torch.utils.data import Dataset
import numpy as np
from .cec2013lsgo import *

class CEC2013LSGO_Dataset(Dataset):
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy'):
        if difficulty != 'easy' and difficulty != 'difficult' and difficulty != 'user':
            raise ValueError(f'{difficulty} difficulty is invalid.')
        func_id = [i for i in range(1, 6)]
        train_set = []
        test_set = []
        if difficulty == 'easy':
            train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            for id in func_id:
                instance = eval(f'F{id}')()
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)
        elif difficulty == 'difficult':
            train_id = [7, 8, 9, 10, 11, 12, 13, 14, 15]
            for id in func_id:
                instance = eval(f'F{id}')()
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)

        return CEC2013LSGO_Dataset(train_set, train_batch_size), CEC2013LSGO_Dataset(test_set, test_batch_size)

    # get a batch of data
    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    # get the number of data
    def __len__(self):
        return self.N

    def __add__(self, other: 'CEC2013LSGO_Dataset'):
        return CEC2013LSGO_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)