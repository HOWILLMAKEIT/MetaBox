import random

from torch.utils.data import Dataset
import numpy as np
from .UF_numpy import *
from .ZDT_numpy import *
from .DTLZ_numpy import *
from .WFG_numpy import *



class MOO_Dataset(Dataset):
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
    def get_datasets(suit,
                     n_objs,
                     n_vars,
                     train_batch_size=1,
                     test_batch_size=1,
                     train_size=0.7,
                     test_size=0.3,
                     instance_seed=3849):
        # get functions ID of indicated suit
        instance_set = []
        if 'WFG' in suit:
            func_id = [i for i in range(1, 10)]#WFG1-9
            for id in func_id:
                for n_obj in n_objs:
                    for n_var in n_vars:
                        instance_set.append(eval(f"WFG{id}")(n_obj = n_obj,n_var = n_var))
        elif suit == 'DTLZ':
            func_id = [i for i in range(1, 8)]#DTLZ1-7
            for id in func_id:
                for n_obj in n_objs:
                    for n_var in n_vars:
                        instance_set.append(eval(f"DTLZ{id}")(n_obj = n_obj,n_var = n_var))
        elif suit == 'UF':
            func_id = [i for i in range(1,11)]#UF1-10
            for id in func_id:
                instance_set.append(eval(f"DTLZ{id}")())
        elif suit == 'ZDT':
            func_id = [i for i in range(1,7)]#ZDT1-6
            for id in func_id:
                instance_set.append(eval(f"ZDT{id}")())
        else:
            raise ValueError(f'{suit} function suit is invalid or is not supported yet.')
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
        train_set = []
        test_set = []
        random.shuffle(instance_set)
        for instance in instance_set:
            train_set.append(instance.PF[:train_size])
            test_set.append(instance.PF[train_size:])
        return MOO_Dataset(train_set, train_batch_size), MOO_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'MOO_Dataset'):
        return MOO_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
