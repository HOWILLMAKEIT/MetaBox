import random

from torch.utils.data import Dataset
import numpy as np
from problem.MOO.MOO_synthetic_numpy.zdt_numpy import *
from problem.MOO.MOO_synthetic_numpy.uf_numpy import *
from problem.MOO.MOO_synthetic_numpy.dtlz_numpy import *
from problem.MOO.MOO_synthetic_numpy.wfg_numpy import *


class Moo_Dataset_numpy(Dataset):
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
    def get_datasets(suit=None,
                     n_objs=None,
                     n_vars=None,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     instance_seed=3849):
        # get functions ID of indicated suit
        instance_set = []
        if suit != None:
            if 'WFG' in suit:
                func_id = [i for i in range(1, 10)]  # WFG1-9
                for id in func_id:
                    for n_obj in n_objs:
                        for n_var in n_vars:
                            instance_set.append(eval(f"WFG{id}")(n_obj=n_obj, n_var=n_var))
            if 'DTLZ' in suit:
                func_id = [i for i in range(1, 8)]  # DTLZ1-7
                for id in func_id:
                    for n_obj in n_objs:
                        for n_var in n_vars:
                            instance_set.append(eval(f"DTLZ{id}")(n_obj=n_obj, n_var=n_var))
            if 'UF' in suit:
                func_id = [i for i in range(1, 11)]  # UF1-10
                for id in func_id:
                    instance_set.append(eval(f"DTLZ{id}")())
            if 'ZDT' in suit:
                func_id = [i for i in range(1, 7)]  # ZDT1-6
                func_id.remove(5)
                for id in func_id:
                    for n_var in n_vars:
                        instance_set.append(eval(f"ZDT{id}")(n_var=n_var))
            else:
                raise ValueError(f'{suit} function suit is invalid or is not supported yet.')
        elif suit == None:
            # UF1-7
            for id in range(1, 8):
                instance_set.append(eval(f"UF{id}")())

            # UF8-10
            for id in range(8, 11):
                instance_set.append(eval(f"UF{id}")())

            # ZDT1-3
            for id in range(1, 4):
                instance_set.append(eval(f"ZDT{id}")(n_var=30))

            # ZDT4 & ZDT6
            for id in [4, 6]:
                instance_set.append(eval(f"ZDT{id}")(n_var=10))

            # DTLZ1
            dtlz1_settings = {
                2: [6],
                3: [7],
                5: [9],
                7: [11],
                8: [12],
                10: [14]
            }
            for n_obj, n_var_list in dtlz1_settings.items():
                for n_var in n_var_list:
                    instance_set.append(eval("DTLZ1")(n_obj=n_obj, n_var=n_var))

            # DTLZ2-6
            for dtlz_id in range(2, 7):
                dtlz_settings = {
                    2: [11],
                    3: [11, 12] if dtlz_id != 3 and dtlz_id != 5 else [12],
                    5: [14],
                    7: [16],
                    8: [17],
                    10: [19]
                }
                for n_obj, n_var_list in dtlz_settings.items():
                    for n_var in n_var_list:
                        instance_set.append(eval(f"DTLZ{dtlz_id}")(n_obj=n_obj, n_var=n_var))

            # DTLZ7
            dtlz7_settings = {
                2: [21],
                3: [22],
                5: [24],
                7: [16, 26],
                8: [27],
                10: [29]
            }
            for n_obj, n_var_list in dtlz7_settings.items():
                for n_var in n_var_list:
                    instance_set.append(eval("DTLZ7")(n_obj=n_obj, n_var=n_var))

            # WFG1-9
            for wfg_id in range(1, 10):
                wfg_settings = {
                    2: [12, 22],
                    3: [12, 14, 24],
                    5: [14, 18, 28],
                    7: [16],
                    8: [24, 34],
                    10: [28, 38]
                }
                for n_obj, n_var_list in wfg_settings.items():
                    for n_var in n_var_list:
                        instance_set.append(eval(f"WFG{wfg_id}")(n_obj=n_obj, n_var=n_var))

            print(f"Total instances: {len(instance_set)}")
        # === 排序 ===
        instance_set.sort(key=lambda x: x.n_obj * x.n_var)
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
        train_set = []
        test_set = []
        if difficulty == 'easy':
            train_set = instance_set[:int(0.8*len(instance_set))]
            test_set = instance_set[int(0.8*len(instance_set)):]
        elif difficulty == 'difficult':
            train_set = instance_set[:int(0.2*len(instance_set))]
            test_set = instance_set[int(0.2*len(instance_set)):]
        return Moo_Dataset_numpy(train_set, train_batch_size), Moo_Dataset_numpy(test_set, test_batch_size)

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

    def __add__(self, other: 'Moo_Dataset_numpy'):
        return Moo_Dataset_numpy(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)

if __name__ == '__main__':
    train_set,test_set = Moo_Dataset_numpy.get_datasets()
    print(train_set,test_set)