from .uav import *
from torch.utils.data import Dataset
from .utils import createmodel

class UAV_Dataset_torch(Dataset):
    def __init__(self, data, batch_size = 1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(train_batch_size = 1,
                     test_batch_size = 1,
                     dv = 5.0,
                     j_pen = 1e4,
                     mode = "standard",
                     seed = 3849,
                     num = 56,
                     difficult = "easy",
                     ):
        # todo 先选择全部读入作为训练集和测试集
        train_set = []
        test_set = []
        if mode == "standard":
            pkl_file = "problem/datafiles/uav_terrain/Model56.pkl"
            with open(pkl_file, 'rb') as f:
                model_data = pickle.load(f)
            func_id = range(56)  # 56
            for id in func_id:
                terrain_data = model_data[id]
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = Terrain(terrain_data, id + 1)
                train_set.append(instance)
                test_set.append(instance)
        elif mode == "custom":
            if difficult == "easy":
                ratio = 0.75
            else:
                ratio = 0.25
            np.random.seed(seed)
            func_id = range(num)
            for id in func_id:
                if id < num * ratio:
                    num_threats = 10
                else:
                    num_threats = 20
                terrain_data = createmodel(map_size = 900,
                                           r = np.random.rand() * 100,
                                           rr = np.random.rand() * 10,
                                           num_threats = num_threats)
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = Terrain(terrain_data, id + 1)
                train_set.append(instance)
                test_set.append(instance)

        return UAV_Dataset_torch(train_set, train_batch_size), \
               UAV_Dataset_torch(test_set, test_batch_size)

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

    def __add__(self, other: 'UAV_Dataset_torch'):
        return UAV_Dataset_torch(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)