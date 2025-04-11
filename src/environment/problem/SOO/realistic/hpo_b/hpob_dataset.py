import numpy as np
from torch.utils.data import Dataset
from .utils import *
import subprocess, sys, os
from .hpo_b import HPOB_Problem
from tqdm import tqdm

class HPOB_Dataset(Dataset):
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
    def get_datasets(datapath=None,
                     train_batch_size=1,
                     test_batch_size=1,
                     cost_normalize=False,):
        # get functions ID of indicated suit
        if datapath is None:
            datapath = 'problem/datafiles/SOO/'
        root_dir = datapath+"HPO-B-main/hpob-data/"
        surrogates_dir = datapath+"HPO-B-main/saved-surrogates/"
        
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) < 7 or not os.path.exists(surrogates_dir) or len(os.listdir(surrogates_dir)) < 1909:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                # check the required package, if not exists, pip install it
                try:
                    subprocess.check_call([sys.executable,'-m', "pip", "install", 'huggingface_hub'])
                    # print("huggingface_hub has been installed successfully!")
                    from huggingface_hub import snapshot_download
                except subprocess.CalledProcessError as e:
                    print(f"Install huggingface_hub leads to errors: {e}")
                    
            snapshot_download(repo_id='GMC-DRL/MetaBox-HPO-B', repo_type="dataset", local_dir=datapath)
            print("Extract data...")
            os.system('tar -xf problem/datafiles/SOO/HPO-B-main.tar.gz -C problem/datafiles/SOO/')
            os.system('rm problem/datafiles/SOO/HPO-B-main.tar.gz')
            os.system('rm problem/datafiles/SOO/.gitattributes')
            
        meta_train_data,meta_vali_data,meta_test_data,bo_initializations,surrogates_stats=get_data(root_dir=root_dir, mode="v3", surrogates_dir=surrogates_dir)
        
        def process_data(data, name, n):
            problems = []
            pbar = tqdm(desc=f'Loading {name}', total=n, leave=False)
            for search_space_id in data.keys():
                for dataset_id in data[search_space_id].keys():
                    bst_model,y_min,y_max=get_bst(surrogates_dir=datapath+'HPO-B-main/saved-surrogates/',search_space_id=search_space_id,dataset_id=dataset_id,surrogates_stats=surrogates_stats)
                    X = np.array(data[search_space_id][dataset_id]["X"])
                    dim = X.shape[1]
                    p=HPOB_Problem(bst_surrogate=bst_model,dim=dim,y_min=y_min,y_max=y_max,normalized=cost_normalize)
                    problems.append(p)
                    pbar.update()
            pbar.close()
            return problems

        train_set = process_data(meta_train_data, 'meta_train_data', 758)
        test_set = process_data(meta_vali_data, 'meta_vali_data', 91) + process_data(meta_test_data, 'meta_test_data', 86)
        
        return HPOB_Dataset(train_set, train_batch_size), HPOB_Dataset(test_set, test_batch_size)

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

    def __add__(self, other: 'HPOB_Dataset'):
        return HPOB_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
        
        

