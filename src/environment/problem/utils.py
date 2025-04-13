# from problem import bbob, bbob_torch, protein_docking
import pickle
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from logger import Logger
from utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import copy

from problem.MMO.classical.cec2013mmo_numpy.cec2013mmo_dataset import CEC2013MMO_Dataset 
from problem.SOO.classical.bbob_numpy.bbob_dataset import BBOB_Dataset as bbob_dataset_numpy
from problem.SOO.classical.bbob_torch.bbob_dataset import BBOB_Dataset as bbob_torch_torch
from problem.SOO.others.bbob_surrogate import bbob_surrogate
from problem.SOO.others.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.others.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_torch.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_numpy.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_numpy
from problem.SOO.realistic.uav_numpy.uav_dataset import UAV_Dataset as UAV_Dataset_numpy
from problem.SOO.realistic.uav_torch.uav_dataset import UAV_Dataset as UAV_Dataset_torch
from problem.SOO.realistic.protein_docking import protein_docking

from problem.MTO.customize.augmented_wcci2020_numpy.augmented_wcci2020_dataset import Augmented_WCCI2020_Dataset

def save_class(dir, file_name, saving_class):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)

def construct_problem_set(config):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob_dataset_numpy.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)
    elif problem in ['bbob-torch', 'bbob-noisy-torch']:
        return bbob_torch_torch.get_datasets(suit=config.problem,
                                                          dim=config.dim,
                                                          upperbound=config.upperbound,
                                                          train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size,
                                                          difficulty=config.difficulty)

    elif problem in ['protein', 'protein-torch']:
        return protein_docking.Protein_Docking_Dataset.get_datasets(version=problem,
                                                                    train_batch_size=config.train_batch_size,
                                                                    test_batch_size=config.test_batch_size,
                                                                    difficulty=config.difficulty)

    elif problem in ['bbob-surrogate']:
        return bbob_surrogate.bbob_surrogate_Dataset.get_datasets(config=config,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)

    elif problem in ['Symbolic_bench','Symbolic_bench-torch']:
        if problem == 'Symbolic_bench':
            return Symbolic_bench_Dataset.get_datasets(upperbound=config.upperbound,
                                                       train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size)
        else:
            return Symbolic_bench_Dataset_torch.get_datasets(upperbound=config.upperbound,
                                                       train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size)
    elif problem in ['lsgo-torch']:
        return LSGO_Dataset_torch.get_datasets(train_batch_size = config.train_batch_size,
                                                 test_batch_size = config.test_batch_size,
                                                 difficulty = config.difficulty)

    elif problem in ['lsgo']:
        return LSGO_Dataset_numpy.get_datasets(train_batch_size = config.train_batch_size,
                                                 test_batch_size = config.test_batch_size,
                                                 difficulty = config.difficulty)
    elif problem in ['uav']:
        return UAV_Dataset_numpy.get_datasets(train_batch_size = config.train_batch_size,
                                              test_batch_size = config.test_batch_size,
                                              dv = 10,
                                              j_pen = 1e4,
                                              mode = "standard",
                                              num = 56,
                                              difficulty = config.difficulty)
    elif problem in ['uav-torch']:
        return UAV_Dataset_torch.get_datasets(train_batch_size = config.train_batch_size,
                                              test_batch_size = config.test_batch_size,
                                              dv = 10,
                                              j_pen = 1e4,
                                              mode = "standard",
                                              num = 56,
                                              difficulty = config.difficulty)
    elif problem in ['MMO', 'MMO-torch']:
        return CEC2013MMO_Dataset.get_datasets(version=problem,
                                            train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=config.difficulty,
                                            user_train_list = config.user_train_list)
    elif problem in ['augmented-WCCI2020']:
        return Augmented_WCCI2020_Dataset.get_datasets(dim=config.dim,
                                                       train_batch_size=config.train_batch_size,
                                                       test_batch_size=config.test_batch_size,
                                                       difficulty=config.difficulty,
                                                       task_cnt=config.task_cnt)

    else:
        raise ValueError(problem + ' is not defined!')
    
        
