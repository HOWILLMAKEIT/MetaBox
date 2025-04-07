# from problem import bbob, bbob_torch, protein_docking
from problem.SOO import bbob_numpy,bbob_surrogate,bbob_torch,protein_docking
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_torch.cec2013lsgo_dataset import CEC2013LSGO_Dataset
from problem.SOO.uav_numpy.uav_dataset import UAV_Dataset_numpy
from problem.SOO.uav_torch.uav_dataset import UAV_Dataset_torch
from problem import bbob, bbob_torch, protein_docking, mmo_dataset


def construct_problem_set(config):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob_numpy.bbob_dataset.BBOB_Dataset.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)
    elif problem in ['bbob-torch', 'bbob-noisy-torch']:
        return bbob_torch.bbob_dataset.BBOB_Dataset_torch.get_datasets(suit=config.problem,
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
        return CEC2013LSGO_Dataset.get_datasets(train_batch_size = config.train_batch_size,
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
    elif problem in ['mmo', 'mmo-torch']:
        return mmo_dataset.MMO_Dataset.get_datasets(version=problem,
                                            train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=config.difficulty,
                                            user_train_list = config.user_train_list)

    else:
        raise ValueError(problem + ' is not defined!')
