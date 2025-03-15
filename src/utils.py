from problem import bbob, bbob_torch, protein_docking
from problem.SOO import bbob_numpy,bbob_surrogate,bbob_torch,protein_docking
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch

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
            
    else:
        raise ValueError(problem + ' is not defined!')
