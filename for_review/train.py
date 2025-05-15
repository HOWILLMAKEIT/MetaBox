from metaevobox import Config, Trainer
# import meta-level agent of MetaBBO you want to meta-train
from metaevobox.baseline.metabbo import *
# import low-level BBO optimizer of MetaBBO you want to meta-train
from metaevobox.environment.optimizer import *
from metaevobox.environment.problem.utils import construct_problem_set

# put user-specific configuration
config = {'train_problem': 'bbob-10D', # specify the problem set you want to train your MetaBBO
          'train_difficulty': 'difficult', # specify the difficulty of the problem set
          'train_batch_size': 16,
          'train_parallel_mode':'subproc', # choose parallel training mode
          }

baseline_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']

dir_results = []

for baseline in baseline_list:
    baseline_config = config.copy()
    if baseline == "GLHF" or baseline == "B2OPT" or baseline == "RNNOPT":
        baseline_config['train_problem'] = "bbob-torch-10D" 

    tmp_config = Config(baseline_config)
    # construct dataset
    tmp_config, datasets = construct_problem_set(tmp_config)
    # initialize your MetaBBO's meta-level agent & low-level optimizer
    metabbo = eval(baseline)(tmp_config)
    metabbo_opt = eval(f"{baseline}_Optimizer")(tmp_config)
    trainer = Trainer(tmp_config, metabbo, metabbo_opt, datasets)
    trainer.train()
    dir_results.append(tmp_config.train_name)

for i in range(len(dir_results)):
    print(f"MetaBBO {baseline_list[i]}: {dir_results[i]}")