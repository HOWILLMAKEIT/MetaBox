import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from metaevobox import Config, Tester
from metaevobox.baseline.metabbo import *
from metaevobox.environment.optimizer import *
from metaevobox.environment.problem.utils import construct_problem_set

baseline_list = ['GLEET', 'GLHF', 'SYMBOL', 'RLDEAFL', 'B2OPT', 'DEDDQN', 'DEDQN', 'RLEPSO']
# 20250421T122939_bbob-10D_difficult
v2_dir_list = [] # todo figure your training model dir - metabox-v2
v1_dir_list = [] # todo figure your training model dir - metabox-v1
config = {'test_problem': 'bbob-10D',
          'test_difficulty': 'difficult', 
          'test_batch_size': 16,
          'test_parallel_mode':'Problem_Testrun',
          }


v2_result_dir = []
v1_result_dir = []

v2_time = {}
v1_time = {}

for v2_dir, v1_dir, baseline in zip(v2_dir_list, v1_dir_list, baseline_list):
    v2_time[baseline] = [0.0] # checkpoint 0
    v1_time[baseline] = [0.0] # checkpoint 0

    v2_path = f"agent_model/train/{baseline}/{v2_dir}/checkpoint_log.txt"
    v1_path = f"agent_model/train/{baseline}/{v1_dir}/checkpoint_log.txt"

    with open(v2_path, 'r') as f:
        for line in f:
            if "Time:" in line:
                time_str = line.strip().split("Time: ")[1].split("s")[0].strip() # get seconds
                v2_time[baseline].append(float(time_str))
                          
    with open(v1_path, 'r') as f:
        for line in f:
            if "Time:" in line:
                time_str = line.strip().split("Time: ")[1].split("s")[0].strip() # get seconds
                v1_time[baseline].append(float(time_str))

# test v2
for v2_dir, baseline in zip(v2_dir_list, baseline_list):
    baseline_config = config.copy()

    tmp_config = Config(baseline_config)

    # Use Tester's rollout
    tmp_config, datasets = construct_problem_set(tmp_config)
    v2_dir = f"agent_model/train/{baseline}/{v2_dir}/"
    
    rollout_opt = eval(f"{baseline}_Optimizer")(tmp_config)

    Tester.rollout_batch(tmp_config, v2_dir, rollout_opt, datasets)
    v2_result_dir.append(f"{tmp_config.rollout_log_dir}_{tmp_config.test_problem}_{tmp_config.test_difficulty}")

# test v1
for v1_dir, baseline in zip(v1_dir_list, baseline_list):
    baseline_config = config.copy()

    tmp_config = Config(baseline_config)

    # Use Tester's rollout
    tmp_config, datasets = construct_problem_set(tmp_config)
    v1_dir = f"agent_model/train/{baseline}/{v1_dir}/"
    
    rollout_opt = eval(f"{baseline}_Optimizer")(tmp_config)

    Tester.rollout_batch(tmp_config, v1_dir, rollout_opt, datasets)
    v1_result_dir.append(f"{tmp_config.rollout_log_dir}_{tmp_config.test_problem}_{tmp_config.test_difficulty}")

# load rollout results

v2_result_mean = [[] for _ in range(len(v2_result_dir))]
v2_result_std = [[] for _ in range(len(v2_result_dir))]

v1_result_mean = [[] for _ in range(len(v1_result_dir))]
v1_result_std = [[] for _ in range(len(v1_result_dir))]


# v2
for v2_dir, baseline in zip(v2_result_dir, baseline_list):
    with open(v2_dir + "rollout.pkl", 'rb') as f:
        results = pickle.load(f)
    rollout_cost = results['cost']

    problem_list = list(rollout_cost.keys())
    rollout_baseline_list = list(rollout_cost[problem_list[0]].keys())
    total_performance_avg = [[] for _ in range(len(rollout_baseline_list))]
    total_performance_std = [[] for _ in range(len(rollout_baseline_list))]


    for problem in problem_list:
        problem_cost = rollout_cost[problem]
        for i, rollout_baseline in enumerate(problem_cost.keys()):
            rollout_cost_baseline = problem_cost[rollout_baseline]
            run_performance = []
            for run in rollout_cost_baseline:
                performance = 1 - (run[-1] / (run[0] + 1e-20))
                run_performance.append(performance)
            
            total_performance_avg[i].append(np.mean(run_performance))
            total_performance_std[i].append(np.std(run_performance))
    
    for i in range(len(rollout_baseline_list)):
        v2_result_mean[baseline_list.index(baseline)].append(np.mean(total_performance_avg[i]))
        v2_result_std[baseline_list.index(baseline)].append(np.mean(total_performance_std[i]))

# v1
for v1_dir, baseline in zip(v1_result_dir, baseline_list):
    with open(v1_dir + "rollout.pkl", 'rb') as f:
        results = pickle.load(f)
    rollout_cost = results['cost']

    problem_list = list(rollout_cost.keys())
    rollout_baseline_list = list(rollout_cost[problem_list[0]].keys())
    total_performance_avg = [[] for _ in range(len(rollout_baseline_list))]
    total_performance_std = [[] for _ in range(len(rollout_baseline_list))]

    for problem in problem_list:
        problem_cost = rollout_cost[problem]
        for i, rollout_baseline in enumerate(problem_cost.keys()):
            rollout_cost_baseline = problem_cost[rollout_baseline]
            run_performance = []
            for run in rollout_cost_baseline:
                performance = 1 - (run[-1] / (run[0] + 1e-20))
                run_performance.append(performance)
            
            total_performance_avg[i].append(np.mean(run_performance))
            total_performance_std[i].append(np.std(run_performance))
    
    for i in range(len(rollout_baseline_list)):
        v1_result_mean[baseline_list.index(baseline)].append(np.mean(total_performance_avg[i]))
        v1_result_std[baseline_list.index(baseline)].append(np.mean(total_performance_std[i]))


# plot

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, baseline in enumerate(baseline_list):
    ax = axes[i]

    # Get time
    x_v2 = v2_time[baseline] / 3600 # convert to hours
    y_v2 = v2_result_mean[i]
    yerr_v2 = v2_result_std[i]

    x_v1 = v1_time[baseline] / 3600
    y_v1 = v1_result_mean[i]
    yerr_v1 = v1_result_std[i]

    ax.plot(x_v2, y_v2, label='v2', color='blue')
    ax.fill_between(x_v2, y_v2 - yerr_v2, y_v2 + yerr_v2, color='blue', alpha=0.2)

    ax.plot(x_v1, y_v1, label='v1', color='orange')
    ax.fill_between(x_v1, y_v1 - yerr_v1, y_v1 + yerr_v1, color='orange', alpha=0.2)
    ax.set_title(baseline)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Performance')
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()




