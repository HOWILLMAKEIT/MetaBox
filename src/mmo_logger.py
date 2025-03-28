import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union
import argparse
params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '3',
    'legend.fontsize': '24',
    'figure.figsize': '20,11',
}
plt.rcParams.update(params)

markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet', 
          'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod', 
          ]

def get_average_data(data_type: str, results: dict, norm: bool=False): # for rollout
    problems=[]
    agents=[]

    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_data={}
    std_data={}
    for agent in agents:
        avg_data[agent]=[]
        std_data[agent]=[]
        for problem in problems:
            if data_type == 'pr' or data_type == 'sr':
                values = results[problem][agent][:,:,3] # accuracy of 1e-4
            elif data_type == 'return' or data_type == 'gbest':
                values = results[problem][agent]
            else:
                raise ValueError('Invalid data type')
            if norm:
                values = (values - np.min((values))) / (np.max(values) - np.min(values))
            std_data[agent].append(np.std(values, -1))
            avg_data[agent].append(np.mean(values, -1))
        avg_data[agent] = np.mean(avg_data[agent], 0)
        std_data[agent] = np.mean(std_data[agent], 0)
    return avg_data, std_data  # {'agent':[] len = n_checkpoints}

def get_test_average_data(data_type: str, results: dict, norm: bool=False): # for test
    problems=[]
    agents=[]

    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_data={}
    std_data={}
    for agent in agents:
        avg_data[agent]=[]
        std_data[agent]=[]
        for problem in problems:
            if data_type == 'pr' or data_type == 'sr':
                values = results[problem][agent][:,3] # accuracy of 1e-4
            else:
                raise ValueError('Invalid data type')
            if norm:
                values = (values - np.min((values))) / (np.max(values) - np.min(values))
            std_data[agent].append(np.std(values, -1))
            avg_data[agent].append(np.mean(values, -1))
        avg_data[agent] = np.mean(avg_data[agent], 0)
        std_data[agent] = np.mean(std_data[agent], 0)
    return avg_data, std_data  


def gen_algorithm_complexity_table(results: dict, out_dir: str) -> None:
    save_list=[]
    t0=results['T0']
    t1=results['T1']
    is_dict=False
    if type(t1) is dict:
        is_dict=True
    t2s=results['T2']
    ratios=[]
    t2_list=[]
    indexs=[]
    columns=['T0','T1','T2','(T2-T1)/T0']

    
    for key,value in t2s.items():
        indexs.append(key)
        t2_list.append(value)
        if is_dict:
            ratios.append((value-t1[key])/t0)
        else:
            ratios.append((value-t1)/t0)
    n=len(t2_list)
    data=np.zeros((n,4))
    data[:,0]=t0
    if is_dict:
        for i,(key,value) in enumerate(t1.items()):
            data[i,1]=value
    else:
        data[:,1]=t1
    
    data[:,2]=t2_list
    data[:,3]=ratios
    table=pd.DataFrame(data=np.round(data,2),index=indexs,columns=columns)
    # table["number_str"] = table["number_str"].astype(long).astype(str)
    #(table)
    table.to_excel(os.path.join(out_dir,'algorithm_complexity.xlsx'))


def gen_agent_performance_table(data_type: str, results: dict, out_dir: str) -> None:
    if data_type == 'pr':
        total_data=results['pr']
    elif data_type == 'sr':
        total_data = results['sr']
    else:
        raise ValueError('Invalid data type')
    table_data={}
    indexs=[]
    columns=['Worst','Best','Median','Mean','Std']
    for problem,value in total_data.items():
        indexs.append(problem)
        problem_data=value
        for alg,alg_data in problem_data.items():
            n_data=[]
            for run in alg_data:
                n_data.append(run[3])
            best=np.min(n_data)
            best=np.format_float_scientific(best,precision=3,exp_digits=3)
            worst=np.max(n_data)
            worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
            median=np.median(n_data)
            median=np.format_float_scientific(median,precision=3,exp_digits=3)
            mean=np.mean(n_data)
            mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
            std=np.std(n_data)
            std=np.format_float_scientific(std,precision=3,exp_digits=3)

            if not alg in table_data:
                table_data[alg]=[]
            table_data[alg].append([worst,best,median,mean,std])
    for alg,data in table_data.items():
        dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
        #print(dataframe)
        if data_type == 'pr':
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_PR_1e-4_table.xlsx'))
        elif data_type == 'sr':
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_SR_1e-4_table.xlsx'))
        else:
            raise ValueError('Invalid data type')

def gen_overall_tab(results: dict, out_dir: str) -> None:
    # get multi-indexes first
    problems = []
    statics = ['gbest','1e-4 PR', '1e-4 SR']
    optimizers = []
    for problem in results['gbest'].keys():
        problems.append(problem)
    for optimizer in results['T2'].keys():
        optimizers.append(optimizer)
    multi_columns = pd.MultiIndex.from_product(
        [problems,statics], names=('Problem', 'metric')
    )
    df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                              index=optimizers,
                              columns=multi_columns)

    # # calculate baseline1 cmaes
    # cmaes_obj = {}
    # for problem in problems:
    #     blobj_problem = results['cost'][problem]['DEAP_CMAES']  # 51 * record_length
    #     objs = []
    #     for run in range(51):
    #         objs.append(blobj_problem[run][-1])
    #     cmaes_obj[problem] = sum(objs) / 51

    # # calculate baseline2 random_search
    # rs_obj = {}
    # for problem in problems:
    #     blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
    #     objs = []
    #     for run in range(51):
    #         objs.append(blobj_problem[run][-1])
    #     rs_obj[problem] = sum(objs) / 51

    # calculate each Obj
    for problem in problems:
        for optimizer in optimizers:
            obj_problem_optimizer = np.array(results['gbest'][problem][optimizer])
            avg_obj = np.mean(obj_problem_optimizer)
            std_obj = np.std(obj_problem_optimizer)
            df_results.loc[optimizer, (problem, 'gbest')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"
            pr_problem_optimizer = np.array(results['PR'][problem][optimizer][:, 3])
            assert len(pr_problem_optimizer) == len(obj_problem_optimizer)
            avg_pr = np.mean(pr_problem_optimizer)
            std_pr = np.std(pr_problem_optimizer)
            df_results.loc[optimizer, (problem, 'PR')] = np.format_float_scientific(avg_pr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_pr, precision=3, exp_digits=1) + ")"
            sr_problem_optimizer = np.array(results['SR'][problem][optimizer][:, 3])
            assert len(sr_problem_optimizer) == len(obj_problem_optimizer)
            avg_sr = np.mean(sr_problem_optimizer)
            std_sr = np.std(sr_problem_optimizer)
            df_results.loc[optimizer, (problem, 'SR')] = np.format_float_scientific(avg_sr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_sr, precision=3, exp_digits=1) + ")"

    df_results.to_excel(out_dir+'overall_table.xlsx')

def to_label(agent_name: str) -> str:
    label = agent_name
    if label == 'BayesianOptimizer':
        return 'BO'
    if label == 'L2L_Agent':
        return 'RNN-OI'
    if len(label) > 6 and (label[-6:] == '_Agent' or label[-6:] == '_agent'):
        label = label[:-6]
    return label

class MMO_Logger:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0

    def draw_test_cost(self, data_type: str, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False) -> None:
        if data_type == 'pr':
            data_name = 'PR'
        elif data_type == 'sr':
            data_name = 'SR'
        else:
            raise ValueError('Invalid data type')
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent][:, :, 3])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel('log ' + data_name)
                    plt.savefig(output_dir + f'{name}_log_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel(data_name)
                    plt.savefig(output_dir + f'{name}_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent][:, :, 3])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel('log ' + data_name)
                    plt.savefig(output_dir + f'learnable_{name}_log_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel(data_name)
                    plt.savefig(output_dir + f'learnable_{name}_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                plt.close()
                
                # todo traditional optimizer

                # plt.figure()
                # for agent in list(data[name].keys()):
                #     if agent not in self.config.t_optimizer_for_cp:
                #         continue
                #     if agent not in self.color_arrangement.keys():
                #         self.color_arrangement[agent] = colors[self.arrange_index]
                #         self.arrange_index += 1
                #     values = np.array(data[name][agent][:,:, 3])
                #     x = np.arange(values.shape[-1])
                #     x = np.array(x, dtype=np.float64)
                #     x *= (self.config.maxFEs / x[-1])
                #     if logged:
                #         values = np.log(np.maximum(values, 1e-8))
                #     std = np.std(values, 0)
                #     mean = np.mean(values, 0)
                #     plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                #     plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                # plt.grid()
                # plt.xlabel('FEs')
                
                # plt.legend()
                # if logged:
                #     plt.ylabel('log ' + data_name)
                #     plt.savefig(output_dir + f'classic_{name}_log_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                # else:
                #     plt.ylabel(data_name)
                #     plt.savefig(output_dir + f'classic_{name}_' + data_name + '_1e-4_curve.png', bbox_inches='tight')
                # plt.close()
    
    # in class Logger for rollout
    def draw_train_logger(self, data_type: str, data: dict, output_dir: str, norm: bool = False) -> None:
        means, stds = get_average_data(data_type, data[data_type], norm=norm)
        plt.figure()
        for agent in means.keys():
            x = np.arange(len(means[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = means[agent]
            s = np.zeros(y.shape[0])
            a = s[0] = y[0]
            norm = self.config.plot_smooth + 1
            for i in range(1, y.shape[0]):
                a = a * self.config.plot_smooth + y[i]
                s[i] = a / norm if norm > 0 else a
                norm *= self.config.plot_smooth
                norm += 1
            if agent not in self.color_arrangement.keys():
                self.color_arrangement[agent] = colors[self.arrange_index]
                self.arrange_index += 1
            plt.plot(x, s, label=to_label(agent), marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
            plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha=0.2, facecolor=self.color_arrangement[agent])
            # plt.plot(x, returns[agent], label=to_label(agent))
        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')    
        if data_type == 'pr':
            plt.ylabel('PR')
            plt.savefig(output_dir + f'avg_PR_1e-4_curve.png', bbox_inches='tight')
        elif data_type == 'sr':
            plt.ylabel('SR')
            plt.savefig(output_dir + f'avg_SR_1e-4_curve.png', bbox_inches='tight')
        elif data_type == 'gbest':
            plt.ylabel('Avg gbest')
            plt.savefig(output_dir + f'avg_gbest_curve.png', bbox_inches='tight')
        elif data_type == 'return':
            plt.ylabel('Avg Return')
            plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        else:
            raise ValueError('Invalid data type')
        plt.close()

    def draw_rank_hist(self, data_type : str, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None) -> None:
        if data_type == 'pr':
            metric, metric_std = get_test_average_data('pr', data['pr'], norm=norm) # (len(agent), )
        elif data_type == 'sr':
            metric, metric_std = get_test_average_data('sr', data['sr'], norm=norm)
        else:
            raise ValueError('Invalid data type')
        X, Y = list(metric.keys()), list(metric.values())
        _, S = list(metric_std.keys()), list(metric_std.values())
        n_agents = len(X)
        for i in range(n_agents):
            X[i] = to_label(X[i])

        plt.figure(figsize=(4*n_agents,15))
        plt.bar(X, Y)
        plt.errorbar(X, Y, S, fmt='s', ecolor='dimgray', ms=1, color='dimgray', elinewidth=5, capsize=30, capthick=5)
        for a,b in zip(X, Y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        plt.xticks(rotation=45, fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylim(0, np.max(np.array(Y) + np.array(S)) * 1.1)
        if data_type == 'pr':
            plt.title(f'The 1e-4 R for {self.config.problem}-{self.config.difficulty}', fontsize=70)
            plt.ylabel('PR', fontsize=60)
            plt.savefig(output_dir + f'PR_1e-4_rank_hist.png', bbox_inches='tight')
        elif data_type == 'sr':
            plt.title(f'The 1e-4 SR for {self.config.problem}-{self.config.difficulty}', fontsize=70)
            plt.ylabel('SR', fontsize=60)
            plt.savefig(output_dir + f'SR_1e-4_rank_hist.png', bbox_inches='tight')
        else:
            raise ValueError('Invalid data type')
        

def post_processing_test_statics(log_dir: str, logger: Logger) -> None:
    with open(log_dir+'test.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(log_dir+'random_search_baseline.pkl', 'rb') as f:
        random = pickle.load(f)
    # Generate excel tables
    if not os.path.exists(log_dir + 'tables/'):
        os.makedirs(log_dir + 'tables/')
    gen_overall_tab(results, log_dir+'tables/') # 
    gen_algorithm_complexity_table(results, log_dir+'tables/') # 
    gen_agent_performance_table('pr', results, log_dir+'tables/') 
    gen_agent_performance_table('sr', results, log_dir + 'tables/')

    # Generate figures
    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    logger.draw_test_cost('pr', results['pr_list'],log_dir + 'pics/', logged=True, categorized=True) #
    logger.draw_test_cost('sr', results['sr_list'],log_dir + 'pics/', logged=True, categorized=True)
    # logger.draw_named_average_test_costs(results['cost'], log_dir + 'pics/',
    #                                     {'MetaBBO-RL': ['DE_DDQN_Agent', 'RL_HPSDE_Agent', 'LDE_Agent', 'QLPSO_Agent', 'RLEPSO_Agent', 'RL_PSO_Agent', 'DEDQN_Agent'],
    #                                      'Classic Optimizer': ['DEAP_DE', 'DEAP_CMAES', 'DEAP_PSO', 'JDE21', 'NL_SHADE_LBC', 'GL_PSO', 'sDMS_PSO', 'MadDE', 'SAHLPSO', 'Random_search']},
    #                                     logged=False) # 各个agent在平均问题上的指标变化曲线
    logger.draw_rank_hist('pr', results, random, log_dir + 'pics/') 
    logger.draw_rank_hist('sr', results, random, log_dir + 'pics/')

def post_processing_rollout_statics(log_dir: str, logger: Logger) -> None:
    with open(log_dir+'rollout.pkl', 'rb') as f:
        results = pickle.load(f)
    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    logger.draw_train_logger('return', results, log_dir + 'pics/', )
    logger.draw_train_logger('gbest', results, log_dir + 'pics/', )
    logger.draw_train_logger('pr', results, log_dir + 'pics/',)
    logger.draw_train_logger('sr', results, log_dir+'pics/',)
