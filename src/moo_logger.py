import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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



def get_average_returns(results: dict, norm: bool=False):
    problems=[]
    agents=[]

    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_return={}
    std_return={}
    # n_checkpoint=len(results[problems[0]][agents[0]])
    for agent in agents:
        avg_return[agent]=[]
        std_return[agent]=[]
        for problem in problems:
            values = results[problem][agent]
            if norm:
                values = (values - np.min((values))) / (np.max(values) - np.min(values))
            std_return[agent].append(np.std(values, -1))
            avg_return[agent].append(np.mean(values, -1))
        avg_return[agent] = np.mean(avg_return[agent], 0)
        std_return[agent] = np.mean(std_return[agent], 0)
        # for checkpoint in range(n_checkpoint):
        #     return_sum=0
        #     for problem in problems:
        #         return_sum+=results[problem][agent][checkpoint]
        #     avg_return[agent].append(return_sum/len(problems))
    return avg_return, std_return  # {'agent':[] len = n_checkpoints}


def get_average_costs(results: dict, norm: bool=False):
    #agent在所有run,所有问题中的最后一代取平均
    problems=[]
    agents=[]
    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_cost = {}
    std_cost = {}
    # n_checkpoint=len(results[problems[0]][agents[0]])
    for agent in agents:
        avg_cost[agent]=[]
        std_cost[agent]=[]
        for problem in problems:
            values = np.array(results[problem][agent])[:, :, -1]
            if norm:
                all_agent_values = np.concatenate([
                    np.array(results[problem][a]).flatten() for a in agents
                ])
                min_val, max_val = np.min(all_agent_values), np.max(all_agent_values)
                
                if max_val > min_val:  # 避免除零错误
                    values = (values - min_val) / (max_val - min_val)
            std_cost[agent].append(np.std(values, -1))
            avg_cost[agent].append(np.mean(values, -1))
        avg_cost[agent] = np.mean(avg_cost[agent], 0)
        std_cost[agent] = np.mean(std_cost[agent], 0)
        # for checkpoint in range(n_checkpoint):
        #      return_sum=0
        #      for problem in problems:
        #           return_sum+=results[problem][agent][checkpoint]
        #      avg_return[agent].append(return_sum/len(problems))
    return avg_cost, std_cost  # {'agent':[] len = n_checkpoints}



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


def gen_agent_performance_table(results: dict, out_dir: str) -> None:
    total_cost=results['cost']
    table_data={}
    indexs=[]
    columns=['Worst','Best','Median','Mean','Std']
    for problem,value in total_cost.items():
        indexs.append(problem)
        problem_cost=value
        for alg,alg_cost in problem_cost.items():
            n_cost=[]
            for run in alg_cost:
                n_cost.append(run[-1])
            # if alg == 'MadDE' and problem == 'F5':
            #     for run in alg_cost:
            #         print(len(run))
            #     print(len(n_cost))
            best=np.min(n_cost)
            best=np.format_float_scientific(best,precision=3,exp_digits=3)
            worst=np.max(n_cost)
            worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
            median=np.median(n_cost)
            median=np.format_float_scientific(median,precision=3,exp_digits=3)
            mean=np.mean(n_cost)
            mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
            std=np.std(n_cost)
            std=np.format_float_scientific(std,precision=3,exp_digits=3)

            if not alg in table_data:
                table_data[alg]=[]
            table_data[alg].append([worst,best,median,mean,std])
    for alg,data in table_data.items():
        dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
        #print(dataframe)
        dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_table.xlsx'))


def gen_overall_tab(results: dict, out_dir: str) -> None:
    # get multi-indexes first
    problems = []
    statics = ['Obj','Gap','FEs']
    optimizers = []
    for problem in results['cost'].keys():
        problems.append(problem)
    for optimizer in results['T2'].keys():
        optimizers.append(optimizer)
    multi_columns = pd.MultiIndex.from_product(
        [problems,statics], names=('Problem', 'metric')
    )
    df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                              index=optimizers,
                              columns=multi_columns)

    # calculate baseline1 cmaes
    cmaes_obj = {}
    for problem in problems:
        blobj_problem = results['cost'][problem]['DEAP_CMAES']  # 51 * record_length
        objs = []
        for run in range(51):
            objs.append(blobj_problem[run][-1])
        cmaes_obj[problem] = sum(objs) / 51

    # calculate baseline2 random_search
    rs_obj = {}
    for problem in problems:
        blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
        objs = []
        for run in range(51):
            objs.append(blobj_problem[run][-1])
        rs_obj[problem] = sum(objs) / 51

    # calculate each Obj
    for problem in problems:
        for optimizer in optimizers:
            obj_problem_optimizer = results['cost'][problem][optimizer]
            objs_ = []
            for run in range(51):
                objs_.append(obj_problem_optimizer[run][-1])
            avg_obj = sum(objs_)/51
            std_obj = np.std(objs_)
            df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"
            # calculate each Gap
            df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1-(rs_obj[problem]-avg_obj) / (rs_obj[problem]-cmaes_obj[problem]+1e-10))
            fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
            avg_fes = np.mean(fes_problem_optimizer)
            std_fes = np.std(fes_problem_optimizer)
            df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(avg_fes, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_fes, precision=3, exp_digits=1) + ")"
    df_results.to_excel(out_dir+'overall_table.xlsx')

def is_pareto_efficient(points):
    """计算帕累托前沿"""
    points = np.array(points)
    pareto_mask = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] = np.any(points[pareto_mask] < p, axis=1)
            pareto_mask[i] = True
    return points[pareto_mask]

def to_label(agent_name: str) -> str:
    label = agent_name
    if label == 'BayesianOptimizer':
        return 'BO'
    if label == 'L2L_Agent':
        return 'RNN-OI'
    if len(label) > 6 and (label[-6:] == '_Agent' or label[-6:] == '_agent'):
        label = label[:-6]
    return label

class moo_Logger:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0
        self.indicators = ['hv','igd']

    # train
    def record(self,data:dict,train_meta_data:dict,problem):
        prefix = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
        if prefix not in data.keys():
            data[prefix] = {}
        for indictor in self.indictors:
            if indictor not in data[prefix].keys():
                data[prefix][indictor] = []
            data[prefix][indictor].append(train_meta_data[indictor])
        if 'return' not in data[prefix].keys():
            data[prefix]['return'] = []
        data[prefix]['return'].append(train_meta_data['return'])
        if 'learn_steps' not in data[prefix].keys():
            data[prefix]['learn_steps'] = []
        data[prefix]['learn_steps'].append(train_meta_data['learn_steps'])
        return data
        
    def save_log(self, epochs, steps, data,log_dir:str):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, data['returns']),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            for indicator in self.indicators:
                if len(data[name][indicator]) == 0:
                    continue
                while len(data[name][indicator]) < len(epochs):
                    data[name][indicator].append(data[name][indicator][-1])
                indictors_record_save = np.stack((epochs, data[name][indicator]),  0)
                np.save(log_dir+name+'_'+indicator,indictors_record_save)
            
    def draw_indicators(self, Name, train_set,log_dir:str):
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in train_set:
            if Name is None:
                name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            elif (isinstance(Name, str) and problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var) != Name) \
                or (isinstance(Name, list) and problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var) not in Name):
                continue
            else:
                name = Name
            for indicator in self.indictors:
                plt.figure()
                plt.title(name + '_' + indicator)
                values = np.load(log_dir + 'log/' + name+'_'+indicator+'.npy')
                x, y, n = values
                plt.plot(x, y)
                plt.savefig(log_dir+f'pic/{name}_'+indicator+'.png')
                plt.close()
    
    def draw_average_indictors(self, train_set,log_dir:str):
        # 这个函数用于画每一个epoch所有问题指标的平均值
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for indictor in self.indicators:
            X = []
            Y = []
            for problem in train_set:
                name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
                values = np.load(log_dir + 'log/' + name+'_'+indictor+'.npy')
                x, y = values
                X.append(x)
                Y.append(y)
            X = np.mean(X, 0)
            Y = np.mean(Y, 0)
            plt.figure()
            plt.title('all problem '+ indictor)
            plt.plot(X, Y)
            plt.savefig(log_dir+f'pic/all_problem_'+indictor+'.png')
            plt.close()

    def draw_return(self,log_dir:str):
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('return')
        values = np.load(log_dir + 'log/return.npy')
        plt.plot(values[0], values[1])
        plt.savefig(log_dir+f'pic/return.png')
        plt.close()
    ## test
    def draw_test_indicator(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, categorized: bool=False) -> None:
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
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])

                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'learnable_{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.t_optimizer_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'classic_{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_indicator(self, data: dict, output_dir: str, named_agents: dict, indicator:str) -> None:
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            Y = {}
            for problem in list(data.keys()):
                # 计算全局最大值和最小值
                all_values = []
                for agent in data[problem].keys():
                    all_values.append(np.array(data[problem][agent]))
                all_values = np.concatenate(all_values, axis=0)  # 拼接所有数据
                global_min = np.min(all_values)  # 计算全局最小值
                global_max = np.max(all_values)  # 计算全局最大值
                
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent][indicator])
                    values = (values - global_min) / (global_max - global_min + 1e-8)  # 避免除零
                
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)

                X = np.arange(mean.shape[-1])
                X = np.array(X, dtype=np.float64)
                X *= (self.config.maxFEs / X[-1])

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('FEs')
            plt.ylabel('Normalized {indicator}')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_{indicator}_curve.png', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None) -> None:
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel('{indicator}')
            plt.savefig(output_dir + f'{agent}_concrete_{indicator}_performance_hist.png', bbox_inches='tight')

    def draw_train_return(self, data: dict, output_dir: str, norm: bool=False) -> None:
        returns, stds = get_average_returns(data['return'], norm=norm)
        plt.figure()
        for agent in returns.keys():
            x = np.arange(len(returns[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = returns[agent]
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
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Return')
        plt.grid()
        plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        plt.close()

    def draw_train_avg_indicator(self, data: dict, output_dir: str,indicator: str, norm: bool=False) -> None:
        costs, stds = get_average_costs(data,norm=norm)
        plt.figure()
        for agent in costs.keys():
            x = np.arange(len(costs[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = costs[agent]
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
        plt.xlabel('Learning Steps')
        plt.ylabel(str(indicator))
        plt.grid()
        plt.savefig(output_dir + f'avg_{indicator}_curve.png', bbox_inches='tight')
        plt.close()

    def draw_boxplot(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None) -> None:
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} {indicator}  Boxplots')
            plt.savefig(output_dir + f'{name}_boxplot.png', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot(self, data: dict, output_dir: str, indicator:str,ignore: Optional[list]=None) -> None:
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel('{indicator} Boxplots')
        plt.savefig(output_dir + f'overall_boxplot.png', bbox_inches='tight')
        plt.close()

    def draw_pareto_fronts(self,data: dict, output_dir: str,Name: Optional[Union[str, list]]=None):
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            fig = plt.figure(figsize=(10, 8))  # 调整画布大小
            markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h', 'v', '<', '>', '|', '_']
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            is_3d = False
            
            for problem, algorithms in data.items():
                for algo, runs in algorithms.items():
                    for generations in runs:
                        last_gen = np.array(generations[-1])  # 选择最后一代
                        obj_values = last_gen.reshape(-1, last_gen.shape[-1])  # 只保留目标值
                        if obj_values.shape[1] == 3:
                            is_3d = True
            
            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=25, azim=135)  # 调整3D视角，确保所有轴清晰可见
                ax.set_proj_type('persp')  # 使用透视投影改善可视化
            else:
                ax = fig.add_subplot(111)
            
            for problem, algorithms in data.items():
                for algo_idx, (algo, runs) in enumerate(algorithms.items()):
                    color = colors[algo_idx % len(colors)]
                    for run_idx, generations in enumerate(runs):
                        last_gen = np.array(generations[-1])  # 选择最后一代
                        obj_values = last_gen.reshape(-1, last_gen.shape[-1])  # 只保留目标值
                        pareto_front = is_pareto_efficient(obj_values)
                        
                        marker = markers[run_idx % len(markers)]  # 不同回合使用不同标记
                        label = f"{problem}-{algo}-Run{run_idx + 1}"  # 标明具体回合
                        if obj_values.shape[1] == 2:
                            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                                    label=label, marker=marker, color=color, edgecolors='k')
                        elif obj_values.shape[1] == 3:
                            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                                    label=label, marker=marker, color=color, edgecolors='k')
            
            ax.set_xlabel('X', fontsize=14, labelpad=25)
            ax.set_ylabel('Y', fontsize=14, labelpad=25)
            if is_3d:
                ax.set_zlabel('Z', fontsize=1, labelpad=10)  # 增加 Z 轴标签的间距
                ax.zaxis.label.set_size(14)  # 增大Z轴标签字体
                ax.zaxis.label.set_rotation(90)  # 旋转Z轴标签使其更清晰可见
                ax.set_box_aspect([1.2, 1, 0.8])  # 调整3D比例以优化视图
            
            plt.legend()
            plt.grid(True)
            plt.title('Pareto Fronts of Different Algorithms Across Runs', fontsize=14)
            plt.savefig(output_dir + f'{name}_pareto_fronts.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            


