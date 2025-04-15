import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union, Callable
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


def data_wrapper_prsr(data, ):
    res = []
    for key in data.keys():
        res.append(np.array(data[key][:, -1, 3]))
    return np.array(res)


def data_wrapper_cost(data, ):
    return np.array(data)[:, :, -1]


def data_wrapper_prsr_test(data, ):
    return np.array(data)[:,-1, 3]


def to_label(agent_name: str) -> str:
    label = agent_name
    if label == 'L2L_Agent':
        return 'RNN-OI'
    if len(label) > 6 and (label[-6:] == '_Agent' or label[-6:] == '_agent'):
        label = label[:-6]
    return label


class Basic_Logger:
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0

    def get_average_data(self, results: dict, norm: bool=False, data_wrapper: Callable = None):
        """
        Get the average and standard deviation of each agent from the results
        :param results  dict: The data to be process
        :param norm     bool: Whether to min-max normalize data
        :param data_wrapper callable: A data pre-processing function wrapper applied to each data item of each agent under each problem
        """
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
                values = results[problem][agent]
                if data_wrapper is not None:
                    values = data_wrapper(values)
                if norm:
                    values = (values - np.min((values))) / (np.max(values) - np.min(values))
                std_data[agent].append(np.std(values, -1))
                avg_data[agent].append(np.mean(values, -1))
            avg_data[agent] = np.mean(avg_data[agent], 0)
            std_data[agent] = np.mean(std_data[agent], 0)
        return avg_data, std_data

    def cal_scores1(self, D: dict, maxf: float):
        """
        Tool function for CEC metric
        """
        SNE = []
        for agent in D.keys():
            values = D[agent]
            sne = 0.5 * np.sum(np.min(values, -1) / maxf)
            SNE.append(sne)
        SNE = np.array(SNE)
        score1 = (1 - (SNE - np.min(SNE)) / SNE) * 50
        return score1

    def get_random_baseline(self, random: dict, fes: Optional[Union[int, float]]):
        """
        Get the results of Random Search for further usage, i.e., for normalization
        """
        baseline = {}
        if isinstance(random['T1'], dict):
            baseline['complexity_avg'] = np.log10(1/ (random['T2']['Random_search'] - random['T1']['Random_search']) / random['T0'])
        else:
            baseline['complexity_avg'] = np.log10(1/ (random['T2']['Random_search'] - random['T1']) / random['T0'])
        baseline['complexity_std'] = 0.005
        problems = random['cost'].keys()
        avg = []
        std = []
        for problem in problems:
            g = np.log10(fes/np.array(random['fes'][problem]['Random_search']))
            avg.append(g.mean())
            std.append(g.std())
        baseline['fes_avg'] = np.mean(avg)
        baseline['fes_std'] = np.mean(std)
        avg = []
        std = []
        for problem in problems:
            g = np.log10(1/(np.array(random['cost'][problem]['Random_search'])[:, -1]+1))
            avg.append(g.mean())
            std.append(g.std()) 
        baseline['cost_avg'] = np.mean(avg)
        baseline['cost_std'] = np.mean(std)
        return baseline

    def gen_algorithm_complexity_table(self, results: dict, out_dir: str) -> None:
        """
        Store algorithm complexity data as excel table 
        """
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
        table.to_excel(os.path.join(out_dir,'algorithm_complexity.xlsx'))

    def gen_agent_performance_table(self, results: dict, out_dir: str) -> None:
        """
        Store the `Worst`, `Best`, `Median`, `Mean` and `Std` of cost results of each agent as excel
        """
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
        """
        Store the overall results inculding `objective values` (costs), `gap` with CMAES and the consumed `FEs` as excel
        """
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

    def aei_cost(self, cost_data: dict, baseline: dict, ignore: Optional[list]=None):
        avg = baseline['cost_avg']
        problems = cost_data.keys()
        agents = cost_data[list(problems)[0]].keys()
        results_cost = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            costs_problem = []
            for problem in problems:
                cost_ = np.log10(1/(np.array(cost_data[problem][agent])[:, -1]+1))
                costs_problem.append(cost_.mean())
            results_cost[agent] = np.exp((costs_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_cost, agents, ignore)
        return results_cost, aei_mean, aei_std
    
    def aei_fes(self, fes_data: dict, baseline: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        avg = baseline['fes_avg']
        problems = fes_data.keys()
        agents = fes_data[list(problems)[0]].keys()
        results_fes = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            fes_problem = []
            for problem in problems:
                if agent == 'L2L':
                    fes_ = np.log10(100/np.array(fes_data[problem][agent]))
                else:
                    fes_ = np.log10(maxFEs/np.array(fes_data[problem][agent]))
                fes_problem.append(fes_.mean())
            results_fes[agent] = np.exp((fes_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_fes, agents, ignore)
        return results_fes, aei_mean, aei_std
    
    def aei_complexity(self, complexity_data: dict, baseline: dict, ignore: Optional[list]=None):
        avg = baseline['complexity_avg']
        std = baseline['complexity_std']
        problems = complexity_data.keys()
        agents = complexity_data[list(problems)[0]].keys()
        results_complex = {}
        for key in agents:
            if (ignore is not None) and (key in ignore):
                continue
            if key not in complexity_data['complexity'].keys():
                t0 = complexity_data['T0']
                if isinstance(complexity_data['T1'], dict):
                    t1 = complexity_data['T1'][key]
                else:
                    t1 = complexity_data['T1']
                t2 = complexity_data['T2'][key]
                complexity_data['complexity'][key] = ((t2 - t1) / t0)
            results_complex[key] = np.exp((np.log10(1/complexity_data['complexity'][key]) - avg)/std/1000 * 1)
        aei_mean, aei_std = self.cal_aei(results_complex, agents, ignore)
        return results_complex, aei_mean, aei_std

    def cal_aei(self, results: dict, agents: dict, ignore: Optional[list]=None):
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return mean, std

    def aei_metric(self, data: dict, random: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        """
        Calculate the AEI metric
        """
        baseline = get_random_baseline(random, maxFEs)
        problems = data['cost'].keys()
        agents = data['cost'][list(problems)[0]].keys()
        
        results_cost, aei_cost_mean, aei_cost_std = self.aei_cost(data['cost'], baseline, ignore)
        results_fes, aei_fes_mean, aei_fes_std = self.aei_fes(data['fes'], baseline, maxFEs, ignore)
        results_complex, aei_clx_mean, aei_clx_std = self.aei_fes(data['complexity'], baseline, ignore)
        
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results_complex[agent] * results_cost[agent] * results_fes[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return mean, std

    def cec_metric(self, data: dict, ignore: Optional[list]=None):
        """
        Calculate the metric adopted in CEC
        """
        score = {}
        M = []
        X = []
        Y = []
        R = []
        data, fes = data['cost'], data['fes']
        for problem in list(data.keys()):
            maxf = 0
            avg_cost = []
            avg_fes = []
            for agent in list(data[problem].keys()):
                if ignore is not None and agent in ignore:
                    continue
                key = to_label(agent)
                if key not in score.keys():
                    score[key] = []
                values = np.array(data[problem][agent])[:, -1]
                score[key].append(values)
                maxf = max(maxf, np.max(values))
                avg_cost.append(np.mean(values))
                avg_fes.append(np.mean(fes[problem][agent]))

            M.append(maxf)
            order = np.lexsort((avg_fes, avg_cost))
            rank = np.zeros(len(avg_cost))
            rank[order] = np.arange(len(avg_cost)) + 1
            R.append(rank)
        sr = 0.5 * np.sum(R, 0)
        score2 = (1 - (sr - np.min(sr)) / sr) * 50
        score1 = cal_scores1(score, M)
        for i, key in enumerate(score.keys()):
            score[key] = score1[i] + score2[i]
        return score

    def draw_ECDF(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True):
        data = data['cost']
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            plt.figure()
            for agent in list(data[name].keys()):
                if agent not in self.color_arrangement.keys():
                    self.color_arrangement[agent] = colors[self.arrange_index]
                    self.arrange_index += 1
                values = np.array(data[name][agent])
                plt.ecdf(values, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('costs')
            plt.legend()
            fig_type = 'pdf' if pdf_fig else 'png'
            plt.savefig(output_dir + f'ECDF_{problem}.{fig_type}', bbox_inches='tight')

    def draw_covergence_curve(self, agent: str, problem: str, metadata_dir: str, output_dir: str, pdf_fig: bool = True):
        def cal_max_distance(X):
            X = np.array(X)
            return np.max(np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, -1)))
        with open(metadata_dir + f'/{problem}.pkl', 'rb') as f:
            metadata = pickle.load(f)[agent]
        plt.figure()
        Xs = []
        n_generations = int(1e9)
        for item in metadata:
            Xs.append(item['X'])
            n_generations = min(n_generations, len(item['X']))
        diameter = np.zeros(n_generations)
        std = np.zeros(n_generations)
        x_axis = np.arange(n_generations)
        for i in range(n_generations):  # episode length
            d = []
            for j in range(len(Xs)):  # test_run
                d.append(cal_max_distance(Xs[j][i]))
            diameter[i] = np.mean(d)
            std[i] = np.std(d)
        plt.plot(x_axis, diameter, marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
        plt.fill_between(x_axis, (diameter - std), (diameter + std), alpha=0.2, facecolor=self.color_arrangement[agent])
        plt.grid()
        plt.xlabel('Optimization Generations')    
        plt.ylabel('Population Diameter')
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'convergence_curve_{agent}_{problem}.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_test_cost(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            # if logged:
            #     plt.title('log cost curve ' + name)
            # else:
            #     plt.title('cost curve ' + name)
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'{name}_log_cost_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'{name}_cost_curve.{fig_type}', bbox_inches='tight')
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'learnable_{name}_log_cost_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'learnable_{name}_cost_curve.{fig_type}', bbox_inches='tight')
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'classic_{name}_log_cost_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'classic_{name}_cost_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
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
                # X = np.log10(X)
                # X[0] = 0

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            # plt.xlabel('log10 FEs')
            plt.xlabel('FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
                D[agent].append(values[:, -1] / values[:, 0])

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
            plt.ylabel('Normalized Costs')
            plt.savefig(output_dir + f'{agent}_concrete_performance_hist.{fig_type}', bbox_inches='tight')

    def draw_boxplot(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
            plt.ylabel(f'{name} Cost Boxplots')
            plt.savefig(output_dir + f'{name}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot(self, data: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
        plt.ylabel('Cost Boxplots')
        plt.savefig(output_dir + f'overall_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_rank_hist(self, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        metric, metric_std = self.aei_metric(data, random, maxFEs=self.config.maxFEs, ignore=ignore)
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
        plt.title(f'The AEI for {self.config.dim}D {self.config.problem}-{self.config.difficulty}', fontsize=70)
        plt.ylabel('AEI', fontsize=60)
        plt.savefig(output_dir + f'rank_hist.{fig_type}', bbox_inches='tight')
        
    def draw_train_logger(self, data_type: str, data: dict, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        means, stds = self.get_average_data(data_type, data, norm=norm, data_wrapper=data_wrapper)
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
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()
        
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = True, pdf_fig: bool = True) -> None:
        with open(log_dir + 'test.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = self.config.agent
        bbo = self.config.t_optimizer
        
        # 可选地读取 random_search_baseline.pkl
        if include_random_baseline:
            with open(log_dir + 'random_search_baseline.pkl', 'rb') as f:
                random = pickle.load(f)

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        gen_overall_tab(results, log_dir + 'tables/')
        gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        # 如果需要，可以为不同的算法绘制图形（例如 cost 图）
        if 'cost' in results:
            self.draw_test_cost(results['cost'], log_dir + 'pics/', logged=True, categorized=True, pdf_fig=pdf_fig)
            self.draw_named_average_test_costs(results['cost'], log_dir + 'pics/',
                                                {'MetaBBO-RL': metabbo,
                                                'Classic Optimizer': bbo},
                                                logged=False, pdf_fig=pdf_fig)

    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['return'], log_dir + 'pics/', pdf_fig=pdf_fig)
        self.draw_train_logger('cost', results['cost'], log_dir + 'pics/', pdf_fig=pdf_fig)

    
class MOO_Logger(Basic_Logger):
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0
        self.indicators = config.indicators
    
    def is_pareto_efficient(self,points):
        """计算帕累托前沿"""
        points = np.array(points)
        pareto_mask = np.ones(points.shape[0], dtype=bool)
        for i, p in enumerate(points):
            if pareto_mask[i]:
                pareto_mask[pareto_mask] = np.any(points[pareto_mask] < p, axis=1)
                pareto_mask[i] = True
        return points[pareto_mask]
    
    def draw_pareto_fronts(self,data: dict, output_dir: str, Name: Optional[Union[str, list]] = None):
        # 输入的数据格式为：dict[problem][algo][run][generation][objective]
        
        for problem in list(data.keys()):
            if Name is not None and ((isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name)):
                continue
            else:
                name = problem

            fig = plt.figure(figsize=(8, 6))  # 更小的画布尺寸
            is_3d = False
            algo_obj_dict = {}

            # 收集每个算法所有回合的最后一代目标值
            for algo, runs in data[problem].items():
                all_obj_values = []
                for generations in runs:
                    last_gen = np.array(generations[-1])
                    obj_values = last_gen.reshape(-1, last_gen.shape[-1])
                    if obj_values.shape[1] == 3:
                        is_3d = True
                    all_obj_values.append(obj_values)
                algo_obj_dict[algo] = np.vstack(all_obj_values)

            # 初始化画布
            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=40, azim=135)  # 更改视角
                ax.set_proj_type('persp')
            else:
                ax = fig.add_subplot(111)

            colors = ['r', 'g', 'b', 'c', 'm', 'y']

            for algo_idx, (algo, obj_values) in enumerate(algo_obj_dict.items()):
                pareto_front = self.is_pareto_efficient(obj_values)
                color = colors[algo_idx % len(colors)]
                label = f"{algo}"

                if obj_values.shape[1] == 2:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                                label=label, color=color, edgecolors='k')
                elif obj_values.shape[1] == 3:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                                label=label, color=color, edgecolors='k')

            if is_3d:
                # 更改坐标轴标签为简写，并减小Z轴标签字体
                ax.set_xlabel('X', fontsize=12, labelpad=10)
                ax.set_ylabel('Y', fontsize=12, labelpad=10)
                ax.set_zlabel('Z', fontsize=12, labelpad=-0.5,color='black')  # 减小labelpad
                

                # 微调Z轴标签的位置，使其靠近坐标轴
                ax.zaxis.set_label_coords(1.05, 0.5)  # 调整位置使标签更靠近右侧

                # 设置3D比例并调整图形位置
                ax.set_box_aspect([1.2, 1.1, 0.9])  # 将Z轴比例稍微缩小，增加Z轴的空间

            else:
                ax.set_xlabel('X', fontsize=14, labelpad=20)
                ax.set_ylabel('Y', fontsize=14, labelpad=20)

            # 调整图形与边缘的距离，特别是右边的边距
            plt.subplots_adjust(right=0.85)

            plt.legend()
            plt.grid(True)
            plt.title(f'Pareto Fronts of Algorithms on {problem}', fontsize=14)

            # 增加边距来确保Z轴标签能显示
            plt.savefig(output_dir + f'{name}_pareto_fronts.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    
    def draw_test_indicator(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, categorized: bool=False, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
                plt.savefig(output_dir + f'classic_{name}_{indicator}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_indicator(self, data: dict, output_dir: str, named_agents: dict, indicator:str,pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
        plt.savefig(output_dir + f'all_problem_{indicator}_curve.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, indicator: Optional[str] = None, Name: Optional[Union[str, list]] = None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        
        # 遍历所有问题
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D:
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1])

        # 绘制图表
        for agent in D.keys():
            plt.figure()
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])

            for a, b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)

            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            
            ylabel = indicator
            plt.ylabel(ylabel)

            plt.savefig(output_dir + f'{agent}_concrete_{indicator}_performance_hist.{fig_type}', bbox_inches='tight')
    
    def draw_boxplot(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
            plt.ylabel(f'{name} {indicator} Boxplots')
            plt.savefig(output_dir + f'{name}_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()
    
    def draw_overall_boxplot(self, data: dict, output_dir: str, indicator:str,ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
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
        plt.ylabel(f'{indicator} Boxplots')
        plt.savefig(output_dir + f'overall_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_train_logger(self, data_type: str, data: dict, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        means, stds = self.get_average_data(data_type, data, norm=norm, data_wrapper=data_wrapper)
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
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()
    
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = False, pdf_fig: bool = True) -> None:
        with open(log_dir + 'test.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = self.config.agent
        bbo = self.config.t_optimizer
        
        # 可选地读取 random_search_baseline.pkl
        if include_random_baseline:
            with open(log_dir + 'random_search_baseline.pkl', 'rb') as f:
                random = pickle.load(f)

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        for indicator in self.indicators:
            self.draw_test_indicator(results[indicator], log_dir + 'pics/', indicator, pdf_fig=pdf_fig)
            self.draw_named_average_test_indicator(results[indicator], log_dir + 'pics/', \
                {'MetaBBO-RL': metabbo, 'Classic Optimizer': bbo}, indicator, pdf_fig=pdf_fig)
    
    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['return'], log_dir + 'pics/', pdf_fig=pdf_fig)
        for indicator in self.indicators:
            self.draw_train_logger(indicator, results[indicator], log_dir + 'pics/', pdf_fig=pdf_fig)



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
                values = np.array(results[problem][agent])[:,:,-1, 3] # accuracy of 1e-4  data_wrapper_prsr
            elif data_type == 'return':
                values = results[problem][agent]
            elif data_type == 'cost':
                values = np.array(results[problem][agent])[:, :, -1]  # data_wrapper_cost
            else:
                raise ValueError('Invalid data type')
            if norm:
                values = (values - np.min((values))) / (np.max(values) - np.min(values))
            std_data[agent].append(np.std(values, -1))
            avg_data[agent].append(np.mean(values, -1))
        avg_data[agent] = np.mean(avg_data[agent], 0)
        std_data[agent] = np.mean(std_data[agent], 0)
    return avg_data, std_data

# mmo
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
                values = np.array(results[problem][agent])[:,-1, 3] # accuracy of 1e-4  data_wrapper_prsr_test
            else:
                raise ValueError('Invalid data type')
            if norm:
                values = (values - np.min((values))) / (np.max(values) - np.min(values))
            std_data[agent].append(np.std(values, -1))
            avg_data[agent].append(np.mean(values, -1))
        avg_data[agent] = np.mean(avg_data[agent], 0)
        std_data[agent] = np.mean(std_data[agent], 0)
    return avg_data, std_data  

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

# 统一moo
def get_average_costs(results: dict, norm: bool=False):
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

            # 统一了moo的要求
            if norm:
                all_agent_values = np.concatenate([np.array(results[problem][a]).flatten() for a in agents]) if 'results' in locals() else values
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

def cal_scores1(D: dict, maxf: float):
    SNE = []
    for agent in D.keys():
        values = D[agent]
        sne = 0.5 * np.sum(np.min(values, -1) / maxf)
        SNE.append(sne)
    SNE = np.array(SNE)
    score1 = (1 - (SNE - np.min(SNE)) / SNE) * 50
    return score1

def get_random_baseline(random: dict, fes: Optional[Union[int, float]]):
    baseline = {}
    if isinstance(random['T1'], dict):
        baseline['complexity_avg'] = np.log10(1/ (random['T2']['Random_search'] - random['T1']['Random_search']) / random['T0'])
    else:
        baseline['complexity_avg'] = np.log10(1/ (random['T2']['Random_search'] - random['T1']) / random['T0'])
    baseline['complexity_std'] = 0.005
    
    problems = random['cost'].keys()
    avg = []
    std = []
    for problem in problems:
        g = np.log10(fes/np.array(random['fes'][problem]['Random_search']))
        avg.append(g.mean())
        std.append(g.std())
    baseline['fes_avg'] = np.mean(avg)
    baseline['fes_std'] = np.mean(std)
    
    avg = []
    std = []
    for problem in problems:
        g = np.log10(1/(np.array(random['cost'][problem]['Random_search'])[:, -1]+1))
        avg.append(g.mean())
        std.append(g.std()) 
    baseline['cost_avg'] = np.mean(avg)
    baseline['cost_std'] = np.mean(std)
    return baseline

# 统一mmo
def gen_algorithm_complexity_table(results: dict, out_dir: str) -> None:
    save_list = []
    
    # 获取 T0, T1, T2 数据
    t0 = results['T0']
    t1 = results['T1']
    is_dict = isinstance(t1, dict)
    t2s = results['T2']
    
    ratios = []
    t2_list = []
    indexs = []
    columns = ['T0', 'T1', 'T2', '(T2-T1)/T0']
    
    # 处理 T0, T1, T2 的计算
    if isinstance(t0, (int, float)):  # 处理 T0 为单一值的情况
        t0 = [t0] * len(t2s)  # 将 t0 扩展为与 t2s 相同长度的列表
    if isinstance(t1, (int, float)):  # 处理 T1 为单一值的情况
        t1 = [t1] * len(t2s)  # 将 t1 扩展为与 t2s 相同长度的列表

    for key, value in t2s.items():
        indexs.append(key)
        t2_list.append(value)
        if is_dict:
            ratios.append((value - t1[key]) / t0)
        else:
            ratios.append((value - t1) / t0)
    
    # 构造数据表
    n = len(t2_list)
    data = np.zeros((n, 4))
    data[:, 0] = t0
    if is_dict:
        for i, (key, value) in enumerate(t1.items()):
            data[i, 1] = value
    else:
        data[:, 1] = t1
    
    data[:, 2] = t2_list
    data[:, 3] = ratios
    
    # 创建 DataFrame 并保存到 Excel
    table = pd.DataFrame(data=np.round(data, 2), index=indexs, columns=columns)
    table.to_excel(os.path.join(out_dir, 'algorithm_complexity.xlsx'))

# 统一mmo
def gen_agent_performance_table(results: dict, out_dir: str, data_type : str = 'cost') -> None:
    table_data = {}
    indexs = []
    columns = ['Worst', 'Best', 'Median', 'Mean', 'Std']
    
    # 根据 data_type 判断数据结构
    if data_type == 'cost':
        total_data = results['cost']
    elif data_type == 'pr' or data_type == 'sr':
        total_data = results
    else:
        raise ValueError('Invalid data type')

    # 遍历所有问题并计算统计数据
    for problem, value in total_data.items():
        indexs.append(problem)
        if data_type == 'cost':
            problem_data = value
        else:
            problem_data = value  # For 'pr' or 'sr', it is expected that problem_data is already structured
            
        for alg, alg_data in problem_data.items():
            n_data = []
            if data_type == 'cost':
                for run in alg_data:
                    n_data.append(run[-1])
            elif data_type in ['pr', 'sr']:
                for run in alg_data:
                    n_data.append(run[-1][3])
            
            # 计算统计值
            best = np.min(n_data)
            best = np.format_float_scientific(best, precision=3, exp_digits=3)
            worst = np.max(n_data)
            worst = np.format_float_scientific(worst, precision=3, exp_digits=3)
            median = np.median(n_data)
            median = np.format_float_scientific(median, precision=3, exp_digits=3)
            mean = np.mean(n_data)
            mean = np.format_float_scientific(mean, precision=3, exp_digits=3)
            std = np.std(n_data)
            std = np.format_float_scientific(std, precision=3, exp_digits=3)

            if alg not in table_data:
                table_data[alg] = []
            table_data[alg].append([worst, best, median, mean, std])

    # 保存结果为 Excel 文件
    for alg, data in table_data.items():
        dataframe = pd.DataFrame(data=data, index=indexs, columns=columns)
        if data_type == 'pr':
            dataframe.to_excel(os.path.join(out_dir, f'{alg}_concrete_performance_PR_1e-4_table.xlsx'))
        elif data_type == 'sr':
            dataframe.to_excel(os.path.join(out_dir, f'{alg}_concrete_performance_SR_1e-4_table.xlsx'))
        elif data_type == 'cost':
            dataframe.to_excel(os.path.join(out_dir, f'{alg}_concrete_performance_table.xlsx'))
        else:
            raise ValueError('Invalid data type')

'''
统一mmo:
添加metrics 参数：此参数控制需要计算哪些指标。
它允许灵活选择指标，可以通过传入 ['Obj', '1e-4 PR', '1e-4 SR'] 来计算这些指标，也可以仅计算 'Obj' 和 'Gap'。
'''
def gen_overall_tab(results: dict, out_dir: str, metrics: Optional[list] = None) -> None:
    # 默认计算的指标
    if metrics is None:
        metrics = ['Obj', 'Gap', 'FEs']
    else:
        # 如果传入自定义指标，使用传入的指标
        pass

    # 获取多级索引
    problems = []
    optimizers = []
    for problem in results['cost'].keys():
        problems.append(problem)
    for optimizer in results['cost'][problems[0]].keys():
        optimizers.append(optimizer)

    # 创建多级列索引
    multi_columns = pd.MultiIndex.from_product([problems, metrics], names=('Problem', 'metric'))
    df_results = pd.DataFrame(np.ones(shape=(len(optimizers), len(problems) * len(metrics))),
                              index=optimizers,
                              columns=multi_columns)

    # 如果需要计算 Gap，则需要计算 CMAES 和 Random Search 的结果
    if 'Gap' in metrics:
        cmaes_obj = {}
        rs_obj = {}
        for problem in problems:
            # 计算 CMAES 的目标值
            blobj_problem = results['cost'][problem]['DEAP_CMAES']
            cmaes_obj[problem] = np.mean([blobj_problem[run][-1] for run in range(51)])

            # 计算 Random Search 的目标值
            blobj_problem = results['cost'][problem]['Random_search']
            rs_obj[problem] = np.mean([blobj_problem[run][-1] for run in range(51)])

    # 计算每个指标
    for problem in problems:
        for optimizer in optimizers:
            # 计算每个算法的 'Obj'
            obj_problem_optimizer = results['cost'][problem][optimizer]
            objs = [obj_problem_optimizer[run][-1] for run in range(len(obj_problem_optimizer))]
            avg_obj = np.mean(objs)
            std_obj = np.std(objs)
            df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"
            
            # 如果有 'Gap' 指标，计算 Gap
            if 'Gap' in metrics:
                df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1 - (rs_obj[problem] - avg_obj) / (rs_obj[problem] - cmaes_obj[problem] + 1e-10))

            # 计算 'FEs'
            fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
            avg_fes = np.mean(fes_problem_optimizer)
            std_fes = np.std(fes_problem_optimizer)
            df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(avg_fes, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_fes, precision=3, exp_digits=1) + ")"

            # 如果有 '1e-4 PR' 或 '1e-4 SR' 指标，计算 PR 和 SR
            if '1e-4 PR' in metrics:
                pr_problem_optimizer = results['pr'][problem][optimizer]
                prs = [pr_problem_optimizer[run][-1][3] for run in range(len(pr_problem_optimizer))]
                avg_pr = np.mean(prs)
                std_pr = np.std(prs)
                df_results.loc[optimizer, (problem, '1e-4 PR')] = np.format_float_scientific(avg_pr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_pr, precision=3, exp_digits=1) + ")"
            
            if '1e-4 SR' in metrics:
                sr_problem_optimizer = results['sr'][problem][optimizer]
                srs = [sr_problem_optimizer[run][-1][3] for run in range(len(sr_problem_optimizer))]
                avg_sr = np.mean(srs)
                std_sr = np.std(srs)
                df_results.loc[optimizer, (problem, '1e-4 SR')] = np.format_float_scientific(avg_sr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_sr, precision=3, exp_digits=1) + ")"

    # 保存结果到 Excel 文件
    df_results.to_excel(out_dir + 'overall_table.xlsx')


class Logger:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0
        self.indicators = ['hv','igd']

    # mmo
    def draw_test_data(self, data_type: str, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False) -> None:
        if data_type == 'pr':
            data_name = 'PR'
        elif data_type == 'sr':
            data_name = 'SR'
        elif data_type == 'cost':
            data_name = 'Cost'
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
                    if data_type == 'pr' or data_type == 'sr':
                        values = np.array(data[name][agent])[:, :, 3]
                    elif data_type == 'cost':
                        values = np.array(data[name][agent])
                    else:
                        raise ValueError('Invalid data type')

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
                    plt.savefig(output_dir + f'{name}_log_' + data_name + '_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel(data_name)
                    plt.savefig(output_dir + f'{name}_' + data_name + '_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if data_type == 'pr' or data_type == 'sr':
                        values = np.array(data[name][agent])[:, :, 3]
                    elif data_type == 'cost':
                        values = np.array(data[name][agent])
                    else:
                        raise ValueError('Invalid data type')
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
                    plt.savefig(output_dir + f'learnable_{name}_log_' + data_name + '_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel(data_name)
                    plt.savefig(output_dir + f'learnable_{name}_' + data_name + '_curve.png', bbox_inches='tight')
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
    
    # mmo：in class Logger for rollout
    def draw_train_logger(self, data_type: str, data: dict, output_dir: str, norm: bool = False) -> None:
        means, stds = get_average_data(data_type, data, norm=norm)
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
        elif data_type == 'cost':
            plt.ylabel('Avg cost')
            plt.savefig(output_dir + f'avg_cost_curve.png', bbox_inches='tight')
        elif data_type == 'return':
            plt.ylabel('Avg Return')
            plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        else:
            raise ValueError('Invalid data type')
        plt.close()

    def draw_rank_hist(self, data_type : str, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None) -> None:
        metric, metric_std = get_test_average_data(data_type, data,) # (len(agent), )
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
            plt.title(f'The 1e-4 PR for {self.config.problem}-{self.config.difficulty}', fontsize=70)
            plt.ylabel('PR', fontsize=60)
            plt.savefig(output_dir + f'PR_1e-4_rank_hist.png', bbox_inches='tight')
        elif data_type == 'sr':
            plt.title(f'The 1e-4 SR for {self.config.problem}-{self.config.difficulty}', fontsize=70)
            plt.ylabel('SR', fontsize=60)
            plt.savefig(output_dir + f'SR_1e-4_rank_hist.png', bbox_inches='tight')
        else:
            raise ValueError('Invalid data type')

    # moo_train
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
    
    # moo
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
    
    # moo
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

    # moo
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
    
    # moo
    def draw_return(self,log_dir:str):
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('return')
        values = np.load(log_dir + 'log/return.npy')
        plt.plot(values[0], values[1])
        plt.savefig(log_dir+f'pic/return.png')
        plt.close()

    # moo
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

    # moo
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

    def draw_test_cost(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False) -> None:
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            # if logged:
            #     plt.title('log cost curve ' + name)
            # else:
            #     plt.title('cost curve ' + name)
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'{name}_cost_curve.png', bbox_inches='tight')
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'learnable_{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'learnable_{name}_cost_curve.png', bbox_inches='tight')
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
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'classic_{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'classic_{name}_cost_curve.png', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False) -> None:
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
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
                # X = np.log10(X)
                # X[0] = 0

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            # plt.xlabel('log10 FEs')
            plt.xlabel('FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.png', bbox_inches='tight')
        plt.close()

    # 与moo统一
    def draw_concrete_performance_hist(self, data: dict, output_dir: str, indicator: Optional[str] = None, Name: Optional[Union[str, list]] = None) -> None:
        D = {}
        X = []
        
        # 遍历所有问题
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)

            for agent in list(data[name].keys()):
                if agent not in D:
                    D[agent] = []
                values = np.array(data[name][agent])
                
                if indicator is not None:
                    D[agent].append(values[:, -1])  # 使用指定的 indicator
                else:
                    D[agent].append(values[:, -1] / values[:, 0])  # 默认归一化处理

        # 绘制图表
        for agent in D.keys():
            plt.figure()
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])

            for a, b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)

            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            
            ylabel = indicator if indicator is not None else 'Normalized Costs'
            plt.ylabel(ylabel)

            plt.savefig(output_dir + f'{agent}_concrete_{indicator if indicator else "performance"}_hist.png', bbox_inches='tight')

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

    # mto
    def draw_mto_train_return(self, data: list, output_dir: str) -> None: 
        log_dir = self.config.log_dir + f'/train'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        return_data = np.array(data,dtype=np.float32) #[epochs, env_cnt]
        x = np.arange(return_data.shape[0])
        y = np.mean(return_data, axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Return')
        plt.grid()
        #plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        plt.savefig(log_dir+f'pic/mto_return.png')
        plt.close()
    # mto
    def draw_mto_train_cost(self, data:list, output_dir: str) -> None:
        log_dir = self.config.log_dir + f'/train'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        cost_data = np.array(data,dtype=np.float32) #[epochs, env_cnt, task_cnt]
        x = np.arange(cost_data.shape[0])
        y = np.mean(np.mean(cost_data, axis=-1), axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Cost')
        plt.grid()
        #plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        plt.savefig(log_dir+f'pic/mto_cost.png')
        plt.close()
    #
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False) -> None:
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
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
                # X = np.log10(X)
                # X[0] = 0

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            # plt.xlabel('log10 FEs')
            plt.xlabel('FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.png', bbox_inches='tight')
        plt.close()

    # moo
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

    def draw_train_avg_cost(self, data: dict, output_dir: str, norm: bool=False) -> None:
        costs, stds = get_average_costs(data['cost'], norm=norm)
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
        plt.ylabel('Avg Cost')
        plt.grid()
        plt.savefig(output_dir + f'avg_cost_curve.png', bbox_inches='tight')
        plt.close()

    # 与moo统一
    def draw_boxplot(self, data: dict, output_dir: str, indicator: Optional[str] = None, Name: Optional[Union[str, list]] = None, ignore: Optional[list] = None) -> None:
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            
            # 遍历每个代理并收集数据
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            
            Y = np.transpose(Y)
            
            # 绘制箱型图
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            
            # 根据 indicator 是否存在来设置 ylabel
            ylabel = f'{name} {indicator} Boxplots' if indicator else f'{name} Cost Boxplots'
            plt.ylabel(ylabel)
            
            # 保存图像
            plt.savefig(output_dir + f'{name}_boxplot.png', bbox_inches='tight')
            plt.close()

    # 与moo统一
    def draw_overall_boxplot(self, data: dict, output_dir: str, indicator: Optional[str] = None, ignore: Optional[list] = None) -> None:
        problems = []
        agents = []
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        
        run = len(data[problems[0]][agents[0]])  
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        
        # 填充 values 数组，并进行归一化
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
        
        # 设置 y 轴标签
        ylabel = f'{indicator} Boxplots' if indicator else 'Cost Boxplots'
        plt.ylabel(ylabel)

        plt.savefig(output_dir + f'overall_boxplot.png', bbox_inches='tight')
        plt.close()

    # moo
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
                
    def draw_rank_hist(self, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None) -> None:
        metric, metric_std = self.aei_metric(data, random, maxFEs=self.config.maxFEs, ignore=ignore)
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
        plt.title(f'The AEI for {self.config.dim}D {self.config.problem}-{self.config.difficulty}', fontsize=70)
        plt.ylabel('AEI', fontsize=60)
        plt.savefig(output_dir + f'rank_hist.png', bbox_inches='tight')

'''
统一mmo
include_random_baseline：这个参数允许用户选择是否包括 random_search_baseline.pkl 数据。
如果设为 False，则仅生成 pr 和 sr 数据的图形，不使用 random_search_baseline 数据
'''
def post_processing_test_statics(log_dir: str, logger: Logger, include_random_baseline: bool = True) -> None:
    with open(log_dir + 'test.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # 可选地读取 random_search_baseline.pkl
    if include_random_baseline:
        with open(log_dir + 'random_search_baseline.pkl', 'rb') as f:
            random = pickle.load(f)

    if not os.path.exists(log_dir + 'tables/'):
        os.makedirs(log_dir + 'tables/')

    gen_overall_tab(results, log_dir + 'tables/')
    gen_algorithm_complexity_table(results, log_dir + 'tables/')

    # 只有在 `pr` 和 `sr` 存在时才生成性能表格
    if 'pr' in results:
        gen_agent_performance_table('pr', results['pr'], log_dir + 'tables/')
    if 'sr' in results:
        gen_agent_performance_table('sr', results['sr'], log_dir + 'tables/')

    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')

    # 根据需要选择图形生成方式
    if include_random_baseline:
        if 'pr' in results:
            logger.draw_rank_hist('pr', results['pr'], random, log_dir + 'pics/')
        if 'sr' in results:
            logger.draw_rank_hist('sr', results['sr'], random, log_dir + 'pics/')
    else:
        if 'pr' in results:
            logger.draw_rank_hist('pr', results['pr'], None, log_dir + 'pics/')
        if 'sr' in results:
            logger.draw_rank_hist('sr', results['sr'], None, log_dir + 'pics/')

    # 画其他数据的图形
    if 'pr' in results:
        logger.draw_test_data('pr', results['pr'], log_dir + 'pics/', logged=False, categorized=True)
    if 'sr' in results:
        logger.draw_test_data('sr', results['sr'], log_dir + 'pics/', logged=False, categorized=True)

    # 如果需要，可以为不同的算法绘制图形（例如 cost 图）
    if 'cost' in results:
        logger.draw_test_cost(results['cost'], log_dir + 'pics/', logged=True, categorized=True)
        logger.draw_named_average_test_costs(results['cost'], log_dir + 'pics/',
                                             {'MetaBBO-RL': ['DE_DDQN_Agent', 'RL_HPSDE_Agent', 'LDE_Agent', 'QLPSO_Agent', 'RLEPSO_Agent', 'RL_PSO_Agent', 'DEDQN_Agent'],
                                              'Classic Optimizer': ['DEAP_DE', 'DEAP_CMAES', 'DEAP_PSO', 'JDE21', 'NL_SHADE_LBC', 'GL_PSO', 'sDMS_PSO', 'MadDE', 'SAHLPSO', 'Random_search']},
                                             logged=False)

def post_processing_rollout_statics(log_dir: str, logger: Logger) -> None:
    with open(log_dir+'rollout.pkl', 'rb') as f:
        results = pickle.load(f)
    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    logger.draw_train_return(results, log_dir + 'pics/', )
    logger.draw_train_avg_cost(results, log_dir + 'pics/', )
    logger.draw_train_logger('return', results['return'], log_dir + 'pics/', )
    logger.draw_train_logger('cost', results['cost'], log_dir + 'pics/', )
    logger.draw_train_logger('pr', results['pr'], log_dir + 'pics/',)
    logger.draw_train_logger('sr', results['sr'], log_dir+'pics/',)

#logger
# class basic_Logger:
class MTO_Logger(Basic_Logger):
    def __init__(self, config):
        super().__init__(config)

    def draw_avg_train_return(self, data: list, output_dir: str) -> None: 
        plt.figure()
        return_data = np.array(data,dtype=np.float32) #[epochs, env_cnt]
        x = np.arange(return_data.shape[0])
        y = np.mean(return_data, axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Return')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_return.png', bbox_inches='tight')
        plt.close()

    def draw_avg_train_cost(self, data:list, output_dir: str) -> None:
        plt.figure()
        cost_data = np.array(data,dtype=np.float32) #[epochs, env_cnt, task_cnt]
        x = np.arange(cost_data.shape[0])
        y = np.mean(np.mean(cost_data, axis=-1), axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Cost')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_cost.png', bbox_inches='tight')
        plt.close()