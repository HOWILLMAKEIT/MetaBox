import copy
from utils import construct_problem_set
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from tqdm import tqdm
import os
from environment.basic_environment import PBO_Env, BBO_Env, MetaBBO_Env
from logger import Logger
from VectorEnv.great_para_env import ParallelEnv
import json
import torch

from agent import (
    # DE_DDQN_Agent,
    # DEDQN_Agent,
    RL_HPSDE_Agent,
    LDE_Agent,
    # QLPSO_Agent,
    RLEPSO_Agent,
    RL_PSO_Agent,
    L2L_Agent,
    # RL_DAS_Agent,
    LES_Agent,
    # NRLPSO_Agent,
    # Symbol_Agent,
)
from optimizer import (
    DE_DDQN_Optimizer,
    DEDQN_Optimizer,
    RL_HPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RL_PSO_Optimizer,
    L2L_Optimizer,
    GLEET_Optimizer,
    RL_DAS_Optimizer,
    LES_Optimizer,
    NRLPSO_Optimizer,
    SYMBOL_Optimizer,
    RLDE_AFL_Optimizer,
    Surr_RLDE_Optimizer,
    RLEMMO_Optimizer,

    DEAP_DE,
    JDE21,
    MadDE,
    NL_SHADE_LBC,

    DEAP_PSO,
    GL_PSO,
    sDMS_PSO,
    SAHLPSO,

    DEAP_CMAES,
    Random_search,
    BayesianOptimizer
)

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLDE_AFL_Agent,
    SYMBOL_Agent,
    RL_DAS_Agent,
    Surr_RLDE_Agent,
    RLEMMO_Agent
)

from VectorEnv.great_para_env import ParallelEnv

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLDE_AFL_Agent,
    SYMBOL_Agent,
    RL_DAS_Agent,
    Surr_RLDE_Agent
)

def cal_t0(dim, fes):
    T0 = 0
    for i in range(10):
        start = time.perf_counter()
        for _ in range(fes):
            x = np.random.rand(dim)
            x + x
            x / (x+2)
            x * x
            np.sqrt(x)
            np.log(x)
            np.exp(x)
        end = time.perf_counter()
        T0 += (end - start) * 1000
    # ms
    return T0/10


def cal_t1(problem, dim, fes):
    T1 = 0
    for i in range(10):
        x = np.random.rand(fes, dim)
        start = time.perf_counter()
        # for i in range(fes):
        #     problem.eval(x[i])
        problem.eval(x)
        end = time.perf_counter()
        T1 += (end - start) * 1000
    # ms
    return T1/10


class Tester(object):
    def __init__(self, config):
        """
        Initializes the tester class with the given configuration.
        Args:
            config (object): Configuration object containing the following attributes:
                - agent (list): List of agent keys to be used.
                - test_log_dir (str): Directory path for storing test logs.
                - problem (str): Problem type to be solved.
                - is_train (bool): Flag indicating whether training is enabled.
                - t_optimizer (list): List of traditional optimizer names.
                - seed (int): Random seed for reproducibility.
        Attributes:
            key_list (list): List of agent keys from the configuration.
            log_dir (str): Directory for storing test logs.
            config (object): Configuration object.
            test_set (object): Constructed problem set for testing.
            seed (range): Range of seeds for testing.
            test_results (dict): Dictionary for logging test results.
            agent_for_cp (list): List of learnable agents.
            agent_name_list (list): List of agent names.
            l_optimizer_for_cp (list): List of learnable optimizers.
            t_optimizer_for_cp (list): List of traditional optimizers.
        Raises:
            KeyError: If a key in `key_list` is missing in the `model.json` file.
        Notes:
            - Initializes directories and problem sets based on the configuration.
            - Loads agents and optimizers from the `model.json` file.
            - Logs the number and types of agents and optimizers.
            - Seeds random number generators for reproducibility.
        """
        self.key_list = config.agent
        self.log_dir = config.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = config

        if self.config.problem[-6:]=='-torch':
            self.config.problem=self.config.problem[:-6]

        if config.problem =='bbob-surrogate':
            config.is_train = False

        _, self.test_set = construct_problem_set(self.config)
        # if 'L2L_Agent' in config.agent_for_cp or 'L2L_Agent' == config.agent:
        #     pre_problem=config.problem
        #     config.problem=pre_problem+'-torch'
        #     _,self.torch_test_set = construct_problem_set(config)
        #     config.problem=pre_problem
        
        self.seed = range(51)
        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T0': 0.,
                             'T1': {},
                             'T2': {},
                             'pr': {},
                             'sr': {}}

        # prepare experimental optimizers and agents
        self.agent_for_cp = []
        self.agent_name_list = []
        self.l_optimizer_for_cp = []
        self.t_optimizer_for_cp = []

        with open('model.json', 'r', encoding = 'utf-8') as f:
            json_data = json.load(f)
        for key in self.key_list:
            if key not in json_data.keys():
                raise KeyError(f"Missing key '{key}' in model.json")

            # get key
            baseline = json_data[key]
            if "Agent" in baseline.keys():
                agent_name = baseline["Agent"]
                l_optimizer = baseline['Optimizer']
                dir = baseline['dir']
                # get agent
                self.agent_name_list.append(key)
                with open(dir, 'rb') as f:
                    self.agent_for_cp.append(pickle.load(f))
                self.l_optimizer_for_cp.append(eval(l_optimizer)(copy.deepcopy(config)))

            else:
                t_optimizer = baseline['Optimizer']
                self.t_optimizer_for_cp.append(eval(t_optimizer)(copy.deepcopy(config)))

        for optimizer in config.t_optimizer:
            self.t_optimizer_for_cp.append(eval(optimizer)(copy.deepcopy(config)))
        # logging
        if len(self.agent_for_cp) == 0:
            print('None of learnable agent')
        else:
            print(f'there are {len(self.agent_for_cp)} agent')
            for a, l_optimizer in zip(self.agent_name_list, self.l_optimizer_for_cp):
                print(f'learnable_agent:{a},l_optimizer:{type(l_optimizer).__name__}')

        if len(self.t_optimizer_for_cp) == 0:
            print('None of traditional optimizer')
        else:
            print(f'there are {len(self.t_optimizer_for_cp)} traditional optimizer')
            for t_optmizer in self.t_optimizer_for_cp:
                print(f't_optimizer:{type(t_optmizer).__name__}')

        for agent_name in self.agent_name_list:
            self.test_results['T1'][agent_name] = 0.
            self.test_results['T2'][agent_name] = 0.
        for optimizer in self.t_optimizer_for_cp:
            self.test_results['T1'][type(optimizer).__name__] = 0.
            self.test_results['T2'][type(optimizer).__name__] = 0.

        for problem in self.test_set.data:
            self.test_results['cost'][problem.__str__()] = {}
            self.test_results['fes'][problem.__str__()] = {}
            for agent_name in self.agent_name_list:
                self.test_results['cost'][problem.__str__()][agent_name] = []  # 51 np.arrays
                self.test_results['fes'][problem.__str__()][agent_name] = []  # 51 scalars
            for optimizer in self.t_optimizer_for_cp:
                self.test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
                self.test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51 scalars

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def test(self):
        """
        Executes the testing process for the configured optimization problems and agents.

        This method evaluates the performance of learnable and traditional optimizers
        on a set of test problems. It calculates metrics such as T0, T1, and T2, and 
        logs the results for further analysis. A progress bar is displayed to track 
        the testing progress.

        Steps:
        1. Calculate T0 based on problem dimensions and maximum function evaluations.
        2. Iterate through the test set of problems.
        3. For each problem:
           - Evaluate learnable optimizers using agents.
           - Optionally evaluate traditional optimizers (commented out in the code).
        4. Log results including cost, function evaluations (FEs), and timing metrics.

        Attributes:
            self.config.run_time (int): The runtime configuration for testing.
            self.config.dim (int): The dimensionality of the optimization problems.
            self.config.maxFEs (int): The maximum number of function evaluations allowed.
            self.test_set (list): A list of test problems to evaluate.
            self.agent_for_cp (list): A list of agents for learnable optimizers.
            self.l_optimizer_for_cp (list): A list of learnable optimizers.
            self.t_optimizer_for_cp (list): A list of traditional optimizers (currently commented out).
            self.seed (list): A list of random seeds for reproducibility.
            self.test_results (dict): A dictionary to store testing results.
            self.log_dir (str): Directory to save test results.

        Notes:
            - The method uses the `tqdm` library to display a progress bar.
            - Some parts of the code are commented out, indicating potential future extensions.
            - Results are saved in pickle files for further analysis.

        Raises:
            Any exceptions raised during the testing process are not explicitly handled
            in this method and will propagate to the caller.
        """
        # todo 测试并行方式有多种 得考虑下

        print(f'start testing: {self.config.run_time}')
        # calculate T0
        T0 = cal_t0(self.config.dim, self.config.maxFEs)
        self.test_results['T0'] = T0
        # calculate T1
        # T1 = cal_t1(self.test_set[0], self.config.dim, self.config.maxFEs)
        # self.test_results['T1'] = T1
        pbar_len = (len(self.t_optimizer_for_cp) + len(self.agent_for_cp)) * self.test_set.N * 51
        with tqdm(range(pbar_len), desc='Testing') as pbar:
            for i,problem in enumerate(self.test_set):

                # run learnable optimizer
                for agent_id,(agent,optimizer) in enumerate(zip(self.agent_for_cp,self.l_optimizer_for_cp)):
                    T1 = 0
                    T2 = 0
                    for run in range(51):
                        env_list = [PBO_Env(p, copy.deepcopy(optimizer)) for p in problem]

                        start = time.perf_counter()
                        info = agent.rollout_batch_episode(envs = env_list,
                                                           seeds = self.seed[run],
                                                           para_mode = 'dummy',
                                                           asynchronous = None,
                                                           num_cpus = 1,
                                                           num_gpus = 0,
                        )
                        # np.random.seed(self.seed[run])
                        # problem.reset() 这里env_list reset有了
                        # construct an ENV for (problem,optimizer)
                        # env = PBO_Env(problem,optimizer)

                        # info = agent.rollout_episode(env)
                        # cost = info['cost']
                        # while len(cost) < 51:
                        #     cost.append(cost[-1])
                        # fes = info['fes']
                        # end = time.perf_counter()
                        # if i == 0:
                        #     T2 += (end - start) * 1000  # ms
                        #     T1 += env.problem.T1
                        # self.test_results['cost'][problem.__str__()][self.agent_name_list[agent_id]].append(cost)
                        # self.test_results['fes'][problem.__str__()][self.agent_name_list[agent_id]].append(fes)
                        pbar_info = {'agent': agent.__str__(),
                                     'run': run,
                                     }

                        # pbar_info = {'problem': problem.__str__(),
                        #              'optimizer': self.agent_name_list[agent_id],
                        #              'run': run,
                        #              'cost': cost[-1],
                        #              'fes': fes}
                        pbar.set_postfix(pbar_info)
                        pbar.update(len(env_list))
                    # if i == 0:
                    #     self.test_results['T1'][self.agent_name_list[agent_id]] = T1/51
                    #     self.test_results['T2'][self.agent_name_list[agent_id]] = T2/51
                    #     if type(agent).__name__ == 'L2L_Agent':
                    #         self.test_results['T1'][self.agent_name_list[agent_id]] *= self.config.maxFEs/100
                    #         self.test_results['T2'][self.agent_name_list[agent_id]] *= self.config.maxFEs/100
                # run traditional optimizer
        #         for optimizer in self.t_optimizer_for_cp:
        #             T1 = 0
        #             T2 = 0
        #             for run in range(51):
        #                 start = time.perf_counter()
        #                 np.random.seed(self.seed[run])
        #
        #                 problem.reset()
        #                 info = optimizer.run_episode(problem)
        #                 cost = info['cost']
        #                 while len(cost) < 51:
        #                     cost.append(cost[-1])
        #                 fes = info['fes']
        #                 end = time.perf_counter()
        #                 if i == 0:
        #                     T1 += problem.T1
        #                     T2 += (end - start) * 1000  # ms
        #                 self.test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
        #                 self.test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
        #                 pbar_info = {'problem': problem.__str__(),
        #                              'optimizer': type(optimizer).__name__,
        #                              'run': run,
        #                              'cost': cost[-1],
        #                              'fes': fes, }
        #                 pbar.set_postfix(pbar_info)
        #                 pbar.update(1)
        #             if i == 0:
        #                 self.test_results['T1'][type(optimizer).__name__] = T1/51
        #                 self.test_results['T2'][type(optimizer).__name__] = T2/51
        #                 if type(optimizer).__name__ == 'BayesianOptimizer':
        #                     self.test_results['T1'][type(optimizer).__name__] *= (self.config.maxFEs/self.config.bo_maxFEs)
        #                     self.test_results['T2'][type(optimizer).__name__] *= (self.config.maxFEs/self.config.bo_maxFEs)
        # with open(self.log_dir + 'test.pkl', 'wb') as f:
        #     pickle.dump(self.test_results, f, -1)
        # random_search_results = test_for_random_search(self.config)
        # with open(self.log_dir + 'random_search_baseline.pkl', 'wb') as f:
        #     pickle.dump(random_search_results, f, -1)
    def test_1(self):
        """
        Executes a testing procedure for evaluating agents and traditional optimizers
        on a set of problems. The method performs parallelized testing for both 
        meta-heuristic agents and traditional optimizers, and tracks progress using 
        a progress bar.

        The testing process involves:
        1. Running each agent on a batch of environments created for each problem.
        2. Running traditional optimizers on the same problems for comparison.

        Attributes:
            self.config.run_time (int): The runtime configuration for the test.
            self.config.test_run (int): Number of test runs for each agent/optimizer.
            self.agent_for_cp (list): List of agents to be tested.
            self.t_optimizer_for_cp (list): List of traditional optimizers to be tested.
            self.test_set.data (list): List of problems to test against.
            self.test_set.N (int): Number of problems in the test set.

        Workflow:
            - For each problem in the test set:
                - For each agent:
                    - Create a batch of environments for the problem.
                    - Perform a batch rollout using the agent.
                    - Update the progress bar with agent and problem information.
                - For each traditional optimizer:
                    - Create a batch of environments for the optimizer.
                    - Run a batch episode using the optimizer.
                    - Update the progress bar with optimizer and problem information.

        Progress Bar:
            - Displays the current testing progress.
            - Shows information about the current agent/optimizer and problem.

        Notes:
            - The method uses `tqdm` for progress tracking.
            - Environments and problems are deep-copied to ensure isolation between tests.
            - Parallelization is achieved using the `ParallelEnv` class.

        """
        # todo 第一种 并行是 agent for 循环
        # todo 每个 agent 做一个问题 x test_run 的列表环境
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        seed_list = list(range(1, test_run + 1))
        pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * self.test_set.N
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set.data):
                for agent_id, (agent, optimizer) in enumerate(zip(self.agent_for_cp, self.l_optimizer_for_cp)):
                    # for agent an env_list [1 * len(test_run)]

                    env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(optimizer)) for _ in range(test_run)]
                    meta_test_data = agent.rollout_batch_episode(envs = env_list,
                                                                 seeds = seed_list,
                                                                 para_mode = 'dummy',
                                                                 asynchronous = None,
                                                                 num_cpus = 1,
                                                                 num_gpus = 0,
                                                                 )
                    # meta_test_data : {cost, fes, return}
                    pbar_info = {'MetaBBO': agent.__str__(),
                                 'problem': problem.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                # run traditional optimizer
                for optimizer in self.t_optimizer_for_cp:
                    # env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run)]
                    # action_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    # env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)

                    env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run)]
                    problem_list = [{'problem': copy.deepcopy(problem)} for _ in range(test_run)]
                    # problem_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)
                    if seed_list is not None:
                        env.seed(seed_list)
                    test_data = env.customized_method('run_batch_episode', problem_list) # List:[dict{cost, fes}] (test_run)
                    pbar_info = {'BBO': type(optimizer).__name__,
                                 'problem': problem.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)

    def test_2(self):
        """
        Perform testing using both MetaBBO agents and traditional optimizers.

        This method evaluates the performance of MetaBBO agents and traditional 
        optimizers on a test set of problems. It uses parallel environments to 
        execute batch episodes for each agent and optimizer, and tracks progress 
        using a progress bar.

        Steps:
        1. For each problem in the test set:
           - For each MetaBBO agent:
             - Create a list of environments for the agent to interact with.
             - Perform batch rollouts using the agent and collect test data.
             - Update the progress bar with MetaBBO-specific information.
           - For each traditional optimizer:
             - Create a list of environments and problem configurations.
             - Perform batch rollouts using the optimizer and collect test data.
             - Update the progress bar with optimizer-specific information.

        Attributes:
            self.config.run_time (int): The runtime configuration for testing.
            self.config.test_run (int): Number of test runs per problem.
            self.config.test_batch_size (int): Batch size for testing.
            self.agent_for_cp (list): List of MetaBBO agents to test.
            self.l_optimizer_for_cp (list): List of optimizers associated with MetaBBO agents.
            self.t_optimizer_for_cp (list): List of traditional optimizers to test.
            self.test_set (iterable): The set of problems to test on.

        Progress Bar:
            - Displays the current testing progress.
            - Updates with information about the current MetaBBO agent or optimizer.

        Notes:
            - The environments and problems are deep-copied to ensure isolation.
            - The method uses a parallel environment for traditional optimizers.

        Raises:
            Any exceptions raised during the testing process will propagate.

        """
        # todo 第二种 并行是 agent for 循环
        # todo 每个 agent 做 bs 个问题 x test_run 的列表环境
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        bs = self.config.test_batch_size

        seed_list = list(range(1, test_run + 1)) * bs # test_run * bs
        pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * (self.test_set.N // bs + self.test_set.N % bs)
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set):
                for agent_id, (agent, optimizer) in enumerate(zip(self.agent_for_cp, self.l_optimizer_for_cp)):
                    # for agent an env_list [bs * len(test_run)]
                    # [F1 F1 F1 F2 F2 F2 F3 F3 F3...]
                    env_list = [
                        PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer))
                        for p in problem  # bs
                        for _ in range(test_run) # test_run
                    ]

                    meta_test_data = agent.rollout_batch_episode(envs = env_list,
                                                                 seeds = seed_list,
                                                                 para_mode = 'dummy',
                                                                 asynchronous = None,
                                                                 num_cpus = 1,
                                                                 num_gpus = 0,
                                                                 )
                    # meta_test_data : {cost, fes, return}
                    pbar_info = {'MetaBBO': agent.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                # run traditional optimizer
                for optimizer in self.t_optimizer_for_cp:
                    env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run) for _ in range(bs)]
                    problem_list = [{'problem': copy.deepcopy(p)} for p in problem for _ in range(test_run)]
                    # problem_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)
                    env.seed(seed_list)
                    test_data = env.customized_method('run_batch_episode', problem_list)  # List:[dict{cost, fes}] (test_run * bs)
                    pbar_info = {'BBO': type(optimizer).__name__,
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)




    def test_3(self):
        """
        Perform testing using both MetaBBO and traditional BBO approaches in parallel.

        This method evaluates the performance of agents and optimizers on a test set of problems
        using parallel environments. It involves running multiple test cases in batches and 
        collecting results for analysis.

        Key Steps:
        - Initialize agents and environments for MetaBBO testing.
        - Run MetaBBO testing in parallel using the `ParallelEnv` class.
        - Initialize optimizers and problems for traditional BBO testing.
        - Run traditional BBO testing in parallel using the `ParallelEnv` class.
        - Update progress bar with testing status for both MetaBBO and BBO.

        Example Workflow:
        - For a batch size (bs) of 3, test runs (test_run) of 2, and agents A1 and A2:
          - MetaBBO testing involves agents interacting with environments in parallel.
          - Traditional BBO testing involves optimizers solving problems in parallel.

        Attributes:
        - self.config.run_time: The runtime configuration for testing.
        - self.config.test_run: Number of test runs to perform.
        - self.config.test_batch_size: Batch size for testing.
        - self.test_set: The set of problems to test.
        - self.agent_for_cp: List of agents for MetaBBO testing.
        - self.l_optimizer_for_cp: List of optimizers for MetaBBO testing.
        - self.t_optimizer_for_cp: List of optimizers for traditional BBO testing.

        Progress Bar:
        - Displays the progress of testing with descriptions for MetaBBO and BBO phases.

        Notes:
        - The method uses deep copies of agents, optimizers, and problems to ensure 
          independence between parallel executions.
        - Parallel execution is managed using the `ParallelEnv` class with `ray` as the backend.

        Raises:
        - Any exceptions raised during the execution of `run_batch_episode` in `ParallelEnv`.

        """
        # todo 第三种 并行是 agent * bs 个问题 * run
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        bs = self.config.test_batch_size

        seed_list = list(range(1, test_run + 1)) # test_run
        pbar_len = 2 * (self.test_set.N // bs + self.test_set.N % bs)
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set):
                # env_list [bs * len(agent) * len(test_run)]
                # agent_list [bs * len(agent) * len(test_run)]


                '''
                    example: bs = 3 test_run = 2 agent A1 A2
                    [   A1   |   A1   |   A1   |   A1   |   A1   |   A1   |   A2   |   A2   |   A2   |   A2   |   A2   |   A2   ]
                    [O1_F1_r1|O1_F1_r2|O1_F2_r1|O1_F2_r2|O1_F3_r1|O1_F3_r2|O2_F1_r1|O2_F1_r2|O2_F2_r1|O2_F2_r2|O2_F3_r1|O2_F3_r2]
                '''
                agent_list = [MetaBBO_Env(copy.deepcopy(agent)) for agent in self.agent_for_cp for _ in range(test_run * bs)]
                env_list = [{'env': PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), 'seed': seed} for optimizer in self.l_optimizer_for_cp for p in problem for seed in seed_list]

                # env_list = []
                #
                # # 拼字典
                # for optimizer in self.t_optimizer_for_cp:
                #     temp_list = [{'env': PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), 'seed': seed} for p in problem for seed in seed_list]
                #     env_list = env_list + temp_list

                # agent parallel

                MetaBBO = ParallelEnv(agent_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)

                meta_test_data = MetaBBO.customized_method('run_batch_episode', env_list)
                pbar_info = {'Testing': "MetaBBO",
                             }
                pbar.set_postfix(pbar_info)
                pbar.update(1)

                # tradition
                optimizer_list = [BBO_Env(copy.deepcopy(optimizer)) for optimizer in self.t_optimizer_for_cp for _ in range(test_run * bs)]
                problem_list = [{'problem': copy.deepcopy(p)} for _ in range(len(self.t_optimizer_for_cp)) for p in problem for _ in range(test_run)]


                # optimizer_list = []
                # problem_list = []
                # for optimizer in self.t_optimizer_for_cp:
                #     for _ in range(test_run * bs):
                #         optimizer_list.append(BBO_Env(copy.deepcopy(optimizer)))
                #     problem_list.append({'problem': copy.deepcopy(p)} for p in problem for _ in range(test_run))
                BBO = ParallelEnv(optimizer_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)
                BBO.seed(seed_list * bs * len(self.agent_for_cp))
                test_data = BBO.customized_method('run_batch_episode', problem_list)
                pbar_info = {'Testing': "BBO",}
                pbar.set_postfix(pbar_info)
                pbar.update(1)

                '''
                    example: bs = 3 test_run = 2 agent A1 A2
                    [   A1   |   A1   |   A1   |   A1   |   A1   |   A1   |   A2   |   A2   |   A2   |   A2   |   A2   |   A2   ]
                    [O1_F1_r1|O1_F1_r2|O1_F2_r1|O1_F2_r2|O1_F3_r1|O1_F3_r2|O2_F1_r1|O2_F1_r2|O2_F2_r1|O2_F2_r2|O2_F3_r1|O2_F3_r2]
                '''
                agent_list = [MetaBBO_Env(copy.deepcopy(agent)) for agent in self.agent_for_cp for _ in range(test_run * bs)]
                env_list = [{'env': PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), 'seed': seed} for optimizer in self.l_optimizer_for_cp for p in problem for seed in seed_list]

                # env_list = []
                #
                # # 拼字典
                # for optimizer in self.t_optimizer_for_cp:
                #     temp_list = [{'env': PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), 'seed': seed} for p in problem for seed in seed_list]
                #     env_list = env_list + temp_list

                # agent parallel

                MetaBBO = ParallelEnv(agent_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)

                meta_test_data = MetaBBO.customized_method('run_batch_episode', env_list)
                pbar_info = {'Testing': "MetaBBO",
                             }
                pbar.set_postfix(pbar_info)
                pbar.update(1)

                # tradition
                optimizer_list = [BBO_Env(copy.deepcopy(optimizer)) for optimizer in self.t_optimizer_for_cp for _ in range(test_run * bs)]
                problem_list = [{'problem': copy.deepcopy(p)} for _ in range(len(self.t_optimizer_for_cp)) for p in problem for _ in range(test_run)]


                # optimizer_list = []
                # problem_list = []
                # for optimizer in self.t_optimizer_for_cp:
                #     for _ in range(test_run * bs):
                #         optimizer_list.append(BBO_Env(copy.deepcopy(optimizer)))
                #     problem_list.append({'problem': copy.deepcopy(p)} for p in problem for _ in range(test_run))
                BBO = ParallelEnv(optimizer_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)
                BBO.seed(seed_list * bs * len(self.agent_for_cp))
                test_data = BBO.customized_method('run_batch_episode', problem_list)
                pbar_info = {'Testing': "BBO",}
                pbar.set_postfix(pbar_info)
                pbar.update(1)




































def rollout_batch(config):
    """
    Executes a batch rollout process for a given configuration.
    This function performs rollouts for a set of problems using specified agents 
    and optimizers. It evaluates the performance of agents across multiple 
    checkpoints and logs the results.
    Args:
        config (object): Configuration object containing the following attributes:
            - run_time (str): Runtime information for logging.
            - problem (str): Name of the problem to solve. If it ends with '-torch', 
              it is adjusted accordingly.
            - is_train (bool): Indicates whether the problem is for training.
            - train_batch_size (int): Batch size for training.
            - agent_for_rollout (list): List of agent names to use for rollouts.
            - agent_load_dir (str): Directory path to load agent checkpoints.
            - n_checkpoint (int): Number of checkpoints to evaluate.
            - optimizer_for_rollout (list): List of optimizer names to use for rollouts.
            - rollout_log_dir (str): Directory path to save rollout results.
    Returns:
        None: The function saves the rollout results to a file in the specified 
        `rollout_log_dir`.
    Notes:
        - The function constructs problem sets and loads agent checkpoints.
        - It evaluates agents on problems using specified optimizers and logs 
          metrics such as cost, pr, sr, and return.
        - Results are saved in a dictionary structure and serialized to a file.
        - A progress bar is displayed during the rollout process.
    Raises:
        FileNotFoundError: If a checkpoint file is not found in the specified directory.
        Exception: For any other issues during the rollout process.
    Example:
        config = {
            'run_time': '2023-01-01',
            'problem': 'bbob-surrogate',
            'agent_for_rollout': ['Agent1', 'Agent2'],
            'agent_load_dir': '/path/to/agents/',
            'n_checkpoint': 5,
            'optimizer_for_rollout': ['Optimizer1', 'Optimizer2'],
            'rollout_log_dir': '/path/to/logs/'
        }
        rollout_batch(config)
    """
    print(f'start rollout: {config.run_time}')

    if config.problem[-6:]=='-torch':
        config.problem=config.problem[:-6]

    if config.problem == 'bbob-surrogate':
        config.is_train = False
    config.train_batch_size = 1
    train_set,_=construct_problem_set(config)
    # if 'L2L_Agent' in config.agent_for_rollout:
    #     pre_problem=config.problem
    #     config.problem=pre_problem+'-torch'
    #     torch_train_set,_ = construct_problem_set(config)
    #     config.problem=pre_problem

    agent_load_dir=config.agent_load_dir
    n_checkpoint=config.n_checkpoint

    train_rollout_results = {'cost': {},
                             'pr': {},
                             'sr': {},
                             'return':{}}

    agent_for_rollout=config.agent_for_rollout

    load_agents={}
    for agent_name in agent_for_rollout:
        load_agents[agent_name]=[]
        for checkpoint in range(0,n_checkpoint+1):
            file_path = agent_load_dir+ agent_name + '/' + 'checkpoint'+str(checkpoint) + '.pkl'
            with open(file_path, 'rb') as f:
                load_agents[agent_name].append(pickle.load(f))

    optimizer_for_rollout=[]
    for optimizer_name in config.optimizer_for_rollout:
        optimizer_for_rollout.append(eval(optimizer_name)(copy.deepcopy(config)))
    for problem in train_set:
        train_rollout_results['cost'][problem.__str__()] = {}
        train_rollout_results['pr'][problem.__str__()] = {}
        train_rollout_results['sr'][problem.__str__()] = {}
        train_rollout_results['return'][problem.__str__()] = {}
        for agent_name in agent_for_rollout:
            train_rollout_results['cost'][problem.__str__()][agent_name] = []
            train_rollout_results['pr'][problem.__str__()][agent_name] = []
            train_rollout_results['sr'][problem.__str__()][agent_name] = []
            train_rollout_results['return'][problem.__str__()][agent_name] = []
            for checkpoint in range(0,n_checkpoint+1):
                train_rollout_results['cost'][problem.__str__()][agent_name].append([])
                train_rollout_results['pr'][problem.__str__()][agent_name].append([])
                train_rollout_results['sr'][problem.__str__()][agent_name].append([])
                train_rollout_results['return'][problem.__str__()][agent_name].append([])

    pbar_len = (len(agent_for_rollout)) * train_set.N * (n_checkpoint+1)
    seed_list = list(range(1, 5 + 1))
    with tqdm(range(pbar_len), desc='Rollouting') as pbar:
        for agent_name,optimizer in zip(agent_for_rollout,optimizer_for_rollout):
            return_list=[]  # n_checkpoint + 1
            agent=None
            for checkpoint in range(0,n_checkpoint+1):
                agent=load_agents[agent_name][checkpoint]
                # return_sum=0
                for i,problem in enumerate(train_set):
                    env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(optimizer)) for _ in range(5)]
                    meta_rollout_data = agent.rollout_batch_episode(envs = env_list,
                                                                seeds = seed_list,
                                                                para_mode = 'dummy',
                                                                asynchronous = None,
                                                                num_cpus = 1,
                                                                num_gpus = 0,
                                                                )
                    cost=meta_rollout_data['cost']
                    pr=meta_rollout_data['pr']
                    sr=meta_rollout_data['sr']
                    R=meta_rollout_data['return']

                    train_rollout_results['cost'][problem.__str__()][agent_name][checkpoint] = np.array(cost)
                    train_rollout_results['pr'][problem.__str__()][agent_name][checkpoint] = np.array(pr)
                    train_rollout_results['sr'][problem.__str__()][agent_name][checkpoint] = np.array(sr)
                    train_rollout_results['return'][problem.__str__()][agent_name][checkpoint] = np.array(R)

                    pbar_info = {'problem': problem.__str__(),
                                'agent': type(agent).__name__,
                                'checkpoint': checkpoint,}
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
            
    log_dir=config.rollout_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'rollout.pkl', 'wb') as f:
        pickle.dump(train_rollout_results, f, -1)


def test_for_random_search(config):
    """
    Tests the performance of a random search optimizer on a given problem set.

    Args:
        config (object): Configuration object containing the following attributes:
            - problem (str): The type of problem to solve (e.g., 'bbob-surrogate').
            - is_train (bool): Flag indicating whether the problem is for training.
            - dim (int): Dimensionality of the problem.
            - maxFEs (int): Maximum number of function evaluations.

    Returns:
        dict: A dictionary containing the test results with the following structure:
            - 'cost': A nested dictionary where keys are problem names and values are 
              dictionaries mapping optimizer names to lists of costs (51 numpy arrays).
            - 'fes': A nested dictionary where keys are problem names and values are 
              dictionaries mapping optimizer names to lists of function evaluations (51 scalars).
            - 'T0': A float representing the calculated T0 value based on problem dimensions and maxFEs.
            - 'T1': A dictionary mapping optimizer names to average T1 values (time taken by the problem).
            - 'T2': A dictionary mapping optimizer names to average T2 values (execution time in milliseconds).

    Notes:
        - The function constructs the problem set, initializes the optimizer, and logs results.
        - It calculates T0 based on the problem's dimensionality and maximum function evaluations.
        - The testing process involves running the optimizer on each problem in the set for 51 runs.
        - Progress is displayed using a tqdm progress bar with detailed information about the current run.
    """
    # get entire problem set
    if config.problem == 'bbob-surrogate':
        config.is_train = False

    train_set, test_set = construct_problem_set(config)
    entire_set = train_set + test_set
    # get optimizer
    optimizer = Random_search(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {'cost': {},
                    'fes': {},
                    'T0': 0.,
                    'T1': {},
                    'T2': {}}
    test_results['T1'][type(optimizer).__name__] = 0.
    test_results['T2'][type(optimizer).__name__] = 0.
    for problem in entire_set:
        test_results['cost'][problem.__str__()] = {}
        test_results['fes'][problem.__str__()] = {}
        test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
        test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51 scalars
    # calculate T0
    test_results['T0'] = cal_t0(config.dim, config.maxFEs)
    # begin testing
    seed = range(51)
    pbar_len = len(entire_set) * 51
    with tqdm(range(pbar_len), desc='test for random search') as pbar:
        for i, problem in enumerate(entire_set):
            T1 = 0
            T2 = 0
            for run in range(51):
                start = time.perf_counter()
                np.random.seed(seed[run])
                info = optimizer.run_episode(problem)
                cost = info['cost']
                while len(cost) < 51:
                    cost.append(cost[-1])
                fes = info['fes']
                end = time.perf_counter()
                if i == 0:
                    T1 += problem.T1
                    T2 += (end - start) * 1000  # ms
                test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
                test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
                pbar_info = {'problem': problem.__str__(),
                             'optimizer': type(optimizer).__name__,
                             'run': run,
                             'cost': cost[-1],
                             'fes': fes, }
                pbar.set_postfix(pbar_info)
                pbar.update(1)
            if i == 0:
                test_results['T1'][type(optimizer).__name__] = T1 / 51
                test_results['T2'][type(optimizer).__name__] = T2 / 51
    return test_results


def name_translate(problem):
    """
    Translates a given problem identifier into a descriptive category name.

    Args:
        problem (str): The identifier of the problem. Expected values are:
            - 'bbob' or 'bbob-torch': Translates to 'Synthetic'.
            - 'bbob-noisy' or 'bbob-noisy-torch': Translates to 'Noisy-Synthetic'.
            - 'protein' or 'protein-torch': Translates to 'Protein-Docking'.

    Returns:
        str: The descriptive category name corresponding to the problem identifier.

    Raises:
        ValueError: If the provided problem identifier is not recognized.
    """
    if problem in ['bbob', 'bbob-torch']:
        return 'Synthetic'
    elif problem in ['bbob-noisy', 'bbob-noisy-torch']:
        return 'Noisy-Synthetic'
    elif problem in ['protein', 'protein-torch']:
        return 'Protein-Docking'
    else:
        raise ValueError(problem + ' is not defined!')


def mgd_test(config):
    """
    Perform the MGD (Model Generalization and Deployment) test for a given configuration.

    This function evaluates the performance of two agents (`model_from` and `model_to`) 
    on a test set of problems using a specified optimizer. It logs the results, calculates 
    metrics, and saves the outcomes for further analysis.

    Args:
        config (object): Configuration object containing the following attributes:
            - run_time (str): Identifier for the runtime of the test.
            - problem (str): The type of problem to test (e.g., 'bbob-surrogate').
            - is_train (bool): Flag indicating whether the model is in training mode.
            - model_from (str): Path to the serialized `model_from` agent.
            - model_to (str): Path to the serialized `model_to` agent.
            - optimizer (str): Name of the optimizer to use.
            - agent (str): Name of the agent being tested.
            - dim (int): Dimensionality of the problem.
            - maxFEs (int): Maximum number of function evaluations.
            - mgd_test_log_dir (str): Directory to save the test logs.
            - problem_from (str): Source problem for MGD.
            - difficulty_from (str): Difficulty level of the source problem.
            - problem_to (str): Target problem for MGD.
            - difficulty_to (str): Difficulty level of the target problem.

    Workflow:
        1. Load the test set of problems and the agents (`model_from` and `model_to`).
        2. Initialize the optimizer and logging structures.
        3. Calculate the baseline time metric (T0).
        4. For each problem in the test set:
            - Evaluate both agents over 51 runs.
            - Log the cost and function evaluations (FEs) for each run.
            - Calculate time metrics (T1 and T2) for the first problem.
        5. Save the test results and random search baseline results to disk.
        6. Compute and log the AEI (Average Efficiency Improvement) metric.

    Outputs:
        - Saves test results to `test.pkl` in the specified log directory.
        - Saves random search baseline results to `random_search_baseline.pkl` in the log directory.
        - Prints AEI and AEI standard deviation metrics.
        - Prints the MGD performance improvement percentage.

    Notes:
        - The function uses `pickle` for loading and saving serialized objects.
        - Progress is displayed using a tqdm progress bar.
        - The AEI metric is computed using a custom `Logger` class.
    """
    print(f'start MGD_test: {config.run_time}')
    # get test set

    if config.problem == 'bbob-surrogate':
        config.is_train = False

    _, test_set = construct_problem_set(config)
    # get agents
    with open(config.model_from, 'rb') as f:
        agent_from = pickle.load(f)
    with open(config.model_to, 'rb') as f:
        agent_to = pickle.load(f)
    # get optimizer
    l_optimizer = eval(config.optimizer)(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {'cost': {},
                    'fes': {},
                    'T0': 0.,
                    'T1': {},
                    'T2': {}}
    agent_name_list = [f'{config.agent}_from', f'{config.agent}_to']
    for agent_name in agent_name_list:
        test_results['T1'][agent_name] = 0.
        test_results['T2'][agent_name] = 0.
    for problem in test_set:
        test_results['cost'][problem.__str__()] = {}
        test_results['fes'][problem.__str__()] = {}
        for agent_name in agent_name_list:
            test_results['cost'][problem.__str__()][agent_name] = []  # 51 np.arrays
            test_results['fes'][problem.__str__()][agent_name] = []  # 51 scalars
    # calculate T0
    test_results['T0'] = cal_t0(config.dim, config.maxFEs)
    # begin mgd_test
    seed = range(51)
    pbar_len = len(agent_name_list) * len(test_set) * 51
    with tqdm(range(pbar_len), desc='MGD_Test') as pbar:
        for i, problem in enumerate(test_set):
            # run model_from and model_to
            for agent_id, agent in enumerate([agent_from, agent_to]):
                T1 = 0
                T2 = 0
                for run in range(51):
                    start = time.perf_counter()
                    np.random.seed(seed[run])
                    # construct an ENV for (problem,optimizer)
                    env = PBO_Env(problem, l_optimizer)
                    info = agent.rollout_episode(env)
                    cost = info['cost']
                    while len(cost) < 51:
                        cost.append(cost[-1])
                    fes = info['fes']
                    end = time.perf_counter()
                    if i == 0:
                        T1 += env.problem.T1
                        T2 += (end - start) * 1000  # ms
                    test_results['cost'][problem.__str__()][agent_name_list[agent_id]].append(cost)
                    test_results['fes'][problem.__str__()][agent_name_list[agent_id]].append(fes)
                    pbar_info = {'problem': problem.__str__(),
                                 'optimizer': agent_name_list[agent_id],
                                 'run': run,
                                 'cost': cost[-1],
                                 'fes': fes}
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                if i == 0:
                    test_results['T1'][agent_name_list[agent_id]] = T1 / 51
                    test_results['T2'][agent_name_list[agent_id]] = T2 / 51
    if not os.path.exists(config.mgd_test_log_dir):
        os.makedirs(config.mgd_test_log_dir)
    with open(config.mgd_test_log_dir + 'test.pkl', 'wb') as f:
        pickle.dump(test_results, f, -1)
    random_search_results = test_for_random_search(config)
    with open(config.mgd_test_log_dir + 'random_search_baseline.pkl', 'wb') as f:
        pickle.dump(random_search_results, f, -1)
    logger = Logger(config)
    aei, aei_std = logger.aei_metric(test_results, random_search_results, config.maxFEs)
    print(f'AEI: {aei}')
    print(f'AEI STD: {aei_std}')
    print(f'MGD({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
          f'{100 * (1 - aei[config.agent+"_from"] / aei[config.agent+"_to"])}%')


def mte_test(config):
    """
    Perform the MTE (Model Transfer Efficiency) test for a given configuration.

    This function evaluates the performance of an agent by comparing its 
    pre-trained and scratch-trained results on a specific problem. It calculates 
    the MTE metric, generates plots for visualization, and saves the results.

    Args:
        config (object): A configuration object containing the following attributes:
            - run_time (str): The runtime identifier for the test.
            - pre_train_rollout (str): Path to the pre-trained rollout data file.
            - scratch_rollout (str): Path to the scratch-trained rollout data file.
            - agent (str): The name of the agent being tested.
            - problem_from (str): The source problem for transfer learning.
            - difficulty_from (str): The difficulty level of the source problem.
            - problem_to (str): The target problem for transfer learning.
            - difficulty_to (str): The difficulty level of the target problem.
            - mte_test_log_dir (str): Directory to save the MTE test results.

    Returns:
        None

    Side Effects:
        - Generates and saves a plot comparing pre-trained and scratch-trained results.
        - Prints the calculated MTE metric to the console.

    Notes:
        - The function preprocesses the input data, calculates average returns, 
          applies smoothing, and computes the MTE metric.
        - The MTE metric is calculated based on the relative performance of the 
          pre-trained and scratch-trained agents over learning steps.
        - The function uses Savitzky-Golay filtering for smoothing the average returns.
        - The generated plot is saved in the directory specified by `config.mte_test_log_dir`.

    Example:
        config = Config(
            run_time="2023-01-01",
            pre_train_rollout="path/to/pre_train.pkl",
            scratch_rollout="path/to/scratch.pkl",
            agent="AgentName",
            problem_from="ProblemA",
            difficulty_from="Easy",
            problem_to="ProblemB",
            difficulty_to="Hard",
            mte_test_log_dir="path/to/log_dir"
        )
        mte_test(config)
    """
    print(f'start MTE_test: {config.run_time}')
    pre_train_file = config.pre_train_rollout
    scratch_file = config.scratch_rollout
    agent = config.agent
    min_max = False

    # preprocess data for agent
    def preprocess(file, agent):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # aggregate all problem's data together
        returns = data['return']
        results = None
        i = 0
        for problem in returns.keys():
            if i == 0:
                results = np.array(returns[problem][agent])
            else:
                results = np.concatenate([results, np.array(returns[problem][agent])], axis=1)
            i += 1
        return np.array(results)

    bbob_data = preprocess(pre_train_file, agent)
    noisy_data = preprocess(scratch_file, agent)
    # calculate min_max avg
    temp = np.concatenate([bbob_data, noisy_data], axis=1)
    if min_max:
        temp_ = (temp - temp.min(-1)[:, None]) / (temp.max(-1)[:, None] - temp.min(-1)[:, None])
    else:
        temp_ = temp
    bd, nd = temp_[:, :90], temp_[:, 90:]
    checkpoints = np.hsplit(bd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    avg = bd.mean(-1)
    avg = savgol_filter(avg, 13, 5)
    std = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    checkpoints = np.hsplit(nd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    std_ = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    avg_ = nd.mean(-1)
    avg_ = savgol_filter(avg_, 13, 5)
    plt.figure(figsize=(40, 15))
    plt.subplot(1, 3, (2, 3))
    x = np.arange(21)
    x = (1.5e6 / x[-1]) * x
    idx = 21
    smooth = 1
    s = np.zeros(21)
    a = s[0] = avg[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg[i]
        s[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1

    s_ = np.zeros(21)
    a = s_[0] = avg_[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg_[i]
        s_[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1
    plt.plot(x[:idx], s[:idx], label='pre-train', marker='*', markersize=30, markevery=1, c='blue', linewidth=5)
    plt.fill_between(x[:idx], s[:idx] - std[:idx], s[:idx] + std[:idx], alpha=0.2, facecolor='blue')
    plt.plot(x[:idx], s_[:idx], label='scratch', marker='*', markersize=30, markevery=1, c='red', linewidth=5)
    plt.fill_between(x[:idx], s_[:idx] - std_[:idx], s_[:idx] + std_[:idx], alpha=0.2, facecolor='red')
    # Search MTE
    scratch = s_[:idx]
    pretrain = s[:idx]
    topx = np.argmax(scratch)
    topy = scratch[topx]
    T = topx / 21
    t = 0
    if pretrain[0] < topy:
        for i in range(1, 21):
            if pretrain[i - 1] < topy <= pretrain[i]:
                t = ((topy - pretrain[i - 1]) / (pretrain[i] - pretrain[i - 1]) + i - 1) / 21
                break
    if np.max(pretrain[-1]) < topy:
        t = 1
    MTE = 1 - t / T

    print(f'MTE({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
          f'{MTE}')

    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(45)
    plt.xticks(fontsize=45, )
    plt.yticks(fontsize=45)
    plt.legend(loc=0, fontsize=60)
    plt.xlabel('Learning Steps', fontsize=55)
    plt.ylabel('Avg Return', fontsize=55)
    plt.title(f'Fine-tuning ({name_translate(config.problem_from)} $\\rightarrow$ {name_translate(config.problem_to)})',
              fontsize=60)
    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.2)
    if not os.path.exists(config.mte_test_log_dir):
        os.makedirs(config.mte_test_log_dir)
    plt.savefig(f'{config.mte_test_log_dir}/MTE_{agent}.png', bbox_inches='tight')
