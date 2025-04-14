"""
This file is used to train the agent.(for the kind of optimizer that is learnable)
"""
import pickle

import torch
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from VectorEnv import *
from logger import Logger
import copy
from utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from basic_agent.utils import save_class
from tensorboardX import SummaryWriter
from agent import (
    # DE_DDQN_Agent,
    # DEDQN_Agent,
    RL_HPSDE_Agent,
    # LDE_Agent,
    # QLPSO_Agent,
    # RLEPSO_Agent,
    # RL_PSO_Agent,
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
    Random_search
)

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLEPSO_Agent,
    RLDE_AFL_Agent,
    LDE_Agent,
    RL_PSO_Agent,
    SYMBOL_Agent,
    RL_DAS_Agent,
    SYMBOL_Agent,
    Surr_RLDE_Agent,
    RLEMMO_Agent
)


matplotlib.use('Agg')


class Trainer(object):
    def __init__(self, config):
        """
        Initializes the trainer with the given configuration.

        Args:
            config (object): Configuration object containing the following attributes:
                - seed (int): Random seed for reproducibility.
                - resume_dir (str or None): Directory to resume training from a saved agent. 
                  If None, a new agent is created.
                - train_agent (str): Name of the training agent class to instantiate or load.
                - train_optimizer (str): Name of the optimizer class to instantiate.
                - problem (str): Problem type, e.g., 'bbob-surrogate'.
                - is_train (bool): Flag indicating whether the mode is training or not.

        Attributes:
            config (object): Stores the provided configuration.
            agent (object): The training agent, either newly created or loaded from a file.
            optimizer (object): The optimizer for training the agent.
            train_set (object): The training dataset constructed based on the problem type.
            test_set (object): The testing dataset constructed based on the problem type.

        Notes:
            - Sets random seeds for reproducibility across PyTorch, CUDA, and NumPy.
            - Configures PyTorch's cuDNN backend for deterministic behavior.
            - If `resume_dir` is provided, loads the agent from a pickle file and updates its settings.
            - Constructs the training and testing datasets based on the problem type.
        """
        self.config = config

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if config.resume_dir is None:
            self.agent = eval(config.train_agent)(config)
        else:
            file_path = config.resume_dir + config.train_agent + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent = pickle.load(f)
            self.agent.update_setting(config)
        self.optimizer = eval(config.train_optimizer)(config)

        if config.problem == 'bbob-surrogate':
            config.is_train = True
        self.train_set, self.test_set = construct_problem_set(config)

    def save_log(self, epochs, steps, cost, returns, normalizer):
        """
        Saves training logs including steps, returns, costs, and normalizers for each problem in the training set.

        Args:
            epochs (list or np.ndarray): The list of epoch numbers.
            steps (list or np.ndarray): The list of steps taken during training.
            cost (dict): A dictionary where keys are problem names and values are lists of costs for each epoch.
            returns (list or np.ndarray): The list of returns achieved during training.
            normalizer (dict): A dictionary where keys are problem names and values are lists of normalizer values for each epoch.

        Behavior:
            - Creates a directory for saving logs if it does not already exist.
            - Saves the steps and returns as a NumPy array in the log directory.
            - For each problem in the training set:
                - Ensures the cost and normalizer lists are the same length as the epochs by appending the last value.
                - Saves the epochs, costs, and normalizer values as a NumPy array in the log directory.

        Note:
            The log directory path is constructed using the configuration settings and the agent's class name.
        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__str__()
            if len(cost[name]) == 0:
                continue
            while len(cost[name]) < len(epochs):
                cost[name].append(cost[name][-1])
                normalizer[name].append(normalizer[name][-1])
            cost_save = np.stack((epochs, cost[name], normalizer[name]),  0)
            np.save(log_dir+name+'_cost', cost_save)
            
    def draw_cost(self, Name=None, normalize=False):
        """
        Plots and saves the cost graph for training problems.

        Args:
            Name (str or list, optional): The name(s) of the problem(s) to plot. If None, plots for all problems in the training set.
                                           If a string, plots for the specific problem matching the name.
                                           If a list, plots for all problems whose names are in the list.
            normalize (bool, optional): If True, normalizes the cost values by dividing by the number of samples (n). Defaults to False.

        Behavior:
            - Loads cost data from a .npy file located in the log directory.
            - Plots the cost data for the specified problem(s).
            - Saves the plot as a .png file in the 'pic/' subdirectory of the log directory.

        Notes:
            - The log directory is determined by the configuration (`self.config.log_dir`) and includes subdirectories for the agent's class name and runtime.
            - If the 'pic/' subdirectory does not exist, it is created automatically.
            - The cost data file is expected to be named in the format '<problem_name>_cost.npy' and located in the 'log/' subdirectory of the log directory.
        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in self.train_set:
            if Name is None:
                name = problem.__str__()
            elif (isinstance(Name, str) and problem.__str__() != Name) or (isinstance(Name, list) and problem.__str__() not in Name):
                continue
            else:
                name = Name
            plt.figure()
            plt.title(name + '_cost')
            values = np.load(log_dir + 'log/' + name+'_cost.npy')
            x, y, n = values
            if normalize:
                y /= n
            plt.plot(x, y)
            plt.savefig(log_dir+f'pic/{name}_cost.png')
            plt.close()
    
    def draw_average_cost(self, normalize=True):
        """
        Draws and saves a plot of the average cost across all problems in the training set.

        Args:
            normalize (bool, optional): If True, normalizes the cost values by dividing 
                by the number of occurrences (n). Defaults to True.

        Behavior:
            - Loads cost data for each problem in the training set from pre-saved `.npy` files.
            - Computes the average cost values across all problems.
            - Plots the average cost over time and saves the plot as an image file.

        File Structure:
            - Expects cost data files to be located in `log_dir/log/` with filenames 
              formatted as `<problem_name>_cost.npy`.
            - Saves the resulting plot in `log_dir/pic/` with the filename 
              `all_problem_cost.png`.

        Note:
            - The `log_dir` is constructed using the configuration and agent class name.
            - Creates the `pic/` directory if it does not already exist.
        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        X = []
        Y = []
        for problem in self.train_set:
            name = problem.__str__()
            values = np.load(log_dir + 'log/' + name+'_cost.npy')
            x, y, n = values
            if normalize:
                y /= n
            X.append(x)
            Y.append(y)
        X = np.mean(X, 0)
        Y = np.mean(Y, 0)
        plt.figure()
        plt.title('all problem cost')
        plt.plot(X, Y)
        plt.savefig(log_dir+f'pic/all_problem_cost.png')
        plt.close()

    def draw_return(self):
        """
        Draws and saves a plot of return values over time.

        This method loads return values from a NumPy file located in the log directory,
        generates a plot with the loaded data, and saves the plot as a PNG image in the
        appropriate directory.

        The log directory is constructed using the configuration's log directory, the
        agent's class name, and the runtime configuration.

        Steps:
        1. Constructs the log directory path.
        2. Creates the necessary subdirectories if they do not exist.
        3. Loads return values from a NumPy file located in the log directory.
        4. Plots the return values.
        5. Saves the plot as a PNG image in the 'pic' subdirectory of the log directory.

        Raises:
            FileNotFoundError: If the required NumPy file ('return.npy') does not exist.
            ValueError: If the loaded data is not in the expected format.

        Note:
            Ensure that the `self.config.log_dir` and `self.config.run_time` are properly
            configured, and the `return.npy` file exists in the expected location.

        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('return')
        values = np.load(log_dir + 'log/return.npy')
        plt.plot(values[0], values[1])
        plt.savefig(log_dir+f'pic/return.png')
        plt.close()

    def train_old(self):
        """
        Trains the agent using the provided training set.

        This method iterates through the training set, shuffling it at each epoch,
        and trains the agent on each problem in the set. It tracks various metrics
        such as cost, normalizer values, and returns for each problem. The training
        process continues until a stopping condition is met, such as exceeding the
        maximum learning steps.

        Attributes:
            config (object): Configuration object containing runtime settings.
            train_set (object): The dataset used for training.
            agent (object): The agent being trained.
            optimizer (object): The optimizer used in the environment.

        Returns:
            None

        Notes:
            - The training process uses a progress bar to display the current
              training status, including loss, learning steps, and returns.
            - Metrics such as cost, normalizer values, and returns are recorded
              for each problem in the training set.
            - The method includes commented-out code for saving the agent and
              generating visualizations, which can be enabled as needed.
            - A fixed seed is used for reproducibility in the optimizer.

        Todo:
            - Add seed configuration to make the seed value adjustable.
            - Implement logging functionality for saving training metrics.
            - Enable visualization of cost and return metrics at specified intervals.
        """
        print(f'start training: {self.config.run_time}')
        # agent_save_dir = self.config.agent_save_dir + self.agent.__class__.__name__ + '/' + self.config.run_time + '/'
        exceed_max_ls = False
        epoch = 0
        cost_record = {}
        normalizer_record = {}
        return_record = []
        learn_steps = []
        epoch_steps = []
        for problem in self.train_set:
            for p in problem:
                cost_record[p.__str__()] = []
                normalizer_record[p.__str__()] = []

        # todo Seed config
        seed = 4
        while not exceed_max_ls:
            learn_step = 0
            self.train_set.shuffle()
            with tqdm(range(self.train_set.N), desc=f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
                for problem_id, problem in enumerate(self.train_set):

                    # env = PBO_Env(problem, self.optimizer)
                    env_list = [PBO_Env(p, copy.deepcopy(self.optimizer)) for p in problem]
                    for env in env_list:
                        env.optimizer.seed(seed)

                    exceed_max_ls, pbar_info_train = self.agent.train_episode(envs = env_list)
                    # exceed_max_ls, pbar_info_train = self.agent.train_episode(env)  # pbar_info -> dict
                    postfix_str = (
                        f"loss={pbar_info_train['loss']:.2e}, "
                        f"learn_steps={pbar_info_train['learn_steps']}, "
                        f"return={[f'{x:.2e}' for x in pbar_info_train['return']]}"
                    )

                    pbar.set_postfix_str(postfix_str)
                    pbar.update(self.config.train_batch_size)
                    learn_step = pbar_info_train['learn_steps']
                    for id, p in enumerate(problem):
                        name = p.__str__()
                        cost_record[name].append(pbar_info_train['gbest'][id])
                        normalizer_record[name].append(pbar_info_train['normalizer'][id])
                        return_record.append(np.mean(pbar_info_train['return']))
                    learn_steps.append(learn_step)
                    if exceed_max_ls:
                        break
                self.agent.train_epoch()
            epoch_steps.append(learn_step)
            # if not os.path.exists(agent_save_dir):
            #     os.makedirs(agent_save_dir)
            # with open(agent_save_dir+'agent_epoch'+str(epoch)+'.pkl', 'wb') as f:
            #     pickle.dump(self.agent, f, -1)

            # todo add log logicality
            # self.save_log(epoch_steps, learn_steps, cost_record, return_record, normalizer_record)
            epoch += 1
            # if epoch % self.config.draw_interval == 0:
            #     self.draw_cost()
            #     self.draw_average_cost()
            #     self.draw_return()

        # self.draw_cost()
        # self.draw_average_cost()
        # self.draw_return()

    def train(self):
        """
        Trains the agent using the specified training configuration and dataset.

        This method orchestrates the training process, including setting up the training environment,
        managing epochs, logging progress, and saving checkpoints. It supports different training modes
        ("single" and "multi") and integrates with TensorBoard for logging.

        Attributes:
            self.config (object): Configuration object containing training parameters such as batch size,
                training mode, seed values, and logging options.
            self.train_set (object): Dataset object containing the training problems.
            self.agent (object): The agent to be trained.
            self.optimizer (object): Optimizer used for training.

        Workflow:
            1. Initializes TensorBoard logger if enabled.
            2. Configures batch size and training mode based on the configuration.
            3. Iteratively trains the agent for each epoch until the stopping condition is met.
            4. Logs training progress using tqdm and TensorBoard.
            5. Saves checkpoints at specified intervals.
            6. Handles random seed management for reproducibility.

        Returns:
            None
        """
        print(f'start training: {self.config.run_time}')
        is_end = False
        # todo tensorboard
        tb_logger = None
        if not self.config.no_tb:
            tb_logger = SummaryWriter(os.path.join('output/tensorboard', self.config.run_time))
            tb_logger.add_scalar("epoch-step", 0, 0)

        epoch = 0
        cost_record = {}
        normalizer_record = {}
        return_record = []
        learn_steps = []
        epoch_steps = []

        # 这里先让train_set bs 一直为1先
        for problem in self.train_set.data:
            cost_record[problem.__str__()] = []
            normalizer_record[problem.__str__] = []

        # 然后根据train_mode 决定 bs
        # single ---> 从train_set 里取出 bs 个问题训练
        # multi ---> 每次从train_set 中取出 1 个问题，copy bs 个 训练
        bs = self.config.train_batch_size
        if self.config.train_mode == "single":
            self.train_set.batch_size = 1
        elif self.config.train_mode == "multi":
            self.train_set.batch_size = bs

        epoch_seed = self.config.epoch_seed
        id_seed = self.config.id_seed
        seed = self.config.seed

        while not is_end:
            learn_step = 0
            self.train_set.shuffle()
            with tqdm(range(self.train_set.N), desc = f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
                for problem_id, problem in enumerate(self.train_set):
                    # set seed
                    seed_list = (epoch * epoch_seed + id_seed * (np.arange(bs) + bs * problem_id) + seed).tolist()

                    # 这里前面已经判断好 train_mode，这里只需要根据 train_mode 构造env就行
                    if self.config.train_mode == "single":
                        env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(self.optimizer)) for _ in range(bs)] # bs
                    elif self.config.train_mode == "multi":
                        env_list = [PBO_Env(copy.deepcopy(p), copy.deepcopy(self.optimizer)) for p in problem] # bs

                    # todo config add para
                    exceed_max_ls, train_meta_data = self.agent.train_episode(envs = env_list,
                                                                              seeds = seed_list,
                                                                              tb_logger = tb_logger,
                                                                              para_mode = "dummy",
                                                                              asynchronous = None,
                                                                              num_cpus = 1,
                                                                              num_gpus = 0,
                                                                              )
                    # exceed_max_ls, pbar_info_train = self.agent.train_episode(env)  # pbar_info -> dict
                    postfix_str = (
                        f"loss={train_meta_data['loss']:.2e}, "
                        f"learn_steps={train_meta_data['learn_steps']}, "
                        f"return={[f'{x:.2e}' for x in train_meta_data['return']]}"
                    )

                    pbar.set_postfix_str(postfix_str)
                    pbar.update(self.train_set.batch_size)
                    learn_step = train_meta_data['learn_steps']
                    # for id, p in enumerate(problem):
                    #     name = p.__str__()
                    #     cost_record[name].append(train_meta_data['gbest'][id])
                    #     normalizer_record[name].append(train_meta_data['normalizer'][id])
                    #     return_record.append(np.mean(train_meta_data['return']))
                    # learn_steps.append(learn_step)

                    if self.config.end_mode == "step" and exceed_max_ls:
                        is_end = True
                        break
                self.agent.train_epoch()
            epoch_steps.append(learn_step)
            epoch += 1

            if not self.config.no_tb:
                tb_logger.add_scalar("epoch-step", learn_step, epoch)

            # todo save
            # save_interval = 5
            # checkpoint0 0
            # checkpoint1 5
            if epoch >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "epoch":
                save_class(self.config.agent_save_dir, 'checkpoint' + str(self.agent.cur_checkpoint), self.agent)
                # 记录 checkpoint 和 total_step
                with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}\n")

                # todo rollout
                # 保存状态
                cpu_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state()
                np_state = np.random.get_state()
                # self.rollout(self.agent.cur_checkpoint)

                # 载入
                torch.random.set_rng_state(cpu_state)
                torch.cuda.set_rng_state(cuda_state)
                np.random.set_state(np_state)

                self.agent.cur_checkpoint += 1
            if self.config.end_mode == "epoch" and epoch >= self.config.max_epoch:
                is_end = True

    def rollout(self, checkpoint, rollout_run = 10):
        def rollout(self, checkpoint, rollout_run=10):
            """
            Perform a rollout operation using a specified checkpoint and number of runs.

            This method loads a pre-trained agent from a checkpoint file, initializes the 
            environment for testing, and performs a batch rollout to evaluate the agent's 
            performance on the test set.

            Args:
                checkpoint (int): The checkpoint index to load the agent from.
                rollout_run (int, optional): The number of rollout runs to perform for each 
                    problem in the test set. Defaults to 10.

            Behavior:
                - Seeds are set for reproducibility using the configuration's seed value.
                - The agent is loaded from a serialized file located in the `agent_save_dir`.
                - A deep copy of the test set is created for the rollout process.
                - The rollout is performed in batches, iterating through the test set.
                - For each problem in the test set, multiple environments are created, and 
                  the agent performs a batch rollout using these environments.
                - Progress is displayed using a progress bar, which updates with the agent's 
                  status.

            Notes:
                - The method uses PyTorch for deterministic behavior by setting seeds and 
                  disabling certain optimizations.
                - The `rollout_batch_episode` method of the agent is called to perform the 
                  rollout in parallel.

            Raises:
                FileNotFoundError: If the checkpoint file does not exist.
                pickle.UnpicklingError: If there is an error while loading the agent.

            """
        # 读取 agent
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with open(self.config.agent_save_dir + 'checkpoint' + str(checkpoint) + ".pkl", "rb") as f:
            agent = pickle.load(f)
        rollout_set = copy.deepcopy(self.test_set)
        bs = self.config.train_batch_size

        seed_list = list(range(1, rollout_run + 1)) * bs
        pbar = (rollout_set.N // bs + rollout_set.N % bs)
        with tqdm(range(pbar), desc = f"Rollout{checkpoint}") as pbar:
            for i, problem in enumerate(rollout_set):
                env_list = [PBO_Env(copy.deepcopy(p), copy.deepcopy(self.optimizer))
                            for p in problem
                            for _ in range(rollout_run)]
                with torch.no_grad():
                    meta_rollout_data = agent.rollout_batch_episode(envs = env_list,
                                                                    seeds = seed_list,
                                                                    para_mode = 'dummy',
                                                                    asynchronous = None,
                                                                    num_cpus = 1,
                                                                    num_gpus = 0,
                                                                    )
                pbar.set_postfix({'MetaBBO': agent.__str__()})
                pbar.update(1)



# class Trainer_l2l(object):
#     def __init__(self, config):
#         self.config = config

#         # two way 
#         self.agent = eval(config.train_agent)(config)
#         self.optimizer = eval(config.train_optimizer)(config)
#         # need to be torch version
#         self.train_set, self.test_set = construct_problem_set(config)


#     def train(self):
#         print(f'start training: {self.config.run_time}')
#         agent_save_dir = self.config.agent_save_dir + self.agent.__class__.__name__ + '/' + self.config.run_time + '/'
#         exceed_max_ls = False
#         epoch = 0
#         cost_record = {}
#         normalizer_record = {}
#         return_record = []
#         learn_steps = []
#         epoch_steps = []
#         for problem in self.train_set:
#             cost_record[problem.__str__()] = []
#             normalizer_record[problem.__str__()] = []
#         while not exceed_max_ls:
#             learn_step = 0
#             self.train_set.shuffle()
#             with tqdm(range(self.train_set.N), desc=f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
#                 for problem_id, problem in enumerate(self.train_set):
                    
#                     env=PBO_Env(problem,self.optimizer)
#                     exceed_max_ls= self.agent.train_episode(env)  # pbar_info -> dict
                    
#                     pbar.update(1)
#                     name = problem.__str__()
                    
#                     learn_steps.append(learn_step)
#                     if exceed_max_ls:
#                         break
#             epoch_steps.append(learn_step)
            
                    
#             epoch += 1
            
        