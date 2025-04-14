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
from agent.utils import save_class
from problem.SOO import bbob_numpy,bbob_surrogate,bbob_torch,protein_docking
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_torch.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_numpy.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_numpy
from problem.SOO.uav_numpy.uav_dataset import UAV_Dataset_numpy
from problem.SOO.uav_torch.uav_dataset import UAV_Dataset_torch
from problem import bbob, bbob_torch, protein_docking, mmo_dataset


def construct_problem_set(config):
    """
    Constructs and returns a dataset based on the specified problem type in the configuration.

    Args:
        config (object): A configuration object containing the following attributes:
            - problem (str): The type of problem to construct the dataset for. 
              Supported values include:
                - 'bbob', 'bbob-noisy'
                - 'bbob-torch', 'bbob-noisy-torch'
                - 'protein', 'protein-torch'
                - 'bbob-surrogate'
                - 'Symbolic_bench', 'Symbolic_bench-torch'
                - 'lsgo', 'lsgo-torch'
                - 'uav', 'uav-torch'
                - 'mmo', 'mmo-torch'
            - dim (int, optional): Dimensionality of the problem (used for certain problem types).
            - upperbound (float, optional): Upper bound for the dataset values (used for certain problem types).
            - train_batch_size (int): Batch size for training data.
            - test_batch_size (int): Batch size for testing data.
            - difficulty (str, optional): Difficulty level of the problem (used for certain problem types).
            - user_train_list (list, optional): User-defined training list (used for 'mmo' and 'mmo-torch').

    Returns:
        object: A dataset object corresponding to the specified problem type.

    Raises:
        ValueError: If the specified problem type is not supported.
    """
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
    elif problem in ['mmo', 'mmo-torch']:
        return mmo_dataset.MMO_Dataset.get_datasets(version=problem,
                                            train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=config.difficulty,
                                            user_train_list = config.user_train_list)

    else:
        raise ValueError(problem + ' is not defined!')
    
    
class Ma_Moo_Trainer:
    def __init__(self, config):
        """
        Initializes the class with the given configuration.

        Args:
            config (object): Configuration object containing the following attributes:
                - resume_dir (str or None): Directory path to resume from a saved agent. 
                  If None, a new agent is created.
                - train_agent (str): Name of the training agent class to be instantiated or loaded.
                - train_optimizer (str): Name of the optimizer class to be instantiated.
                - indicators (list): List of indicators for training or evaluation.

        Attributes:
            config (object): Stores the provided configuration object.
            agent (object): The training agent, either newly created or loaded from a file.
            optimizer (object): The optimizer instance created based on the configuration.
            train_set (object): The training dataset constructed from the configuration.
            test_set (object): The testing dataset constructed from the configuration.
            indicators (list): List of indicators for training or evaluation.

        Raises:
            FileNotFoundError: If the specified resume file does not exist.
            Exception: If there is an error during agent loading or initialization.
        """
        self.config = config
        if config.resume_dir is None:
            self.agent = eval(config.train_agent)(config)
        else:
            file_path = config.resume_dir + config.train_agent + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent = pickle.load(f)
            self.agent.update_setting(config)
        self.optimizer = eval(config.train_optimizer)(config)
        self.train_set, self.test_set = construct_problem_set(config)
        self.indicators = config.indicators

    def save_log(self, epochs, steps, indicators_record, returns):
        """
        Saves training logs, including returns and performance indicators, to disk.

        Args:
            epochs (list or np.ndarray): A list or array of epoch numbers.
            steps (list or np.ndarray): A list or array of step counts corresponding to the epochs.
            indicators_record (dict): A dictionary containing performance indicators for each problem.
                The keys are problem names, and the values are dictionaries where:
                    - Keys are indicator names (e.g., "accuracy", "loss").
                    - Values are lists of indicator values recorded during training.
            returns (list or np.ndarray): A list or array of return values corresponding to the steps.

        Behavior:
            - Creates a directory structure for saving logs based on the agent's class name and runtime configuration.
            - Saves the return values as a NumPy array in the log directory.
            - Iterates through the training set problems and saves performance indicators for each problem.
            - Ensures that the length of indicator records matches the number of epochs by appending the last value if necessary.
            - Saves the indicators as NumPy arrays in the log directory.

        Notes:
            - The log directory is constructed using the `log_dir` attribute from the configuration object.
            - The problem names and indicator names are used to generate unique filenames for the saved logs.
        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            for indicator in self.indicators:
                if len(indicators_record[name][indicator]) == 0:
                    continue
                while len(indicators_record[name][indicator]) < len(epochs):
                    indicators_record[name][indicator].append(indicators_record[name][indicator][-1])
                indictors_record_save = np.stack((epochs, indicators_record[name][indicator]),  0)
                np.save(log_dir+name+'_'+indicator,indictors_record_save)
            
    def draw_indicators(self, Name=None, normalize=False):
        """
        Draws and saves indicator plots for the training problems in the dataset.

        Parameters:
            Name (str or list, optional): Specifies the name(s) of the problem(s) to process. 
                If None, all problems in the training set are processed. If a string, only the 
                problem with the matching name is processed. If a list, only the problems with 
                names in the list are processed.
            normalize (bool, optional): If True, normalizes the y-values of the indicators 
                by dividing them by the corresponding `n` value. Defaults to False.

        Behavior:
            - Creates a directory structure for saving plots if it does not already exist.
            - Iterates through the training problems in `self.train_set` and processes each 
              problem based on the `Name` parameter.
            - For each problem and indicator, loads the corresponding data from a `.npy` file, 
              optionally normalizes the y-values, and generates a plot.
            - Saves the generated plot as a `.png` file in the appropriate directory.

        Notes:
            - The method assumes that the indicator data is stored in `.npy` files within a 
              `log` subdirectory of the `log_dir` path.
            - The generated plots are saved in a `pic` subdirectory of the `log_dir` path.
        """
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in self.train_set:
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
                if normalize:
                    y /= n
                plt.plot(x, y)
                plt.savefig(log_dir+f'pic/{name}_'+indicator+'.png')
                plt.close()
    
    def draw_average_indictors(self, normalize=True):
        """
        Draws and saves plots of the average values of specified indicators for all problems 
        in the training set across epochs.

        Args:
            normalize (bool, optional): A flag indicating whether to normalize the data. 
                                        Currently unused in the function. Defaults to True.

        Description:
            - This function iterates over a list of indicators and computes the average 
              values of these indicators for all problems in the training set.
            - For each indicator, it loads the corresponding data from `.npy` files, 
              calculates the mean across all problems, and generates a plot.
            - The plots are saved as PNG files in a directory structure based on the 
              configuration and runtime information.

        Notes:
            - The function assumes the existence of a `config` attribute with `log_dir` 
              and `run_time` properties, and an `agent` attribute with a class name.
            - The function also assumes the existence of a `train_set` attribute containing 
              problem instances, and an `indicators` attribute listing the indicators to process.

        File Output:
            - Saves the generated plots in the directory: 
              `<log_dir>/train/<agent_class_name>/<run_time>/pic/`.

        Raises:
            - FileNotFoundError: If the required `.npy` files for the indicators are not found.
            - OSError: If there are issues creating the output directories.

        """
        # 这个函数用于画每一个epoch所有问题指标的平均值
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for indictor in self.indicators:
            X = []
            Y = []
            for problem in self.train_set:
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

    def draw_return(self):
        """
        Generates and saves a plot of return values for the agent's training process.

        This method loads return values from a NumPy file, creates a plot of the values,
        and saves the plot as a PNG image in a specified directory. If the required
        directories do not exist, they are created.

        The plot is saved in the 'pic/' subdirectory under the log directory, which is
        constructed based on the agent's class name and runtime configuration.

        Raises:
            FileNotFoundError: If the required NumPy file ('return.npy') does not exist.
            OSError: If there is an issue creating the required directories.

        Notes:
            - The method assumes that the return values are stored in a NumPy file
              with two arrays: one for the x-axis (e.g., episodes) and one for the
              y-axis (e.g., return values).
            - The plot is titled 'return' and saved as 'return.png'.

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

    def train(self):
        """
        Trains the agent using the specified training configuration and dataset.

        This method initializes training parameters, iterates through the training
        dataset, and performs training episodes for the agent. It supports both
        single and multi-problem training modes, and tracks various metrics such as
        loss, learning steps, and performance indicators.

        Key Features:
        - Configures batch size and training mode (single or multi).
        - Shuffles the training dataset and processes each problem in the dataset.
        - Constructs environments for training episodes based on the training mode.
        - Tracks and records metrics such as loss, learning steps, returns, and
          performance indicators (e.g., IGD, HV).
        - Supports saving agent checkpoints at specified intervals.
        - Generates visualizations for performance indicators and returns.

        Returns:
            None
        """
        print(f'start training: {self.config.run_time}')
        if self.indictors is None:
            self.indicators = ['best_igd','best_hv']
        indictors_record = {}
        for problem in self.train_set:
            problem_name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            indictors_record[problem_name] = {}
            for indictor in self.indictors:
                indictors_record[problem_name][indictor] = []
        is_end = False
        epoch = 0
        return_record = []
        learn_steps = []
        epoch_steps = []
        q_values_record = []
        loss_record = []

        # # 这里先让train_set bs 一直为1先
        # for problem in self.train_set.data:
        #     cost_record[problem.__str__()] = []
        #     normalizer_record[problem.__str__] = []

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
                    for indictor in self.indictors:
                        indictors_record[problem_name][indictor].append(train_meta_data[indictor])
                    return_record.append(train_meta_data['return'])
                    learn_steps.append(learn_step)
                    q_values_record.append(train_meta_data['q_values'])
                    loss_record.append(train_meta_data['loss'])
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

            # todo save
            # save_interval = 5
            # checkpoint0 0
            # checkpoint1 5
            if epoch >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "epoch":
                save_class(self.config.agent_save_dir, 'checkpoint' + str(self.agent.cur_checkpoint), self.agent)
                # 记录 checkpoint 和 total_step
                with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}\n")

                self.agent.cur_checkpoint += 1
            if self.config.end_mode == "epoch" and epoch >= self.config.max_epoch:
                is_end = True
            # if not os.path.exists(agent_save_dir):
            #     os.makedirs(agent_save_dir)
            # with open(agent_save_dir+'agent_epoch'+str(epoch)+'.pkl', 'wb') as f:
            #     pickle.dump(self.agent, f, -1)
            self.save_log(epoch_steps, learn_steps, indictors_record, return_record)
            epoch += 1
            if epoch % self.config.draw_interval == 0:
                self.draw_indictors()
                self.draw_average_indictors()
                self.draw_return()
        self.draw_indictors()
        self.draw_average_indictors()
        self.draw_return()
        
