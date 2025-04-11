"""
This file is used to train the agent.(for the kind of optimizer that is learnable)
"""
import pickle

import torch
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from environment.VectorEnv import *
from logger import Logger
import copy
from environment.problem.utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from rl.utils import save_class
from tensorboardX import SummaryWriter

from environment.optimizer import (
    DE_DDQN_Optimizer,
    DEDQN_Optimizer,
    RL_HPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RL_PSO_Optimizer,
    L2L_Optimizer,
    GLEET_Optimizer,
    RLDAS_Optimizer,
    LES_Optimizer,
    NRLPSO_Optimizer,
    SYMBOL_Optimizer,
    RLDE_AFL_Optimizer,
    Surr_RLDE_Optimizer,
    RLEMMO_Optimizer,
)

from baseline.bbo import (
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
)

from baseline.metabbo import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLDE_AFL_Agent,
    SYMBOL_Agent,
    RLDAS_Agent,
    Surr_RLDE_Agent,
    RLEMMO_Agent
)


matplotlib.use('Agg')


class Trainer(object):
    def __init__(self, config):
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
        self.logger = Logger(config)
        
        if config.problem == 'bbob-surrogate':
            config.is_train = True
        self.train_set, self.test_set = construct_problem_set(config)

    def save_log(self, epochs, steps, cost, returns, normalizer):
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
            
    def save_class(dir, file_name, saving_class):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file_name+'.pkl', 'wb') as f:
            pickle.dump(saving_class, f, -1)

    def train(self):
        print(f'start training: {self.config.run_time}')
        is_end = False
        # todo tensorboard
        tb_logger = None
        if not self.config.no_tb:
            tb_logger = SummaryWriter(os.path.join('output/tensorboard', self.config.run_time))
            tb_logger.add_scalar("epoch-step", 0, 0)

        epoch = 0
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
            # epoch_steps.append(learn_step)
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
            
        