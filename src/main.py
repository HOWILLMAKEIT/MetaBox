import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLS_NUM_THREADS'] = '1'
# os.environ['GOTO_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['TORCH_NUM_THREADS'] = '1'
# os.environ['RAY_num_server_call_thread'] = '1'

import torch
from trainer import Trainer
from tester import *
from config import get_config
from logger import *
import shutil
import warnings
import json
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    config = get_config()
    assert ((config.train is not None) +
            (config.rollout is not None) +
            (config.test is not None) +
            (config.run_experiment is not None) +
            (config.mgd_test is not None) +
            (config.mte_test is not None)) == 1, \
        'Among train, rollout, test, run_experiment, mgd_test & mte_test, only one mode can be given at one time.'
    torch.set_default_dtype(torch.float64)
    if config.train_problem in ['mmo', 'mmo-torch']:
        logger = MMO_Logger(config)
    elif config.train_problem in  ['wcci2020', 'cec2017mto']:
        logger = MTO_Logger(config)
    elif config.train_problem in  ['moo-synthetic']:
        logger = MOO_Logger(config)
    else:
        logger = Basic_Logger(config)
        
    # train
    if config.train:
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train()

    # rollout
    if config.rollout:
        torch.set_grad_enabled(False)
        rollout_batch(config)
        logger.post_processing_rollout_statics(config.rollout_log_dir)

    # test
    if config.test:
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test()
        logger.post_processing_test_statics(config.test_log_dir)

    # run_experiment
    if config.run_experiment:
        # train
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train()

        # rollout
        agent_save_dir = config.agent_save_dir  # user defined agent_save_dir + agent name + run_time
        rollout_save_dir = os.path.join(agent_save_dir, config.train_agent)  # user defined agent_save_dir + agent name + run_time + agent_name
        if not os.path.exists(rollout_save_dir):
            os.makedirs(rollout_save_dir)
        # copy models from agent_save_dir to rollout_save_dir
        for filename in os.listdir(agent_save_dir):
            if os.path.isfile(os.path.join(agent_save_dir, filename)):
                shutil.copy(os.path.join(agent_save_dir, filename), rollout_save_dir)
        # test_agent_load_dir = None
        # if config.agent_load_dir is not None:
        #     test_agent_load_dir = config.agent_load_dir
        # config.agent_load_dir = agent_save_dir  # let config.agent_load_dir = config.agent_save_dir to load model
        config.agent_for_rollout = [config.train_agent]
        config.optimizer_for_rollout = [config.train_optimizer]
        torch.set_grad_enabled(False)
        rollout_batch(config)
        shutil.rmtree(rollout_save_dir)  # remove rollout model files after rollout
        logger.post_processing_rollout_statics(config.rollout_log_dir)

        # test
        # if test_agent_load_dir is not None:
        #     config.agent_load_dir = test_agent_load_dir
        with open('model.json', 'r', encoding = 'utf-8') as f:
            json_data = json.load(f)
        json_data[config.train_agent] = {'Agent': config.train_agent, 'Optimizer': config.train_optimizer, 'dir': os.path.join(agent_save_dir, 'checkpoint20.pkl')}
        with open('model.json', 'w', encoding = 'utf-8') as f:
            json.dump(json_data, f)
        if (config.train_agent != config.agent) and (config.train_agent not in config.agent):
            config.agent.append(config.train_agent)
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test()
        logger.post_processing_test_statics(config.test_log_dir)

    # mgd_test
    if config.mgd_test:
        torch.set_grad_enabled(False)
        tester.mgd_test()

    # mte_test
    if config.mte_test:
        tester.mte_test()
