# Flexible usage gallery

some examples, tutorial and some other advanced usage of Metabox

```{toctree}

:maxdepth: 1
:hidden
Own_algorithm
Own_dataset
Config

```

## 1. Implement your own algorithm in Metabox

### MetaBBO

#### 1. Create your own Agent

```{tip}
MetaBOX not only supports RL-based MetaBBO methods, \
but also supports MetaBBO methods based on SL, NE, and ICL. \
Below, we use the RL-based method as an example; \
for other methods, please refer to xxx. to **Gallery > Config** for details.
```

##### 1.1. Create your own RL

```{important}
MetaBOX has pre-implemented various RL methods — refer to **Gallery > Config** for details. \
You just need to inherit it and design your own Agent — Jump directly to [Create your own Agent](#create-your-own-optimizer) ！
```


1️⃣ Import Required Packages

```python
import torch
from metaevobox.rl import basic_agent
```

2️⃣ Initialize the RL Class

```python
class MyRL(basic_agent):
     
     def __init__(self, config):
    
    # If rl contains the network
    # def __init__(self, config, networks: dict, learning_rates: float):
          super().__init___(config)
          self.config = config
          # Init parameters
          # xxx
          
          # If rl contains the network
          # self.set_network(networks, learning_rates)
          
          # Init learning time
          self.learning_time = 0
          self.cur_checkpoint = 0
          
          # Save init agent
          save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
          self.cur_checkpoint += 1
```

3️⃣ Initialize the Network (Optional)

```{note}
This function is designed for rl methods that require networks and is not necessary.
```

```python
    def set_network(self, networks: dict, learning_rates: float):
        pass
```

4️⃣ Set update rules

```{code}
    def update_setting(self, config):
        pass
```

5️⃣ The Main Function for Training 

```python
def train_episode(self, 
                 envs,
                 seeds: Optional[Union[int, List[int], np.ndarray]],
                 para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                 compute_resource = {},
                 tb_logger = None,
                 required_info = {}):
   
   num_cpus = None
   num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
   if 'num_cpus' in compute_resource.keys():
       num_cpus = compute_resource['num_cpus']
   if 'num_gpus' in compute_resource.keys():
       num_gpus = compute_resource['num_gpus']
   env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
   env.seed(seeds) 

   state = env.reset()
   state = torch.FloatTensor(state)

   R = torch.zeros(len(env))

   while not env.all_done():
       # Get actions based on specific methods
       # action = ....

       # State transient
       next_state, reward, is_end, info = env.step(action)
       R += reward

       # Specific operations
       # xxxx

       # Store info
       self.learning_time += 1
       if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
           save_class(self.config.agent_save_dir, 'checkpoint-'+str(self.cur_checkpoint), self)
           self.cur_checkpoint += 1

       return self.learning_time >= self.config.max_learning_step, return_info

   # Return the necessary training data
   is_train_ended = self.learning_time >= self.config.max_learning_step
   return_info = {'return': R, 'learn_steps': self.learning_time, }
   env_cost = env.get_env_attr('cost')
   return_info['gbest'] = env_cost[-1]
   for key in required_info.keys():
       return_info[key] = env.get_env_attr(required_info[key])
   env.close()
   return is_train_ended, return_info
```

6️⃣ The Main Function for Testing

```python
def rollout_episode(self, env, seed=None, required_info = {}):
   with torch.no_grad():
       if seed is not None:
           env.seed(seed)
       is_done = False
       state = env.reset()
       R = 0
       
       while not is_done:
           # Get actions based on specific methods
           # action = ....
           
           # State transient
           next_state, reward, is_end, info = env.step(action)
           R += reward
          
       # Return the necessary test data
       env_cost = env.get_env_attr('cost')
       env_fes = env.get_env_attr('fes')
       env_metadata = env.get_env_attr('metadata') 
       results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
       for key in required_info.keys():
           results[key] = getattr(env, required_info[key])
       return results
```
       
6️⃣ The Main Function to Record Data for Analysis

```python
def log_to_tb_train(self, tb_logger):
   # Record the training data to tensorboard
   # Exp：tb_logger.add_scalar('loss', loss.item(), mini_step)
   pass
```

```{tip}
Not familiar with tensorboard? Click this [link](https://www.tensorflow.org/tensorboard/get_started).
```

##### 1.2. Create your own Agent

```{important}
MetaBOX has pre-implemented various RL methods — refer to **Gallery > Config** for details. \
You just need to inherit it and design your own Agent ！\
Here we take the rl method inherited from MetaBOX as an example.
```

1️⃣ Inheritance and Initialization

```python
from metaevobox.rl import xxx

class MyAgent(xxx):
     def __init__(self, config):
         super().__init__(self.config):
         self.config = config

         # Init parameters
         # XXX

     def __str__(self):
         return "MyAgent"
```

2️⃣ Modify train_episode according to specific work (Optional)

```{note}
This is designed for those with special training needs, not necessary
```

```python
def train_episode(self, 
                 envs,
                 seeds: Optional[Union[int, List[int], np.ndarray]],
                 para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                 compute_resource = {},
                 tb_logger = None,
                 required_info = {}):
   
   num_cpus = None
   num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
   if 'num_cpus' in compute_resource.keys():
       num_cpus = compute_resource['num_cpus']
   if 'num_gpus' in compute_resource.keys():
       num_gpus = compute_resource['num_gpus']
   env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
   env.seed(seeds) 

   state = env.reset()
   state = torch.FloatTensor(state)

   R = torch.zeros(len(env))

   while not env.all_done():
       # Get actions based on specific methods
       # action = ....

       # State transient
       next_state, reward, is_end, info = env.step(action)
       R += reward

       # Specific operations
       # xxxx

       # Store info
       self.learning_time += 1
       if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
           save_class(self.config.agent_save_dir, 'checkpoint-'+str(self.cur_checkpoint), self)
           self.cur_checkpoint += 1

       return self.learning_time >= self.config.max_learning_step

   # Return the necessary training data
   is_train_ended = self.learning_time >= self.config.max_learning_step
   return_info = {'return': R, 'learn_steps': self.learning_time, }
   env_cost = env.get_env_attr('cost')
   return_info['gbest'] = env_cost[-1]
   for key in required_info.keys():
       return_info[key] = env.get_env_attr(required_info[key])
   env.close()
   return is_train_ended, return_info
```

2️⃣ Modify rollout_episode according to specific work (Optional)

```{note}
This is designed for those with special test needs, not necessary
```

```python
def rollout_episode(self, env, seed=None, required_info = {}):
   with torch.no_grad():
       if seed is not None:
           env.seed(seed)
       is_done = False
       state = env.reset()
       R = 0
       
       while not is_done:
           # Get actions based on specific methods
           # action = ....
           
           # State transient
           next_state, reward, is_end, info = env.step(action)
           R += reward
          
       # Return the necessary test data
       env_cost = env.get_env_attr('cost')
       env_fes = env.get_env_attr('fes')
       env_metadata = env.get_env_attr('metadata') 
       results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
       for key in required_info.keys():
           results[key] = getattr(env, required_info[key])
       return results
```

#### 2. Create your own Optimizer

1️⃣ Inheritance and Initialization

```python
from metaevobox.environment.optimizer import Learnale_Optimizer

class MyOptimizer(Learnale_Optimizer):
     def __init__(self, config):
         super().__init__(config)
         
         self.config = config
         self.max_fes = config.maxFEs
         self.fes = None
         self.cost = None
         self.log_index = None
         self.log_interval = config.log_interval

         # Init parameters
         # XXX

     def __str__(self):
         return "MyOptimizer"
```     

2️⃣ Initialize the population

```python
def init_population(self, problem):

    # Specific operations
    # xxxx

    if self.config.full_meta_data:
        self.meta_X = [population.copy()]
        # population is all individuals in each generation
        self.meta_Cost = [all_cost.copy()]
        # all_cost is all evaluation values in each generation

    return # According to your specific needs

```
3️⃣ The Main function for updating

```python
def update(self, action, problem):

    # Specific operations
    # xxxx

    # Record all individuals in each generation
    # and their corresponding evaluation values
    if self.full_meta_data:
        self.meta_X.append(population.copy())
        # population is all individuals in each generation
        self.meta_Cost.append(all_cost.copy())
        # all_cost is all evaluation values in each generation

    # In order to ensure that the logger data format is correct,
    # there is only one stop mechanism.
    is_end = self.fes >= self.max_fes

    if self.fes >= self.log_interval * self.log_index:
        self.log_index += 1
        self.cost.append(self.gbest_cost)
        # gbest_cost is the optimal evaluation value

    if is_end:
        if len(self.cost) >= self.config.n_logpoint + 1:
            self.cost[-1] = self.gbest_cost
        else:
            while len(self.cost) < self.config.n_logpoint + 1:
                self.cost.append(self.gbest_cost)

    info = {}
    return next_state, reward, is_end, info
```

```{important}
Since optimizer is extremely flexible, the above functions are only necessary \
and need to be adjusted appropriately **according to specific tasks**.
```

### BBO

```{tip}
Considering that you may need to compare with other bbo,\
we also open the bbo design interface to you! 😉
```

1️⃣ Inheritance and Initialization

```python
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer

class MyBBO(Basic_Optimizer):
    def __init__(self, config):
        super(self).__init__(config)
         
        self.config = config
        self.max_fes = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        # Init parameters
        # XXX

    def __str__(self):
        return "MyBBO"
```

2️⃣ The Main Function for Testing

```python
def run_episode(self, problem):

    # Update
    is_end = False
    while not is_end:

        if self.full_meta_data:
            self.meta_X.append(population.copy())
            # population is all individuals in each generation
            self.meta_Cost.append(all_cost.copy())
            # all_cost is all evaluation values in each generation

        if self.fes >= log_index * self.log_interval:
            log_index += 1
            self.cost.append(self.gbest_cost)
            # gbest_cost is the optimal evaluation value

        is_end = self.fes >= self.config.maxFEs

     # Record
     if len(self.cost) >= self.__n_logpoint + 1:
         self.cost[-1] = self.gbest
     else:
         self.cost.append(self.gbest)
     results = {'cost': self.cost, 'fes': self.fes}

     if self.full_meta_data:
         metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
         results['metadata'] = metadata
      return results
```
## 2. Implement your own dataset in Metabox

```{important}
The dataset needs to be constructed in **two versions: torch and numpy**, \
to facilitate the use of different optimizers. \ 
The following is an example of the numpy version
```

1️⃣ Inheritance and Initialization
```python
from metaevobox.environment.problem.basic_problem import Basic_Problem
# torch version : from metaevobox.environment.problem.basic_problem import Basic_Problem_Torch
class MyProblem(Basic_Problem):
# torch version : class MyProblem(Basic_Problem_Torch)
    def __init__(self):
        # Init parameters
        self.opt = None
        self.optimum = 0.0

    def get_optimial(self)
        return self.opt

    def func(self, x):
        raise NotImplementedError
```

2️⃣ Instantiation Function
```python
class MyFunction(MyProblem):
    def __init__(self):
        super().__init__()

        self.opt = # Specific setting

    def __str__(self):
        return 'MyFunction'
    
    def func(self, x):
        # Specific Operation
        return result
```

2️⃣ Instantiation Dataset
```python
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        func_id = [i for i in range(1, # The number of functions)]
        train_set = []
        test_set = []

        if difficulty == 'easy':
            train_id = [# Specific setting]
            for id in func_id:
                if version == 'numpy':
                    instance = eval(f'F{id}')()
                else:
                    instance = eval(f'F{id}_Torch')()
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)

         # difficulty == 'difficult' is the same
         elif difficulty == 'all':
             for id in func_id:
                 if version == 'numpy':
                     instance = eval(f'F{id}')()
                 else:
                     instance = eval(f'F{id}_Torch')()
                 train_set.append(instance)
                 test_set.append(instance)
         elif difficulty is None:
             train_id = user_train_list
             test_id = user_test_list
             for id in func_id:
                 if version == 'numpy':
                     instance = eval(f'F{id}')()
                 else:
                     instance = eval(f'F{id}_Torch')()
                 if id in train_id:
                     train_set.append(instance)
                 elif id in test_id:
                     test_set.append(instance)

        return MyDataset(train_set, train_batch_size), MyDataset(test_set, test_batch_size)
    
    # get a batch of data
    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    # get the number of data
    def __len__(self):
        return self.N

    def __add__(self, other: 'MyDataset'):
        return MyDataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
```

## 3. Config

### The Config of Problem Setup
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| train_problem                   | Specify the problem suite for training                                 | ['bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D', 'bbob-noisy-30D', 'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D', 'hpo-b', 'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav', 'uav-torch', 'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic'] | 'bbob'                                                                                                                       |
| test_problem                    | Specify the problem suite for testing, default to be consistent with training | ['None', 'bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D', 'bbob-noisy-30D', 'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D', 'hpo-b', 'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav', 'uav-torch', 'ne', 'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic'] | None                                                                                                                         |
| train_difficulty                | Difficulty level for training problems                                 | ['all', 'easy', 'difficult', 'user-define']                                                                                        | 'easy'                                                                                                                       |
| test_difficulty                 | Difficulty level for testing problems, default to be consistent with training | ['all', 'easy', 'difficult', 'user-define']                                                                                        | None                                                                                                                         |
| upperbound                      | Upperbound of search space                                             | float value                                                                                                                       | 5                                                                                                                           |
| user_train_problem_list         | User-defined training problem list                                     | List of problems                                                                                                                 | None                                                                                                                         |
| user_test_problem_list          | User-defined testing problem list                                      | List of problems                                                                                                                 | None                                                                                                                         |
| device                          | Device to use                                                          | ['cpu', 'cuda']                                                                                                                  | 'cpu'                                                                                                                        |

### The Config of Training Mode
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| train                           | Switch to train mode                                                   | None or action                                                                                                                   | None                                                                                                                         |
| test                            | Switch to inference mode                                               | None or action                                                                                                                   | None                                                                                                                         |
| rollout                         | Switch to rollout mode                                                 | None or action                                                                                                                   | None                                                                                                                         |
| run_experiment                  | Switch to run_experiment mode                                          | None or action                                                                                                                   | None                                                                                                                         |
| mgd_test                        | Switch to mgd_test mode                                                | None or action                                                                                                                   | None                                                                                                                         |
| mte_test                        | Switch to mte_test mode                                                | None or action                                                                                                                   | None                                                                                                                         |
| task_cnt                        | Number of tasks in multitask                                           | int value                                                                                                                        | 10                                                                                                                           |
| generation                      | Total generations for L2O                                              | int value                                                                                                                        | 250                                                                                                                          |
| full_meta_data                  | Store the metadata                                                     | True or False                                                                                                                    | True                                                                                                                         |
| max_learning_step               | The maximum learning step for training                                | int value                                                                                                                        | 1500000                                                                                                                       |
| train_batch_size                | Batch size of train set                                                | int value                                                                                                                        | 1                                                                                                                             |
| train_agent                     | Agent for training                                                     | Agent name                                                                                                                       | None                                                                                                                         |
| train_optimizer                 | Optimizer for training                                                 | Optimizer name                                                                                                                   | None                                                                                                                         |
| agent_save_dir                  | Save your own trained agent model                                      | string path                                                                                                                      | 'agent_model/train/'                                                                                                         |
| log_dir                         | Logging testing output                                                 | string path                                                                                                                      | 'output/'                                                                                                                     |
| draw_interval                   | Interval epochs in drawing figures                                     | int value                                                                                                                        | 3                                                                                                                             |
| agent_for_plot_training         | Learnable optimizer to compare                                         | List of agent names                                                                                                              | ['RL_HPSDE_Agent']                                                                                                           |
| n_checkpoint                    | Number of training checkpoints                                         | int value                                                                                                                        | 20                                                                                                                            |
| resume_dir                      | Directory to load previous checkpoint model                            | string path                                                                                                                      | None                                                                                                                         |
| train_parallel_mode             | The parallel processing method for batch env step in training          | ['dummy', 'subproc', 'ray']                                                                                                      | 'dummy'                                                                                                                       |
### The Config of Testing Mode
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| agent                           | Key written in key.json                                                | List of agent names                                                                                                              | []                                                                                                                            |
| t_optimizer                     | Traditional optimizer                                                  | List of traditional optimizer names                                                                                               | []                                                                                                                            |
| test_batch_size                 | Batch size of test set                                                 | int value                                                                                                                        | 1                                                                                                                             |
| parallel_batch                  | The parallel processing mode for testing                               | ['Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch']                                                                          | 'Batch'                                                                                                                       |
### The Config of Rollout Mode
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| agent_for_rollout               | Learnable agent for rollout                                            | Learnable agent name                                                                                                             | None                                                                                                                         |
| checkpoints_for_rollout         | The index of checkpoints for rollout                                   | List of checkpoint indices                                                                                                       | None                                                                                                                         |
| plot_smooth                     | A float between 0 and 1 to control the smoothness of figure curves      | float value between 0 and 1                                                                                                      | 0.8                                                                                                                          |
### Zero-Shot & Transfer Learning
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| problem_from                    | Source problem set in zero-shot and transfer learning                  | ['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch']                                               | None                                                                                                                         |
| problem_to                      | Target problem set in zero-shot and transfer learning                  | ['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch']                                               | None                                                                                                                         |
| difficulty_from                 | Difficulty of source problem set in zero-shot and transfer learning    | ['easy', 'difficult']                                                                                                             | 'easy'                                                                                                                       |
| difficulty_to                   | Difficulty of target problem set in zero-shot and transfer learning    | ['easy', 'difficult']                                                                                                             | 'easy'                                                                                                                       |
### Zero-Shot Parameters
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| model_from                      | The model trained on source problem set                                | string value                                                                                                                      | None                                                                                                                         |
| model_to                        | The model trained on target problem set                                | string value                                                                                                                      | None                                                                                                                         |
### Transfer Learning Parameters
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| pre_train_rollout               | Key of pre-train models rollout in model.json                          | string value                                                                                                                      | None                                                                                                                         |
| scratch_rollout                 | Key of scratch models rollout result in model.json                     | string value                                                                                                                      | None                                                                                                                         |

### General Parameters
| **Config Name**                | **Description**                                                        | **Possible Values**                                                                                                               | **Default**                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| max_epoch                       | Maximum epoch for training                                             | int value                                                                                                                        | 100                                                                                                                           |
| seed                            | Random seed for experiment                                             | int value                                                                                                                        | 3849                                                                                                                          |
| epoch_seed                      | Epoch seed for specific experiments                                    | int value                                                                                                                        | 100                                                                                                                           |
| id_seed                         | ID seed for model generation                                           | int value                                                                                                                        | 5                                                                                                                             |
| train_mode                      | Training mode                                                           | ['single', 'multi']                                                                                                               | 'single'                                                                                                                      |
| end_mode                        | End mode                                                                | ['step', 'epoch']                                                                                                                | 'epoch'                                                                                                                       |
| test_run                        | Test run number                                                         | int value                                                                                                                        | 51                                                                                                                            |
| rollout_run                     | Rollout run number                                                      | int value                                                                                                                        | 10                                                                                                                            |
| no_tb                           | Disable tensorboard logging                                            | True or False                                                                                                                    | False                                                                                                                         |
| log_step                        | Log every log_step steps                                               | int value                                                                                                                        | 50                                                                                                                            |

## Flexible Usage

### 1. Flexible implementation of MetaBBO across different learning paradigms.

```{tip}
MetaBOX not only supports RL-based MetaBBO methods, but also supports MetaBBO methods based on SL, NE, and ICL.
```

Exp:
- **MetaBBO-RL**：GLEET
- **MetaBBO-SL**：GLHF
- **MetaBBO-NE**：LES
- **MetaBBO-ICL**：OPRO

Compared to RL, other methods differ in that they do not require an `rl` class and are instead built directly by inheriting from `basic_agent`.

1️⃣ Create the Agent

```{python}
from metaevobox.rl import basic_agent
class MyAgent(basic_agent)
    # Specific Operation
```

The procedure can be referred to in [Create your own agent](#12-create-your-own-agent).

2️⃣ Create the Optimizer

The procedure can be referred to in [Create your own optimizer](#2-create-your-own-optimizer).


### 2. Flexibly leverage parallelism to accelerate your training and testing.

### 3. Flexibly utilize metadata to visualize your results.

### 4. Flexibly integrate other EC libraries.
```{tip}
There are many excellent libraries in our community that have significantly accelerated our research. \ 
MetaBOX pays tribute to these outstanding libraries and supports their flexible integration.
```
#### EvoX
```{note}
[EvoX](https://github.com/EMI-Group/evox) is a distributed GPU-accelerated evolutionary computation framework compatible with PyTorch. 
```

Usage example: Using EvoX to construct a dataset.

#### PyPop7
```{note}
[PyPop7](https://github.com/Evolutionary-Intelligence/pypop) is a Python library of POPulation-based OPtimization for single-objective, real-parameter, unconstrained black-box problems.
```

Usage example: Using PyPop7 to construct a baseline.
