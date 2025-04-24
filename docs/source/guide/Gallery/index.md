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

       if self.learning_time >= self.config.max_learning_step:
           return_info = {'return': _R, 'learn_steps': self.learning_time, }
           env_cost = env.get_env_attr('cost')
           return_info['gbest'] = env_cost[-1]
           for key in required_info.keys():
               return_info[key] = env.get_env_attr(required_info[key])
           env.close()
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
from metabox.rl import xxx

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

       if self.learning_time >= self.config.max_learning_step:
           return_info = {'return': _R, 'learn_steps': self.learning_time, }
           env_cost = env.get_env_attr('cost')
           return_info['gbest'] = env_cost[-1]
           for key in required_info.keys():
               return_info[key] = env.get_env_attr(required_info[key])
           env.close()
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
from metabox.environment.optimizer import Learnale_Optimizer

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
        self.meta_X = [self.population.copy()]
        self.meta_Cost = [self.cost.copy()]

    return # According to your specific needs

```
3️⃣ The Main function for updating

```python
def update(self, action, problem):

    # Specific operations
    # xxxx

    # In order to ensure that the logger data format is correct,
    # there is only one stop mechanism.
    is_end = self.fes >= self.max_fes

    if self.fes >= self.log_interval * self.log_index:
        self.log_index += 1
        self.cost.append(self.gbest_val)

    if is_end:
        if len(self.cost) >= self.config.n_logpoint + 1:
            self.cost[-1] = self.gbest_val
        else:
            while len(self.cost) < self.config.n_logpoint + 1:
                self.cost.append(self.gbest_val)

    info = {}
    return next_state, reward, is_end, info
```

```{important}
Since optimizer is extremely flexible, the above functions are only necessary \
and need to be adjusted appropriately **according to specific tasks**.
```

### BBO

```{tip}
Considering that you may need to compare with other bbo,
we also open the bbo design interface to you! 😉
```

1️⃣ Inheritance and Initialization

```python
from environment.optimizer.basic_optimizer import Basic_Optimizer

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



```
## 2. Implement your own dataset in Metabox

## 3. Config

## BashBuilder

You can use the following module to generate a bash command with options. This is a simple command builder that allows you to select options and generate a command dynamically.

```{raw} html
<div class="command-generator">
  <h3>Bash 命令生成器</h3>

  <div class="control-group">
    <label for="difficulty">难度级别:</label>
    <select id="difficulty" class="form-control">
      <option value="easy">简单</option>
      <option value="difficult">困难</option>
      <option value="full">完整数据集</option>
        <option value="customized">自定义</option>
    </select>
  </div>

  <div class="control-group">
    <label for="problem-set">问题集:</label>
    <select id="problem-set" class="form-control">
      <option value="problem1">问题1</option>
      <option value="problem2">问题2</option>
      <option value="problem2">问题3</option>
    </select>
  </div>

  <div class="control-group">
    <label for="action">操作类型:</label>
    <select id="action" class="form-control">
      <option value="train">训练</option>
      <option value="test">测试</option>
      <option value="rollout">rollout</option>
      <option value="experiment">experiment</option>
    </select>
  </div>

  <div class="generated-command">
    <strong>生成的命令:</strong>
    <code id="bash-command">$ ./script.sh --difficulty   --problem   --action  </code>
    <button id="copy-btn" class="btn-copy">复制</button>
  </div>
</div>

<style>
.command-generator {
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  padding: 16px;
  margin: 20px 0;
  background-color: #f6f8fa;
}

.control-group {
  margin-bottom: 12px;
}

.control-group label {
  display: inline-block;
  width: 100px;
  font-weight: 600;
}

.form-control {
  padding: 6px 8px;
  border: 1px solid #d1d5da;
  border-radius: 3px;
  width: 200px;
}

.generated-command {
  margin-top: 20px;
  padding: 12px;
  background-color: #24292e;
  border-radius: 3px;
  color: white;
}

.generated-command code {
  font-family: monospace;
}

.btn-copy {
  margin-left: 10px;
  padding: 3px 10px;
  background-color: #0366d6;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
}

.btn-copy:hover {
  background-color: #035fc7;
}
</style>

<script>
// 获取DOM元素
const difficultySelect = document.getElementById('difficulty');
const problemSetSelect = document.getElementById('problem-set');
const actionSelect = document.getElementById('action');
const bashCommand = document.getElementById('bash-command');
const copyBtn = document.getElementById('copy-btn');

// 命令模板
const commandTemplate = (level, set, action) =>
  `$ ./script.sh --difficulty ${level} --problem ${set} --action ${action}`;

// 更新命令函数
function updateCommand() {
  const difficulty = difficultySelect.value;
  const problem = problemSetSelect.value;
  const action = actionSelect.value;
  bashCommand.textContent = commandTemplate(difficulty, problem, action);
}

// 复制命令函数
function copyToClipboard() {
  const command = bashCommand.textContent;
  navigator.clipboard.writeText(command.trim());

  // 临时改变按钮文本
  copyBtn.textContent = '已复制!';
  setTimeout(() => {
    copyBtn.textContent = '复制';
  }, 2000);
}

// 添加事件监听
[difficultySelect, problemSetSelect, actionSelect].forEach(select => {
  select.addEventListener('change', updateCommand);
});

copyBtn.addEventListener('click', copyToClipboard);

// 初始化命令
updateCommand();
</script>
```

## flexible usage

### 1. Transferring an algorithm to another type of problem

### 2. parrallel

### 3. using metadata (metric)

### 4. How to use other EC open source libraries
