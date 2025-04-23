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

#### 1. Create your own RL

     from metabox.rl.basic_agent

     class MyRL(basic_agent)
       pass

#### 1. Create your own Agent

> [!TIP]
> MetaBOX has pre-implemented various RL methods — refer to **Gallery > Config** for details. \
> You just need to inherit it and design your own Agent！

     from metabox.rl import xxx
     
     class MyAgent(xxx):
       pass

#### 2. Create your own Optimizer

     from metabox.environment.optimizer import Learnale_Optimizer
     
     class MyOptimizer(xxx):
       pass

    

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
