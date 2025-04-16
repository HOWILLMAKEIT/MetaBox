# Quickstart

the fastest way to begin your journey of Metabox.

## Installation
### Linux
    ```bash bash_to_install_metabox_linux
### Windows
    ```bash bash_to_install_metabox_windows

## Metabox's core concept

### Environment
- **Environment**: A collection of problems, datasets, classic optimizer, Metabbo's low-level optimizer and parrallel environment(advanced) that can be used to train and evaluate machine learning algorithms. It provides a standardized interface for interacting with different datasets and problems.

### RL
- **RL**: Reinforcement Learning, which is surported by Metabox for make decisions as an agent by taking actions in an environment to maximize some notion of cumulative reward.

### Baseline
- **Baseline**: Includes the classic optimizers(bbo) and Metabbo. It is a set of several type algorithms or methods used as a reference point for evaluating the performance of other algorithms.

### Trainer
- **Trainer**: A module responsible for training a metabbo.

### Tester
- **Tester**: A module responsible for testing a classic bbo or a trained metabbo. 

## Common Usage
### 1. Train one algorithm on Metabox's one Dataset with high customization

### 2. Test one algorithm on Metabox's one Dataset with high customization

### 3. basic data

### 4. Trying to use Metabox's common function in this example




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