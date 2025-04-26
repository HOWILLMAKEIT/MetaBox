# Quickstart

```{toctree}

:maxdepth: 1

Common_Usage
Core_Concept

```

# Quickstart

🚀 Hi！Here is **the fastest way to begin your journey of MetaBox**. 🚀

## Installation

```bash
conda create -n metaevobox_env python=3.11.5 -y
conda activate metaevobox_env
pip install metaevobox
```

## MetaBBO's Core Concept

> 💡 **MetaBox is an all-in-one platform for using and developing the algorithms in Meta-Black-Box Optimization (MetaBBO)**. 💡

MetaBBO, an emerging research direction in recent years, aims to automate the design of BBO algorithms by constructing intelligent agents as replacements for human experts.

```{image} /_static/metabbo.png
:alt: MetaBBO
:align: center
```

Its dual-layer architecture synergizes:

- Low-level​​: Standard BBO algorithms for solving optimization problems.
- Meta-level​​: A parameterized AI agent that adjusts low-level algorithms in real-time based on their optimization status info.

Through meta-learning on target problem distribution, MetaBBO shifts from human-expertise-driven design to data-driven automation, delivering unprecedented generalization power and design efficiency, and the performance
exceeds that of traditional BBO.

For further exploration, we recommend reading the comprehensive survey : "[Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization](https://arxiv.org/abs/2411.00625)" and exploring the curated repository [Awesome-MetaBBO](https://github.com/GMC-DRL/Awesome-MetaBBO), which aggregates MetaBBO-related research papers and code implementations.

## Common Usage

### 1. Train one MetaBox's algorithm on MetaBox's one dataset

```{note}
**The following code demonstrates the core training logic.**
Numerous configurable options are available — refer to **Gallery > Config** for details.
```

<!-- ```{note} Notes require **no** arguments, so content can start here.
```
```{tip} Notes require **no** arguments, so content can start here.
```
```{warning} Notes require **no** arguments, so content can start here.
```
:::{note}
This text is **standard** _Markdown_
:::
:::{warning}
This text is **standard** _Markdown_
:::
```{admonition} Here's my title
:class: note

Here's my admonition content

``` -->

🧪 General Training Code

```python
from metaevobox import Trainer, Config
from metaevobox.baseline.metabbo import XXX
from metaevobox.baseline.metabbo import XXX_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

user_config = {"train_problem": "xxx",
                   "train_difficulty": "xxx"
                   }
config = Config(user_config)
config, dataset = construct_problem_set(config)

agent = XXX(config)
optimizer = XXX_Optimizer(config)

trainer = Trainer(config, agent, optimizer, dataset)
trainer.train()
```

🎯 Example: Train GLEET on COCO's BBOB (10D, easy)

```python
from metaevobox import Trainer, Config
from metaevobox.baseline.metabbo import GLEET
from metaevobox.baseline.metabbo import GLEET_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

user_config = {"train_problem": "bbob-10D",
               "train_difficulty": "easy"
               }
config = Config(user_config)
config, dataset = construct_problem_set(config)

agent = GLEET(config)
optimizer = GLEET_Optimizer(config)

trainer = Trainer(user_config, agent, optimizer, dataset)
trainer.train()
```

<!-- > [!TIP]
> **Train your algorithm on MetaBox** — refer to  **Gallery > Config** for details. -->

```{tip} **Train your algorithm on MetaBox** — refer to  **Gallery > Config** for details.

```

### 2. Test one MetaBox's algorithm on MetaBox's one Dataset

<!-- > [!NOTE]
> **The following code demonstrates the core test logic.**
> Numerous configurable options are available — refer to **Gallery > Config** for details. -->

```{note} **The following code demonstrates the core test logic.**
Numerous configurable options are available — refer to **Gallery > Config** for details.
```

🧪 General Tester Code

```python
import pickle
from metaevobox import Tester, Config
from metaevobox.environment.problem.utils import construct_problem_set
user_config = {"test_problem": "xxx",
               "test_difficulty": "xxx"
               }
config = Config(user_config)
config, dataset = construct_problem_set(config)

dir = "xxx"
with open(dir, 'rb') as f:
     agent = pickle.load(f)
opt = XXX_Optimizer(config)

tester = Tester(config, user_agents: [agent], user_loptimizers: [opt], user_datasets = dataset)
tester.test()
```

🎯 Example: Test GLEET and CMAES on COCO's BBOB (10D, easy)

Assume the GLEET agent is saved in "agent_model/train/GLEET/20250426T113530_bbob-10D_easy/checkpoint-0.pk1"
```python
from metaevobox import Tester, Config
from metaevobox.environment.problem.utils import construct_problem_set
from metaevobox.environment.optimizer import GLEET_Optimizer
from metaevobox.bbo import CMAES

user_config = {"train_problem": "bbob-10D",
                "train_difficulty": "easy",
                }
config = Config(user_config)
config, dataset = construct_problem_set(config)

dir = "agent_model/train/GLEET/20250426T113530_bbob-10D_easy/checkpoint-0.pk1"
with open(dir, 'rb') as f:
     agent = pickle.load(f)
opt = GLEET_Optimizer(config)

tester = Tester(config, user_agents: [agent], user_loptimizers: [opt], user_toprimizers：[CMAES], user_datasets = dataset)
tester.test()
```

<!-- > [!TIP]
> **Test your algorithm on MetaBox** — refer to  **Gallery > Config** for details.\
> **Test two or more algorithms** — refer to  **Gallery > Config** for details. -->

```{tip} **Test your algorithm on MetaBox** — refer to  **Gallery > Config** for details.\
**Test two or more algorithms** — refer to  **Gallery > Config** for details.
```

### 3. The builder of user_config

<!-- > [!IMPORTANT]
> MetaBox provides granularly configurable parameters that empower you to **​​tailor training and testing workflows**​​. \
> ​​Feeling overwhelmed?​​ Navigate to **​​Gallery > Config**​​ to access the intuitive **user_config builder**. \
> This tool is designed to accelerate your process and simplify configuration management." -->

```{important} MetaBox provides granularly configurable parameters that empower you to **​​tailor training and testing workflows**​​.
​​Feeling overwhelmed?​​ Navigate to **​​Gallery > Config**​​ to access the intuitive **user_config builder**.
This tool is designed to accelerate your process and simplify configuration management."
```
