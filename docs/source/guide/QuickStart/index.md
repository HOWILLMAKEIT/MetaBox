# Quickstart

```{toctree}

:maxdepth: 1

Common_Usage
Core_Concept

```

# Quickstart

Hi！Here is the fastest way to begin your journey of Metabox.

## Installation
### Linux
    bash bash_to_install_metabox_linux
### Windows
    bash bash_to_install_metabox_windows

## MetaBBO's Core Concept

> 💡  **MetaBox is an all-in-one platform for using and developing the algorithms in Meta-Black-Box Optimization (MetaBBO)**.  💡

MetaBBO, an emerging research direction in recent years, aims to automate the design of BBO algorithms by constructing intelligent agents as replacements for human experts. Its dual-layer architecture synergizes:

- Low-level​​: Standard BBO algorithms for solving optimization problems.
- Meta-level​​: A parameterized AI agent that adjusts low-level algorithms in real-time based on their optimization status info.

Through meta-learning on target problem distribution, MetaBBO shifts from human-expertise-driven design to data-driven automation, delivering unprecedented generalization power and design efficiency, and the performance
exceeds that of traditional BBO.

For further exploration, we recommend reading the comprehensive survey : "[Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization](https://arxiv.org/abs/2411.00625)" and exploring the curated repository [Awesome-MetaBBO](https://github.com/GMC-DRL/Awesome-MetaBBO), which aggregates MetaBBO-related research papers and code implementations.

## Common Usage
### 1. Train one algorithm on MetaBox's one Dataset

> [!NOTE]
> **The following command demonstrates the core training logic.**  
> Numerous configurable options are available — refer to **Gallery > Config** for details.
> 
🧪 General Training Command

    python MetaBox/src/main.py --train 
        --train_problem xxx 
        --train_difficulty xxx 
        --train_agent xxx 
        --train_optimizer xxx

🎯 Example: Train GLEET on COCO's BBOB (10D, easy)

    python MetaBox/src/main.py --train 
    --train_problem bbob-10D 
    --train_difficulty easy 
    --train_agent GLEET 
    --train_optimizer GLEET_Optimizer

### 2. Test one algorithm on MetaBox's one Dataset

### 3. Bash Builder
