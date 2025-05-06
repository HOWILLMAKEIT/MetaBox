# Problem & Baseline

introduce the problem set and baseline we use

```{toctree}

:maxdepth: 1
:hidden
Problem
Baseline

```

## Problem

| 🚩 **Problem Category**           | 📚 **Problem set**                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Single-Objective Optimization** | COCO-BBOB、[BBOB-Surrogate](#bbob-surrogate)、HPO-B、NeuroEvolution、Protein-Docking、UAV、[CEC2013LSGO](<(#cec2013lsgo)>) |
| **Multi-Objective Optimization**  | [MOO-Synthetic (dtlz、uf、wfg、zdt)](#moo-synthetic)                                                                       |
| **Multi-Modal Optimization**      | [CEC2013MMO](#cec2013mmo)                                                                                                  |
| **Multi-Task Optimization**       | [CEC2017MTO](#cec2017mto)、[WCCI2020](#wcci2020)                                                                           |

### Single-Object Optimization

#### COCO-BBOB

#### BBOB-Surrogate

- **Introduction**：\
  BBOB-Surrogate investigates the integration of surrogate modeling techniques into MetaBBO , enabling data-driven approximation of expensive objective functions while maintaining optimization fidelity.
- **Original paper**：\
  "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
- **Official Implementation**： [BBOB-Surrogate](https://github.com/GMC-DRL/Surr-RLDE)
- **License**：None
- **Problem Suite Composition**：\
  BBOB-Surrogate contains a total of 72 optimization problems, corresponding to three dimensions (2, 5, 10), each dimension contains 24 problems. Each problem consists of a trained KAN or MLP network, which is used to fit 24 black box functions in the COCO-BBOB benchmark. The network here is a surrogate model of the original function.

#### HPO-B

#### NeuroEvolution

#### Protein-Docking

#### UAV

#### CEC2013LSGO

- **Introduction**：\
  CEC2013LSGO proposes 15 large-scale benchmark problems to represent a wider range of realworld large-scale optimization problems.
- **Original paper**：\
  "[Benchmark functions for the CEC 2013 special session and competition on large-scale global optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf)." gene 7.33 (2013): 8.
- **Official Implementation**： [CEC2013LSGO](https://github.com/dmolina/cec2013lsgo)
- **License**：GPL-3.0
- **Problem Suite Composition**：\
  CEC2013LSGO contains four major categories of large-scale problems:
  1. Fully-separable functions (F1-F3)
  2. Two types of partially separable functions:
     1. Partially separable functions with a set of non-separable subcomponents and one fully-separable subcomponents (F4-F7)
     2. Partially separable functions with only a set of non-separable subcomponents and no fullyseparable subcomponent (F8-F11)
  3. Two types of overlapping functions:
     1. Overlapping functions with conforming subcomponents (F12-F13)
     2. Overlapping functions with conflicting subcomponents (F14)
  4. Fully-nonseparable functions (F15)

### Multi-Objective Optimization

#### MOO-Synthetic

- **Introduction**：\
  MOO-Synthetic provides a more comprehensive problem set for multi-objective optimization by combining multiple mainstream problem sets (ZDT、UF、DTLZ、WFG).
- **Original paper**：
  - **ZDT**："[Comparison of multiobjective evolutionary algorithms: Empirical results](https://ieeexplore.ieee.org/abstract/document/6787994)." Evolutionary computation 8.2 (2000): 173-195.
  - **UF**: "[Multiobjective optimization test instances for the CEC 2009 special session and competition](https://www.al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf)." (2008): 1-30.
  - **DTLZ**: "[Scalable multi-objective optimization test problems](https://ieeexplore.ieee.org/abstract/document/1007032)." Proceedings of the 2002 congress on evolutionary computation. CEC'02 (Cat. No. 02TH8600). Vol. 1. IEEE, 2002.
  - **WFG**: "[A review of multiobjective test problems and a scalable test problem toolkit](https://ieeexplore.ieee.org/abstract/document/1705400)." IEEE Transactions on Evolutionary Computation 10.5 (2006): 477-506.
- **Official Implementation**： [pymoo](https://github.com/anyoptimization/pymoo)
- **License**：Apache-2.0
- **Problem Suite Composition**：\
  MOO-Synthetic contains 187 questions, consisting of the ZDT, UF, DTLZ, and WFG question sets.

### Multi-Modal Optimization

#### CEC2013MMO

- **Introduction**：\
  CEC2013MMO is a problem set for evaluating multi-modal optimization algorithms.
- **Original paper**：
  "[Benchmark functions for CEC’2013 special session and competition on niching methods for multimodal function optimization](https://web.xidian.edu.cn/xlwang/files/20150312_175833.pdf)." RMIT University, evolutionary computation and machine learning Group, Australia, Tech. Rep (2013).
- **Official Implementation**： [CEC2013MMO](https://github.com/mikeagn/CEC2013)
- **License**：View
- **Problem Suite Composition**：\
  CEC2013MMO includes 20 functions covering different dimensions and the number of global optima. Among them, F1 to F5 are simple functions, F6 to F10 are scalable functions with many global optima, and F11 to F20 are composition functions with challenging landscapes.

### Multi-Modal Optimization

#### CEC2017MTO

- **Introduction**：\
  CEC2017MTO is a problem set for evaluating multi-task optimization algorithms.
- **Original paper**：
  "[Evolutionary multitasking for single-objective continuous optimization: Benchmark problems, performance metric, and baseline results](https://arxiv.org/abs/1706.03470)." arXiv preprint arXiv:1706.03470 (2017).
- **Official Implementation**： [CEC2017MTO](http://www.bdsc.site/websites/MTO/index.html)
- **License**：None
- **Problem Suite Composition**：\
  CEC2017MTO has 9 multi-task questions, each of which contains two basic questions.

  <p align="center">
  <img src="https://github.com/GMC-DRL/MetaBox/blob/v2.0.0/docs/pic/CEC2017MTO.png" width="700"/>
  </p>

#### WCCI2020

- **Introduction**：\
  WCCI2020 is a problem set for evaluating multi-task optimization algorithms.
- **Original paper**：
  "[WCCI2020 competition on evolutionary multi-task optimization](http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html)." IEEE World Congress on Computational Intelligence 2020. 2020.
- **Official Implementation**： [WCCI2020](http://www.bdsc.site/websites/MTO/index.html)
- **License**：None
- **Problem Suite Composition**：\
  The benchmark comprises 10 multi-task problems, each integrating 50 uniformly 50-dimensional base tasks.
  <p align="center">
  <img src="https://github.com/GMC-DRL/MetaBox/blob/v2.0.0/docs/pic/WCCI2020.png" width="600"/>
  </p>
  For example, P1 is composed of a single base problem, which consists of 50 differently shifted and rotated Sphere functions. In contrast, composite problems like P4 formed by multiple base problems are constructed by cyclically incorporating differently shifted and rotated Sphere, Rosenbrock, and Ackley functions to form a multi-task optimization problem.

## Baseline

|                     | SOO                                                                                                                                                                                                                                                                                      | MOO             | MMO             | MTO  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------------- | ---- |
| [MetaBBO](#metabbo) | [B2Opt](#b2opt)、[DEDDQN](#deddqn)、[DEDQN](#dedqn)、[GLEET](#gleet)、[GLHF](#glhf)、[LDE](#lde)、[LES](#les)、[LGA](#lga)、[NRLPSO](#nrlpso)、[QLPSO](#qlpso)、[RLDAS](#rldas)、[RLDEAFL](#rldeafl)、[RLEPSO](#rlepso)、[RLPSO](#rlpso)、[SurrRLDE](#surrrlde)、[SYMBOL](#symbol)、OPRO | MADAC           | RLEMMO、PSORLNS | L2T  |
| [BBO](#bbo)         | [CMA-ES](#cma-es)、DE、[GLPSO](#glpso)、[JDE21](#jde21)、[MADDE](#madde)、[NL-SHADE-LBC](#nl-shade-lbc)、PSO、[SHADE](#shade)、[SAHLPSO](#sahlpso)、SDMSPSO                                                                                                                              | [MOEAD](#moead) |                 | MFEA |

### MetaBBO

#### Single-Object Optimization

##### B2OPT

- **Introduction**：\
  B2Opt: Learning to Optimize Black-box Optimization with Little Budget.
- **Original paper**：
  "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).
- **Official Implementation**：[B2Opt](https://github.com/ninja-wm/B2Opt)

##### DEDDQN

- **Introduction**：\
  DE-DDQN is an adaptive operator selection method based on Double Deep Q-Learning (DDQN), a Deep Reinforcement Learning method, to control the mutation strategies of Differential Evolution (DE).
- **Original paper**：
  "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).
- **Official Implementation**：[DE-DDQN](https://github.com/mudita11/DE-DDQN)

##### DEDQN

- **Introduction**：\
  DEDQN is a mixed mutation strategy Differential Evolution (DE) algorithm based on deep Q-network (DQN), in which a deep reinforcement learning approach realizes the adaptive selection of mutation strategy in the evolution process.
- **Original paper**：
  "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing (2021).
- **Official Implementation**： None

##### GLEET

- **Introduction**：\
  GLEET is a **G**eneralizable **L**earning-based **E**xploration-**E**xploitation **T**radeoff framework, which could explicitly control the exploration-exploitation tradeoff hyper-parameters of a given EC algorithm to solve a class of problems via reinforcement learning.
- **Original paper**：
  "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://dl.acm.org/doi/abs/10.1145/3638529.3653996)." Proceedings of the Genetic and Evolutionary Computation Conference (2024).
- **Official Implementation**：[GLEET](https://github.com/GMC-DRL/GLEET)

##### GLHF

- **Introduction**：\
  GLHF: General Learned Evolutionary Algorithm Via Hyper Functions
- **Original paper**：
  "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).
- **Official Implementation**：[GLHF](https://github.com/ninja-wm/POM/)

##### LDE

- **Introduction**：\
  LDE：Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient
- **Original paper**：
  "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation (2021).
- **Official Implementation**：[LDE](https://github.com/yierh/LDE)

##### LES

- **Introduction**：\
  **L**earned **E**volution **S**trategy (LES) is a novel self-attention-based evolution strategies parametrization, and discover effective update rules for ES via meta-learning.
- **Original paper**：
  "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. (2023).
- **Official Implementation**：[LES](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/les.py)

##### LGA

- **Introduction**：\
  **L**earned **G**enetic **A**lgorithm parametrizes selection and mutation rate adaptation as cross- and self-attention modules and use MetaBBO to evolve their parameters on a set of diverse optimization tasks.
- **Original paper**：
  "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. (2023).
- **Official Implementation**：[LGA](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/lga.py)

##### NRLPSO

- **Introduction**：\
  NRLPSO is a reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy.
- **Original paper**：
  "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation (2023).
- **Official Implementation**：None

##### QLPSO

- **Introduction**：\
  QLPSO is a problem-free PSO which integrates a reinforcement learning method.
- **Original paper**：
  "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications (2020).
- **Official Implementation**：None

##### RLDAS

- **Introduction**：\
  RLDAS is a deep reinforcement learning-based dynamic algorithm selection framework.
- **Original paper**：
  "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).
- **Official Implementation**：[RL-DAS](https://github.com/GMC-DRL/RL-DAS)

##### RLDEAFL

- **Introduction**：\
   RLDEAFL is an algorithm that supports automated feature learning during the meta-learning process, which integrates a learnable feature extraction module into a reinforcement learning-based DE method to learn both the feature encoding and meta-level policy.
- **Original paper**："[**Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning**](http://arxiv.org/abs/2503.18061)."
- **Official Implementation**：[RLDE-AFL](https://github.com/MetaEvo/RLDE-AFL)

##### RLEPSO

- **Introduction**：\
   RLEPSO is a new particle swarm optimization algorithm that combines reinforcement learning.
- **Original paper**：
  "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. (2021).
- **Official Implementation**：None

##### RLPSO

- **Introduction**：\
   RLPSO develops a reinforcement learning strategy to enhance PSO in convergence by replacing the uniformly distributed random number in the updating function with a random number generated from a selected normal distribution.
- **Original paper**：
  "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization (2022).Intelligence. (2021).
- **Official Implementation**：None

##### SurrRLDE

- **Introduction**：\
   SurrRLDE is a novel MetaBBO framework which combines surrogate learning process and reinforcement learning-aided Differential Evolution (DE) algorithm.
- **Original paper**：
  "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
- **Official Implementation**：[SurrRLDE](https://github.com/GMC-DRL/Surr-RLDE)

##### SYMBOL

- **Introduction**：\
   SYMBOL is a novel framework that promotes the automated discovery of black-box optimizers through symbolic equation learning.
- **Original paper**：
  "[**Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning**](https://openreview.net/forum?id=vLJcd43U7a)." The Twelfth International Conference on Learning Representations. (2024).
- **Official Implementation**：[SYMBOL](https://github.com/GMC-DRL/Symbol)

#### Multi-Modal Optimization

##### MADAC

- **Introduction**：\
  Multi-agent dynamic algorithm configuration in which one agent works for one type of configuration hyperparameter.It rmulates the dynamic configuration of a complex algorithm with multiple types of hyperparameters as a contextual multi-agent Markov decision process and solves it by a cooperative multi-agent RL (MARL) algorithm.
- **Original paper**：
  "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.
- **Official Implementation**：[MADAC](https://github.com/lamda-bbo/madac)

#### Multi-Task Optimization

### BBO

#### Single-Object Optimization

##### CMA-ES

- **Introduction**：\
  A novel evolutionary optimization strategy based on the derandomized evolution strategy with covariance matrix adaptation. This is accomplished by efficientlyincorporating the available information from a large population, thus significantly re-ducing the number of generations needed to adapt the covariance matrix.
- **Original paper**：
  "[**Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)**](https://ieeexplore.ieee.org/abstract/document/6790790/)." Evolutionary Computation 11.1 (2003): 1-18.
- **Official Implementation**：None

##### GLPSO

- **Introduction**：\
  The PSO algorithm is hybridized with genetic evolution mechanisms. In this approach, genetic operators—specifically crossover, mutation, and selection—are incorporated into the PSO framework to construct promising exemplars and enhance the search performance.
- **Original paper**：
  "[**Genetic learning particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/7271066/)." IEEE Transactions on Cybernetics 46.10 (2015): 2277-2290.
- **Official Implementation**：[GLPSO](http://www.ai.sysu.edu.cn/GYJ/glpso/c_co)

##### JDE21

- **Introduction**：\
  A DE for solving single-objective real-parameter bound-constrained optimization problems. It uses several mechanisms to tackle optimization problems efficiently: two populations with different sizes, restart mechanism in both populations, self-adaptive control parameters F and CR, the extended range of values for CR in thebigger population, migration of the best individual from the big population into the small population, modified mutation strategy in the bigger population, crowding mechanism and population size reduction in the bigger population.
- **Original paper**：
  "[**Self-adaptive differential evolution algorithm with population size reduction for single objective bound-constrained optimization: Algorithm j21**](https://ieeexplore.ieee.org/abstract/document/9504782/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.
- **Official Implementation**：None

##### MADDE

- **Introduction**：\
  A variant of the DE algorithm leverages the power of the multiple adaptation strategy(Mad) with respect to the control parameters and search mechanisms.
- **Original paper**：
  "[**Improving differential evolution through Bayesian hyperparameter optimization**](https://ieeexplore.ieee.org/abstract/document/9504792/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.
- **Official Implementation**：[MADDE](https://github.com/subhodipbiswas/MadDE)

##### NL-SHADE-LBC

- **Introduction**：\
  Non-Linear population size reduction Success-History Adaptive Differential Evolution with Linear Bias Change.It combines selective pressure, biased parameter adaptation with linear bias change, current-to-pbest strategy, resampling of solutions as bound constraint handling techniques, as well as the non-linear population size reduction.
- **Original paper**：
  "[**NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization**](https://ieeexplore.ieee.org/abstract/document/9870295/)." 2022 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2022.
- **Official Implementation**：None

##### SHADE

- **Introduction**：\
  A parameter adaptation technique for DE which uses a historical memory of successful control parameter settings to guide the selection of future control parameter values.
- **Original paper**：
  "[**Success-history based parameter adaptation for differential evolution**](https://ieeexplore.ieee.org/abstract/document/6557555/)." 2013 IEEE Congress on Evolutionary Computation. IEEE, 2013.
- **Official Implementation**：None

##### SAHLPSO

- **Introduction**：\
  Self-Adaptive two roles hybrid learn-ing strategies-based particle swarm optimization.It uses exploration-role and exploitation-role learning strategies with self-adaptively updating parameters manner.
- **Original paper**：
  "[**Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization**](https://www.sciencedirect.com/science/article/pii/S0020025521006988)." Information Sciences 578 (2021): 457-481.
- **Official Implementation**：None

##### SDMSPSO

- **Introduction**：\
  The sDMS-PSO is a self-adaptive dynamic multi-swarm particle swarm optimizer that incorporates parameter adaptation, cooperative coevolution among multiple swarms, and a quasi-Newton local search to enhance convergence speed and optimization performance.
- **Original paper**：
  "[**A self-adaptive dynamic particle swarm optimizer**](https://ieeexplore.ieee.org/abstract/document/7257290/)." 2015 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2015.
- **Official Implementation**：None

#### Multi-Objective Optimization

##### MOEAD

- **Introduction**：\
   MOEAD is a multiobjective evolutionary algorithm based on decomposition.It decomposes a multiobjective optimization problem into a number of scalar optimization subproblems and optimizes them simultaneously. Each subproblem is optimized by only using information from its several neighboring subproblems.
- **Original paper**：
  "[**MOEA/D: A multiobjective evolutionary algorithm based on decomposition**](https://ieeexplore.ieee.org/abstract/document/4358754/)." IEEE Transactions on Evolutionary Computation 11.6 (2007): 712-731.
- **Official Implementation**：None

#### Multi-Modal Optimization

#### Multi-Task Optimization
