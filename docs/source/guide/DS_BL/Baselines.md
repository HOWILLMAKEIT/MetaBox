# Baselines

|                     | SOO                                                                                                                                                                                                                                                                                 | MOO             | MMO             | MTO  |
| ------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------- | --------------- | ---- |
| [MetaBBO](#metabbo) | [B2Opt](#b2opt)、[DEDDQN](#deddqn)、[DEDQN](#dedqn)、[GLEET](#gleet)、[GLHF](#glhf)、[LDE](#lde)、[LES](#les)、[LGA](#lga)、[NRLPSO](#nrlpso)、[QLPSO](#qlpso)、[RLDAS](#rldas)、[RLDEAFL](#rldeafl)、[RLEPSO](#rlepso)、[RLPSO](#rlpso)、[SurrRLDE](#surrrlde)、[SYMBOL](#symbol)、OPRO、RNN-OPT、RL-HPSDE | MADAC           | RLEMMO、PSORLNS | L2T  |
| [BBO](#bbo)         | [CMA-ES](#cma-es)、DE、[GLPSO](#glpso)、[JDE21](#jde21)、[MADDE](#madde)、[NL-SHADE-LBC](#nl-shade-lbc)、PSO、[SHADE](#shade)、[SAHLPSO](#sahlpso)、SDMSPSO                                                                                                                                  | [MOEAD](#moead) |                 | MFEA |

## MetaBBO

### Single-Object Optimization

#### B2OPT

- **Introduction**：\
  B2Opt: Learning to Optimize Black-box Optimization with Little Budget.
- **Original paper**：
  "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).
- **Official Implementation**：[B2Opt](https://github.com/ninja-wm/B2Opt)

#### DEDDQN

- **Introduction**：\
  DE-DDQN is an adaptive operator selection method based on Double Deep Q-Learning (DDQN), a Deep Reinforcement Learning method, to control the mutation strategies of Differential Evolution (DE).
- **Original paper**：
  "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).
- **Official Implementation**：[DE-DDQN](https://github.com/mudita11/DE-DDQN)

#### DEDQN

- **Introduction**：\
  DEDQN is a mixed mutation strategy Differential Evolution (DE) algorithm based on deep Q-network (DQN), in which a deep reinforcement learning approach realizes the adaptive selection of mutation strategy in the evolution process.
- **Original paper**：
  "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing (2021).
- **Official Implementation**： None

#### GLEET

- **Introduction**：\
  GLEET is a **G**eneralizable **L**earning-based **E**xploration-**E**xploitation **T**radeoff framework, which could explicitly control the exploration-exploitation tradeoff hyper-parameters of a given EC algorithm to solve a class of problems via reinforcement learning.
- **Original paper**：
  "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://dl.acm.org/doi/abs/10.1145/3638529.3653996)." Proceedings of the Genetic and Evolutionary Computation Conference (2024).
- **Official Implementation**：[GLEET](https://github.com/GMC-DRL/GLEET)

#### GLHF

- **Introduction**：\
  GLHF: General Learned Evolutionary Algorithm Via Hyper Functions
- **Original paper**：
  "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).
- **Official Implementation**：[GLHF](https://github.com/ninja-wm/POM/)

#### LDE

- **Introduction**：\
  LDE：Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient
- **Original paper**：
  "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation (2021).
- **Official Implementation**：[LDE](https://github.com/yierh/LDE)

#### LES

- **Introduction**：\
  **L**earned **E**volution **S**trategy (LES) is a novel self-attention-based evolution strategies parametrization, and discover effective update rules for ES via meta-learning.
- **Original paper**：
  "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. (2023).
- **Official Implementation**：[LES](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/les.py)

#### LGA

- **Introduction**：\
  **L**earned **G**enetic **A**lgorithm parametrizes selection and mutation rate adaptation as cross- and self-attention modules and use MetaBBO to evolve their parameters on a set of diverse optimization tasks.
- **Original paper**：
  "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. (2023).
- **Official Implementation**：[LGA](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/lga.py)

#### NRLPSO

- **Introduction**：\
  NRLPSO is a reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy.
- **Original paper**：
  "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation (2023).
- **Official Implementation**：None

#### QLPSO

- **Introduction**：\
  QLPSO is a problem-free PSO which integrates a reinforcement learning method.
- **Original paper**：
  "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications (2020).
- **Official Implementation**：None

#### RLDAS

- **Introduction**：\
  RLDAS is a deep reinforcement learning-based dynamic algorithm selection framework.
- **Original paper**：
  "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).
- **Official Implementation**：[RL-DAS](https://github.com/GMC-DRL/RL-DAS)

#### RLDEAFL

- **Introduction**：\
   RLDEAFL is an algorithm that supports automated feature learning during the meta-learning process, which integrates a learnable feature extraction module into a reinforcement learning-based DE method to learn both the feature encoding and meta-level policy.
- **Original paper**："[**Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning**](http://arxiv.org/abs/2503.18061)."
- **Official Implementation**：[RLDE-AFL](https://github.com/MetaEvo/RLDE-AFL)

#### RLEPSO

- **Introduction**：\
   RLEPSO is a new particle swarm optimization algorithm that combines reinforcement learning.
- **Original paper**：
  "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. (2021).
- **Official Implementation**：None

#### RLPSO

- **Introduction**：\
   RLPSO develops a reinforcement learning strategy to enhance PSO in convergence by replacing the uniformly distributed random number in the updating function with a random number generated from a selected normal distribution.
- **Original paper**：
  "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization (2022).Intelligence. (2021).
- **Official Implementation**：None

#### SurrRLDE

- **Introduction**：\
   SurrRLDE is a novel MetaBBO framework which combines surrogate learning process and reinforcement learning-aided Differential Evolution (DE) algorithm.
- **Original paper**：
  "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
- **Official Implementation**：[SurrRLDE](https://github.com/GMC-DRL/Surr-RLDE)

#### SYMBOL

- **Introduction**：\
   SYMBOL is a novel framework that promotes the automated discovery of black-box optimizers through symbolic equation learning.
- **Original paper**：
  "[**Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning**](https://openreview.net/forum?id=vLJcd43U7a)." The Twelfth International Conference on Learning Representations. (2024).
- **Official Implementation**：[SYMBOL](https://github.com/GMC-DRL/Symbol)

### Multi-Modal Optimization

#### MADAC

- **Introduction**：\
  Multi-agent dynamic algorithm configuration in which one agent works for one type of configuration hyperparameter.It rmulates the dynamic configuration of a complex algorithm with multiple types of hyperparameters as a contextual multi-agent Markov decision process and solves it by a cooperative multi-agent RL (MARL) algorithm.
- **Original paper**：
  "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.
- **Official Implementation**：[MADAC](https://github.com/lamda-bbo/madac)

### Multi-Task Optimization

## BBO

### Single-Object Optimization

#### CMA-ES

- **Introduction**：\
  A novel evolutionary optimization strategy based on the derandomized evolution strategy with covariance matrix adaptation. This is accomplished by efficientlyincorporating the available information from a large population, thus significantly re-ducing the number of generations needed to adapt the covariance matrix.
- **Original paper**：
  "[**Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)**](https://ieeexplore.ieee.org/abstract/document/6790790/)." Evolutionary Computation 11.1 (2003): 1-18.
- **Official Implementation**：None

#### GLPSO

- **Introduction**：\
  The PSO algorithm is hybridized with genetic evolution mechanisms. In this approach, genetic operators—specifically crossover, mutation, and selection—are incorporated into the PSO framework to construct promising exemplars and enhance the search performance.
- **Original paper**：
  "[**Genetic learning particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/7271066/)." IEEE Transactions on Cybernetics 46.10 (2015): 2277-2290.
- **Official Implementation**：[GLPSO](http://www.ai.sysu.edu.cn/GYJ/glpso/c_co)

#### JDE21

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
