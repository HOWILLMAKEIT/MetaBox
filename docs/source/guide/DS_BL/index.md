# Problem & Baseline

introduce the problem set and baseline we use

```{toctree}

:maxdepth: 1
:hidden
Problem
Baseline

```

## Problem

| 🚩 **Problem Category**         | 📚 **Problem set**                                                               |
|-------------------------------|-------------------------------------------------------------------------------|
| **Single-Objective Optimization** | COCO-BBOB、[BBOB-Surrogate](#bbob-surrogate)、HPO-B、NeuroEvolution、Protein-Docking、UAV、[CEC2013LSGO]((#cec2013lsgo)) |
| **Multi-Objective Optimization**  | [MOO-Synthetic(dtlz、uf、wfg、zdt)](#moo-synthetic)                                   |
|  **Multi-Modal Optimization**      | [CEC2013MMO](#cec2013mmo)                                                                    |
|  **Multi-Task Optimization**       | CEC2017MTO、WCCI2020                                                          |



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
  
## Baseline
