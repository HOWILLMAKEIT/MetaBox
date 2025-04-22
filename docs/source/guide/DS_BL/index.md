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
| **Single-Objective Optimization** | COCO-BBOB、[BBOB-Surrogate](#bbob-surrogate)、HPO-B、NeuroEvolution、Protein-Docking、UAV、CEC2013LSGO |
| **Multi-Objective Optimization**  | MOO-Synthetic（dtlz、uf、wfg、zdt）                                            |
|  **Multi-Modal Optimization**      | CEC2013MMO                                                                    |
|  **Multi-Task Optimization**       | CEC2017MTO、WCCI2020                                                          |



### Single-Object Optimization

#### COCO-BBOB

#### BBOB-Surrogate

- **Introduction**：\
  BBOB-Surrogate investigates the integration of surrogate modeling techniques into MetaBBO , enabling data-driven approximation of expensive objective functions while maintaining optimization fidelity.
- **Original paper**：\
  Ma, Zeyuan, et al. "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
- **Official Implementation**： [BBOB-Surrogate](https://github.com/GMC-DRL/Surr-RLDE)
- **License**：None
- **Problem Suite Composition**：\
  BBOB-Surrogate contains a total of 72 optimization problems, corresponding to three dimensions (2, 5, 10), each dimension contains 24 problems. Each problem consists of a trained KAN or MLP network, which is used to fit 24 black box functions in the COCO-BBOB benchmark. The network here is a surrogate model of the original function.


## Baseline
