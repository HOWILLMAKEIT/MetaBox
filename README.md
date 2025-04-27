<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-title.png" width="50%">
</div>
<h2 align="center">
  <div style="font-size: 0.9em; margin-top: 12px">
    Benchmarking Meta-Black-Box Optimization under<br/>
    Diverse Optimization Scenarios with Efficiency and Flexibility
  </div>
</h2>

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-b31b1b.svg)]([https://proceedings.neurips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html]) **MetaBox-v1 has been published at NeurIPS 2023！**

<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-NEW.png" width="80%">
</div>

<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-features.png" width="99%">
</div>

we propose MetaBox 2.0 version (MetaBox-v2) as a major upgradation of [MetaBox-v1](https://github.com/MetaEvo/MetaBox/tree/v1.0.0). MetaBox-v2 now supports plentiful optimization scenarios to embrace users from single-objective optimization, multi-objective optimization, multi-modal optimization, multi-task optimization and etc. Correspondingly, **11 optimization problem sets** (synthetic + realistic) and **36 baseline methods** (traditional optimizers + up-to-date MetaBBOs) are reproduced within MetaBox-v2 to assist various research ideas and comprehensive comparison. To address MetaBBO's inherent efficiency issue, we have optimized low-level implementation of MetaBox-v2 to support parallel meta-training and evaluation, which reduces the running cost from days to hours. More importantly, we have optimized MetaBox-v2's sourcecode to support **sufficient development flexbility**, with clear and sound tutotials correspondingly. Enjoy your journey of learning and using MetaBBO from here!   


## Quick Start
### Installation

> [!Important]
> Below we install a cpu-version torch for you, if you need install any other versions, \
> see [torch](https://pytorch.org/get-started) and replace the corresponding installation instruction below.

```bash
## create a venv
conda create -n metaevobox_env python=3.11.5 -y
conda activate metaevobox_env
## install pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
## install metabox
pip install metaevobox
```
### Common Usage
#### Train a MetaBBO baseline
#### Test BBO/MetaBBO baselines

### High-level Development Usage
We sincerely suggest researchers with interests to check out **[Online Documentation](https://metaboxdoc.readthedocs.io/en/latest/index.html)** for further flexible usege of MetaBox-v2, such as implementing your own MetaBBO, customized experimental design & analysis, using pre-collected metadata and seamless API calling with other famous optimization repos.

<details>
<summary><h2>Available Optimization Problem Sets in MetaBox-v2</h2></summary>

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Type</th> <!-- Center the Type column -->
      <th colspan="3" align="center">Problem Set</th> <!-- Center the Problem Set columns -->
      <th rowspan="2" align="center">Description</th> <!-- Center the Description column -->
    </tr>
    <tr>
      <th align="center">Name</th>
      <th align="center">Paper</th>
      <th align="center">Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" align="center">Single-Objective Optimization</td> <!-- Center the Type column -->
      <td align="center">COCO-BBOB-10D</td>
      <td><a href="#">Paper</a></td>
      <td><a href="#">Code</a></td>
      <td>A problem set for single-objective optimization with benchmark problems.</td>
    </tr>
    <tr>
      
      <td align="center">COCO-BBOB-30D</td>
      <td><a href="#">Paper</a></td>
      <td><a href="#">Code</a></td>
      <td>A problem set for single-objective optimization with benchmark problems.</td>
    </tr>
    <tr>
      <td align="center">BBOB-Surrogate</td>
      <td><a href="https://arxiv.org/abs/2503.18060">Paper</a></td>
      <td><a href="https://github.com/GMC-DRL/Surr-RLDE">Code</a></td>
      <td>Integration of surrogate modeling techniques for expensive objective functions.</td>
    </tr>
    <tr>
      <td align="center">HPO-B</td>
      <td><a href="#">Paper</a></td>
      <td><a href="#">Code</a></td>
      <td>Problem set designed for hyperparameter optimization.</td>
    </tr>
    <tr>
      <td align="center">Neuroevolution</td>
      <td><a href="#">Paper</a></td>
      <td><a href="#">Code</a></td>
      <td>Problem set for optimization through neuroevolutionary techniques.</td>
    </tr>
    <tr>
      <td align="center">Protein-Docking</td>
      <td><a href="#">Paper</a></td>
      <td><a href="#">Code</a></td>
      <td>Focuses on protein structure docking in computational biology.</td>
    </tr>
    <tr>
      <td align="center">CEC2013LSGO</td>
      <td><a href="https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf">Paper</a></td>
      <td><a href="https://github.com/dmolina/cec2013lsgo">Code</a></td>
      <td>A large-scale benchmark suite for global optimization.</td>
    </tr>
    <tr>
      <td rowspan="1" align="center">Multi-Objective Optimization</td> <!-- Center the Type column -->
      <td align="center">MOO-Synthetic</td>
      <td>
        <a href="https://ieeexplore.ieee.org/abstract/document/6787994">ZDT</a><br>
        <a href="https://www.al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf">UF</a><br>
        <a href="https://ieeexplore.ieee.org/abstract/document/1007032">DTLZ</a><br>
        <a href="https://ieeexplore.ieee.org/abstract/document/1705400">WFG</a>
      </td>
      <td><a href="https://github.com/anyoptimization/pymoo">Code</a></td>
      <td>Combines multiple objective optimization problems like ZDT, UF, DTLZ, and WFG.</td>
    </tr>
    <tr>
      <td rowspan="1" align="center">Multi-Model Optimization</td> <!-- Center the Type column -->
      <td align="center">CEC2013MMO</td>
      <td><a href="https://web.xidian.edu.cn/xlwang/files/20150312_175833.pdf">Paper</a></td>
      <td><a href="https://github.com/mikeagn/CEC2013">Code</a></td>
      <td>A problem set for multi-modal optimization algorithms.</td>
    </tr>
    <tr>
      <td rowspan="2" align="center">Multi-Task Optimization</td> <!-- Center the Type column -->
      <td align="center">CEC2017MTO</td>
      <td><a href="https://arxiv.org/abs/1706.03470">Paper</a></td>
      <td><a href="http://www.bdsc.site/websites/MTO/index.html">Code</a></td>
      <td>Multi-task optimization benchmark for evaluating multitasking algorithms.</td>
    </tr>
    <tr>
      <td align="center">WCCI2020</td>
      <td><a href="http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html">Paper</a></td>
      <td><a href="http://www.bdsc.site/websites/MTO/index.html">Code</a></td>
      <td>Focuses on multi-task optimization problems with 50-dimensional tasks.</td>
    </tr>
  </tbody>
</table>

</details>



## Available BBO/MetaBBO Baselines in MetaBox

## Citing MetaBox

The PDF version of the paper is available [here](https://arxiv.org/abs/2310.08252). If you find our MetaBox useful, please cite it in your publications or projects.

```latex
@inproceedings{metabox,
author={Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang},
title={MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning},
booktitle = {Advances in Neural Information Processing Systems},
year={2023},
volume = {36}
}
```

## 😁Contact Us
👨‍💻👩‍💻We are a research team mainly focus on Meta-Black-Box-Optimization (MetaBBO), which assists automated algorithm design for Evolutionary Computation. 

Here is our [homepage](https://metaevo.github.io/) and [github](https://github.com/MetaEvo). **🥰🥰🥰Please feel free to contact us—any suggestions are welcome!**

If you have any question or want to contact us：
- 🌱Fork, Add, and Merge
- ❓️Report an [issue](https://github.com/MetaEvo/MetaBox/issues)
- 📧Contact WenJie Qiu ([wukongqwj@gmail.com](mailto:wukongqwj@gmail.com))
- 🚨**We warmly invite you to join our QQ group for further communication (Group Number: 952185139).**



