<h1 align="center">
  <span style="font-size: 2.2em">⭐ MetaBox-v2 ⭐</span><br/>
  <div style="font-size: 0.9em; margin-top: 12px">
    Benchmarking Meta-Black-Box Optimization under Diverse<br/>
    Optimization Scenarios with Efficiency and Flexibility
  </div>
</h1>

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-b31b1b.svg)]([https://proceedings.neurips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html]) **MetaBox-v1 has been published at NeurIPS 2023！**

we propose MetaBox 2.0 version (MetaBox-v2) as a major upgradation of [MetaBox-v1](https://github.com/MetaEvo/MetaBox/tree/v1.0.0). MetaBox-v2 now supports plentiful optimization scenarios to embrace users from single-objective optimization, multi-objective optimization, multi-modal optimization, multi-task optimization and etc. Correspondingly, **11 optimization problem sets** (synthetic + realistic) and **36 baseline methods** (traditional optimizers + up-to-date MetaBBOs) are reproduced within MetaBox-v2 to assist various research ideas and comprehensive comparison. To address MetaBBO's inherent efficiency issue, we have optimized low-level implementation of MetaBox-v2 to support parallel meta-training and evaluation, which reduces the running cost from days to hours. More importantly, we have optimized MetaBox-v2's sourcecode to support **sufficient development flexbility**, with clear and sound tutotials correspondingly. Enjoy your journey of learning and using MetaBBO from here!   

<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-NEW.png" width="80%">
</div>

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
We sincerely suggest researchers with interests to check out **[Online Documentation](https://metaboxdoc.readthedocs.io/en/latest/index.html)** for further flexible usege of MetaBox-v2.

## Available Optimization Problem Set in MetaBox
<details>
<summary>Single-Objective Optimization</summary>

- [COCO-BB08](#coco-bb08)
- [BB08-Surrogate](#bb08-surrogate)
</details>

## Available BBO/MetaBBO Baselines in MetaBox

## Researches that used MetaBox



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



