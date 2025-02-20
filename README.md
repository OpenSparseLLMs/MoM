<div align="center">

# Mixture-of-Memories

</div>

Welcome to MoM! This repository provides the implementation of [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685). MoM is compatible with all kinds of linear sequence modeling methods like: linear attention, SSM, linear RNNs, etc. You can find a introduction artical about MoM on [Zhihu](https://zhuanlan.zhihu.com/p/25066090353).

<p align="center">
  <img src="assets/mom_fig1.png" width="70%" />
</p>
<div align="center">
Figure 1: MoM Architecture
</div>

## 🛠 Installation

First, create a new virtual environment and install the required dependencies:
```bash
conda create --name mom python=3.10
conda activate mom
pip install -r requirements.txt
```

## 🚀 Getting Started

### 📂 Data Preparasion
Before training, make sure to preprocess your data by following the steps outlined in [training/README.md](training/README.md).

### 🎯 Train MoM

#### Train with default configuration:
To start training with the default setup, simply run:
```bash
cd trainning
bash cmd_mom.sh
```

#### ⚙️ customization
- Modify the script to adjust the training configuration.
- Modify the [training/configs/mom.json](training/configs/mom.json) to adjust the MoM structure.

### 📊 Evaluation

#### Commonsense Reasoning Tasks

Evaluate MoM on commonsense reasoning benchmarks using the provided script:
```bash
bash eval.sh
```

#### Recall-intensive Tasks

For recall-intensive tasks, please follow the instructions in [Prefix Linear Attention](https://github.com/HazyResearch/prefix-linear-attention)

## 🙌 Acknowledgements
This project builds upon the work of [FLA](https://github.com/fla-org/flash-linear-attention).

Happy experimenting! 🚀🔥

# Citation
If you find this repo useful, please consider citing our work:
```bib
@article{du2025mom,
  title={MoM: Linear Sequence Modeling with Mixture-of-Memories},
  author={Du, Jusen and Sun, Weigao and Lan, Disen and Hu, Jiaxi and Cheng, Yu},
  journal={arXiv preprint arXiv:2502.13685},
  year={2025}
}
```