# 📚 MoM: Linear Sequence Modeling with Mixture-of-Memories

Welcome to MoM! This repository provides an efficient approach for linear sequence modeling using a Mixture-of-Memories mechanism.

<div align="center">
  <img height="150" alt="image" src="assets/MoM.jpg">
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
This project builds upon the work of [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). Huge thanks to the original authors for their contributions!🎉

Happy experimenting! 🚀🔥