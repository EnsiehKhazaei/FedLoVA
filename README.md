# FedLoVA

PyTorch implementation of the paper **"FedLoVA: Value-Only Aggregation for Federated LoRA Fine-Tuning of Large Language Models"**, accepted at NeurIPS 2026.

📄 Paper: [link will be added shortly]()

## Abstract

Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning (PEFT) method for fine-tuning of large language models (LLMs), especially in federated learning, due to its strong performance and communication efficiency. In practice, LoRA is applied to the query and value projections of transformer layers without considering the distinct roles of these components. In this work, we analyze the impact of these projections in federated fine-tuning of LLMs using LoRA and observe that the value projection converges significantly faster than the query projection. Motivated by this finding, we propose \scheme, a federated fine-tuning framework that trains LoRA adapters on both query and value projections locally while sharing only the value projection with the server. This approach reduces communication overhead while improving performance and convergence efficiency. Experiments on natural language understanding, multilingual sentiment analysis, and generation tasks show that FedLoVA reduces the communication cost by up to $7.8\times$ compared to existing federated LLM fine-tuning baselines while achieving comparable performance.

## Overview

In each communication round, every client trains LoRA adapters on both the query (Q) and value (V) projections. Only the value-projection adapters are sent to the server and aggregated; the query-projection adapters remain local and are never transmitted.

## Installation

This repository is built on the [FedSA-LoRA repository](https://github.com/Pengxin-Guo/FedSA-LoRA), which is based on [FederatedScope](https://github.com/alibaba/FederatedScope). Our code uses Python 3.10 and PyTorch 2.1.0. Install all dependencies with:

```bash
conda create -n fedlova python=3.10
conda activate fedlova
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .[llm]
```

## Run Code

### GLUE (natural language understanding)

```bash
python federatedscope/main.py --cfg federatedscope/glue/yamls/fedlova.yaml
```

### Multilingual sentiment analysis

For federated fine-tuning in which clients hold data in different languages.

```bash
python federatedscope/main.py --cfg federatedscope/mls/yamls/fedlova.yaml
```

### Generation (GSM8K)

```bash
python federatedscope/main.py --cfg federatedscope/llm/yamls/fedlova.yaml
```

### Baselines

Baseline configurations (FedLoRA, FFA-LoRA, FedSA-LoRA, LoRA-FAIR, FedLoQA) are provided in the same `yamls` folders. Replace `fedlova.yaml` with the corresponding file, e.g.:

```bash
python federatedscope/main.py --cfg federatedscope/glue/yamls/fedsa-lora.yaml
```

Key settings such as the dataset, number of clients, Dirichlet parameter α, and LoRA rank can be changed in the YAML files.

## Output

Experimental results, including the test accuracy at each communication round, are saved in the `exp` folder.

## Acknowledgements

This code is built on [FedSA-LoRA](https://github.com/Pengxin-Guo/FedSA-LoRA) and [FederatedScope](https://github.com/alibaba/FederatedScope). We thank the authors for making their code publicly available. If you use this repository, please also consider citing their work.

## Citation

Please cite our paper if you find this repository helpful.

```bibtex
@inproceedings{khazaei2026fedlova,
  title     = {FedLoVA: Value-Only Aggregation for Federated LoRA Fine-Tuning of Large Language Models},
  author    = {Khazaei, Ensieh and Buyukates, Baturalp and Hatzinakos, Dimitrios},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
