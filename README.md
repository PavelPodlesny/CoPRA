# 🥥 CoPRA

<p align="left">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white" />
  </a>
  <a href="https://lightning.ai/">
    <img src="https://img.shields.io/badge/Lightning-792EE5?style=flat&logo=PyTorch-Lightning&logoColor=white" />
  </a>
  <a href="https://arxiv.org/abs/2409.03773">
    <img src="https://img.shields.io/badge/AAAI 2025-arxiv.2409.03773-red?style=flat" />
  </a>
  <a href="https://mamba.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Mamba-2CCEEE?style=flat" />
  </a>
  <a href="https://huggingface.co/">
    <img src="https://img.shields.io/badge/Huggingface-FFD21E?style=flat" />
  </a>
</p>
This is the official implementation of CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction (AAAI 2025)

<img src="./assets/model_overview.jpg" alt="Overview of CoPRA" width="800">



CoPRA is a state-of-the-art predictor of protein-RNA binding affinity. The framework of CoPRA is based on a protein language model and an RNA-language model, with complex structure as input. The model was originally pre-trained on the PRI30k dataset via a bi-scope stratege and fine-tuned on PRA310, and could also be redirected to predict mutation effects on the mCSM_RNA dataset — see [our paper](https://arxiv.org/abs/2409.03773) for those results. **This repository currently implements only the PRA310/PRA201 dG-regression finetuning/inference path** (the `pretune`/`ddG` training code and configs have been removed from `run.py`); the five-fold checkpoints below were produced by that original pretrain+finetune pipeline.

Please do not hesitate to contact us or create an issue/PR if you have any questions or suggestions!

## 🛠️ Installation

**Step 1**. Clone this repository and setup the environment. We recommend you to install the dependencies via the fast package management tool [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) (you can also replace the command 'mamba' with 'conda' to install them). Generally, CoPRA works with Python 3.10.14 and PyTorch version 2.1.2.
```
git@github.com:hanrthu/CoPRA.git
cd CoPRA
mamba env create -f environment.yml
```

**Step 2**. Install flash-attn and rinalmo with the following command, you may also need to download Rinalmo-650M model and place it at `./weights` folder of this repo.
```
# Download flash-attn-2.6.3 wheel file at https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
git clone git@github.com:lbcb-sci/RiNALMo.git
cd RiNALMo
pip install -e .
```

**Step 3 (mandatory)**. CoPRA computes an interface-energy feature via a frozen InNA model at data-preparation time. Clone the InNA repository as a sibling of `CoPRA/` (i.e. `../InNA` relative to this repo) and place its checkpoint under `../InNA/model/`. The dataset configs (`config/datasets/PRA310.yml`, `PRA201.yml`) point at it via `inna_repo_path`/`inna_weights` — update those paths if you place InNA elsewhere. `naskit`, InNA's structure-parsing dependency, is already installed via `environment.yml`.

## 📖 Datasets and model weights for Protein-RNA binding affinity prediction
Here, we first provide our proposed datasets, including PRA310, PRA201 and PRI30k together with an mCSM_RNA dataset, you can easily access them through 🤗Huggingface: [/Jesse7/CoPRA_data](https://huggingface.co/datasets/Jesse7/CoPRA_data/tree/main). The only difference between PRA201 and PRA310 are the selected samples, thus the PRA201 labels and splits are in PRA310/splits/PRA201.csv. Download these datasets and place them at `./datasets` folder.

The number of samples of the original dataset is shown below, we take PRA as the abbreviation of Protein-RNA binding affinity:

| Dataset | Type | Size |
| :---: | :---: | :---: |
| PRA310 | PRA | 310 |
| PRA201 | PRA (pair-only) | 201 |
| PRI30k | Unsupervised complexes | 30006 |
| mCSM-RNA | Mutation effect on PRA | 79 |


We also provide a five-fold model checkpoints after pretraining Co-Former with PRI30k and finetune it with PRA310, and they can also be downloaded through 🤗Huggingface: [/Jesse7/CoPRA](https://huggingface.co/Jesse7/CoPRA). This repository also contains a pretrained RiNALMo-650M weights. Download these weights at place them at `./weights` folder.

The performance of 5-fold cross validation on PRA310 reaches state-of-the-art, and here is the comparison:

<img src="./assets/results_on_PRA.png" alt="Results on PRA" width="800">



## 🚀 Training on the protein-RNA datasets

**Note1:** Before finetuning or testing on a dataset, run the offline precache step once — it parses every structure's PDB and precomputes backbone geometry plus the InNA interface-energy map, writing one file per structure to `prepared_dir`. This is a required step, not an optimization; training/testing will fail if it hasn't been run.
```
python run.py precache --data_config ./config/datasets/PRA310.yml
```

**Note2:** We also support LoRA tuning and all-param tuning. For LoRA tuning, just specify `lora_tune: true` in `./config/models/copra.yml`. For all-param tuning, just specify `fix_lms: false` in `./config/models/copra.yml`.

### Run 5-fold inference on PRA310
```
python run.py test --model_config ./config/models/copra.yml --data_config ./config/datasets/PRA310.yml --run_config ./config/runs/test_basic.yml
```

### Run finetune on PRA310
```
python run.py finetune --model_config ./config/models/copra.yml --data_config ./config/datasets/PRA310.yml --run_config ./config/runs/finetune_struct.yml
```

### Run finetune on PRA201
```
python run.py finetune --model_config ./config/models/copra.yml --data_config ./config/datasets/PRA201.yml --run_config ./config/runs/finetune_struct.yml
```

## 🖌️ Citation
If you find our repo useful, please kindly consider citing:
```
@article{han2024copra,
  title={CoPRA: Bridging Cross-domain Pretrained Sequence Models with Complex Structures for Protein-RNA Binding Affinity Prediction},
  author={Han, Rong and Liu, Xiaohong and Pan, Tong and Xu, Jing and Wang, Xiaoyu and Lan, Wuyang and Li, Zhenyu and Wang, Zixuan and Song, Jiangning and Wang, Guangyu and others},
  journal={arXiv preprint arXiv:2409.03773},
  year={2024}
}
```