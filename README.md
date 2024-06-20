
<p align="center">
<img src="figures/icon.png" width="150"/>
</h1>


<h1 align="center">Stealth edits for provably fixing or attacking large language models</h1>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qinghua-zhou/stealth-edits/blob/main/demos/colab_demo.ipynb)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/qinghua-zhou/stealth-edits)

Implementation and source code of algorithms from paper: ***"Stealth edits for provably fixing or attacking large language models"***. 


### Getting Started

1. Before attempting stealth edits, please first install the environment:

    ```bash
    conda env create --name=llm-sa -f environment.yml
    conda activate llm-sa
    ```

2. The model `llama-3-8b` requires you to apply for access. Please follow the instructions [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B). You will also need to install `huggingface-cli` and input an [user access token](https://huggingface.co/docs/huggingface_hub/en/guides/cli).


3. To start playing with stealth edit and attacks, please refer to the [Colab Demo](https://colab.research.google.com/github/qinghua-zhou/stealth-edits/blob/main/demos/colab_demo.ipynb) and the [Huggingface Demo](https://huggingface.co/spaces/qinghua-zhou/stealth-edits).

### Experiments

To reproduce experiments in the paper, please first run the extraction script:

  ```bash
  bash scripts/extract.sh
  ```

and then run edits and/or attacks and evaluation with the following scripts:

  ```bash
  bash scripts/edit.sh
  bash scripts/eval.sh
  ```

It is recommended to distribute the experiments on multiple nodes.

### How to Cite

```bibtex
@article{sutton2024stealth,
  title = {Stealth Edits for Provably Fixing or Attacking Large Language Models},
  author = {Sutton, Oliver J. and Zhou, Qinghua and Wang, Wei and Higham, Desmond J. and Gorban, Alexander N. and Bastounis, Alexander and Tyukin, Ivan Y.},
  year = {2024},
  month = jun,
  number = {arXiv:2406.12670},
  eprint = {2406.12670},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2406.12670},
  urldate = {2024-06-20},
  archiveprefix = {arXiv},
}
```