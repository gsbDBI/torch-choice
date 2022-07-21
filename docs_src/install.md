# Installation
This page will guide you through the installation procedure of `torch-choice` and `bemb`.

There are two parts of this project: the `torch_choice` library consisting of data management modules, logit and nested-logit models for consumer choice modelling.

For researchers wish to use the Bayesian Embedding (BEMB) model, they need to install an additional `bemb` package, which was built on the top of `torch_choice`.

**Note** Since this project is still on its pre-release stage and subject to changes, we have not uploaded our packages to PIP or CONDA. Researchers need to install these packages from Github source code.

## Option 1: Install using Source Code from Github Repository
To install `torch_choice` and `bemb` from source,
1. Clone the repositories of both `torch_choice` and `bemb` to your local machine.
2. Install required dependencies (e.g., PyTorch and PyTorch-Lightning).
3. For each of repositories, run `python3 ./setup.py develop` to add the package to your Python environment.
4. Check installation by running `python3 -c "import torch_choice; print(torch_choice.__version__)"`.
5. Check installation by running `python3 -c "import bemb; print(bemb.__version__)"`.

## Option 2: Install using Pip
The `torch-choice` is available on PIP now [here](https://pypi.org/project/torch-choice/)! You can use `pip install torch-choice` to install it.
**Note**: We are working on publishing BEMB to PIP.