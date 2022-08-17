# torch-choice

> Authors: Tianyu Du and Ayush Kanodia; PI: Susan Athey; Contact: tianyudu@stanford.edu

`torch-choice` is a flexible, fast choice modeling with PyTorch: logit and nested logit models, designed for both estimation and prediction. See the [complete documentation](https://gsbdbi.github.io/torch-choice/) for more details.
Unique features:
1. GPU support via torch for speed
2. Specify customized models
3. Specify availability sets
4. Report standard errors

## Installation
1. Clone the repository to your local machine or server.
2. Install required dependencies using: `pip3 install -r requirements.txt`.
3. Run `pip3 install torch-choice`.
4. Check installation by running `python3 -c 'import torch_choice; print(torch_choice.__version__)'`.

[The installation page](https://gsbdbi.github.io/torch-choice/install/) provides more details on installation.

In this demonstration, we will guide you through a minimal example of fitting a conditional logit model using our package. We will be referencing to R code and Stata code as well to deliver a smooth knowledge transfer.

## Mode Canada Example
In this demonstration, we will guide you through a minimal example of fitting a conditional logit model using our package. We will be referencing R code as well to deliver a smooth knowledge transfer.

More information about the [ModeCanada: Mode Choice for the Montreal-Toronto Corridor](https://www.rdocumentation.org/packages/mlogit/versions/1.1-1/topics/ModeCanada).

In this example, we are estimating the utility for user $u$ to choose transport method $i$ in session $s$ as

$$
U_{uis} = \alpha_i + \beta_i \text{income}_s + \gamma \text{cost} + \delta \text{freq} + \eta \text{ovt} + \iota_i \text{ivt} + \varepsilon
$$

###  Mode Canada with Torch-Choice


```python
# load packages.
import pandas as pd
import torch_choice

# load data.
df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/ModeCanada.csv?token=GHSAT0AAAAAABRGHCCSNNQARRMU63W7P7F4YWYP5HA').query('noalt == 4').reset_index(drop=True)

# format data.
data = torch_choice.utils.easy_data_wrapper.EasyDatasetWrapper(
    main_data=df,
    purchase_record_column='case',
    choice_column='choice',
    item_name_column='alt',
    user_index_column='case',
    session_index_column='case',
    session_observable_columns=['income'],
    price_observable_columns=['cost', 'freq', 'ovt', 'ivt'])

# define the conditional logit model.
model = torch_choice.model.ConditionalLogitModel(
    coef_variation_dict={'price_cost': 'constant',
                         'price_freq': 'constant',
                         'price_ovt': 'constant',
                         'session_income': 'item',
                         'price_ivt': 'item-full',
                         'intercept': 'item'},
    num_items=4)
# fit the conditional logit model.
torch_choice.utils.run_helper.run(model, data.choice_dataset, num_epochs=5000, learning_rate=0.01, batch_size=-1)
```

## Mode Canada with R

We include the R code for the ModeCanada example as well.
```{r}
# load packages.
library("mlogit")

# load data.
ModeCanada <- read.csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/ModeCanada.csv?token=GHSAT0AAAAAABRGHCCSNNQARRMU63W7P7F4YWYP5HA')
ModeCanada <- select(ModeCanada, -X)
ModeCanada$alt <- as.factor(ModeCanada$alt)

# format data.
MC <- dfidx(ModeCanada, subset = noalt == 4)

# fit the data.
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')
summary(ml.MC1)
```

# What's in the package?
Overall, the `torch-choice` package offers the following features:

1. The package includes a data management module called `ChoiceDataset`, which is built upon PyTorch's dataset module. Our dataset implementation allows users to easily move data between CPU and GPU. Unlike traditional long or wide formats, the `ChoiceDataset` offers a memory-efficient way to manage observables.

2. The package provides a (1) conditional logit model and (2) a nested logit model for consumer choice modeling.

3. The package leverage GPU acceleration using PyTorch and easily scale to large dataset of millions of choice records. All models are trained using state-of-the-art optimizers by in PyTorch. These optimization algorithms are tested to be scalable by modern machine learning practitioners. However, you can rest assure that the package runs flawlessly when no GPU is used as well.

4. Setting up the PyTorch training pipelines can be frustrating. We provide easy-to-use [PyTorch lightning](https://www.pytorchlightning.ai) wrapper of models to free researchers from the hassle from setting up PyTorch optimizers and training loops.


# More Details on Models

Models in the package fit individual choices by capturing the utility $U$ of user $u$ from choosing item $i$ in each occasion (called sessions) $s$.

The utility admits functional form consisting of product of coefficients (often called learnable parameters in computer science literature) and observables (also called features in CS literature).

$$
U_{ui} = \alpha + \beta^\top X + \gamma^\top Y + \dots
$$

We allow coefficients ($\alpha$, $\beta$ and $\gamma$) to be constant, user-specific (i.e., $\alpha=\alpha_u$), item-specific (i.e., $\alpha=\alpha_i$), session-specific (i.e., $\alpha=\alpha_t$), or (session, item)-specific (i.e., $\alpha=\alpha_{ti}$). For example, specifying $\alpha$ to be item-specific is equivalent to adding an item-level fixed effect.

Observables ($X$ and $Y$) can be constant, user-specific, item-specific, session-specific, or (session, item)-specific as well.

The flexibility in coefficients and features allows for more than 20 types of additive term to $U_{ui}$, which enables modelling rich structures.

We highly recommend users to go through tutorials we prepared to get a better understanding of what the package is offering.
