# torch-choice

> Authors: Tianyu Du, Ayush Kanodia and Susan Athey; Contact: tianyudu@stanford.edu

> Acknowledgements: We would like to thank Erik Sverdrup, Charles Pebereau and Keshav Agrawal for their feedback.

`torch-choice` is a library for flexible, fast choice modeling with PyTorch: it has logit and nested logit models, designed for both estimation and prediction. See the [complete documentation](https://gsbdbi.github.io/torch-choice/) for more details.
Unique features:
1. GPU support via torch for speed
2. Specify customized models
3. Specify availability sets
4. Maximum Likelihood Estimation (MLE) (optionally, reporting standard errors or MAP inference with Bayesian Priors on coefficients)
5. Estimation via minimization of Cross Entropy Loss (optionally with L1/L2 regularization)

# Choice Models Supported

This package allows learning of flexible choice models for multinomial choice.  Denote item observables by $X$ and user observables by $Y$. Define utility $U_{uis}$ for user $u$ from choosing item $i$ in session $s$ as,

$$
U_{uis} = \alpha + \beta^\top X + \gamma^\top Y + \dots + \epsilon_{uis}
$$

The probability that user $u$ chooses item $i$ in session $s$ is given by the logistic function (if we assume iid extreme value type 1 errors for $\epsilon_{uis}$, as shown by [McFadden](https://en.wikipedia.org/wiki/Choice_modelling), and as in [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)), is

$$
P_{uis} = \frac{e^{U_{uis}}}{\Sigma_{j \in A_{us}}e^{U_{ujs}}}
$$

where $A_{us}$ is the set of items available for user $u$ in session $s$.

We implement a fully flexible setup, where we allow 
1. coefficients ($\alpha$, $\beta$, $\gamma$, $\dots$) to be constant, user-specific (i.e., $\alpha=\alpha_u$), item-specific (i.e., $\alpha=\alpha_i$), session-specific (i.e., $\alpha=\alpha_t$), or (session, item)-specific (i.e., $\alpha=\alpha_{ti}$). For example, specifying $\alpha$ to be item-specific is equivalent to adding an item-level fixed effect.
2. Observables ($X$, $Y$, $\dots$) to be constant, user-specific, item-specific, session-specific, or (session, item)-specific as well.
3. Specifying availability sets $A_{us}$

This flexibility in coefficients and features allows for more than 20 types of additive terms to $U_{uis}$, which enables modelling rich structures.

Utility Form Examples
1. Mode Canada [(Detailed Tutorial)](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb)

$$
U_{uit} = \beta^0_i + \beta^{1\top} X^{price: (cost, freq, ovt)}_{it} + \beta^2_i X^{session:income}_t + \beta^3_i X_{it}^{price:ivt} + \epsilon_{uit}
$$

This is also described as a conditional logit model in Econometrics.


2. MNIST classification [(Upcoming Detailed Tutorial)]()

$$
U_{it} = \beta_i X^{session:pixelvalues} + \epsilon_{it}
$$

This is a classic problem used for exposition in Computer Science to motivate various Machine Learning models. There is no concept of a user in this setup. Our package allows for models of this nature and is fully usable for Machine Learning problems with added flexibility over [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

We highly recommend users to go through [tutorials](https://github.com/gsbDBI/torch-choice/blob/main/tutorials) we prepared to get a better understanding of what the package is offering. We present multiple examples, and for each case we specify the utility form.

## Installation
1. Clone the repository to your local machine or server.
2. Install required dependencies using: `pip3 install -r requirements.txt`.
3. Run `pip3 install torch-choice`.
4. Check installation by running `python3 -c 'import torch_choice; print(torch_choice.__version__)'`.

[The installation page](https://gsbdbi.github.io/torch-choice/install/) provides more details on installation.

In this demonstration, we will guide you through a minimal example of fitting a conditional logit model using our package. We will be referencing to R code and Stata code as well to deliver a smooth knowledge transfer.

## An Example on Transportation Choice Dataset
In this demonstration, we will guide you through a minimal example of fitting a conditional logit model using our package. We will be referencing R code as well to deliver a smooth knowledge transfer.

We are modelling people's choices on transportation modes using the publicly available `ModeCanada` dataset.
More information about the [ModeCanada: Mode Choice for the Montreal-Toronto Corridor](https://www.rdocumentation.org/packages/mlogit/versions/1.1-1/topics/ModeCanada).

In this example, we are estimating the utility for user $u$ to choose transport method $i$ in session $s$ as

$$
U_{uis} = \alpha_i + \beta_i \text{income}_s + \gamma \text{cost} + \delta \text{freq} + \eta \text{ovt} + \iota_i \text{ivt} + \varepsilon
$$

### Modelling Transportation Choice with Torch-Choice


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

### Modelling Transportation Choice with R

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



```python

```
