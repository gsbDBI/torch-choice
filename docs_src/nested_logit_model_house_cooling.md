# Random Utility Model (RUM) Part II: Nested Logit Model
Author: Tianyu Du

The package implements the nested logit model as well, which allows researchers to model choices as a two-stage process: the user first picks a category of purchase and then picks the item from the chosen category that generates the most utility.

Examples here are modified from [Exercise 2: Nested logit model by Kenneth Train and Yves Croissant](https://cran.r-project.org/web/packages/mlogit/vignettes/e2nlogit.html).

The House Cooling (HC) dataset from `mlogit` contains data in R format on the choice of heating and central cooling system for 250 single-family, newly built houses in California.

The dataset is small and serve as a demonstration of the nested logit model.


The alternatives are:

- Gas central heat with cooling `gcc`,
- Electric central resistence heat with cooling `ecc`,
- Electric room resistence heat with cooling `erc`,
- Electric heat pump, which provides cooling also `hpc`,
- Gas central heat without cooling `gc`,
- Electric central resistence heat without cooling `ec`,
- Electric room resistence heat without cooling `er`.
- Heat pumps necessarily provide both heating and cooling such that heat pump without cooling is not an alternative.

The variables are:

- `depvar` gives the name of the chosen alternative,
- `ich.alt` are the installation cost for the heating portion of the system,
- `icca` is the installation cost for cooling
- `och.alt` are the operating cost for the heating portion of the system
- `occa` is the operating cost for cooling
- `income` is the annual income of the household

Note that the full installation cost of alternative gcc is ich.gcc+icca, and similarly for the operating cost and for the other alternatives with cooling.

## Nested Logit Model: Background
The following code block provides an example initialization of the `NestedLogitModel` (please refer to examples below for details).
```python
model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)
```

The nested logit model decompose the utility of choosing item $i$ into the (1) item-specific values and (2) category specify values.  For simplicity, suppose item $i$  belongs to category $k \in \{1, \dots, K\}$: $i \in B_k$.

$$
U_{uit} = W_{ukt} + Y_{uit}
$$

Where both $W$ and $Y$ are estimated using linear models from as in the conditional logit model.

The log-likelihood for user $u$ to choose item $i$ at time/session $t$ decomposes into the item-level likelihood and category-level likelihood.

$$
\log P(i \mid u, t) = \log P(i \mid u, t, B_k) + \log P(k \mid u, t) \\
= \log \left(\frac{\exp(Y_{uit}/\lambda_k)}{\sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)}\right) + \log \left( \frac{\exp(W_{ukt} + \lambda_k I_{ukt})}{\sum_{\ell=1}^K \exp(W_{u\ell t} + \lambda_\ell I_{u\ell t})}\right)
$$

The **inclusive value** of category $k$, $I_{ukt}$ is defined as $\log \sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)$, which is the *expected utility from choosing the best alternative from category $k$*.

The `category_to_item` keyword defines a dictionary of the mapping $k \mapsto B_k$, where keys of `category_to_item`  are integer $k$'s and  `category_to_item[k]`  is a list consisting of IDs of items in $B_k$.

The `{category, item}_coef_variation_dict` provides specification to $W_{ukt}$ and $Y_{uit}$ respectively, `torch_choice` allows for empty category level models by providing an empty dictionary (in this case, $W_{ukt} = \epsilon_{ukt}$) since the inclusive value term $\lambda_k I_{ukt}$ will be used to model the choice over categories. However, by specifying an empty second stage model ($Y_{uit} = \epsilon_{uit}$), the nested logit model reduces to a conditional logit model of choices over categories. Hence, one should never use the `NestedLogitModel` class with an empty item-level model.

Similar to the conditional logit model, `{category, item}_num_param_dict` specify the dimension (number of observables to be multiplied with the coefficient) of coefficients. The above code initializes a simple model built upon item-time-specific observables $X_{it} \in \mathbb{R}^7$,

$$
Y_{uit} = \beta^\top X_{it} + \epsilon_{uit} \\
W_{ukt} = \epsilon_{ukt}
$$

The research may wish to enfoce the *elasiticity* $\lambda_k$ to be constant across categories, setting `shared_lambda=True` enforces $\lambda_k = \lambda\ \forall k \in [K]$.

## Load Essential Packages
We firstly read essential packages for this tutorial.


```python
import argparse

import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.run_helper import run
```

We then select the appropriate device to run the model on, our package supports both CPU and GPU.


```python

if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    DEVICE = 'cuda'
else:
    print('Running tutorial on CPU')
    DEVICE = 'cpu' 
    
```

    CUDA device used: NVIDIA GeForce RTX 3090


## Load Datasets
We firstly read the dataset for this tutorial, the `csv` file can be found at `./public_datasets/HC.csv`.


```python
df = pd.read_csv('./public_datasets/HC.csv', index_col=0)
df = df.reset_index(drop=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>depvar</th>
      <th>icca</th>
      <th>occa</th>
      <th>income</th>
      <th>ich</th>
      <th>och</th>
      <th>idx.id1</th>
      <th>idx.id2</th>
      <th>inc.room</th>
      <th>inc.cooling</th>
      <th>int.cooling</th>
      <th>cooling.modes</th>
      <th>room.modes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>24.50</td>
      <td>4.09</td>
      <td>1</td>
      <td>ec</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>27.28</td>
      <td>2.95</td>
      <td>20</td>
      <td>7.86</td>
      <td>4.09</td>
      <td>1</td>
      <td>ecc</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>7.37</td>
      <td>3.85</td>
      <td>1</td>
      <td>er</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>27.28</td>
      <td>2.95</td>
      <td>20</td>
      <td>8.79</td>
      <td>3.85</td>
      <td>1</td>
      <td>erc</td>
      <td>20</td>
      <td>20</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>24.08</td>
      <td>2.26</td>
      <td>1</td>
      <td>gc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The raw dataset is in a long-format (i.e., each row contains information of one item).


```python
df['idx.id2'].value_counts()
```




    ec     250
    ecc    250
    er     250
    erc    250
    gc     250
    gcc    250
    hpc    250
    Name: idx.id2, dtype: int64




```python
# what was actually chosen.
item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
num_items = df['idx.id2'].nunique()
# cardinal encoder.
encoder = dict(zip(item_names, range(num_items)))
item_index = item_index.map(lambda x: encoder[x])
item_index = torch.LongTensor(item_index)
```

Because we will be training our model with `PyTorch`, we need to encode item names to integers (from 0 to 6).
We do this manually in this exercise given the small amount of items, for more items, one can use [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) to encode.

Raw item names will be encoded as the following.


```python
encoder
```




    {'ec': 0, 'ecc': 1, 'er': 2, 'erc': 3, 'gc': 4, 'gcc': 5, 'hpc': 6}



### Category Level Dataset
We firstly construct the category-level dataset, however, there is no observable that is constant within the same category, so we don't need to include any observable tensor to the `category_dataset`.

All we need to do is adding the `item_index` (i.e., which item is chosen) to the dataset, so that `category_dataset` knows the total number of choices made.


```python
# category feature: no category feature, all features are item-level.
category_dataset = ChoiceDataset(item_index=item_index.clone()).to(DEVICE)
```

    No `session_index` is provided, assume each choice instance is in its own session.


### Item Level Dataset
For simplicity, we treat each purchasing record as its own session. Moreover, we treat all observables as price observables (i.e., varying by both session and item).

Since there are 7 observables in total, the resulted `price_obs` has shape (250, 7, 7) corresponding to `number_of_sessions` by `number_of_items` by `number_of_observables`. 


```python
# item feature.
item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
price_obs.shape
```




    torch.Size([250, 7, 7])



Then, we construct the item level dataset by providing both `item_index` and `price_obs`.

We move `item_dataset` to the appropriate device as well. This is only necessary if we are using GPU to accelerate the model.


```python
item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs).to(DEVICE)
```

    No `session_index` is provided, assume each choice instance is in its own session.


Finally, we chain the category-level and item-level dataset into a single `JointDataset`.


```python
dataset = JointDataset(category=category_dataset, item=item_dataset)
```

One can print the joint dataset to see its contents, and tensors contained in each of these sub-datasets.


```python
print(dataset)
```

    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )


## Examples
There are multiple ways to group 7 items into categories, different classification will result in different utility functions and estimations (see the background of nested logit models).

We will demonstrate the usage of our package by presenting three different categorization schemes and corresponding model estimations.

### Example 1
In the first example, the model is specified to have the *cooling alternatives* `{gcc, ecc, erc, hpc}` in one category and the *non-cooling alternatives* `{gc, ec, er}` in another category.

We create a `category_to_item` dictionary to inform the model our categorization scheme. The dictionary should have keys ranging from `0` to `number_of_categories - 1`, each integer corresponds to a category. The value of each key is a list of item IDs in the category, the encoding of item names should be exactly the same as in the construction of `item_index`.


```python
category_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                    1: ['gc', 'ec', 'er']}

# encode items to integers.
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)
```

In this example, we have item `[1, 3, 5, 6]` in the first category (category `0`) and the rest of items in the second category (category `1`).


```python
print(category_to_item)
```

    {0: [1, 3, 5, 6], 1: [0, 2, 4]}


Next, let's create the `NestedLogitModel` class!

The first thing to put in is the `category_to_item` dictionary we just built.

For `category_coef_variation_dict`, `category_num_param_dict`, since we don't have any category-specific observables, we can simply put an empty dictionary there.

Coefficients for all observables are constant across items, and there are 7 observables in total.

As for `shared_lambda=True`, please refer to the background recap for nested logit model.


```python
model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)

model = model.to(DEVICE)
```

You can print the model to get summary information of the `NestedLogitModel` class.


```python
print(model)
```

    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )


**NOTE**: We are computing the standard errors using $\sqrt{\text{diag}(H^{-1})}$, where $H$ is the
hessian of negative log-likelihood with respect to model parameters. This leads to slight different
results compared with R implementation.


```python
run(model, dataset, num_epochs=10000)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 1000: Log-likelihood=-187.43597412109375
    Epoch 2000: Log-likelihood=-179.69964599609375
    Epoch 3000: Log-likelihood=-178.70831298828125
    Epoch 4000: Log-likelihood=-178.28799438476562
    Epoch 5000: Log-likelihood=-178.17779541015625
    Epoch 6000: Log-likelihood=-178.13650512695312
    Epoch 7000: Log-likelihood=-178.12576293945312
    Epoch 8000: Log-likelihood=-178.14144897460938
    Epoch 9000: Log-likelihood=-178.12478637695312
    Epoch 10000: Log-likelihood=-178.13674926757812
    ==================== model results ====================
    Training Epochs: 10000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -178.13674926757812
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     0.585981 |   0.167168  |
    | item_price_obs_0 |    -0.555577 |   0.145414  |
    | item_price_obs_1 |    -0.85812  |   0.238405  |
    | item_price_obs_2 |    -0.224599 |   0.111092  |
    | item_price_obs_3 |    -1.08912  |   1.04131   |
    | item_price_obs_4 |    -0.379067 |   0.101126  |
    | item_price_obs_5 |     0.250203 |   0.0522721 |
    | item_price_obs_6 |    -5.99917  |   4.85404   |





    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )



#### R Output
Here we provide the output from `mlogit` model in `R` for estimation reference.

Coefficient names reported are slightly different in `Python` and `R`, please use the following table for comparison. Please note that the `lambda_weight_0` in `Python` (at the top) corresponds to the `iv` (inclusive value) in `R` (at the bottom). Orderings of coefficients for observables should be the same in both languages.

| Coefficient (Python) |   Coefficient (R) |
|:-----------------|:-------------:|
| lambda_weight_0  |     iv |
| item_price_obs_0 |    ich |
| item_price_obs_1 |    och  |
| item_price_obs_2 |    icca |
| item_price_obs_3 |    occa  |
| item_price_obs_4 |    inc.room |
| item_price_obs_5 |    inc.cooling  |
| item_price_obs_6 |    int.cooling  |

```
## 
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room + 
##     inc.cooling + int.cooling | 0, data = HC, nests = list(cooling = c("gcc", 
##     "ecc", "erc", "hpc"), other = c("gc", "ec", "er")), un.nest.el = TRUE)
## 
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc 
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104 
## 
## bfgs method
## 11 iterations, 0h:0m:0s 
## g'(-H)^-1g = 7.26E-06 
## successive function values within tolerance limits 
## 
## Coefficients :
##              Estimate Std. Error z-value  Pr(>|z|)    
## ich         -0.554878   0.144205 -3.8478 0.0001192 ***
## och         -0.857886   0.255313 -3.3601 0.0007791 ***
## icca        -0.225079   0.144423 -1.5585 0.1191212    
## occa        -1.089458   1.219821 -0.8931 0.3717882    
## inc.room    -0.378971   0.099631 -3.8038 0.0001425 ***
## inc.cooling  0.249575   0.059213  4.2149 2.499e-05 ***
## int.cooling -6.000415   5.562423 -1.0787 0.2807030    
## iv           0.585922   0.179708  3.2604 0.0011125 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Log-Likelihood: -178.12
```

### Example 2
The second example is similar to the first one, but we change the way we group items into different categories.
Re-estimate the model with the room alternatives in one nest and the central alternatives in another nest. (Note that a heat pump is a central system.)


```python
category_to_item = {0: ['ec', 'ecc', 'gc', 'gcc', 'hpc'],
                    1: ['er', 'erc']}
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)

model = NestedLogitModel(category_to_item=category_to_item,
                            category_coef_variation_dict={},
                            category_num_param_dict={},
                            item_coef_variation_dict={'price_obs': 'constant'},
                            item_num_param_dict={'price_obs': 7},
                            shared_lambda=True
                            )

model = model.to(DEVICE)
```


```python
run(model, dataset, num_epochs=5000, learning_rate=0.3)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 500: Log-likelihood=-193.73406982421875
    Epoch 1000: Log-likelihood=-185.25933837890625
    Epoch 1500: Log-likelihood=-183.55142211914062
    Epoch 2000: Log-likelihood=-181.8164825439453
    Epoch 2500: Log-likelihood=-180.4320526123047
    Epoch 3000: Log-likelihood=-180.04095458984375
    Epoch 3500: Log-likelihood=-180.7447509765625
    Epoch 4000: Log-likelihood=-180.39688110351562
    Epoch 4500: Log-likelihood=-180.27947998046875
    Epoch 5000: Log-likelihood=-181.1483612060547
    ==================== model results ====================
    Training Epochs: 5000
    
    Learning Rate: 0.3
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -181.1483612060547
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     1.61072  |    0.787735 |
    | item_price_obs_0 |    -1.34719  |    0.631206 |
    | item_price_obs_1 |    -2.16109  |    1.0451   |
    | item_price_obs_2 |    -0.393868 |    0.255138 |
    | item_price_obs_3 |    -2.53253  |    2.2719   |
    | item_price_obs_4 |    -0.884873 |    0.379626 |
    | item_price_obs_5 |     0.496491 |    0.248118 |
    | item_price_obs_6 |   -15.6477   |    9.88054  |





    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )



#### R Output

You can use the table for converting coefficient names reported by `Python` and `R`:

| Coefficient (Python) |   Coefficient (R) |
|:-----------------|:-------------:|
| lambda_weight_0  |     iv |
| item_price_obs_0 |    ich |
| item_price_obs_1 |    och  |
| item_price_obs_2 |    icca |
| item_price_obs_3 |    occa  |
| item_price_obs_4 |    inc.room |
| item_price_obs_5 |    inc.cooling  |
| item_price_obs_6 |    int.cooling  |

```
## 
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room + 
##     inc.cooling + int.cooling | 0, data = HC, nests = list(central = c("ec", 
##     "ecc", "gc", "gcc", "hpc"), room = c("er", "erc")), un.nest.el = TRUE)
## 
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc 
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104 
## 
## bfgs method
## 10 iterations, 0h:0m:0s 
## g'(-H)^-1g = 5.87E-07 
## gradient close to zero 
## 
## Coefficients :
##              Estimate Std. Error z-value Pr(>|z|)  
## ich          -1.13818    0.54216 -2.0993  0.03579 *
## och          -1.82532    0.93228 -1.9579  0.05024 .
## icca         -0.33746    0.26934 -1.2529  0.21024  
## occa         -2.06328    1.89726 -1.0875  0.27681  
## inc.room     -0.75722    0.34292 -2.2081  0.02723 *
## inc.cooling   0.41689    0.20742  2.0099  0.04444 *
## int.cooling -13.82487    7.94031 -1.7411  0.08167 .
## iv            1.36201    0.65393  2.0828  0.03727 *
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Log-Likelihood: -180.02

```

### Example 3
For the third example, we now group items into three categories. Specifically, we have items `gcc`, `ecc` and `erc` in the first category (category `0` in the `category_to_item` dictionary), `hpc` in a category (category `1`) alone, and items `gc`, `ec` and `er` in the last category (category `2`).


```python
category_to_item = {0: ['gcc', 'ecc', 'erc'],
                    1: ['hpc'],
                    2: ['gc', 'ec', 'er']}
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)

model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True
                         )

model = model.to(DEVICE)
```


```python
run(model, dataset)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 500: Log-likelihood=-187.12100219726562
    Epoch 1000: Log-likelihood=-182.98468017578125
    Epoch 1500: Log-likelihood=-181.72171020507812
    Epoch 2000: Log-likelihood=-181.3906707763672
    Epoch 2500: Log-likelihood=-181.2037353515625
    Epoch 3000: Log-likelihood=-181.0186767578125
    Epoch 3500: Log-likelihood=-180.83331298828125
    Epoch 4000: Log-likelihood=-180.6610107421875
    Epoch 4500: Log-likelihood=-180.51480102539062
    Epoch 5000: Log-likelihood=-180.40383911132812
    ==================== model results ====================
    Training Epochs: 5000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -180.40383911132812
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     0.939528 |   0.193704  |
    | item_price_obs_0 |    -0.823672 |   0.0973065 |
    | item_price_obs_1 |    -1.31387  |   0.182701  |
    | item_price_obs_2 |    -0.305365 |   0.12726   |
    | item_price_obs_3 |    -1.89104  |   1.14781   |
    | item_price_obs_4 |    -0.559503 |   0.0734163 |
    | item_price_obs_5 |     0.310081 |   0.0551569 |
    | item_price_obs_6 |    -7.68508  |   5.09592   |





    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )



#### R Output

You can use the table for converting coefficient names reported by `Python` and `R`:

| Coefficient (Python) |   Coefficient (R) |
|:-----------------|:-------------:|
| lambda_weight_0  |     iv |
| item_price_obs_0 |    ich |
| item_price_obs_1 |    och  |
| item_price_obs_2 |    icca |
| item_price_obs_3 |    occa  |
| item_price_obs_4 |    inc.room |
| item_price_obs_5 |    inc.cooling  |
| item_price_obs_6 |    int.cooling  |

```
## 
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room + 
##     inc.cooling + int.cooling | 0, data = HC, nests = list(n1 = c("gcc", 
##     "ecc", "erc"), n2 = c("hpc"), n3 = c("gc", "ec", "er")), 
##     un.nest.el = TRUE)
## 
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc 
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104 
## 
## bfgs method
## 8 iterations, 0h:0m:0s 
## g'(-H)^-1g = 3.71E-08 
## gradient close to zero 
## 
## Coefficients :
##               Estimate Std. Error z-value  Pr(>|z|)    
## ich          -0.838394   0.100546 -8.3384 < 2.2e-16 ***
## och          -1.331598   0.252069 -5.2827 1.273e-07 ***
## icca         -0.256131   0.145564 -1.7596   0.07848 .  
## occa         -1.405656   1.207281 -1.1643   0.24430    
## inc.room     -0.571352   0.077950 -7.3297 2.307e-13 ***
## inc.cooling   0.311355   0.056357  5.5247 3.301e-08 ***
## int.cooling -10.413384   5.612445 -1.8554   0.06354 .  
## iv            0.956544   0.180722  5.2929 1.204e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Log-Likelihood: -180.26
```
