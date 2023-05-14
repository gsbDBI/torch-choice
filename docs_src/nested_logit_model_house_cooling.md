# Random Utility Model (RUM) Part II: Nested Logit Model
Author: Tianyu Du

The package implements the nested logit model as well, which allows researchers to model choices as a two-stage process: the user first picks a nest of purchase and then picks the item from the chosen nest that generates the most utility.

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
model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_coef_variation_dict={},
                         nest_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)
```

The nested logit model decompose the utility of choosing item $i$ into the (1) item-specific values and (2) nest specify values.  For simplicity, suppose item $i$  belongs to nest $k \in \{1, \dots, K\}$: $i \in B_k$.

$$
U_{uit} = W_{ukt} + Y_{uit}
$$

Where both $W$ and $Y$ are estimated using linear models from as in the conditional logit model.

The log-likelihood for user $u$ to choose item $i$ at time/session $t$ decomposes into the item-level likelihood and nest-level likelihood.

$$
\log P(i \mid u, t) = \log P(i \mid u, t, B_k) + \log P(k \mid u, t) \\
= \log \left(\frac{\exp(Y_{uit}/\lambda_k)}{\sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)}\right) + \log \left( \frac{\exp(W_{ukt} + \lambda_k I_{ukt})}{\sum_{\ell=1}^K \exp(W_{u\ell t} + \lambda_\ell I_{u\ell t})}\right)
$$

The **inclusive value** of nest $k$, $I_{ukt}$ is defined as $\log \sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)$, which is the *expected utility from choosing the best alternative from nest $k$*.

The `nest_to_item` keyword defines a dictionary of the mapping $k \mapsto B_k$, where keys of `nest_to_item`  are integer $k$'s and  `nest_to_item[k]`  is a list consisting of IDs of items in $B_k$.

The `{nest, item}_coef_variation_dict` provides specification to $W_{ukt}$ and $Y_{uit}$ respectively, `torch_choice` allows for empty nest level models by providing an empty dictionary (in this case, $W_{ukt} = \epsilon_{ukt}$) since the inclusive value term $\lambda_k I_{ukt}$ will be used to model the choice over nests. However, by specifying an empty second stage model ($Y_{uit} = \epsilon_{uit}$), the nested logit model reduces to a conditional logit model of choices over nests. Hence, one should never use the `NestedLogitModel` class with an empty item-level model.

Similar to the conditional logit model, `{nest, item}_num_param_dict` specify the dimension (number of observables to be multiplied with the coefficient) of coefficients. The above code initializes a simple model built upon item-time-specific observables $X_{it} \in \mathbb{R}^7$,

$$
Y_{uit} = \beta^\top X_{it} + \epsilon_{uit} \\
W_{ukt} = \epsilon_{ukt}
$$

The research may wish to enfoce the *elasiticity* $\lambda_k$ to be constant across nests, setting `shared_lambda=True` enforces $\lambda_k = \lambda\ \forall k \in [K]$.

## Load Essential Packages
We firstly read essential packages for this tutorial.


```python
import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.run_helper import run
print(torch.__version__)
```

    2.0.0


We then select the appropriate device to run the model on, our package supports both CPU and GPU.


```python
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    DEVICE = 'cuda'
else:
    print('Running tutorial on CPU')
    DEVICE = 'cpu'
    
```

    Running tutorial on CPU


## Load Datasets
We firstly read the dataset for this tutorial, the `csv` file can be found at `./public_datasets/HC.csv`.
Alternatively, we load the dataset directly from the Github website.


```python
df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv', index_col=0)
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



### Nest Level Dataset
We firstly construct the nest-level dataset, however, there is no observable that is constant within the same nest, so we don't need to include any observable tensor to the `nest_dataset`.

All we need to do is adding the `item_index` (i.e., which item is chosen) to the dataset, so that `nest_dataset` knows the total number of choices made.


```python
# nest feature: no nest feature, all features are item-level.
nest_dataset = ChoiceDataset(item_index=item_index.clone()).to(DEVICE)
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


Finally, we chain the nest-level and item-level dataset into a single `JointDataset`.


```python
dataset = JointDataset(nest=nest_dataset, item=item_dataset)
```

One can print the joint dataset to see its contents, and tensors contained in each of these sub-datasets.


```python
print(dataset)
```

    JointDataset with 2 sub-datasets: (
    	nest: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cpu)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cpu)
    )


## Examples
There are multiple ways to group 7 items into nests, different classification will result in different utility functions and estimations (see the background of nested logit models).

We will demonstrate the usage of our package by presenting three different categorization schemes and corresponding model estimations.

### Example 1
In the first example, the model is specified to have the *cooling alternatives* `{gcc, ecc, erc, hpc}` in one nest and the *non-cooling alternatives* `{gc, ec, er}` in another nest.

We create a `nest_to_item` dictionary to inform the model our categorization scheme. The dictionary should have keys ranging from `0` to `number_of_nests - 1`, each integer corresponds to a nest. The value of each key is a list of item IDs in the nest, the encoding of item names should be exactly the same as in the construction of `item_index`.


```python
nest_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                1: ['gc', 'ec', 'er']}

# encode items to integers.
for k, v in nest_to_item.items():
    v = [encoder[item] for item in v]
    nest_to_item[k] = sorted(v)
```

In this example, we have item `[1, 3, 5, 6]` in the first nest (i.e., the nest with ID `0`) and the rest of items in the second nest (i.e., the nest with ID `1`).


```python
print(nest_to_item)
```

    {0: [1, 3, 5, 6], 1: [0, 2, 4]}


Next, let's create the `NestedLogitModel` class!

The first thing to put in is the `nest_to_item` dictionary we just built.

For `nest_coef_variation_dict`, `nest_num_param_dict`, since we don't have any nest-specific observables, we can simply put an empty dictionary there.

Coefficients for all observables are constant across items, and there are 7 observables in total.

As for `shared_lambda=True`, please refer to the background recap for nested logit model.


```python
model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_coef_variation_dict={},
                         nest_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_formula='',
                         item_formula='(price_obs|constant)',
                         dataset=dataset,
                         shared_lambda=True)

model = model.to(DEVICE)
```

You can print the model to get summary information of the `NestedLogitModel` class.


```python
print(model)
```

    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
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
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	nest: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cpu)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cpu)
    )
    ==================== training the model ====================
    Epoch 1000: Log-likelihood=-179.78282165527344
    Epoch 2000: Log-likelihood=-178.6439666748047
    Epoch 3000: Log-likelihood=-178.45376586914062
    Epoch 4000: Log-likelihood=-178.30226135253906
    Epoch 5000: Log-likelihood=-178.19009399414062
    Epoch 6000: Log-likelihood=-178.1377716064453
    Epoch 7000: Log-likelihood=-178.1256866455078
    Epoch 8000: Log-likelihood=-178.124755859375
    Epoch 9000: Log-likelihood=-178.12757873535156
    Epoch 10000: Log-likelihood=-178.12527465820312
    ==================== model results ====================
    Training Epochs: 10000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -178.12527465820312
    
    Coefficients:
    
    | Coefficient                |   Estimation |   Std. Err. |
    |:---------------------------|-------------:|------------:|
    | lambda_weight_0            |     0.585844 |    0.166706 |
    | item_price_obs[constant]_0 |    -0.555026 |    0.144731 |
    | item_price_obs[constant]_1 |    -0.858004 |    0.237756 |
    | item_price_obs[constant]_2 |    -0.224923 |    0.110701 |
    | item_price_obs[constant]_3 |    -1.08933  |    1.03791  |
    | item_price_obs[constant]_4 |    -0.379122 |    0.100874 |
    | item_price_obs[constant]_5 |     0.249721 |    0.051977 |
    | item_price_obs[constant]_6 |    -5.99982  |    4.83646  |





    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
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
The second example is similar to the first one, but we change the way we group items into different nests.
Re-estimate the model with the room alternatives in one nest and the central alternatives in another nest. (Note that a heat pump is a central system.)


```python
nest_to_item = {0: ['ec', 'ecc', 'gc', 'gcc', 'hpc'],
                    1: ['er', 'erc']}
for k, v in nest_to_item.items():
    v = [encoder[item] for item in v]
    nest_to_item[k] = sorted(v)

# these two initializations are equivalent.
model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_coef_variation_dict={},
                         nest_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)
print(model)

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_formula='',
                         item_formula='(price_obs|constant)',
                         dataset=dataset,
                         shared_lambda=True)
print(model)
model = model.to(DEVICE)
```

    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
      )
    )
    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
      )
    )



```python
run
```




    <function torch_choice.utils.run_helper.run(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000, report_frequency=None, compute_std=True, return_final_training_log_likelihood=False)>




```python
run(model, dataset, num_epochs=50000, learning_rate=0.3)
```

    ==================== received model ====================
    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	nest: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cpu)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cpu)
    )
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-180.560791015625
    Epoch 10000: Log-likelihood=-180.80062866210938
    Epoch 15000: Log-likelihood=-181.21275329589844
    Epoch 20000: Log-likelihood=-180.3982696533203
    Epoch 25000: Log-likelihood=-180.29925537109375
    Epoch 30000: Log-likelihood=-182.28366088867188
    Epoch 35000: Log-likelihood=-180.1341552734375
    Epoch 40000: Log-likelihood=-182.2633514404297
    Epoch 45000: Log-likelihood=-180.19305419921875
    Epoch 50000: Log-likelihood=-180.68240356445312
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.3
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -180.68240356445312
    
    Coefficients:
    
    | Coefficient                |   Estimation |   Std. Err. |
    |:---------------------------|-------------:|------------:|
    | lambda_weight_0            |     1.63154  |    0.678117 |
    | item_price_obs[constant]_0 |    -1.34966  |    0.531558 |
    | item_price_obs[constant]_1 |    -2.17924  |    0.894518 |
    | item_price_obs[constant]_2 |    -0.412631 |    0.243317 |
    | item_price_obs[constant]_3 |    -2.61227  |    2.06289  |
    | item_price_obs[constant]_4 |    -0.885769 |    0.337734 |
    | item_price_obs[constant]_5 |     0.49301  |    0.199931 |
    | item_price_obs[constant]_6 |   -16.0524   |    9.32373  |





    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
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
For the third example, we now group items into three nests. Specifically, we have items `gcc`, `ecc` and `erc` in the first nest (nest `0` in the `nest_to_item` dictionary), `hpc` in a nest (nest `1`) alone, and items `gc`, `ec` and `er` in the last nest (nest `2`).


```python
nest_to_item = {0: ['gcc', 'ecc', 'erc'],
                1: ['hpc'],
                2: ['gc', 'ec', 'er']}
for k, v in nest_to_item.items():
    v = [encoder[item] for item in v]
    nest_to_item[k] = sorted(v)

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_coef_variation_dict={},
                         nest_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_formula='',
                         item_formula='(price_obs|constant)',
                         dataset=dataset,
                         shared_lambda=True)

model = model.to(DEVICE)
```


```python
run(model, dataset, num_epochs=50000, learning_rate=0.3)
```

    ==================== received model ====================
    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	nest: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cpu)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cpu)
    )
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-180.84153747558594
    Epoch 10000: Log-likelihood=-182.17794799804688
    Epoch 15000: Log-likelihood=-181.74029541015625
    Epoch 20000: Log-likelihood=-182.3179931640625
    Epoch 25000: Log-likelihood=-182.50352478027344
    Epoch 30000: Log-likelihood=-181.481201171875
    Epoch 35000: Log-likelihood=-181.8275604248047
    Epoch 40000: Log-likelihood=-180.5753173828125
    Epoch 45000: Log-likelihood=-182.4506072998047
    Epoch 50000: Log-likelihood=-185.08358764648438
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.3
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -185.08358764648438
    
    Coefficients:
    
    | Coefficient                |   Estimation |   Std. Err. |
    |:---------------------------|-------------:|------------:|
    | lambda_weight_0            |     0.949264 |   0.19245   |
    | item_price_obs[constant]_0 |    -0.852556 |   0.100724  |
    | item_price_obs[constant]_1 |    -1.35082  |   0.188374  |
    | item_price_obs[constant]_2 |    -0.248292 |   0.14014   |
    | item_price_obs[constant]_3 |    -1.41068  |   1.2839    |
    | item_price_obs[constant]_4 |    -0.581716 |   0.0771356 |
    | item_price_obs[constant]_5 |     0.336492 |   0.0656387 |
    | item_price_obs[constant]_6 |   -10.5186   |   5.71641   |





    NestedLogitModel(
      (nest_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs[constant]): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cpu).
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
