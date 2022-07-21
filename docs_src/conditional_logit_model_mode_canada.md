# Tutorial: Conditional Logit Model on ModeCanada Dataset
**Author: Tianyu Du (tianyudu@stanford.edu)**

**Update: May. 3, 2022**

**Reference:** This tutorial is modified from the [Random utility model and the multinomial logit model](https://cran.r-project.org/web/packages/mlogit/vignettes/c3.rum.html) in th documentation of `mlogit` package in R.

Please note that the dataset involved in this example is fairly small (2,779 choice records), so we don't expect the performance to be faster than the R implementation.

We provide this tutorial mainly to check the correctness of our prediction. The fully potential of PyTorch is better exploited on much larger dataset.

The executable Jupyter notebook for this tutorial is located at [Random Utility Model (RUM) 1: Conditional Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb).

Let's first import essential Python packages.


```python
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from torch_choice.utils.run_helper import run
```

This tutorial will run both with and without graphic processing unit (GPU). However, our package is *much* faster with GPU.


```python
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    device = 'cuda'
else:
    print('Running tutorial on CPU.')
    device = 'cpu'
```

    Running tutorial on CPU.


## Load Dataset
We have included the `ModeCanada` dataset in our package, which is located at `./public_datasets/`.

The `ModeCanada` dataset contains individuals' choice on traveling methods.

The raw dataset is in a long-format, in which the `case` variable identifies each choice.
Using the terminology mentioned in the data management tutorial, each choice is called a *purchasing record* (i.e., consumer bought the ticket of a particular travelling mode), and the total number of choices made is denoted as $B$.

For example, the first four row below (with `case == 109`) corresponds to the first choice, the `alt` column lists all alternatives/items available.

The `choice` column identifies which alternative/item is chosen. The second row in the data snapshot below, we have `choice == 1` and `alt == 'air'` for `case == 109`. This indicates the travelling mode chosen in `case = 109` was `air`.

Now we convert the raw dataset into the format compatible with our model, for a detailed tutorial on the compatible formats, please refer to the data management tutorial.

We focus on cases when four alternatives were available by filtering `noalt == 4`.


```python
df = pd.read_csv('./public_datasets/ModeCanada.csv')
df = df.query('noalt == 4').reset_index(drop=True)
df.sort_values(by='case', inplace=True)
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
      <th>Unnamed: 0</th>
      <th>case</th>
      <th>alt</th>
      <th>choice</th>
      <th>dist</th>
      <th>cost</th>
      <th>ivt</th>
      <th>ovt</th>
      <th>freq</th>
      <th>income</th>
      <th>urban</th>
      <th>noalt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>304</td>
      <td>109</td>
      <td>train</td>
      <td>0</td>
      <td>377</td>
      <td>58.25</td>
      <td>215</td>
      <td>74</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>305</td>
      <td>109</td>
      <td>air</td>
      <td>1</td>
      <td>377</td>
      <td>142.80</td>
      <td>56</td>
      <td>85</td>
      <td>9</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>306</td>
      <td>109</td>
      <td>bus</td>
      <td>0</td>
      <td>377</td>
      <td>27.52</td>
      <td>301</td>
      <td>63</td>
      <td>8</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>307</td>
      <td>109</td>
      <td>car</td>
      <td>0</td>
      <td>377</td>
      <td>71.63</td>
      <td>262</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>308</td>
      <td>110</td>
      <td>train</td>
      <td>0</td>
      <td>377</td>
      <td>58.25</td>
      <td>215</td>
      <td>74</td>
      <td>4</td>
      <td>70</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Since there are 4 rows corresponding to each *purchasing record*, the length of the long-format data is $4 \times B$.
Please refer to the data management tutorial for notations.


```python
df.shape
```




    (11116, 12)



### Construct the `item_index` tensor
The first thing is to construct the `item_index` tensor identifying which item (i.e., travel mode) was chosen in each purchasing record.

We can now construct the `item_index` array containing which item was chosen in each purchasing record.


```python
item_index = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
print(item_index)
```

    0       air
    1       air
    2       air
    3       air
    4       air
           ...
    2774    car
    2775    car
    2776    car
    2777    car
    2778    car
    Name: alt, Length: 2779, dtype: object


Since we will be training our model using `PyTorch`, we need to encode `{'air', 'bus', 'car', 'train'}` into integer values.

| Travel Mode Name      | Encoded Integer Values |
| :---     | :----:  |
| air      | 0       |
| bus      | 1       |
| car      | 2       |
| train    | 3       |

The generated `item_index` would be a tensor of shape 2,778 (i.e., number of purchasing records in this dataset) with values `{0, 1, 2, 3}`.


```python
item_names = ['air', 'bus', 'car', 'train']
num_items = 4
encoder = dict(zip(item_names, range(num_items)))
print(f"{encoder=:}")
item_index = item_index.map(lambda x: encoder[x])
item_index = torch.LongTensor(item_index)
print(f"{item_index=:}")
```

    encoder={'air': 0, 'bus': 1, 'car': 2, 'train': 3}
    item_index=tensor([0, 0, 0,  ..., 2, 2, 2])


### Construct Observables

Then let's constrct tensors for observables.
As mentioned in the data management tutorial, the *session* is capturing the temporal dimension of our data.
Since we have different values `cost`, `freq` and `ovt` for each purchasing record and for each item, it's natural to say each purchasing record has its own session.

Consequently, these three variables are `price` observables since they vary by both item and session.
The tensor holding these observables has shape $(\text{numer of purchasing records}, \text{number of items}, 3)$

We do the same for variable `ivt`, we put `ivt` into a separate tensor because we want to model its coefficient differently later.


```python
price_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                    values=['cost', 'freq', 'ovt'])
print(f'{price_cost_freq_ovt.shape=:}')

price_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
print(f'{price_ivt.shape=:}')
```

    price_cost_freq_ovt.shape=torch.Size([2779, 4, 3])
    price_ivt.shape=torch.Size([2779, 4, 1])


In contrast, the `income` variable varies only by session (i.e., purchasing record), but not by item. `income` is therefore naturally a `session` variable.


```python
session_income = df.groupby('case')['income'].first()
session_income = torch.Tensor(session_income.values).view(-1, 1)
print(f'{session_income.shape=:}')
```

    session_income.shape=torch.Size([2779, 1])


To summarize, the `ChoiceDataset` constructed contains 2779 choice records. Since the original dataset did not reveal the identity of each decision maker, we consider all 2779 choices were made by a single user but in 2779 different sessions to handle variations.

In this case, the `cost`, `freq` and `ovt` are observables depending on both sessions and items, we created a `price_cost_freq_ovt` tensor with shape `(num_sessions, num_items, 3) = (2779, 4, 3)` to contain these variables.
In contrast, the `income` information depends only on session but not on items, hence we create the `session_income` tensor to store it.

Because we wish to fit item-specific coefficients for the `ivt` variable, which varies by both sessions and items as well, we create another `price_ivt` tensor in addition to the `price_cost_freq_ovt` tensor.

Lastly, we put all tensors we created to a single `ChoiceDataset` object, and move the dataset to the appropriate device.


```python
dataset = ChoiceDataset(item_index=item_index,
                        price_cost_freq_ovt=price_cost_freq_ovt,
                        session_income=session_income,
                        price_ivt=price_ivt
                        ).to(device)
```

You can `print(dataset)` to check shapes of tensors contained in the `ChoiceDataset`.


```python
print(dataset)
```

    ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cpu)


## Create the Model
We now construct the `ConditionalLogitModel` to fit the dataset we constructed above.

To start with, we aim to estimate the following model formulation:

$$
U_{uit} = \beta^0_i + \beta^{1\top} X^{price: (cost, freq, ovt)}_{it} + \beta^2_i X^{session:income}_t + \beta^3_i X_{it}^{price:ivt} + \epsilon_{uit}
$$

We now initialize the `ConditionalLogitModel` to predict choices from the dataset.
Please see the documentation for a complete description of the `ConditionalLogitModel` class.

At it's core, the `ConditionalLogitModel` constructor requires the following four components.

### Define variation of each $\beta$ using `coef_variation_dict`
The keyword `coef_variation_dict` is a dictionary with variable names (defined above while constructing the dataset) as keys and values from `{constant, user, item, item-full}`.

For instance, since we wish to have constant coefficients for `cost`, `freq` and `ovt` observables, and these three observables are stored in the `price_cost_freq_ovt` tensor of the choice dataset, we set `coef_variation_dict['price_cost_freq_ovt'] = 'constant'` (corresponding to the $\beta^{1\top} X^{price: (cost, freq, ovt)}_{it}$ term above).

The models allows for the option of zeroing coefficient for one item.
The variation of $\beta^3$ above is specified as `item-full` which indicates 4 values of $\beta^3$ is learned (one for each item).
In contrast, $\beta^0, \beta^2$ are specified to have variation `item` instead of `item-full`. In this case, the $\beta$ correspond to the first item (i.e., the baseline item, which is encoded as 0 in the label tensor, `air` in our example) is force to be zero.

The researcher needs to declare `intercept` explicitly for the model to fit an intercept as well, otherwise the model assumes zero intercept term.

### Define the dimension of each $\beta$ using `num_param_dict`
The `num_param_dict` is a dictionary with keys exactly the same as the `coef_variation_dict`.
Each of dictionary values tells the dimension of the corresponding observables, hence the dimension of the coefficient.
For example, the `price_cost_freq_ovt` consists of three observables and we set the corresponding to three.

Even the model can infer `num_param_dict['intercept'] = 1`, but we recommend the research to include it for completeness.

### Number of items
The `num_items` keyword informs the model how many alternatives users are choosing from.

### Number of users
The `num_users` keyword is an optional integer informing the model how many users there are in the dataset. However, in this example we implicitly assume there is only one user making all the decisions and we do not have any `user_obs` involved, hence `num_users` argument is not supplied.


```python
model = ConditionalLogitModel(coef_variation_dict={'price_cost_freq_ovt': 'constant',
                                                   'session_income': 'item',
                                                   'price_ivt': 'item-full',
                                                   'intercept': 'item'},
                              num_param_dict={'price_cost_freq_ovt': 3,
                                              'session_income': 1,
                                              'price_ivt': 1,
                                              'intercept': 1},
                              num_items=4)
```

Then we move the model to the appropriate device.


```python
model = model.to(device)
```

One can print the `ConditionalLogitModel` object to obtain a summary of the model.


```python
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
      )
    )
    Conditional logistic discrete choice model, expects input features:

    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.


## Train the Model

We provide an easy-to-use helper function `run()` imported from `torch_choice.utils.run_helper` to fit the model with a particular dataset.

We provide an easy-to-use model runner for both `ConditionalLogitModel` and `NestedLogitModel` (see later) instances.

The `run()` mehtod supports mini-batch updating as well, for small datasets like the one we are dealing right now, we can use `batch_size = -1` to conduct full-batch gradient update.


```python
start_time = time()
run(model, dataset, num_epochs=50000, learning_rate=0.01, batch_size=-1)
print('Time taken:', time() - start_time)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
      )
    )
    Conditional logistic discrete choice model, expects input features:

    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cpu)
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-1875.552490234375
    Epoch 10000: Log-likelihood=-1892.94775390625
    Epoch 15000: Log-likelihood=-1877.9156494140625
    Epoch 20000: Log-likelihood=-1881.0845947265625
    Epoch 25000: Log-likelihood=-1884.7335205078125
    Epoch 30000: Log-likelihood=-1874.423828125
    Epoch 35000: Log-likelihood=-1875.3016357421875
    Epoch 40000: Log-likelihood=-1874.3779296875
    Epoch 45000: Log-likelihood=-1875.703125
    Epoch 50000: Log-likelihood=-1899.8175048828125
    ==================== model results ====================
    Training Epochs: 50000

    Learning Rate: 0.01

    Batch Size: 2779 out of 2779 observations in total

    Final Log-likelihood: -1899.8175048828125

    Coefficients:

    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 |  -0.0342194  |  0.00731707 |
    | price_cost_freq_ovt_1 |   0.092262   |  0.00520946 |
    | price_cost_freq_ovt_2 |  -0.0439827  |  0.00342765 |
    | session_income_0      |  -0.0901207  |  0.0205214  |
    | session_income_1      |  -0.0272581  |  0.00385396 |
    | session_income_2      |  -0.0390468  |  0.00428838 |
    | price_ivt_0           |   0.0592097  |  0.0102933  |
    | price_ivt_1           |  -0.00753696 |  0.00496264 |
    | price_ivt_2           |  -0.00604297 |  0.00193414 |
    | price_ivt_3           |  -0.00207518 |  0.00123286 |
    | intercept_0           |   0.700786   |  1.39368    |
    | intercept_1           |   1.85016    |  0.728283   |
    | intercept_2           |   3.2782     |  0.648064   |
    Time taken: 179.84411025047302


### Parameter Estimation from `R`
The following is the R-output from the `mlogit` implementation, the estimation, standard error, and log-likelihood from our `torch_choice` implementation is the same as the result from `mlogit` implementation.

We see that the final log-likelihood of models estimated using two packages are all around `-1874`.

The `run()` method calculates the standard deviation using $\sqrt{\text{diag}(H^{-1})}$, where $H$ is the hessian of negative log-likelihood with repsect to model parameters.

Names of coefficients are slightly different, one can use the following conversion table to compare estimations and standard deviations reported by both packages.

| Coefficient Name in Python |  Estimation |   Std. Err. |  Coeffcient Name in R | R Estimation | R Std. Err. |
|:---------------------:|-------------:|------------:| :--------------: | ----------: | ------: |
| price_cost_freq_ovt_0 |  -0.0342194  |  0.00731707 | cost             | -0.0333389  |0.0070955|
| price_cost_freq_ovt_1 |   0.092262   |  0.00520946 | freq             |  0.0925297  |0.0050976|
| price_cost_freq_ovt_2 |  -0.0439827  |  0.00342765 | ovt              | -0.0430036  |0.0032247|
| session_income_0      |  -0.0901207  |  0.0205214  | income:bus       | -0.0890867  |0.0183471|
| session_income_1      |  -0.0272581  |  0.00385396 | income:car       | -0.0279930  |0.0038726|
| session_income_2      |  -0.0390468  |  0.00428838 | ivt:train        | -0.0014504  |0.0011875|
| price_ivt_0           |   0.0592097  |  0.0102933  | ivt:air          |  0.0595097  |0.0100727|
| price_ivt_1           |  -0.00753696 |  0.00496264 | ivt:bus          | -0.0067835  |0.0044334|
| price_ivt_2           |  -0.00604297 |  0.00193414 | ivt:car          | -0.0064603  |0.0018985|
| price_ivt_3           |  -0.00207518 |  0.00123286 | ivt:train        | -0.0014504  |0.0011875|
| intercept_0           |   0.700786   |  1.39368    | (Intercept):bus  |  0.6983381  |1.2802466|
| intercept_1           |   1.85016    |  0.728283   | (Intercept):car  |  1.8441129  |0.7085089|
| intercept_2           |   3.2782     |  0.648064   | (Intercept):train|  3.2741952  |0.6244152|

### R Output
```r
install.packages("mlogit")
library("mlogit")
data("ModeCanada", package = "mlogit")
MC <- dfidx(ModeCanada, subset = noalt == 4)
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')

summary(ml.MC1)
```
```
Call:
mlogit(formula = choice ~ cost + freq + ovt | income | ivt, data = MC,
    reflevel = "air", method = "nr")

Frequencies of alternatives:choice
      air     train       bus       car
0.3738755 0.1666067 0.0035984 0.4559194

nr method
9 iterations, 0h:0m:0s
g'(-H)^-1g = 0.00014
successive function values within tolerance limits

Coefficients :
                    Estimate Std. Error  z-value  Pr(>|z|)
(Intercept):train  3.2741952  0.6244152   5.2436 1.575e-07 ***
(Intercept):bus    0.6983381  1.2802466   0.5455 0.5854292
(Intercept):car    1.8441129  0.7085089   2.6028 0.0092464 **
cost              -0.0333389  0.0070955  -4.6986 2.620e-06 ***
freq               0.0925297  0.0050976  18.1517 < 2.2e-16 ***
ovt               -0.0430036  0.0032247 -13.3356 < 2.2e-16 ***
income:train      -0.0381466  0.0040831  -9.3426 < 2.2e-16 ***
income:bus        -0.0890867  0.0183471  -4.8556 1.200e-06 ***
income:car        -0.0279930  0.0038726  -7.2286 4.881e-13 ***
ivt:air            0.0595097  0.0100727   5.9080 3.463e-09 ***
ivt:train         -0.0014504  0.0011875  -1.2214 0.2219430
ivt:bus           -0.0067835  0.0044334  -1.5301 0.1259938
ivt:car           -0.0064603  0.0018985  -3.4029 0.0006668 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Log-Likelihood: -1874.3
McFadden R^2:  0.35443
Likelihood ratio test : chisq = 2058.1 (p.value = < 2.22e-16)
```


```python

```
