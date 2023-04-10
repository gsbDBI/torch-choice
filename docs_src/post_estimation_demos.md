# Tutorial: Post-Estimations

**Author: Tianyu Du (tianyudu@stanford.edu)**

This tutorial covers the toolkit in `torch-choice` for visualizing and analyzing models after model estimation.

**Note**: models demonstrated in this tutorial are for demonstration purpose only, hence we don't estimate them in this tutorial. Instead, this tutorial focuses on APIs to visualize and analyze models.


```python
# import required dependencies.
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.run_helper import run

import seaborn as sns
```


```python
# let's get a helper
def print_dict_shape(d):
    for key, val in d.items():
        if torch.is_tensor(val):
            print(f'dict.{key}.shape={val.shape}')
```

## Creating  `ChoiceDataset` Object

We first create a dummy `ChoiceDataset` object, please refer to the **data management** tutorial for more details.


```python
# Feel free to modify it as you want.
num_users = 100
num_items = 25
num_sessions = 500

length_of_dataset = 10000
# create observables/features, the number of parameters are arbitrarily chosen.
# generate 128 features for each user, e.g., race, gender.
user_obs = torch.randn(num_users, 128)
# generate 64 features for each user, e.g., quality.
item_obs = torch.randn(num_items, 64)
# generate 10 features for each session, e.g., weekday indicator. 
session_obs = torch.randn(num_sessions, 10)
# generate 12 features for each session user pair, e.g., the budget of that user at the shopping day.
itemsession_obs = torch.randn(num_sessions, num_items, 12)
item_index = torch.LongTensor(np.random.choice(num_items, size=length_of_dataset))
user_index = torch.LongTensor(np.random.choice(num_users, size=length_of_dataset))
session_index = torch.LongTensor(np.random.choice(num_sessions, size=length_of_dataset))
# assume all items are available in all sessions.
item_availability = torch.ones(num_sessions, num_items).bool()

# initialize a ChoiceDataset object.
dataset = ChoiceDataset(
    # pre-specified keywords of __init__
    item_index=item_index,  # required.
    # optional:
    num_users=num_users,
    num_items=num_items,
    user_index=user_index,
    session_index=session_index,
    item_availability=item_availability,
    # additional keywords of __init__
    user_obs=user_obs,
    item_obs=item_obs,
    session_obs=session_obs,
    itemsession_obs=itemsession_obs)
```


```python
print(dataset)
```

    ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 25], user_obs=[100, 128], item_obs=[25, 64], session_obs=[500, 10], itemsession_obs=[500, 25, 12], device=cpu)


# Conditional Logit Model

Suppose that we are creating a very complicated dummy model as the following. Please note that model and dataset here are for demonstration purpose only, the model is unlikely to converge if one estimate it on this dataset.

$$
U_{uis} = \alpha + \beta_i + \gamma_u + \delta_i^\top \textbf{x}^{(user)}_u + \eta^\top \textbf{y}^{(item)}_i + \theta_u^\top \textbf{z}^{(session)}_{ui} + \kappa_i^\top \textbf{w}^{(user-item)}_{ui} + \epsilon_{uis}
$$


```python
model = ConditionalLogitModel(formula='(1|constant) + (1|item) + (1|user) + (user_obs|item) + (item_obs|constant) + (session_obs|user) + (itemsession_obs|user)',
                              dataset=dataset,
                              num_users=num_users,
                              num_items=num_items)

# estimate the model.
```


```python
model
```




    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (intercept[constant]): Coefficient(variation=constant, num_items=25, num_users=100, num_params=1, 1 trainable parameters in total, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=25, num_users=100, num_params=1, 24 trainable parameters in total, device=cpu).
        (intercept[user]): Coefficient(variation=user, num_items=25, num_users=100, num_params=1, 100 trainable parameters in total, device=cpu).
        (user_obs): Coefficient(variation=item, num_items=25, num_users=100, num_params=128, 3072 trainable parameters in total, device=cpu).
        (item_obs): Coefficient(variation=constant, num_items=25, num_users=100, num_params=64, 64 trainable parameters in total, device=cpu).
        (session_obs): Coefficient(variation=user, num_items=25, num_users=100, num_params=10, 1000 trainable parameters in total, device=cpu).
        (itemsession_obs): Coefficient(variation=user, num_items=25, num_users=100, num_params=12, 1200 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[intercept[constant]] with 1 parameters, with constant level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    X[intercept[user]] with 1 parameters, with user level variation.
    X[user_obs] with 128 parameters, with item level variation.
    X[item_obs] with 64 parameters, with constant level variation.
    X[session_obs] with 10 parameters, with user level variation.
    X[itemsession_obs] with 12 parameters, with user level variation.
    device=cpu



# Retrieving Model Parameters with the `get_coefficient()` method.

In the model representation above, we can see that the model has coefficients from `intercept[constant]` to `itemsession_obs`. 
The `get_coefficient()` method allows users to retrieve the coefficient values from the model using the general syntax `model.get_coefficient(COEFFICIENT_NAME)`.

For example, `model.get_coefficient('intercept[constant]')` will return the value of $\alpha$, which is a scalar.


```python
model.get_coefficient('intercept[constant]')
```




    tensor([0.0972])



`model.get_coefficient('intercept[user]')` returns the array of $\gamma_u$'s, which is a 1D array of length `num_users`.


```python
model.get_coefficient('intercept[user]').shape
```




    torch.Size([100, 1])



`model.get_coefficient('session_obs')` returns the corresponding coefficient `theta_u`, which is a 2D array of shape `(num_users, num_session_features)`. Each row of the returned tensor corresponds to the coefficient vector of a user.


```python
model.get_coefficient('session_obs').shape
```




    torch.Size([100, 10])




```python
model.get_coefficient('itemsession_obs').shape
```




    torch.Size([100, 12])




