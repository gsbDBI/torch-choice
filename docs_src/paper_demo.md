# Materials for Torch-Choice Paper


```python
from time import time
import numpy as np
import torch
import torch_choice
from tqdm import tqdm
from typing import List
from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel
```

## The Car Choice Dataset Example


```python
import pandas as pd
import torch 
import torch_choice
import torch_choice.utils
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper
from torch_choice.utils.run_helper import run
```

## Performance Benchmark
**Copy the following description to the paper**.
We designed a simple performance benchmark based on the transportation choice dataset: we duplicates $K$ copies of the original dataset of 2779 observations and compare time taken by various implementations. We compared the time cost of only the estimation process, since there are ample possibilities for further optimizing the estimation process (e.g., tuning learning rates, early stopping), we could under-estimate performances here. However, we wish to highlight how K 
The metric $\frac{\text{log-likelihood}}{K}$ is used to check that various optimizers converged to the same solution.


```python
! mkdir -p './benchmark_data'
```


```python
num_copies = 3
df = pd.read_csv('./public_datasets/ModeCanada.csv')
df_list = list()
num_cases = df['case'].max()
for i in range(num_copies):
    df_copy = df.copy()
    df_copy['case'] += num_cases * i
    df_list.append(df_copy)
df = pd.concat(df_list, ignore_index=True)
```


```python
df
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
      <td>1</td>
      <td>1</td>
      <td>train</td>
      <td>0</td>
      <td>83</td>
      <td>28.25</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>car</td>
      <td>1</td>
      <td>83</td>
      <td>15.77</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>train</td>
      <td>0</td>
      <td>83</td>
      <td>28.25</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>car</td>
      <td>1</td>
      <td>83</td>
      <td>15.77</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>train</td>
      <td>0</td>
      <td>83</td>
      <td>28.25</td>
      <td>50</td>
      <td>66</td>
      <td>4</td>
      <td>70</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>46555</th>
      <td>15516</td>
      <td>12970</td>
      <td>car</td>
      <td>1</td>
      <td>347</td>
      <td>65.93</td>
      <td>267</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46556</th>
      <td>15517</td>
      <td>12971</td>
      <td>train</td>
      <td>0</td>
      <td>323</td>
      <td>60.60</td>
      <td>193</td>
      <td>200</td>
      <td>3</td>
      <td>45</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>46557</th>
      <td>15518</td>
      <td>12971</td>
      <td>car</td>
      <td>1</td>
      <td>323</td>
      <td>61.37</td>
      <td>278</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>46558</th>
      <td>15519</td>
      <td>12972</td>
      <td>train</td>
      <td>0</td>
      <td>150</td>
      <td>28.50</td>
      <td>63</td>
      <td>105</td>
      <td>1</td>
      <td>70</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>46559</th>
      <td>15520</td>
      <td>12972</td>
      <td>car</td>
      <td>1</td>
      <td>150</td>
      <td>28.50</td>
      <td>134</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>46560 rows × 12 columns</p>
</div>




```python
def duplicate_mode_canada_datasets(num_copies: int):
    df = pd.read_csv('./public_datasets/ModeCanada.csv', index_col=0)
    df_list = list()
    num_cases = df['case'].max()
    for i in range(num_copies):
        df_copy = df.copy()
        df_copy['case'] += num_cases * i
        df_list.append(df_copy)
    df = pd.concat(df_list, ignore_index=True)
    df = df.query('noalt == 4').reset_index(drop=True)
    df.sort_values(by='case', inplace=True)
    item_index = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
    item_names = ['air', 'bus', 'car', 'train']
    num_items = 4
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)
    price_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                        values=['cost', 'freq', 'ovt'])
    price_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
    session_income = df.groupby('case')['income'].first()
    session_income = torch.Tensor(session_income.values).view(-1, 1)

    # session_index = torch.arange(len(session_income))
    
    dataset = ChoiceDataset(
        # item_index=item_index.repeat(num_copies),
        item_index=item_index,
        session_index=torch.arange(len(session_income)),
        price_cost_freq_ovt=price_cost_freq_ovt,
        session_income=session_income,
        price_ivt=price_ivt)
    return df, dataset.clone()
```


```python
df, dataset = duplicate_mode_canada_datasets(10)
```


```python
performance_records = list()
# k_range = [1, 5, 10, 100, 1_000, 10_000]
k_range = [50, 500, 5_000]
dataset_at_k = dict()
for k in tqdm(k_range):
    df, dataset = duplicate_mode_canada_datasets(k)
    dataset_at_k[k] = dataset.clone()
    # df.to_csv(f'./benchmark_data/mode_canada_{k}.csv', index=False)
```

    100%|██████████| 3/3 [03:08<00:00, 62.69s/it]



```python
for k in k_range:
    # run for 3 times.
    for _ in range(3):
        dataset = duplicate_mode_canada_datasets(k)
        model = model = ConditionalLogitModel(
            formula='(price_cost_freq_ovt|constant) + (session_income|item) + (price_ivt|item-full) + (intercept|item)',
            dataset=dataset,
            num_items=4)
        # only time the model estimation.
        start_time = time()
        model, ll = run(model, dataset, batch_size=-1, learning_rate=0.03 , num_epochs=1000, compute_std=True, return_final_training_log_likelihood=True)
        end_time = time()
        performance_records.append(dict(k=k, time=end_time - start_time, ll=ll))
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cpu).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cpu
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[27790], provided_num_items=[], user_index=[], session_index=[27790], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cpu)
    ==================== training the model ====================
    Epoch 100: Log-likelihood=-44074.7265625
    Epoch 200: Log-likelihood=-29754.7890625
    Epoch 300: Log-likelihood=-23659.138671875
    Epoch 400: Log-likelihood=-21073.43359375
    Epoch 500: Log-likelihood=-18857.978515625
    Epoch 600: Log-likelihood=-18838.84375
    Epoch 700: Log-likelihood=-18827.517578125
    Epoch 800: Log-likelihood=-18819.5078125
    Epoch 900: Log-likelihood=-18813.14453125
    Epoch 1000: Log-likelihood=-18807.544921875



```python
performance_records
```




    []



### Simulation Setup (Depreciated)
Example utility construction:
$$
U_{uis} = \lambda_i + \beta_u^\top \bm{x}_\text{item}^{(i)} + \gamma^\top \bm{x}_\text{session}^{(s)} + \epsilon
$$


```python
num_items = 10
num_users = 5
num_sessions = 1
N = 50000
# generate a random user ui.
user_index = torch.LongTensor(np.random.choice(num_users, size=N))
# construct users.
# item_index = torch.LongTensor(np.random.choice(num_items, size=N))
# construct sessions.
session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
rational_prob = 0.99
```


    Running cells with 'Python 3.9.15 ('python-dev')' requires ipykernel package.


    Run the following command to install 'ipykernel' into the Python environment. 


    Command: 'conda install -n python-dev ipykernel --update-deps --force-reinstall'



```python
user_obs = torch.rand(num_users, 3)
item_obs = torch.rand(num_items, 3)
session_obs = torch.rand(num_sessions, 3)
# price_obs = torch.randn(num_sessions, num_items, 12)
item_index = torch.LongTensor(np.random.choice(num_items, size=N))
user_index = torch.LongTensor(np.random.choice(num_users, size=N))
session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
item_availability = torch.ones(num_sessions, num_items).bool()
```


```python
lambda_item = torch.rand(num_items) * 10
lambda_item[0] = 0
beta_user = torch.rand(num_users, item_obs.shape[-1]) * 10
gamma_constant = torch.rand(session_obs.shape[-1]) * 10
```


```python
item_index = list()

for n in tqdm(range(N)):
    u, s = user_index[n], session_index[n]
    if np.random.rand() <= rational_prob:
        # (num_items, 1)
        # utilities = lambda_item + (beta_user[u].view(1, -1).expand(num_items, -1) * item_obs).sum(dim=-1) + (gamma_constant.view(1, -1).expand(num_items, -1) * session_obs[s].view(1, -1).expand(num_items, -1)).sum(dim=-1)
        utilities = lambda_item
        p = torch.nn.functional.softmax(utilities, dim=0).detach().numpy()
        item_index.append(np.random.choice(num_items, p=p))
        # item_index.append(int(np.argmax(utilities)))
    else:
        item_index.append(int(np.random.choice(num_items, size=1)))
item_index = torch.LongTensor(item_index)
```

    100%|██████████| 50000/50000 [00:00<00:00, 64854.88it/s]



```python
df = pd.DataFrame(data={'item_index': item_index, 'user_index': user_index, 'session_index': session_index})
df.to_csv('./benchmark_data/choice_data.csv', index=False)
```


```python
dataset = ChoiceDataset(item_index=item_index, user_index=user_index, session_index=session_index, item_obs=item_obs, user_obs=user_obs, session_obs=session_obs, num_items=num_items)
dataset
```




    ChoiceDataset(label=[], item_index=[50000], provided_num_items=[1], user_index=[50000], session_index=[50000], item_availability=[], item_obs=[10, 3], user_obs=[5, 3], session_obs=[1, 3], device=cpu)




```python
# model = ConditionalLogitModel(formula='(1|item-full) + (item_obs|user) + (session_obs|constant)', dataset=dataset, num_items=num_items, num_users=num_users)
model = ConditionalLogitModel(formula='(1|item)', dataset=dataset, num_items=num_items, num_users=num_users)
print(np.mean((model(dataset).argmax(dim=1) == item_index).float().numpy()))
model
```

    0.00134





    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (intercept): Coefficient(variation=item, num_items=10, num_users=5, num_params=1, 9 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[intercept] with 1 parameters, with item level variation.
    device=cpu




```python
model = run(model, dataset, batch_size=-1, learning_rate=0.3 , num_epochs=1000, compute_std=False)
np.mean((model(dataset).argmax(dim=1) == item_index).float().numpy())
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (intercept): Coefficient(variation=item, num_items=10, num_users=5, num_params=1, 9 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[intercept] with 1 parameters, with item level variation.
    device=cpu
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[50000], provided_num_items=[1], user_index=[50000], session_index=[50000], item_availability=[], item_obs=[10, 3], user_obs=[5, 3], session_obs=[1, 3], device=cpu)
    ==================== training the model ====================
    Epoch 100: Log-likelihood=-81310.9375
    Epoch 200: Log-likelihood=-81305.359375
    Epoch 300: Log-likelihood=-81305.2421875
    Epoch 400: Log-likelihood=-81305.2265625
    Epoch 500: Log-likelihood=-81305.234375
    Epoch 600: Log-likelihood=-81305.2265625
    Epoch 700: Log-likelihood=-81308.171875
    Epoch 800: Log-likelihood=-81305.2265625
    Epoch 900: Log-likelihood=-81339.671875
    Epoch 1000: Log-likelihood=-81305.234375





    0.40572



# Verify Parameter


```python
beta_user
```




    tensor([[4.7301, 6.1908, 7.1181],
            [3.4178, 8.8197, 4.9632],
            [4.9116, 4.5997, 0.7213],
            [8.3757, 0.5155, 4.8729],
            [8.5097, 6.4045, 2.3534]])




```python
model.coef_dict['item_obs'].coef
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /var/folders/r3/rj0t5xcj557855yt3xr0qwnh0000gn/T/ipykernel_79509/1385705425.py in <module>
    ----> 1 model.coef_dict['item_obs'].coef
    

    ~/miniforge3/envs/ml/lib/python3.9/site-packages/torch/nn/modules/container.py in __getitem__(self, key)
        324     @_copy_to_script_wrapper
        325     def __getitem__(self, key: str) -> Module:
    --> 326         return self._modules[key]
        327 
        328     def __setitem__(self, key: str, module: Module) -> None:


    KeyError: 'item_obs'



```python
lambda_item
```




    tensor([0.0000, 6.0579, 0.8783, 7.2887, 6.3035, 1.2217, 4.7925, 6.6317, 4.6998,
            5.0522])




```python
model.coef_dict['intercept'].coef.squeeze()
```




    tensor([ 3.1896, -1.4328,  4.4139,  3.4304, -1.5620,  1.9348,  3.7506,  1.8674,
             2.2129], grad_fn=<SqueezeBackward0>)




```python
gamma_constant
```




    tensor([1.1350, 8.2167, 7.8468])




```python
model.coef_dict['session_obs'].coef.squeeze()
```




    tensor([-1.0625, -2.9416, -0.7111], grad_fn=<SqueezeBackward0>)




```python
np.mean((model(dataset).argmax(dim=1) == item_index).float().numpy())
```




    0.40572




