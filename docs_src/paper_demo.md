# Replication Materials for the Torch-Choice Paper

> Author: Tianyu Du
> 
> Email: `tianyudu@stanford.edu`

This repository contains the replication materials for the paper "Torch-Choice: A Library for Choice Models in PyTorch". Due to the limited space in the main paper, we have omitted some codes and outputs in the paper. This repository contains the full version of codes mentioned in the paper.


```python
import warnings
warnings.filterwarnings("ignore")

from time import time
import numpy as np
import pandas as pd
import torch
import torch_choice
from torch_choice import run
from tqdm import tqdm
from torch_choice.data import ChoiceDataset, JointDataset, utils, load_mode_canada_dataset, load_house_cooling_dataset_v1
from torch_choice.model import ConditionalLogitModel, NestedLogitModel
```

# Data Structure


```python
car_choice = pd.read_csv("https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/car_choice.csv")
car_choice.head()
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
      <th>record_id</th>
      <th>session_id</th>
      <th>consumer_id</th>
      <th>car</th>
      <th>purchase</th>
      <th>gender</th>
      <th>income</th>
      <th>speed</th>
      <th>discount</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>American</td>
      <td>1</td>
      <td>1</td>
      <td>46.699997</td>
      <td>10</td>
      <td>0.94</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Japanese</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>8</td>
      <td>0.94</td>
      <td>110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>European</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>7</td>
      <td>0.94</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Korean</td>
      <td>0</td>
      <td>1</td>
      <td>46.699997</td>
      <td>8</td>
      <td>0.94</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>American</td>
      <td>1</td>
      <td>1</td>
      <td>26.100000</td>
      <td>10</td>
      <td>0.95</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



### Adding Observables, Method 1: Observables Derived from Columns of the Main Dataset


```python
user_observable_columns=["gender", "income"]
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper
data_wrapper_from_columns = EasyDatasetWrapper(
    main_data=car_choice,
    purchase_record_column='record_id',
    choice_column='purchase',
    item_name_column='car',
    user_index_column='consumer_id',
    session_index_column='session_id',
    user_observable_columns=['gender', 'income'],
    item_observable_columns=['speed'],
    session_observable_columns=['discount'],
    itemsession_observable_columns=['price'])

data_wrapper_from_columns.summary()
dataset = data_wrapper_from_columns.choice_dataset
# ChoiceDataset(label=[], item_index=[885], provided_num_items=[], user_index=[885], session_index=[885], item_availability=[885, 4], item_speed=[4, 1], user_gender=[885, 1], user_income=[885, 1], session_discount=[885, 1], itemsession_price=[885, 4, 1], device=cpu)
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.
    * purchase record index range: [1 2 3] ... [883 884 885]
    * Space of 4 items:
                       0         1         2       3
    item name  American  European  Japanese  Korean
    * Number of purchase records/cases: 885.
    * Preview of main data frame:
          record_id  session_id  consumer_id       car  purchase  gender  \
    0             1           1            1  American         1       1   
    1             1           1            1  Japanese         0       1   
    2             1           1            1  European         0       1   
    3             1           1            1    Korean         0       1   
    4             2           2            2  American         1       1   
    ...         ...         ...          ...       ...       ...     ...   
    3155        884         884          884  Japanese         1       1   
    3156        884         884          884  European         0       1   
    3157        885         885          885  American         1       1   
    3158        885         885          885  Japanese         0       1   
    3159        885         885          885  European         0       1   
    
             income  speed  discount  price  
    0     46.699997     10      0.94     90  
    1     46.699997      8      0.94    110  
    2     46.699997      7      0.94     50  
    3     46.699997      8      0.94     10  
    4     26.100000     10      0.95    100  
    ...         ...    ...       ...    ...  
    3155  20.900000      8      0.89    100  
    3156  20.900000      7      0.89     40  
    3157  30.600000     10      0.81    100  
    3158  30.600000      8      0.81     50  
    3159  30.600000      7      0.81     40  
    
    [3160 rows x 10 columns]
    * Preview of ChoiceDataset:
    ChoiceDataset(label=[], item_index=[885], user_index=[885], session_index=[885], item_availability=[885, 4], item_speed=[4, 1], user_gender=[885, 1], user_income=[885, 1], session_discount=[885, 1], itemsession_price=[885, 4, 1], device=cpu)


### Adding Observables, Method 2: Added as Separated DataFrames


```python
# create dataframes for gender and income. The dataframe for user-specific observable needs to have the `consumer_id` column.
gender = car_choice.groupby('consumer_id')['gender'].first().reset_index()
income = car_choice.groupby('consumer_id')['income'].first().reset_index()
# alternatively, put gender and income in the same dataframe.
gender_and_income = car_choice.groupby('consumer_id')[['gender', 'income']].first().reset_index()
# speed as item observable, the dataframe requires a `car` column.
speed = car_choice.groupby('car')['speed'].first().reset_index()
# discount as session observable. the dataframe requires a `session_id` column.
discount = car_choice.groupby('session_id')['discount'].first().reset_index()
# create the price as itemsession observable, the dataframe requires both `car` and `session_id` columns.
price = car_choice[['car', 'session_id', 'price']]
# fill in NANs for (session, item) pairs that the item was not available in that session.
price = price.pivot('car', 'session_id', 'price').melt(ignore_index=False).reset_index()
```


```python
data_wrapper_from_dataframes = EasyDatasetWrapper(
    main_data=car_choice,
    purchase_record_column='record_id',
    choice_column='purchase',
    item_name_column='car',
    user_index_column='consumer_id',
    session_index_column='session_id',
    user_observable_data={'gender': gender, 'income': income},
    # alternatively, supply gender and income as a single dataframe.
    # user_observable_data={'gender_and_income': gender_and_income},
    item_observable_data={'speed': speed},
    session_observable_data={'discount': discount},
    itemsession_observable_data={'price': price})

# the second method creates exactly the same ChoiceDataset as the previous method.
assert data_wrapper_from_dataframes.choice_dataset == data_wrapper_from_columns.choice_dataset
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.



```python
data_wrapper_mixed = EasyDatasetWrapper(
    main_data=car_choice,
    purchase_record_column='record_id',
    choice_column='purchase',
    item_name_column='car',
    user_index_column='consumer_id',
    session_index_column='session_id',
    user_observable_data={'gender': gender, 'income': income},
    item_observable_data={'speed': speed},
    session_observable_data={'discount': discount},
    itemsession_observable_columns=['price'])

# these methods create exactly the same choice dataset.
assert data_wrapper_mixed.choice_dataset == data_wrapper_from_columns.choice_dataset == data_wrapper_from_dataframes.choice_dataset
```

    Creating choice dataset from stata format data-frames...
    Note: choice sets of different sizes found in different purchase records: {'size 4': 'occurrence 505', 'size 3': 'occurrence 380'}
    Finished Creating Choice Dataset.


## Constructing a Choice Dataset, Method 2: Building from Tensors


```python
N = 10_000
num_users = 10
num_items = 4
num_sessions = 500


user_obs = torch.randn(num_users, 128)
item_obs = torch.randn(num_items, 64)
session_obs = torch.randn(num_sessions, 10)
itemsession_obs = torch.randn(num_sessions, num_items, 12)
item_index = torch.LongTensor(np.random.choice(num_items, size=N))
user_index = torch.LongTensor(np.random.choice(num_users, size=N))
session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
item_availability = torch.ones(num_sessions, num_items).bool()

dataset = ChoiceDataset(
    # required:
    item_index=item_index,
    # optional:
    user_index=user_index, session_index=session_index, item_availability=item_availability,
    # observable tensors are supplied as keyword arguments with special prefixes.
    user_obs=user_obs, item_obs=item_obs, session_obs=session_obs, itemsession_obs=itemsession_obs)
```


```python
print(dataset)
```

    ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], itemsession_obs=[500, 4, 12], device=cpu)


## Functionalities of the Choice Dataset


```python
print(f'{dataset.num_users=:}')
# dataset.num_users=10
print(f'{dataset.num_items=:}')
# dataset.num_items=4
print(f'{dataset.num_sessions=:}')
# dataset.num_sessions=500
print(f'{len(dataset)=:}')
# len(dataset)=10000
```

    dataset.num_users=10
    dataset.num_items=4
    dataset.num_sessions=500
    len(dataset)=10000



```python
# clone
print(dataset.item_index[:10])
# tensor([2, 2, 3, 1, 3, 2, 2, 1, 0, 1])
dataset_cloned = dataset.clone()
# modify the cloned dataset.
dataset_cloned.item_index = 99 * torch.ones(num_sessions)
print(dataset_cloned.item_index[:10])
# the cloned dataset is changed.
# tensor([99., 99., 99., 99., 99., 99., 99., 99., 99., 99.])
print(dataset.item_index[:10])
# the original dataset does not change.
# tensor([2, 2, 3, 1, 3, 2, 2, 1, 0, 1])
```

    tensor([1, 2, 0, 0, 3, 0, 3, 1, 0, 2])
    tensor([99., 99., 99., 99., 99., 99., 99., 99., 99., 99.])
    tensor([1, 2, 0, 0, 3, 0, 3, 1, 0, 2])



```python
# move to device
print(f'{dataset.device=:}')
# dataset.device=cpu
print(f'{dataset.device=:}')
# dataset.device=cpu
print(f'{dataset.user_index.device=:}')
# dataset.user_index.device=cpu
print(f'{dataset.session_index.device=:}')
# dataset.session_index.device=cpu

dataset = dataset.to('cuda')

print(f'{dataset.device=:}')
# dataset.device=cuda:0
print(f'{dataset.item_index.device=:}')
# dataset.item_index.device=cuda:0
print(f'{dataset.user_index.device=:}')
# dataset.user_index.device=cuda:0
print(f'{dataset.session_index.device=:}')
# dataset.session_index.device=cuda:0

dataset._check_device_consistency()
```

    dataset.device=cpu
    dataset.device=cpu
    dataset.user_index.device=cpu
    dataset.session_index.device=cpu
    dataset.device=cuda:0
    dataset.item_index.device=cuda:0
    dataset.user_index.device=cuda:0
    dataset.session_index.device=cuda:0



```python
def print_dict_shape(d):
    for key, val in d.items():
        if torch.is_tensor(val):
            print(f'dict.{key}.shape={val.shape}')
print_dict_shape(dataset.x_dict)
```

    dict.user_obs.shape=torch.Size([10000, 4, 128])
    dict.item_obs.shape=torch.Size([10000, 4, 64])
    dict.session_obs.shape=torch.Size([10000, 4, 10])
    dict.itemsession_obs.shape=torch.Size([10000, 4, 12])



```python
# __getitem__ to get batch.
# pick 5 random sessions as the mini-batch.
dataset = dataset.to('cpu')
indices = torch.Tensor(np.random.choice(len(dataset), size=5, replace=False)).long()
print(indices)
# tensor([1118,  976, 1956,  290, 8283])
subset = dataset[indices]
print(dataset)
# ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
print(subset)
# ChoiceDataset(label=[], item_index=[5], user_index=[5], session_index=[5], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
```

    tensor([1865, 6236, 4548, 5486, 1771])
    ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], itemsession_obs=[500, 4, 12], device=cpu)
    ChoiceDataset(label=[], item_index=[5], user_index=[5], session_index=[5], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], itemsession_obs=[500, 4, 12], device=cpu)



```python
print(subset.item_index)
# tensor([0, 1, 0, 0, 0])
print(dataset.item_index[indices])
# tensor([0, 1, 0, 0, 0])

subset.item_index += 1  # modifying the batch does not change the original dataset.

print(subset.item_index)
# tensor([1, 2, 1, 1, 1])
print(dataset.item_index[indices])
# tensor([0, 1, 0, 0, 0])
```

    tensor([1, 1, 2, 2, 2])
    tensor([1, 1, 2, 2, 2])
    tensor([2, 2, 3, 3, 3])
    tensor([1, 1, 2, 2, 2])



```python
print(subset.item_obs[0, 0])
# tensor(-1.5811)
print(dataset.item_obs[0, 0])
# tensor(-1.5811)

subset.item_obs += 1
print(subset.item_obs[0, 0])
# tensor(-0.5811)
print(dataset.item_obs[0, 0])
# tensor(-1.5811)
```

    tensor(-0.6857)
    tensor(-0.6857)
    tensor(0.3143)
    tensor(-0.6857)



```python
print(id(subset.item_index))
# 140339656298640
print(id(dataset.item_index[indices]))
# 140339656150528
# these two are different objects in memory.
```

    139766186199856
    139766186203856


## Chaining Multiple Datasets with JointDataset


```python
item_level_dataset = dataset.clone()
nest_level_dataset = dataset.clone()
joint_dataset = JointDataset(
    item=item_level_dataset,
    nest=nest_level_dataset)

print(joint_dataset)
```

    JointDataset with 2 sub-datasets: (
    	item: ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], itemsession_obs=[500, 4, 12], device=cpu)
    	nest: ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], itemsession_obs=[500, 4, 12], device=cpu)
    )



```python
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
shuffle = False  # for demonstration purpose.
batch_size = 32

# Create sampler.
sampler = BatchSampler(
    RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
    batch_size=batch_size,
    drop_last=False)

dataloader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler,
                                         collate_fn=lambda x: x[0],
                                         pin_memory=(dataset.device == 'cpu'))
```


```python
print(f'{item_obs.shape=:}')
# item_obs.shape=torch.Size([4, 64])
item_obs_all = item_obs.view(1, num_items, -1).expand(len(dataset), -1, -1)
item_obs_all = item_obs_all.to(dataset.device)
item_index_all = item_index.to(dataset.device)
print(f'{item_obs_all.shape=:}')
# item_obs_all.shape=torch.Size([10000, 4, 64])
```

    item_obs.shape=torch.Size([4, 64])
    item_obs_all.shape=torch.Size([10000, 4, 64])



```python
for i, batch in enumerate(dataloader):
    first, last = i * batch_size, min(len(dataset), (i + 1) * batch_size)
    idx = torch.arange(first, last)
    assert torch.all(item_obs_all[idx, :, :] == batch.x_dict['item_obs'])
    assert torch.all(item_index_all[idx] == batch.item_index)
```


```python
batch.x_dict['item_obs'].shape
# torch.Size([16, 4, 64])
```




    torch.Size([16, 4, 64])




```python
print_dict_shape(dataset.x_dict)
# dict.user_obs.shape=torch.Size([10000, 4, 128])
# dict.item_obs.shape=torch.Size([10000, 4, 64])
# dict.session_obs.shape=torch.Size([10000, 4, 10])
# dict.price_obs.shape=torch.Size([10000, 4, 12])
```

    dict.user_obs.shape=torch.Size([10000, 4, 128])
    dict.item_obs.shape=torch.Size([10000, 4, 64])
    dict.session_obs.shape=torch.Size([10000, 4, 10])
    dict.itemsession_obs.shape=torch.Size([10000, 4, 12])



```python
dataset.__len__()
# 10000
```




    10000



# Conditional Logit Model


```python
dataset = load_mode_canada_dataset() 
```

    No `session_index` is provided, assume each choice instance is in its own session.



```python
dataset
```




    ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], itemsession_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], itemsession_ivt=[2779, 4, 1], device=cpu)




```python
model = ConditionalLogitModel(
    formula='(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)',
    dataset=dataset,
    num_items=4)
```


```python
model = ConditionalLogitModel(
    coef_variation_dict={'itemsession_cost_freq_ovt': 'constant',
                         'session_income': 'item',
                         'itemsession_ivt': 'item-full',
                         'intercept': 'item'},
    num_param_dict={'itemsession_cost_freq_ovt': 3,
                    'session_income': 1,
                    'itemsession_ivt': 1,
                    'intercept': 1},
    num_items=4)
```


```python
model = ConditionalLogitModel(
    coef_variation_dict={'itemsession_cost_freq_ovt': 'constant',
                         'session_income': 'item',
                         'itemsession_ivt': 'item-full',
                         'intercept': 'item'},
    num_param_dict={'itemsession_cost_freq_ovt': 3,
                    'session_income': 1,
                    'itemsession_ivt': 1,
                    'intercept': 1},
    num_items=4,
    regularization="L1", regularization_weight=0.5)
```


```python
from torch_choice import run
run(model, dataset, batch_size=-1, learning_rate=0.01, num_epochs=1000, model_optimizer="LBFGS")
```

    GPU available: True (cuda), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    
      | Name  | Type                  | Params
    ------------------------------------------------
    0 | model | ConditionalLogitModel | 13    
    ------------------------------------------------
    13        Trainable params
    0         Non-trainable params
    13        Total params
    0.000     Total estimated model params size (MB)


    ==================== model received ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_cost_freq_ovt[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cpu).
        (session_income[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (itemsession_ivt[item-full]): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_cost_freq_ovt[constant]] with 3 parameters, with constant level variation.
    X[session_income[item]] with 1 parameters, with item level variation.
    X[itemsession_ivt[item-full]] with 1 parameters, with item-full level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu
    ==================== data set received ====================
    [Train dataset] ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], itemsession_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], itemsession_ivt=[2779, 4, 1], device=cuda:0)
    [Validation dataset] None
    [Test dataset] None



    Training: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Time taken for training: 23.79274034500122
    Skip testing, no test dataset is provided.
    ==================== model results ====================
    Log-likelihood: [Training] -1874.63623046875, [Validation] N/A, [Test] N/A
    
    | Coefficient                           |   Estimation |   Std. Err. |       z-value |    Pr(>|z|) | Significance   |
    |:--------------------------------------|-------------:|------------:|--------------:|------------:|:---------------|
    | itemsession_cost_freq_ovt[constant]_0 | -0.0372827   |  0.00709507 |  -5.25473     | 1.48243e-07 | ***            |
    | itemsession_cost_freq_ovt[constant]_1 |  0.0934419   |  0.00509598 |  18.3364      | 0           | ***            |
    | itemsession_cost_freq_ovt[constant]_2 | -0.0427658   |  0.00322177 | -13.274       | 0           | ***            |
    | session_income[item]_0                | -0.0862181   |  0.0183006  |  -4.71123     | 2.46226e-06 | ***            |
    | session_income[item]_1                | -0.0269176   |  0.00384876 |  -6.99383     | 2.67497e-12 | ***            |
    | session_income[item]_2                | -0.0370536   |  0.0040631  |  -9.11952     | 0           | ***            |
    | itemsession_ivt[item-full]_0          |  0.0593798   |  0.0100866  |   5.88698     | 3.93307e-09 | ***            |
    | itemsession_ivt[item-full]_1          | -0.00634217  |  0.0042797  |  -1.48192     | 0.138361    |                |
    | itemsession_ivt[item-full]_2          | -0.00583443  |  0.00189438 |  -3.07986     | 0.00207095  | **             |
    | itemsession_ivt[item-full]_3          | -0.00137758  |  0.00118694 |  -1.16061     | 0.245801    |                |
    | intercept[item]_0                     | -1.91536e-07 |  1.26821    |  -1.51029e-07 | 1           |                |
    | intercept[item]_1                     |  1.32858     |  0.703745   |   1.88787     | 0.0590437   |                |
    | intercept[item]_2                     |  2.82011     |  0.618218   |   4.56167     | 5.07483e-06 | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1





    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_cost_freq_ovt[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cpu).
        (session_income[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (itemsession_ivt[item-full]): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_cost_freq_ovt[constant]] with 3 parameters, with constant level variation.
    X[session_income[item]] with 1 parameters, with item level variation.
    X[itemsession_ivt[item-full]] with 1 parameters, with item-full level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu




```python
! tensorboard --logdir ./lightning_logs --port 6006
```

    2023-05-14 21:27:08.325570: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX512F AVX512_VNNI
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-05-14 21:27:08.389919: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-05-14 21:27:09.137097: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2023-05-14 21:27:09.184401: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2023-05-14 21:27:09.185605: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.11.0 at http://localhost:6006/ (Press CTRL+C to quit)
    ^C


# Nested Logit Model
The code demo for nested logit models in the paper was abstract, please refer to the nested-logit model tutorial for executable code.


