# Tutorial: Data Management
**Author: Tianyu Du (tianyudu@stanford.edu)**

**Note**: please go through the introduction tutorial [here](https://gsbdbi.github.io/torch-choice/intro/) before proceeding.

This notebook aims to help users understand the functionality of `ChoiceDataset` object.
The `ChoiceDataset` is an instance of the more general PyTorch dataset object holding information of consumer choices. The `ChoiceDataset` offers easy, clean and efficient data management. The Jupyter-notebook version of this tutorial can be found [here](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/data_management.ipynb).

This tutorial provides in-depth explanations on how the `torch-choice` library manages data. We are also providing an easy-to-use data wrapper converting long-format dataset to `ChoiceDataset` [here](https://gsbdbi.github.io/torch-choice/easy_data_management/), you can harness the `torch-choice` library without going through this tutorial. 

**Note**: since this package was initially proposed for modelling consumer choices, attribute names of `ChoiceDataset` are borrowed from the consumer choice literature.


### Observables
Observables are tensors with specific shapes, we classify observables into four categories based on their variations.

#### Basic Usage
Optionally, the researcher can incorporate observables of, for example, users and items. Currently, the package support the following types of observables, where $K_{...}$ denote the number of observables.

1. `user_obs` $\in \mathbb{R}^{U\times K_{user}}$: user observables such as user age.
2. `item_obs` $\in \mathbb{R}^{I\times K_{item}}$: item observables such as item quality.
3. `session_obs` $\in \mathbb{R}^{S \times K_{session}}$: session observable such as whether the purchase was made on weekdays.
4. `price_obs` $\in \mathbb{R}^{S \times I \times K_{price}}$, price observables are values depending on **both** session and item such as the price of item.

The researcher should supply them with as appropriate keyword arguments while constructing the `ChoiceDataset` object.

#### Advanced Usage: Additional Observables
In some cases, the researcher may wish to handle different parts of `user_obs` (or other observable tensors) differently.
For example, the researcher wishes to model the utility for user $u$ to purchase item $i$ in session $s$ as the following:

$$
U_{usi} = \beta_{i} X^{(u)}_{user\ income} + \gamma X^{(u)}_{user\ market\ membership}
$$

The coefficient for user income is item-specific so that it captures the nature of the product (i.e., a luxury or an essential good). Additionally, the utility representation admits an user market membership becomes shoppers with active memberships tend to purchase more, and the coefficient of this term is constant across all items.
As we will cover later in the modelling section, we need to supply two user observable tensors in this case for the model to build coefficient with different levels of variations (i.e., item-specific coefficients versus constant coefficients). In this case, the researcher needs to supply two tensors `user_income` and `user_market_membership` as keyword arguments to the `ChoiceDataset` constructor.
The `ChoiceDataset` handles multiple user/item/session/price observables internally, for example, every keyword arguments passed into `ChoiceDataset` with name starting with `item_` (except for the reserved `item_availability`) will be treated as item observable tensors. All keywords with names starting `user_`, `session_` and `price_` (except for reserved names like `user_index` and `session_index` mentioned above) will be interpreted as user/session/price observable tensors.


```python
# import required dependencies.
import numpy as np
import torch
from torch_choice.data import ChoiceDataset, JointDataset
```


```python
# let's get a helper
def print_dict_shape(d):
    for key, val in d.items():
        if torch.is_tensor(val):
            print(f'dict.{key}.shape={val.shape}')
```

## Creating  `ChoiceDataset` Object


```python
# Feel free to modify it as you want.
num_users = 10
num_items = 4
num_sessions = 500

length_of_dataset = 10000
```

### Step 1: Generate some random purchase records and observables
We will be creating a randomly generated dataset with 10000 purchase records from 10 users, 4 items and 500 sessions.

The first step is to randomly generate the purchase records using the following code. For simplicity, we assume all items are available in all sessions.


```python
# create observables/features, the number of parameters are arbitrarily chosen.
# generate 128 features for each user, e.g., race, gender.
user_obs = torch.randn(num_users, 128)
# generate 64 features for each user, e.g., quality.
item_obs = torch.randn(num_items, 64)
# generate 10 features for each session, e.g., weekday indicator. 
session_obs = torch.randn(num_sessions, 10)
# generate 12 features for each session user pair, e.g., the budget of that user at the shopping day.
price_obs = torch.randn(num_sessions, num_items, 12)
```

We then generate random observable tensors for users, items, sessions and price observables, the size of observables of each type (i.e., the last dimension in the shape) is arbitrarily chosen.

**Notes on Encodings** Since we will be using PyTorch to train our model, we represent their identities with *consecutive* integer values instead of the raw human-readable names of items (e.g., Dell 24-inch LCD monitor). Similarly, you would need to encode user indices and session indices as well.
Raw item names can be encoded easily with [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) (The [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) works as well).


```python
item_index = torch.LongTensor(np.random.choice(num_items, size=length_of_dataset))
user_index = torch.LongTensor(np.random.choice(num_users, size=length_of_dataset))
session_index = torch.LongTensor(np.random.choice(num_sessions, size=length_of_dataset))

# assume all items are available in all sessions.
item_availability = torch.ones(num_sessions, num_items).bool()
```

### Step 2: Initialize the `ChoiceDataset`.
You can construct a choice set using the following code, which manage all information for you.


```python
dataset = ChoiceDataset(
    # pre-specified keywords of __init__
    item_index=item_index,  # required.
    # optional:
    user_index=user_index,
    session_index=session_index,
    item_availability=item_availability,
    # additional keywords of __init__
    user_obs=user_obs,
    item_obs=item_obs,
    session_obs=session_obs,
    price_obs=price_obs)
```

## What you can do with the `ChoiceDataset`?

### `print(dataset)` and `dataset.__str__`
The command `print(dataset)` will provide a quick overview of shapes of tensors included in the object as well as where the dataset is located (i.e., host memory or GPU memory).


```python
print(dataset)
```

    ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)


### `dataset.num_{users, items, sessions}`
You can use the `num_{users, items, sessions}` attribute to obtain the number of users, items, and sessions, they are determined automatically from the `{user, item, session}_obs` tensors provided while initializing the dataset object.

**Note**: the print `=:` operator requires Python3.8 or higher, you can remove `=:` if you are using an earlier copy of Python.


```python
print(f'{dataset.num_users=:}')
print(f'{dataset.num_items=:}')
print(f'{dataset.num_sessions=:}')
print(f'{len(dataset)=:}')
```

    dataset.num_users=10
    dataset.num_items=4
    dataset.num_sessions=500
    len(dataset)=10000


### `dataset.clone()`
The `ChoiceDataset` offers a `clone` method allow you to make copy of the dataset, you can modify the cloned dataset arbitrarily without changing the original dataset.


```python
# clone
print(dataset.item_index[:10])
dataset_cloned = dataset.clone()
dataset_cloned.item_index = 99 * torch.ones(num_sessions)
print(dataset_cloned.item_index[:10])
print(dataset.item_index[:10])  # does not change the original dataset.
```

    tensor([2, 2, 3, 1, 3, 2, 2, 1, 0, 1])
    tensor([99., 99., 99., 99., 99., 99., 99., 99., 99., 99.])
    tensor([2, 2, 3, 1, 3, 2, 2, 1, 0, 1])


### `dataset.to('cuda')` and `dataset._check_device_consistency()`.
One key advantage of the `torch_choice` and `bemb` is their compatibility with GPUs, you can easily move tensors in a `ChoiceDataset` object between host memory (i.e., cpu memory) and device memory (i.e., GPU memory) using `dataset.to()` method.
Please note that the following code runs only if your machine has a compatible GPU and GPU-compatible version of PyTorch installed.

Similarly, one can move data to host-memory using `dataset.to('cpu')`.
The dataset also provides a `dataset._check_device_consistency()` method to check if all tensors are on the same device.
If we only move the `label` to cpu without moving other tensors, this will result in an error message.


```python
# move to device
print(f'{dataset.device=:}')
print(f'{dataset.device=:}')
print(f'{dataset.user_index.device=:}')
print(f'{dataset.session_index.device=:}')

dataset = dataset.to('cuda')

print(f'{dataset.device=:}')
print(f'{dataset.item_index.device=:}')
print(f'{dataset.user_index.device=:}')
print(f'{dataset.session_index.device=:}')
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
dataset._check_device_consistency()
```


```python
# # NOTE: this cell will result errors, this is intentional.
dataset.item_index = dataset.item_index.to('cpu')
dataset._check_device_consistency()
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-56-40d626c6d436> in <module>
          1 # # NOTE: this cell will result errors, this is intentional.
          2 dataset.item_index = dataset.item_index.to('cpu')
    ----> 3 dataset._check_device_consistency()
    

    ~/Development/torch-choice/torch_choice/data/choice_dataset.py in _check_device_consistency(self)
        180                 devices.append(val.device)
        181         if len(set(devices)) > 1:
    --> 182             raise Exception(f'Found tensors on different devices: {set(devices)}.',
        183                             'Use dataset.to() method to align devices.')
        184 


    Exception: ("Found tensors on different devices: {device(type='cuda', index=0), device(type='cpu')}.", 'Use dataset.to() method to align devices.')



```python
# create dictionary inputs for model.forward()
# collapse to a dictionary object.
print_dict_shape(dataset.x_dict)
```

    dict.user_obs.shape=torch.Size([10000, 4, 128])
    dict.item_obs.shape=torch.Size([10000, 4, 64])
    dict.session_obs.shape=torch.Size([10000, 4, 10])
    dict.price_obs.shape=torch.Size([10000, 4, 12])


### Subset method
One can use `dataset[indices]` with `indices` as an integer-valued tensor or array to get the corresponding rows of the dataset.
The example code block below queries the 6256-th, 4119-th, 453-th, 5520-th, and 1877-th row of the dataset object.
The `item_index`, `user_index`, `session_index` of the resulted subset will be different from the original dataset, but other tensors will be the same.


```python
# __getitem__ to get batch.
# pick 5 random sessions as the mini-batch.
dataset = dataset.to('cpu')
indices = torch.Tensor(np.random.choice(len(dataset), size=5, replace=False)).long()
print(indices)
subset = dataset[indices]
print(dataset)
print(subset)
# print_dict_shape(subset.x_dict)

# assert torch.all(dataset.x_dict['price_obs'][indices, :, :] == subset.x_dict['price_obs'])
# assert torch.all(dataset.item_index[indices] == subset.item_index)
```

    tensor([1118,  976, 1956,  290, 8283])
    ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
    ChoiceDataset(label=[], item_index=[5], user_index=[5], session_index=[5], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)


The subset method internally creates a copy of the datasets so that any modification applied on the subset will **not** be reflected on the original dataset.
The researcher can feel free to do in-place modification to the subset.


```python
print(subset.item_index)
print(dataset.item_index[indices])

subset.item_index += 1  # modifying the batch does not change the original dataset.

print(subset.item_index)
print(dataset.item_index[indices])
```

    tensor([0, 1, 0, 0, 0])
    tensor([0, 1, 0, 0, 0])
    tensor([1, 2, 1, 1, 1])
    tensor([0, 1, 0, 0, 0])



```python
print(subset.item_obs[0, 0])
print(dataset.item_obs[0, 0])
subset.item_obs += 1
print(subset.item_obs[0, 0])
print(dataset.item_obs[0, 0])
```

    tensor(-1.5811)
    tensor(-1.5811)
    tensor(-0.5811)
    tensor(-1.5811)



```python
print(id(subset.item_index))
print(id(dataset.item_index[indices]))
```

    140339656298640
    140339656150528


## Using Pytorch dataloader for the training loop.
The `ChoiceDataset` object natively support batch samplers from PyTorch. For demonstration purpose, we turned off the shuffling option.


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
                                         num_workers=1,
                                         collate_fn=lambda x: x[0],
                                         pin_memory=(dataset.device == 'cpu'))
```


```python
print(f'{item_obs.shape=:}')
item_obs_all = item_obs.view(1, num_items, -1).expand(len(dataset), -1, -1)
item_obs_all = item_obs_all.to(dataset.device)
item_index_all = item_index.to(dataset.device)
print(f'{item_obs_all.shape=:}')
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
```




    torch.Size([16, 4, 64])




```python
print_dict_shape(dataset.x_dict)
```

    dict.user_obs.shape=torch.Size([10000, 4, 128])
    dict.item_obs.shape=torch.Size([10000, 4, 64])
    dict.session_obs.shape=torch.Size([10000, 4, 10])
    dict.price_obs.shape=torch.Size([10000, 4, 12])



```python
dataset.__len__()
```




    10000



## Chaining Multiple Datasets: `JointDataset` Examples


```python
dataset1 = dataset.clone()
dataset2 = dataset.clone()
joint_dataset = JointDataset(the_dataset=dataset1, another_dataset=dataset2)
```


```python
joint_dataset
```




    JointDataset with 2 sub-datasets: (
    	the_dataset: ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
    	another_dataset: ChoiceDataset(label=[], item_index=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
    )


