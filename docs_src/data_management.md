# Tutorial: Data Management
**Author: Tianyu Du (tianyudu@stanford.edu)**

**Note**: please go through the introduction tutorial [here](https://gsbdbi.github.io/torch-choice/intro/) before proceeding.

This notebook aims to help users understand the functionality of `ChoiceDataset` object.
The `ChoiceDataset` is an instance of the more general PyTorch dataset object holding information of consumer choices. The `ChoiceDataset` offers easy, clean and efficient data management. The Jupyter-notebook version of this tutorial can be found [here](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/data_management.ipynb).

This tutorial provides in-depth explanations on how the `torch-choice` library manages data. We are also providing an easy-to-use data wrapper converting long-format dataset to `ChoiceDataset` [here](https://gsbdbi.github.io/torch-choice/easy_data_management/), you can harness the `torch-choice` library without going through this tutorial. 

**Note**: since this package was initially proposed for modelling consumer choices, attribute names of `ChoiceDataset` are borrowed from the consumer choice literature.

**Note**: PyTorch uses the term **tensor** to denote high dimensional matrices, we will be using **tensor** and **matrix** interchangeably.

After walking through this tutorial, you should be abel to initiate a `ChoiceDataset` object as the following and use it to manage data.
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

### Observables
Observables are tensors with specific shapes, we classify observables into four categories based on their variations.

#### Basic Usage
Optionally, the researcher can incorporate observables of, for example, users and items. Currently, the package support the following types of observables, where $K_{...}$ denote the number of observables.

1. `user_obs` $\in \mathbb{R}^{U\times K_{user}}$: user observables such as user age.
2. `item_obs` $\in \mathbb{R}^{I\times K_{item}}$: item observables such as item quality.
3. `session_obs` $\in \mathbb{R}^{S \times K_{session}}$: session observable such as whether the purchase was made on weekdays.
4. `price_obs` $\in \mathbb{R}^{S \times I \times K_{price}}$, price observables are values depending on **both** session and item such as the price of item.

The researcher should supply them with as appropriate keyword arguments while constructing the `ChoiceDataset` object.

#### (Optional) Advanced Usage: Additional Observables
In some cases, the researcher have multiple sets of user (or item, or session, or price) observables, say *user income* (a scalar variable) and *user market membership*. The *user income* a matrix in $\mathbb{R}^{U\times 1}$. Further, suppose there are four types of market membership: no-membership, silver-membership, gold-membership, and diamond-membership. The *user market membership* is a binary matrix in $\{0, 1\}^{U\times 4}$ if we one-hot encode users' membership status.

In this case, the researcher can either
1. concatenate `user_income` and `user_market_membership` to a $\mathbb{R}^{U\times (1+4)}$ matrix and supply it as a single `user_obs` as the following:
```python
dataset = ChoiceDataset(..., user_obs=torch.cat([user_income, user_market_membership], dim=1), ...)
```
2. Or, supply these two sets of observables separately, namely a `user_income` $\in \mathbb{R}^{U \times 1}$ matrix and a `user_market_membership` $\in \mathbb{R}^{U \times 4}$ matrix as the following:
```python
dataset = ChoiceDataset(..., user_income=user_income, user_market_membership=user_market_membership, ...)
```

Supplying two separate sets of observables is particularly useful when the researcher wants different kinds of coefficients for different kinds of observables.

For example, the researcher wishes to model the utility for user $u$ to purchase item $i$ in session $s$ as the following:

$$
U_{usi} = \beta_{i} X^{(u)}_{user\ income} + \gamma X^{(u)}_{user\ market\ membership} + \varepsilon
$$

Please note that the $\beta_i$ coefficient has an $i$ subscript, which means it's item specific. The $\gamma$ coefficient has no subscript, which means it's the same for all items.

The coefficient for user income is item-specific so that it captures the nature of the product (i.e., a luxury or an essential good). Additionally, the utility representation admits an user market membership becomes shoppers with active memberships tend to purchase more, and the coefficient of this term is constant across all items.

As we will cover later in the modelling section, we need to supply two user observable tensors in this case for the model to build coefficient with different levels of variations (i.e., item-specific coefficients versus constant coefficients). In this case, the researcher needs to supply two tensors `user_income` and `user_market_membership` as keyword arguments to the `ChoiceDataset` constructor.

Generally, the `ChoiceDataset` handles multiple user/item/session/price observables internally, the `ChoiceDataset` class identifies the variation of observables by their prefixes. For example, every keyword arguments passed into `ChoiceDataset` with name starting with `item_` (except for the reserved `item_availability`) will be treated as item observable tensors.
Similarly, all keywords with names starting `user_`, `session_` and `price_` (except for reserved names like `user_index` and `session_index` mentioned above) will be interpreted as user/session/price observable tensors.


```python
# import required dependencies.
import numpy as np
import pandas as pd
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

We use the term **purchase record** to denote the observation in the dataset due to the convention in Stata documentation (because *observation* meant something else in the Stata documentation and we don't want to confuse existing Stata users).

As mentioned in the introduction tutorial, one purchase record consists of *who* (i.e., user) bought *what* (i.e., item) *when* and *where* (i.e., session). 

The length of the dataset equals the number of purchase records in it.

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


### `dataset.summary()`
The `summary` method provides preliminary summarization of the dataset.


```python
print(pd.DataFrame(dataset.user_index).value_counts())
```

    4    1038
    8    1035
    5    1024
    1    1010
    2     997
    0     990
    6     981
    9     980
    3     974
    7     971
    dtype: int64



```python
print(pd.DataFrame(dataset.item_index).value_counts())
```

    0    2575
    1    2539
    2    2467
    3    2419
    dtype: int64



```python
dataset.summary()
```

    ChoiceDataset with 500 sessions, 4 items, 10 users, 10000 purchase records (observations) .
    The most frequent user is 4 with 1038 observations; the least frequent user is 7 with 971 observations; on average, there are 1000.00 observations per user.
    5 most frequent users are: 4(1038 times), 8(1035 times), 5(1024 times), 1(1010 times), 2(997 times).
    5 least frequent users are: 7(971 times), 3(974 times), 9(980 times), 6(981 times), 0(990 times).
    The most frequent item is 0, it was chosen 2575 times; the least frequent item is 3 it was 2419 times; on average, each item was purchased 2500.00 times.
    4 most frequent items are: 0(2575 times), 1(2539 times), 2(2467 times), 3(2419 times).
    4 least frequent items are: 3(2419 times), 2(2467 times), 1(2539 times), 0(2575 times).
    Attribute Summaries:
    Observable Tensor 'user_obs' with shape torch.Size([10, 128])
                 0          1          2          3          4          5    \
    count  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000   
    mean    0.687878  -0.339077  -0.375829   0.086242   0.250604  -0.344643   
    std     0.738520   1.259936   0.844018   0.766233   0.802785   0.645239   
    min    -0.578577  -2.135251  -1.335928  -0.911508  -1.396776  -1.519729   
    25%     0.264708  -0.889820  -0.845100  -0.414891  -0.132619  -0.699887   
    50%     0.902505  -0.603065  -0.638757  -0.289223   0.297693  -0.405371   
    75%     1.155211   0.021188  -0.190907   0.712183   0.768554   0.117107   
    max     1.623162   2.217712   1.624211   1.252059   1.273116   0.571998   
    
                 6          7          8          9    ...        118        119  \
    count  10.000000  10.000000  10.000000  10.000000  ...  10.000000  10.000000   
    mean    0.423672   0.325855   0.258114  -0.199072  ...  -0.165618  -0.378175   
    std     1.304160   0.815934   0.938925   1.344848  ...   1.135625   0.940863   
    min    -1.440672  -1.068176  -1.280547  -2.819688  ...  -1.567793  -1.604171   
    25%    -0.535055   0.051598  -0.178302  -0.801871  ...  -1.114392  -1.066492   
    50%     0.502826   0.369002   0.230939  -0.576039  ...  -0.114789  -0.587483   
    75%     1.227700   0.899518   0.740881   0.820789  ...   0.602045   0.160254   
    max     2.462891   1.440098   1.828760   1.866570  ...   1.854828   1.386001   
    
                 120        121        122        123        124        125  \
    count  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000   
    mean   -0.557321   0.402392  -0.070746  -0.770201   0.594842   0.572671   
    std     1.128886   0.899030   0.757537   1.044478   0.956856   0.883374   
    min    -3.131332  -0.907885  -1.296398  -2.159384  -1.244177  -0.462607   
    25%    -0.834223  -0.059528  -0.222124  -1.332558   0.234198  -0.008799   
    50%    -0.613761   0.117478  -0.109676  -0.984450   0.656855   0.466357   
    75%     0.040239   1.136383   0.416972  -0.285216   1.246513   0.772441   
    max     1.087999   1.757588   1.022053   1.486507   2.010775   2.162550   
    
                 126        127  
    count  10.000000  10.000000  
    mean    0.226993  -0.064205  
    std     1.463179   0.602277  
    min    -1.731004  -0.865115  
    25%    -0.951169  -0.418553  
    50%     0.174763  -0.112277  
    75%     0.773072   0.353951  
    max     2.991696   0.804881  
    
    [8 rows x 128 columns]
    Observable Tensor 'item_obs' with shape torch.Size([4, 64])
                 0         1         2         3         4         5         6   \
    count  4.000000  4.000000  4.000000  4.000000  4.000000  4.000000  4.000000   
    mean   0.287015 -0.180256 -0.239000  0.169168  0.159036  0.385342 -1.142672   
    std    1.339318  1.603530  0.722772  0.473407  0.392562  1.327739  0.566069   
    min   -1.138152 -2.212473 -1.051363 -0.538771 -0.330795 -0.517352 -1.770297   
    25%   -0.558802 -0.990083 -0.745828  0.132031 -0.006671 -0.485835 -1.397787   
    50%    0.170810 -0.012201 -0.154058  0.385432  0.174086 -0.125969 -1.199654   
    75%    1.016628  0.797626  0.352770  0.422569  0.339793  0.745208 -0.944538   
    max    1.944591  1.515852  0.403479  0.444577  0.618768  2.310656 -0.401083   
    
                 7         8         9   ...        54        55        56  \
    count  4.000000  4.000000  4.000000  ...  4.000000  4.000000  4.000000   
    mean   0.581071 -0.169341  0.076562  ...  0.055457 -0.002887 -0.160406   
    std    0.972295  0.978922  1.116274  ...  0.777132  0.903879  1.140101   
    min   -0.596834 -1.309131 -1.563906  ... -0.481757 -0.997574 -1.721709   
    25%   -0.025344 -0.718815 -0.153971  ... -0.442894 -0.340660 -0.631280   
    50%    0.745386 -0.177989  0.514336  ... -0.240767 -0.105541  0.117918   
    75%    1.351801  0.371485  0.744870  ...  0.257583  0.232232  0.588793   
    max    1.430348  0.987744  0.841483  ...  1.185118  1.197110  0.844249   
    
                 57        58        59        60        61        62        63  
    count  4.000000  4.000000  4.000000  4.000000  4.000000  4.000000  4.000000  
    mean   0.149579  0.199678  0.088542 -0.356379  1.004674  0.095064 -0.548665  
    std    0.963564  0.744614  1.170228  0.833992  0.559029  0.912057  0.730697  
    min   -0.760765 -0.419252 -1.038935 -0.989042  0.442226 -0.989018 -1.445138  
    25%   -0.268040 -0.383280 -0.604213 -0.970008  0.592259 -0.492793 -0.790356  
    50%   -0.075941  0.036190 -0.142981 -0.611959  0.966522  0.230826 -0.546745  
    75%    0.341678  0.619148  0.549774  0.001670  1.378937  0.818683 -0.305054  
    max    1.510964  1.145585  1.679067  0.787444  1.643426  0.907622  0.343970  
    
    [8 rows x 64 columns]
    Observable Tensor 'session_obs' with shape torch.Size([500, 10])
                    0           1           2           3           4           5  \
    count  500.000000  500.000000  500.000000  500.000000  500.000000  500.000000   
    mean    -0.025211   -0.018355   -0.002907    0.091295   -0.061911   -0.046364   
    std      0.976283    1.029875    0.959884    0.968500    1.020114    1.010222   
    min     -2.642895   -3.091050   -3.572037   -2.406249   -3.147900   -3.357277   
    25%     -0.745162   -0.685578   -0.636044   -0.629955   -0.754234   -0.732924   
    50%     -0.018775    0.017807   -0.018642    0.112322   -0.090321   -0.070502   
    75%      0.652438    0.646001    0.601829    0.722870    0.640275    0.652521   
    max      3.044069    3.191774    2.521059    2.695970    3.166039    2.714594   
    
                    6           7           8           9  
    count  500.000000  500.000000  500.000000  500.000000  
    mean     0.000907    0.001370    0.070499   -0.007936  
    std      1.015561    1.032878    1.036212    0.936091  
    min     -2.677915   -3.489751   -2.953354   -2.424499  
    25%     -0.679291   -0.671086   -0.582997   -0.681405  
    50%      0.002569   -0.009368    0.087901    0.010856  
    75%      0.703671    0.732814    0.737692    0.618773  
    max      2.528283    3.259835    2.827300    2.492085  
    Observable Tensor 'price_obs' with shape torch.Size([500, 4, 12])
    device=cpu


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


