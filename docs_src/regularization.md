# Regularization: $L_1$ and $L_2$

Author: Tianyu Du
Date: Sept. 28, 2022

Also known as **weight decay** or **penalized regression**. Adding the regularization loss term would shrink coefficient magnitudes and better prevent over-fitting.

Specifically, we add the $L_1$ or $L_2$ norm of coefficients to the loss (negative log-likelihood) function.

$$
\text{Loss} = \text{NegativeLogLikelihood} + \alpha \sum_{c \in \text{model coefficients}} ||c||_p \quad p \in \{1, 2\}
$$

Readers can adjust the $\alpha$ weight to control the strength of regularization.


```python
import numpy as np
import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.run_helper import run
```


```python
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    device = 'cuda'
else:
    print('Running tutorial on CPU.')
    device = 'cpu'
```

    CUDA device used: NVIDIA GeForce RTX 3090


## Conditional Logit Model


```python
df = pd.read_csv('./public_datasets/ModeCanada.csv')
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
dataset = ChoiceDataset(item_index=item_index,
                        price_cost_freq_ovt=price_cost_freq_ovt,
                        session_income=session_income,
                        price_ivt=price_ivt
                        ).to(device)
print(dataset)
```

    No `session_index` is provided, assume each choice instance is in its own session.
    ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)



```python
# shuffle the dataset.
N = len(dataset)
shuffle_index = np.random.permutation(N)

train_index = shuffle_index[:int(0.7 * N)]
test_index = shuffle_index[int(0.7 * N):]

# splits of dataset.
dataset_train, dataset_test = dataset[train_index], dataset[test_index]
```


```python
conditional_logit_common_arguments = {
    "coef_variation_dict": {'price_cost_freq_ovt': 'constant',
                            'session_income': 'item',
                            'price_ivt': 'item-full',
                            'intercept': 'item'},
    "num_param_dict": {'price_cost_freq_ovt': 3,
                       'session_income': 1,
                       'price_ivt': 1,
                       'intercept': 1},
    "num_items": 4,
}
```


```python
def train_conditional_logit_model(regularization, regularization_weight):
    model = ConditionalLogitModel(**conditional_logit_common_arguments,
                                regularization=regularization,
                                regularization_weight=regularization_weight).to(device)

    run(model, dataset_train, dataset_test=dataset_test, num_epochs=50000, learning_rate=0.003, batch_size=-1)
    # report total model weight
    print('Total weight L2 norm:', sum([torch.norm(param, p=2) for param in model.parameters()]))
```


```python
train_conditional_logit_model(regularization=None, regularization_weight=None)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cuda:0).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cuda:0).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cuda:0
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[1945], user_index=[], session_index=[1945], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-1322.9208984375
    Epoch 10000: Log-likelihood=-1322.427490234375
    Epoch 15000: Log-likelihood=-1322.361572265625
    Epoch 20000: Log-likelihood=-1322.354736328125
    Epoch 25000: Log-likelihood=-1322.4718017578125
    Epoch 30000: Log-likelihood=-1331.5247802734375
    Epoch 35000: Log-likelihood=-1322.3544921875
    Epoch 40000: Log-likelihood=-1322.421142578125
    Epoch 45000: Log-likelihood=-1322.3602294921875
    Epoch 50000: Log-likelihood=-1322.495849609375
    Test set log-likelihood:  -554.70849609375
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.003
    
    Batch Size: 1945 out of 1945 observations in total
    
    Final Log-likelihood: -1322.495849609375
    
    Coefficients:
    
    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 |  -0.0308257  |  0.00839731 |
    | price_cost_freq_ovt_1 |   0.0945616  |  0.00598799 |
    | price_cost_freq_ovt_2 |  -0.0397223  |  0.00373588 |
    | session_income_0      |  -0.0716898  |  0.0195864  |
    | session_income_1      |  -0.0273578  |  0.00459898 |
    | session_income_2      |  -0.038647   |  0.00484347 |
    | price_ivt_0           |   0.0564822  |  0.0117201  |
    | price_ivt_1           |  -0.00936753 |  0.00582746 |
    | price_ivt_2           |  -0.00678837 |  0.00222236 |
    | price_ivt_3           |  -0.00175041 |  0.00139018 |
    | intercept_0           |   0.899362   |  1.53674    |
    | intercept_1           |   2.24992    |  0.848803   |
    | intercept_2           |   3.50811    |  0.747974   |
    Total weight L2 norm: tensor(2.6599, device='cuda:0', grad_fn=<AddBackward0>)



```python
train_conditional_logit_model(regularization='L1', regularization_weight=5)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cuda:0).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cuda:0).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cuda:0
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[1945], user_index=[], session_index=[1945], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-1327.5283203125
    Epoch 10000: Log-likelihood=-1327.5472412109375
    Epoch 15000: Log-likelihood=-1327.5458984375
    Epoch 20000: Log-likelihood=-1327.5452880859375
    Epoch 25000: Log-likelihood=-1327.54931640625
    Epoch 30000: Log-likelihood=-1327.9013671875
    Epoch 35000: Log-likelihood=-1327.5465087890625
    Epoch 40000: Log-likelihood=-1327.6224365234375
    Epoch 45000: Log-likelihood=-1327.5556640625
    Epoch 50000: Log-likelihood=-1333.43359375
    Test set log-likelihood:  -556.6971435546875
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.003
    
    Batch Size: 1945 out of 1945 observations in total
    
    Final Log-likelihood: -1333.43359375
    
    Coefficients:
    
    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 | -0.0485882   |  0.0084985  |
    | price_cost_freq_ovt_1 |  0.0963804   |  0.00600474 |
    | price_cost_freq_ovt_2 | -0.0381796   |  0.00383793 |
    | session_income_0      | -0.0766308   |  0.0208468  |
    | session_income_1      | -0.0225714   |  0.00444105 |
    | session_income_2      | -0.0326763   |  0.00488883 |
    | price_ivt_0           |  0.0531795   |  0.0118078  |
    | price_ivt_1           | -0.0166434   |  0.0080002  |
    | price_ivt_2           | -0.00397061  |  0.00221348 |
    | price_ivt_3           | -0.00189491  |  0.00140921 |
    | intercept_0           |  0.000167495 |  1.69499    |
    | intercept_1           |  0.000309494 |  0.833982   |
    | intercept_2           |  1.2901      |  0.729501   |
    Total weight L2 norm: tensor(1.3817, device='cuda:0', grad_fn=<AddBackward0>)



```python
train_conditional_logit_model(regularization='L2', regularization_weight=5)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cuda:0).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cuda:0).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cuda:0
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[1945], user_index=[], session_index=[1945], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-1327.98876953125
    Epoch 10000: Log-likelihood=-1327.377197265625
    Epoch 15000: Log-likelihood=-1327.3466796875
    Epoch 20000: Log-likelihood=-1327.345458984375
    Epoch 25000: Log-likelihood=-1327.433349609375
    Epoch 30000: Log-likelihood=-1327.3453369140625
    Epoch 35000: Log-likelihood=-1327.34521484375
    Epoch 40000: Log-likelihood=-1327.3885498046875
    Epoch 45000: Log-likelihood=-1327.3486328125
    Epoch 50000: Log-likelihood=-1327.34765625
    Test set log-likelihood:  -555.1453857421875
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.003
    
    Batch Size: 1945 out of 1945 observations in total
    
    Final Log-likelihood: -1327.34765625
    
    Coefficients:
    
    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 |  -0.0482729  |  0.0083645  |
    | price_cost_freq_ovt_1 |   0.0967298  |  0.00595309 |
    | price_cost_freq_ovt_2 |  -0.0376925  |  0.0037188  |
    | session_income_0      |  -0.0749973  |  0.019634   |
    | session_income_1      |  -0.0231255  |  0.00446823 |
    | session_income_2      |  -0.032398   |  0.00475483 |
    | price_ivt_0           |   0.0534635  |  0.0117147  |
    | price_ivt_1           |  -0.0153539  |  0.00731768 |
    | price_ivt_2           |  -0.00426721 |  0.00219745 |
    | price_ivt_3           |  -0.00154632 |  0.00138443 |
    | intercept_0           |  -0.201299   |  1.60544    |
    | intercept_1           |   0.00875631 |  0.823289   |
    | intercept_2           |   1.29872    |  0.715818   |
    Total weight L2 norm: tensor(1.5968, device='cuda:0', grad_fn=<AddBackward0>)



```python
train_conditional_logit_model(regularization='L1', regularization_weight=1E5)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cuda:0).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cuda:0).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cuda:0).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cuda:0
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[1945], user_index=[], session_index=[1945], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)
    ==================== training the model ====================
    Epoch 5000: Log-likelihood=-2680.06005859375
    Epoch 10000: Log-likelihood=-2431.19091796875
    Epoch 15000: Log-likelihood=-2651.45849609375
    Epoch 20000: Log-likelihood=-2578.85107421875
    Epoch 25000: Log-likelihood=-2525.41650390625
    Epoch 30000: Log-likelihood=-2554.415283203125
    Epoch 35000: Log-likelihood=-2570.41845703125
    Epoch 40000: Log-likelihood=-2658.0556640625
    Epoch 45000: Log-likelihood=-2560.906005859375
    Epoch 50000: Log-likelihood=-2677.46826171875
    Test set log-likelihood:  -1136.294921875
    ==================== model results ====================
    Training Epochs: 50000
    
    Learning Rate: 0.003
    
    Batch Size: 1945 out of 1945 observations in total
    
    Final Log-likelihood: -2677.46826171875
    
    Coefficients:
    
    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 |  0.000446639 | 0.00574829  |
    | price_cost_freq_ovt_1 | -0.000407603 | 0.00415769  |
    | price_cost_freq_ovt_2 |  0.000226522 | 0.0021607   |
    | session_income_0      | -4.7971e-05  | 0.00383794  |
    | session_income_1      |  0.00117954  | 0.00375016  |
    | session_income_2      |  0.00041626  | 0.00359678  |
    | price_ivt_0           | -0.000192594 | 0.00875022  |
    | price_ivt_1           | -0.000618745 | 0.000871537 |
    | price_ivt_2           | -0.000398202 | 0.00165723  |
    | price_ivt_3           |  0.000407054 | 0.00104901  |
    | intercept_0           | -0.000648632 | 0.567814    |
    | intercept_1           | -0.000525868 | 0.580968    |
    | intercept_2           | -0.000405973 | 0.505175    |
    Total weight L2 norm: tensor(1.3426, device='cuda:0', grad_fn=<AddBackward0>)


## On Nested Logit Model


```python
df = pd.read_csv('./public_datasets/HC.csv', index_col=0)
df = df.reset_index(drop=True)
df.head()

# what was actually chosen.
item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
num_items = df['idx.id2'].nunique()
# cardinal encoder.
encoder = dict(zip(item_names, range(num_items)))
item_index = item_index.map(lambda x: encoder[x])
item_index = torch.LongTensor(item_index)

# category feature: no category feature, all features are item-level.
category_dataset = ChoiceDataset(item_index=item_index.clone()).to(device)

# item feature.
item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)

item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs).to(device)

dataset = JointDataset(category=category_dataset, item=item_dataset)

category_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                    1: ['gc', 'ec', 'er']}

# encode items to integers.
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)
```

    No `session_index` is provided, assume each choice instance is in its own session.
    No `session_index` is provided, assume each choice instance is in its own session.



```python
def train_nested_logit_model(regularization, regularization_weight):
    model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         regularization=regularization,
                         regularization_weight=regularization_weight,
                         shared_lambda=True).to(device)
    run(model, dataset, num_epochs=10000)
```


```python
train_nested_logit_model(None, None)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cuda:0).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 1000: Log-likelihood=-226.63345336914062
    Epoch 2000: Log-likelihood=-189.08030700683594
    Epoch 3000: Log-likelihood=-181.08639526367188
    Epoch 4000: Log-likelihood=-179.11544799804688
    Epoch 5000: Log-likelihood=-178.78994750976562
    Epoch 6000: Log-likelihood=-178.64102172851562
    Epoch 7000: Log-likelihood=-178.50711059570312
    Epoch 8000: Log-likelihood=-178.36279296875
    Epoch 9000: Log-likelihood=-178.23562622070312
    Epoch 10000: Log-likelihood=-178.15724182128906
    ==================== model results ====================
    Training Epochs: 10000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -178.15724182128906
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     0.569814 |   0.163447  |
    | item_price_obs_0 |    -0.5397   |   0.141929  |
    | item_price_obs_1 |    -0.834805 |   0.233345  |
    | item_price_obs_2 |    -0.242956 |   0.110592  |
    | item_price_obs_3 |    -1.27541  |   1.03548   |
    | item_price_obs_4 |    -0.368249 |   0.0986935 |
    | item_price_obs_5 |     0.247266 |   0.0513082 |
    | item_price_obs_6 |    -4.78207  |   4.7152    |



```python
train_nested_logit_model("L1", 10)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cuda:0).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 1000: Log-likelihood=-186.81593322753906
    Epoch 2000: Log-likelihood=-187.0428924560547
    Epoch 3000: Log-likelihood=-188.46871948242188
    Epoch 4000: Log-likelihood=-187.3245849609375
    Epoch 5000: Log-likelihood=-187.10488891601562
    Epoch 6000: Log-likelihood=-187.18087768554688
    Epoch 7000: Log-likelihood=-187.34005737304688
    Epoch 8000: Log-likelihood=-187.11846923828125
    Epoch 9000: Log-likelihood=-187.3697509765625
    Epoch 10000: Log-likelihood=-187.0865478515625
    ==================== model results ====================
    Training Epochs: 10000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -187.0865478515625
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |  0.0530321   |   0.0531535 |
    | item_price_obs_0 | -0.0512223   |   0.0514528 |
    | item_price_obs_1 | -0.0779116   |   0.078385  |
    | item_price_obs_2 | -0.187379    |   0.087971  |
    | item_price_obs_3 | -0.00119437  |   0.863954  |
    | item_price_obs_4 | -0.0346545   |   0.0350824 |
    | item_price_obs_5 |  0.183375    |   0.034789  |
    | item_price_obs_6 |  0.000892786 |   3.57438   |



```python
train_nested_logit_model("L2", 10)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total, device=cuda:0).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], device=cuda:0)
    	item: ChoiceDataset(label=[], item_index=[250], user_index=[], session_index=[250], item_availability=[], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 1000: Log-likelihood=-219.621826171875
    Epoch 2000: Log-likelihood=-200.87660217285156
    Epoch 3000: Log-likelihood=-192.0721435546875
    Epoch 4000: Log-likelihood=-183.12820434570312
    Epoch 5000: Log-likelihood=-182.87225341796875
    Epoch 6000: Log-likelihood=-183.52407836914062
    Epoch 7000: Log-likelihood=-183.50723266601562
    Epoch 8000: Log-likelihood=-183.5075225830078
    Epoch 9000: Log-likelihood=-183.50465393066406
    Epoch 10000: Log-likelihood=-183.5073699951172
    ==================== model results ====================
    Training Epochs: 10000
    
    Learning Rate: 0.01
    
    Batch Size: 250 out of 250 observations in total
    
    Final Log-likelihood: -183.5073699951172
    
    Coefficients:
    
    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |    0.181474  |   0.108225  |
    | item_price_obs_0 |   -0.174871  |   0.102564  |
    | item_price_obs_1 |   -0.265047  |   0.156401  |
    | item_price_obs_2 |   -0.258935  |   0.0949367 |
    | item_price_obs_3 |   -0.151668  |   0.898396  |
    | item_price_obs_4 |   -0.118241  |   0.0697575 |
    | item_price_obs_5 |    0.193267  |   0.0380327 |
    | item_price_obs_6 |   -0.0374295 |   3.90292   |

