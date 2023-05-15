# Introduction
Welcome to the deep choice documentation site, we will guide you through basics of our package and how to use it.

Author: Tianyu Du

Date: Jun. 22, 2022

Update: Jul. 10, 2022


```python
__author__ = 'Tianyu Du'
```

In this demonstration, we will guide you through a minimal example of fitting a conditional logit model using our package. We will be referencing to R code and Stata code as well to deliver a smooth knowledge transfer.

First thing first, let's import a couple of modules from our package.

# Step 0: Import Modules

## Python


```python
import pandas as pd
from torch_choice.utils import EasyDatasetWrapper, run_helper
from torch_choice.model import ConditionalLogitModel
```

## R
```{r}
library("mlogit")
```

# Step 1: Load Data
We have include a copy of the `ModeCanada` dataset in our package: `./public_datasets/ModeCanada.csv`, it's a very small dataset and please feel free to investigate it using softwares like Microsoft Excel.

Let's load the mode canada dataset (TODO: add reference to it).

## Python


```python
df = pd.read_csv('./public_datasets/ModeCanada.csv').query('noalt == 4').reset_index(drop=True)
```

## R
```{r}
ModeCanada <- read.csv('./public_datasets/ModeCanada.csv')
ModeCanada <- select(ModeCanada, -X)
ModeCanada$alt <- as.factor(ModeCanada$alt)
```

# Step 2: Format Data-Frame
TODO: add why we need to do it (every package is doing it).
## Python
Tell the `EasyDatasetWrapper` about observables

1. price observable: cost, freq, ovt, ivt
2. session observables: income.


```python
data = EasyDatasetWrapper(
    main_data=df,
    purchase_record_column='case',
    choice_column='choice',
    item_name_column='alt',
    user_index_column='case',
    session_index_column='case',
    session_observable_columns=['income'],
    price_observable_columns=['cost', 'freq', 'ovt', 'ivt']
)

```

    Creating choice dataset from stata format data-frames...
    Finished Creating Choice Dataset.


## R
```{r}
MC <- dfidx(ModeCanada, subset = noalt == 4)
```

# Step 3: Define and Fit the Conditional Logit Model
## Python


```python
model = ConditionalLogitModel(
    coef_variation_dict={
        'itemsession_cost': 'constant',
        'itemsession_freq': 'constant',
        'itemsession_ovt': 'constant',
        'session_income': 'item',
        'itemsession_ivt': 'item-full',
        'intercept': 'item'
    },
    num_items=4
)
```


```python
run_helper.run(model, data.choice_dataset, num_epochs=5000, learning_rate=0.01, batch_size=-1)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_cost): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (itemsession_freq): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (itemsession_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (itemsession_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_cost] with 1 parameters, with constant level variation.
    X[itemsession_freq] with 1 parameters, with constant level variation.
    X[itemsession_ovt] with 1 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[itemsession_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cpu
    ==================== received dataset ====================
    ChoiceDataset(label=[], item_index=[2779], provided_num_items=[], user_index=[2779], session_index=[2779], item_availability=[], session_income=[2779, 1], itemsession_cost=[2779, 4, 1], itemsession_freq=[2779, 4, 1], itemsession_ovt=[2779, 4, 1], itemsession_ivt=[2779, 4, 1], device=cpu)
    ==================== training the model ====================
    Epoch 500: Log-likelihood=-1980.04736328125
    Epoch 1000: Log-likelihood=-1883.31298828125
    Epoch 1500: Log-likelihood=-1878.42333984375
    Epoch 2000: Log-likelihood=-1878.1141357421875
    Epoch 2500: Log-likelihood=-1879.6005859375
    Epoch 3000: Log-likelihood=-1881.0731201171875
    Epoch 3500: Log-likelihood=-1876.06494140625
    Epoch 4000: Log-likelihood=-1877.595703125
    Epoch 4500: Log-likelihood=-1875.7891845703125
    Epoch 5000: Log-likelihood=-1880.450439453125
    ==================== model results ====================
    Training Epochs: 5000
    
    Learning Rate: 0.01
    
    Batch Size: 2779 out of 2779 observations in total
    
    Final Log-likelihood: -1880.450439453125
    
    Coefficients:
    
    | Coefficient        |   Estimation |   Std. Err. |
    |:-------------------|-------------:|------------:|
    | itemsession_cost_0 | -0.0395517   |  0.00698674 |
    | itemsession_freq_0 |  0.094687    |  0.00504918 |
    | itemsession_ovt_0  | -0.0427526   |  0.00314028 |
    | session_income_0   | -0.0867186   |  0.0174223  |
    | session_income_1   | -0.0268471   |  0.00385441 |
    | session_income_2   | -0.0359928   |  0.00396057 |
    | itemsession_ivt_0  |  0.0597122   |  0.0100132  |
    | itemsession_ivt_1  | -0.00648056  |  0.00417645 |
    | itemsession_ivt_2  | -0.00567451  |  0.00187769 |
    | itemsession_ivt_3  | -0.000954159 |  0.00116984 |
    | intercept_0        | -0.202089    |  1.22288    |
    | intercept_1        |  0.95435     |  0.691519   |
    | intercept_2        |  2.51871     |  0.60307    |





    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_cost): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (itemsession_freq): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (itemsession_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, device=cpu).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (itemsession_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_cost] with 1 parameters, with constant level variation.
    X[itemsession_freq] with 1 parameters, with constant level variation.
    X[itemsession_ovt] with 1 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[itemsession_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    device=cpu



## R
```{r}
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')
summary(ml.MC1)
```

R output:
```
Call:
mlogit(formula = choice ~ cost + freq + ovt | income | ivt, data = MC, 
    reflevel = "air", method = "nr")

Frequencies of alternatives:choice
      air       bus       car     train 
0.3738755 0.0035984 0.4559194 0.1666067 

nr method
9 iterations, 0h:0m:0s 
g'(-H)^-1g = 0.00014 
successive function values within tolerance limits 

Coefficients :
                    Estimate Std. Error  z-value  Pr(>|z|)    
(Intercept):bus    0.6983381  1.2802466   0.5455 0.5854292    
(Intercept):car    1.8441129  0.7085089   2.6028 0.0092464 ** 
(Intercept):train  3.2741952  0.6244152   5.2436 1.575e-07 ***
cost              -0.0333389  0.0070955  -4.6986 2.620e-06 ***
freq               0.0925297  0.0050976  18.1517 < 2.2e-16 ***
ovt               -0.0430036  0.0032247 -13.3356 < 2.2e-16 ***
income:bus        -0.0890867  0.0183471  -4.8556 1.200e-06 ***
income:car        -0.0279930  0.0038726  -7.2286 4.881e-13 ***
income:train      -0.0381466  0.0040831  -9.3426 < 2.2e-16 ***
ivt:air            0.0595097  0.0100727   5.9080 3.463e-09 ***
ivt:bus           -0.0067835  0.0044334  -1.5301 0.1259938    
ivt:car           -0.0064603  0.0018985  -3.4029 0.0006668 ***
ivt:train         -0.0014504  0.0011875  -1.2214 0.2219430    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Log-Likelihood: -1874.3
McFadden R^2:  0.35443 
Likelihood ratio test : chisq = 2058.1 (p.value = < 2.22e-16)
```
