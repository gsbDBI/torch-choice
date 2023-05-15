# Tutorial: Optimization Algorithms
**Author: Tianyu Du (tianyudu@stanford.edu)**

**Update: May. 14, 2023**

Let's first import essential Python packages.


```python
import pandas as pd
import torch
import torch.nn.functional as F

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from torch_choice import run
```

    /Users/tianyudu/miniforge3/envs/dev/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: dlopen(/Users/tianyudu/miniforge3/envs/dev/lib/python3.9/site-packages/torchvision/image.so, 0x0006): Symbol not found: __ZN2at4_ops19empty_memory_format4callEN3c108ArrayRefIxEENS2_8optionalINS2_10ScalarTypeEEENS5_INS2_6LayoutEEENS5_INS2_6DeviceEEENS5_IbEENS5_INS2_12MemoryFormatEEE
      Referenced from: <B3E58761-2785-34C6-A89B-F37110C88A05> /Users/tianyudu/miniforge3/envs/dev/lib/python3.9/site-packages/torchvision/image.so
      Expected in:     <AE6DCE26-A528-35ED-BB3D-88890D27E6B9> /Users/tianyudu/miniforge3/envs/dev/lib/python3.9/site-packages/torch/lib/libtorch_cpu.dylib
      warn(f"Failed to load image Python extension: {e}")



```python
print(torch.__version__)
print(f"{torch.cuda.is_available()=:}")
```

    2.0.0
    torch.cuda.is_available()=False



```python
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    device = 'cuda'
else:
    print('Running tutorial on CPU.')
    device = 'cpu'
```

    Running tutorial on CPU.



```python
df = pd.read_csv('./public_datasets/ModeCanada.csv')
df = df.query('noalt == 4').reset_index(drop=True)
df.sort_values(by='case', inplace=True)
df.head()
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
```

    No `session_index` is provided, assume each choice instance is in its own session.



```python
print(dataset)
```

    ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cpu)



```python
import warnings
warnings.filterwarnings("ignore")
device = "cpu"
model = ConditionalLogitModel(
    formula='(price_cost_freq_ovt|constant) + (session_income|item) + (price_ivt|item-full) + (intercept|item)',
    dataset=dataset,
    num_items=4).to(device)
run(model, dataset, num_epochs=500, learning_rate=0.01, batch_size=-1, model_optimizer="LBFGS", device=device)
```

    GPU available: True (mps), used: False
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
        (price_cost_freq_ovt[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cpu).
        (session_income[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (price_ivt[item-full]): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt[constant]] with 3 parameters, with constant level variation.
    X[session_income[item]] with 1 parameters, with item level variation.
    X[price_ivt[item-full]] with 1 parameters, with item-full level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu
    ==================== data set received ====================
    [Train dataset] ChoiceDataset(label=[], item_index=[2779], user_index=[], session_index=[2779], item_availability=[], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cpu)
    [Validation dataset] None
    [Test dataset] None
    Epoch 499: 100%|██████████| 1/1 [00:00<00:00, 40.10it/s, loss=1.87e+03, v_num=15] 

    `Trainer.fit` stopped: `max_epochs=500` reached.


    Epoch 499: 100%|██████████| 1/1 [00:00<00:00, 38.63it/s, loss=1.87e+03, v_num=15]
    Time taken for training: 12.536703109741211
    Skip testing, no test dataset is provided.
    ==================== model results ====================
    Log-likelihood: [Training] -1874.3427734375, [Validation] N/A, [Test] N/A
    
    | Coefficient                     |   Estimation |   Std. Err. |    z-value |    Pr(>|z|) | Significance   |
    |:--------------------------------|-------------:|------------:|-----------:|------------:|:---------------|
    | price_cost_freq_ovt[constant]_0 |  -0.0333376  |  0.00709551 |  -4.69841  | 2.62196e-06 | ***            |
    | price_cost_freq_ovt[constant]_1 |   0.0925288  |  0.00509756 |  18.1516   | 0           | ***            |
    | price_cost_freq_ovt[constant]_2 |  -0.0430023  |  0.0032247  | -13.3353   | 0           | ***            |
    | session_income[item]_0          |  -0.0891035  |  0.018348   |  -4.85631  | 1.19595e-06 | ***            |
    | session_income[item]_1          |  -0.0279937  |  0.00387255 |  -7.22876  | 4.87388e-13 | ***            |
    | session_income[item]_2          |  -0.038145   |  0.00408308 |  -9.34222  | 0           | ***            |
    | price_ivt[item-full]_0          |   0.059507   |  0.0100727  |   5.90777  | 3.46776e-09 | ***            |
    | price_ivt[item-full]_1          |  -0.00678584 |  0.00443389 |  -1.53045  | 0.125905    |                |
    | price_ivt[item-full]_2          |  -0.00646072 |  0.00189849 |  -3.40309  | 0.000666291 | ***            |
    | price_ivt[item-full]_3          |  -0.00145041 |  0.00118748 |  -1.22142  | 0.221927    |                |
    | intercept[item]_0               |   0.699403   |  1.28026    |   0.546298 | 0.584861    |                |
    | intercept[item]_1               |   1.84431    |  0.708509   |   2.60309  | 0.00923886  | **             |
    | intercept[item]_2               |   3.2741     |  0.624415   |   5.24347  | 1.57586e-07 | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1





    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total, device=cpu).
        (session_income[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
        (price_ivt[item-full]): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[price_cost_freq_ovt[constant]] with 3 parameters, with constant level variation.
    X[session_income[item]] with 1 parameters, with item level variation.
    X[price_ivt[item-full]] with 1 parameters, with item-full level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu



### Parameter Estimation from `R`
The following is the R-output from the `mlogit` implementation, the estimation, standard error, and log-likelihood from our `torch_choice` implementation is the same as the result from `mlogit` implementation.

We see that the final log-likelihood of models estimated using two packages are all around `-1874`.

The `run()` method calculates the standard deviation using $\sqrt{\text{diag}(H^{-1})}$, where $H$ is the hessian of negative log-likelihood with repsect to model parameters.

Names of coefficients are slightly different, one can use the following conversion table to compare estimations and standard deviations reported by both packages.

<!-- | Coefficient Name in Python |  Estimation |   Std. Err. |  Coeffcient Name in R | R Estimation | R Std. Err. | 
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
| intercept_2           |   3.2782     |  0.648064   | (Intercept):train|  3.2741952  |0.6244152| -->

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
