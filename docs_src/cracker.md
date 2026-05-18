# Cracker brand choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of household cracker brand choice
on the Jain–Vilcassim–Chintagunta (JBES, 1994) scanner panel. We fit the same
model twice — once in R with `mlogit`, once in Python with `torch-choice` — and
show that the two implementations recover the same coefficients to numerical
precision. We then explore richer alternative specifications and discuss
which is theoretically preferred.

Cracker is the *brand-loyalty / state-dependence* companion to Yogurt: same
JBES 1994 paper, different CPG category, same wide 4-brand panel structure.
If you already worked through `tutorials/yogurt/yogurt.ipynb`, this one will
look very familiar — that's by design, so you can compare the two categories
side by side.

## 1. About this dataset

**Domain.** Consumer packaged goods / brand choice. 136 households make 3,292
cracker purchases over time, choosing one of four brands per trip:

| Brand | Notes |
|---|---|
| `nabisco` | Reference (dominant national brand, ~54% market share in the panel) |
| `keebler` | National competitor |
| `private` | Store / private label |
| `sunshine` | Regional |

Each row is one purchase occasion. For every occasion we observe the **price**
charged for all four brands (in cents per pound × 100 in the raw R data —
treat the units as opaque; only the coefficient sign and relative magnitude
matter), a **display** indicator (`disp.*` = 1 if the brand was on an in-store
display) and a **feature** indicator (`feat.*` = 1 if the brand was in a
newspaper feature/promotion that week). The chosen brand is in the `choice`
column.

**Reference.** Jain, D. C., Vilcassim, N. J., Chintagunta, P. K. (1994).
"A Random-Coefficients Logit Brand-Choice Model Applied to Panel Data."
*Journal of Business & Economic Statistics* 12(3), 317–328.
([doi:10.1080/07350015.1994.10524547](https://doi.org/10.1080/07350015.1994.10524547))

**License.** Distributed in the R package `mlogit` under GPL-2, redistributable
with citation.

### How to download

The CSV in this folder was extracted from the cran/mlogit GitHub mirror:

```python
import requests, pyreadr, tempfile
url = 'https://raw.githubusercontent.com/cran/mlogit/master/data/Cracker.rda'
r = requests.get(url)
with tempfile.NamedTemporaryFile(suffix='.rda', delete=False) as tf:
    tf.write(r.content); tmp_path = tf.name
cracker = list(pyreadr.read_r(tmp_path).values())[0]
```

Or, in R:

```r
data(Cracker, package = 'mlogit')
write.csv(Cracker, 'cracker.csv', row.names = FALSE)
```

The notebook reads `cracker.csv` from this folder.


```python
# Make `tutorials/_mlogit_compare.py` importable from this folder.
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from _mlogit_compare import run_or_load_mlogit, compare_coefs

torch.manual_seed(0)   # reproducibility across re-runs (LBFGS line search)
HERE = Path.cwd()
```


```python
df = pd.read_csv(HERE / 'cracker.csv')
print(f"shape={df.shape}, n_households={df['id'].nunique()}, n_occasions={len(df)}")
df.head()
```

    shape=(3292, 14), n_households=136, n_occasions=3292





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
      <th>id</th>
      <th>disp.sunshine</th>
      <th>disp.keebler</th>
      <th>disp.nabisco</th>
      <th>disp.private</th>
      <th>feat.sunshine</th>
      <th>feat.keebler</th>
      <th>feat.nabisco</th>
      <th>feat.private</th>
      <th>price.sunshine</th>
      <th>price.keebler</th>
      <th>price.nabisco</th>
      <th>price.private</th>
      <th>choice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>98.000002</td>
      <td>88.0</td>
      <td>120.000000</td>
      <td>70.999998</td>
      <td>nabisco</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>99.000001</td>
      <td>109.0</td>
      <td>99.000001</td>
      <td>70.999998</td>
      <td>nabisco</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49.000001</td>
      <td>109.0</td>
      <td>109.000000</td>
      <td>77.999997</td>
      <td>sunshine</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>103.000000</td>
      <td>109.0</td>
      <td>88.999999</td>
      <td>77.999997</td>
      <td>nabisco</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>109.000000</td>
      <td>109.0</td>
      <td>119.000010</td>
      <td>63.999999</td>
      <td>nabisco</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook MNL on the wide-format CSV using `mlogit::mlogit`, with
`nabisco` as the reference alternative (it's the dominant national brand —
the natural reference for an alt-specific intercept comparison):

```r
suppressPackageStartupMessages(library(mlogit))
df   <- read.csv('cracker.csv')
data <- dfidx(df, shape='wide', choice='choice', varying=2:13,
              sep='.', idnames=c('chid','alt'))
mod  <- mlogit(choice ~ price + disp + feat, data=data, reflevel='nabisco')
summary(mod)
```

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to the
cached output in `mlogit_output.json` (also next to this notebook), so the
comparison still renders for readers without an R installation.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / 'fit_mlogit.R',
    csv_path=HERE / 'cracker.csv',
    cache_path=HERE / 'mlogit_output.json',
)
print(f'R/mlogit train log-likelihood: {mlogit_ll:.4f}')
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -3347.7133





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
      <th>name</th>
      <th>estimate</th>
      <th>std_err</th>
      <th>z_value</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Intercept):keebler</td>
      <td>-1.961608</td>
      <td>0.072354</td>
      <td>-27.111087</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Intercept):private</td>
      <td>-1.792814</td>
      <td>0.100107</td>
      <td>-17.909029</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Intercept):sunshine</td>
      <td>-2.455213</td>
      <td>0.080015</td>
      <td>-30.684293</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>price</td>
      <td>-0.031247</td>
      <td>0.002089</td>
      <td>-14.961536</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>disp</td>
      <td>0.091917</td>
      <td>0.062093</td>
      <td>1.480309</td>
      <td>1.387909e-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>feat</td>
      <td>0.496126</td>
      <td>0.095430</td>
      <td>5.198834</td>
      <td>2.005428e-07</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_price|constant) + (itemsession_disp|constant)
  + (itemsession_feat|constant) + (intercept|item)
```

— one shared coefficient on `price`, one shared on `disp`, one shared on
`feat`, plus an alt-specific intercept (the first item, `nabisco`, is the
reference, with intercept pinned to zero, which matches R's
`reflevel="nabisco"`).

### Build a `ChoiceDataset`

The CSV is in *wide* format (one row per occasion, with brand-suffixed
`price.{brand}`, `disp.{brand}`, `feat.{brand}` columns). We reshape to long
format and pivot into the 3D `(num_sessions, num_items, num_features)`
tensors `torch-choice` expects.


```python
# Brand encoding: index 0 is the reference (matches R's reflevel='nabisco').
BRAND_ORDER = ['nabisco', 'keebler', 'private', 'sunshine']
brand_to_idx = {b: i for i, b in enumerate(BRAND_ORDER)}

# Wide -> long: 3,292 occasions x 4 brands = 13,168 rows.
df = df.copy()
df['occasion'] = np.arange(len(df))
long_records = []
for brand, idx in brand_to_idx.items():
    long_records.append(
        df[['occasion', f'price.{brand}', f'disp.{brand}', f'feat.{brand}']]
        .rename(columns={
            f'price.{brand}': 'price',
            f'disp.{brand}':  'disp',
            f'feat.{brand}':  'feat',
        })
        .assign(brand=idx)
    )
long_df = pd.concat(long_records, ignore_index=True).sort_values(['occasion', 'brand'])

# Pivot into (num_sessions, num_items, num_features) tensors.
itemsession_price = utils.pivot3d(long_df, dim0='occasion', dim1='brand', values='price')
itemsession_disp  = utils.pivot3d(long_df, dim0='occasion', dim1='brand', values='disp')
itemsession_feat  = utils.pivot3d(long_df, dim0='occasion', dim1='brand', values='feat')

# Per-occasion chosen brand index, household id, and session id.
item_index    = torch.LongTensor(df['choice'].map(brand_to_idx).values)
user_index    = torch.LongTensor(df['id'].astype(int).values - 1)  # 0-indexed
session_index = torch.arange(len(df))
NUM_USERS     = int(user_index.max().item()) + 1   # 136 households
N_OCC         = len(item_index)                    # 3292 occasions

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=4,
    num_users=NUM_USERS,
    user_index=user_index,
    session_index=session_index,
    itemsession_price=itemsession_price,
    itemsession_disp=itemsession_disp,
    itemsession_feat=itemsession_feat,
)
print(dataset)
```

    ChoiceDataset(num_items=4, num_users=136, num_sessions=3292, label=[], item_index=[3292], user_index=[3292], session_index=[3292], item_availability=[], itemsession_price=[3292, 4, 1], itemsession_disp=[3292, 4, 1], itemsession_feat=[3292, 4, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


### Fit (Spec A — pooled MNL)

We use full-batch LBFGS for 5,000 epochs at `learning_rate=0.001`. The
Cracker price column lives on a numerically larger scale than e.g. Yogurt's
(prices ranging ~50–150 vs ~6–12), so the default `learning_rate=0.01`
is too aggressive — LBFGS line-search blows up early. Dropping the rate
10x (and bumping epochs to 5,000 to compensate) lets the optimizer match
`mlogit`'s Newton-Raphson optimum to 1e-3 precision. `model.fit(...)`
returns an `EstimationOutput` whose `coef_summary` mirrors a standard
regression table.


```python
model = ConditionalLogitModel(
    formula=(
        '(itemsession_price|constant) + (itemsession_disp|constant)'
        ' + (itemsession_feat|constant) + (intercept|item)'
    ),
    dataset=dataset,
    num_items=4,
)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_price[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_disp[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_feat[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_price[constant]] with 1 parameters, with constant level variation.
    X[itemsession_disp[constant]] with 1 parameters, with constant level variation.
    X[itemsession_feat[constant]] with 1 parameters, with constant level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu



```python
result = model.fit(
    dataset,
    batch_size=-1,
    learning_rate=0.001,
    num_epochs=5000,
    model_optimizer='LBFGS',
    backend='lightning',
    print_summary=False,
)
print(result)
print(f'\ntorch-choice train log-likelihood: {result.train_ll:.4f}')
```

    GPU available: True (mps), used: False


    TPU available: False, using: 0 TPU cores


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.


    Starting PyTorch Lightning training loop.


    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    
      | Name  | Type                  | Params | Mode  | FLOPs
    ----------------------------------------------------------------
    0 | model | ConditionalLogitModel | 6      | train | 0    
    ----------------------------------------------------------------
    6         Trainable params
    0         Non-trainable params
    6         Total params
    0.000     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=5000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -3347.713, [Validation] None, [Test] None
    
    | Coefficient                   |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_price[constant]_0 |   -0.0312435 |  0.00208846 |   -14.96  | < 2e-16    | ***            |
    | itemsession_disp[constant]_0  |    0.0919333 |  0.0620922  |     1.481 | 0.139      |                |
    | itemsession_feat[constant]_0  |    0.496172  |  0.0954292  |     5.199 | 2.000e-07  | ***            |
    | intercept[item]_0             |   -1.96154   |  0.072353   |   -27.111 | < 2e-16    | ***            |
    | intercept[item]_1             |   -1.79263   |  0.100104   |   -17.908 | < 2e-16    | ***            |
    | intercept[item]_2             |   -2.45496   |  0.08001    |   -30.683 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -3347.7134


## 4. Side-by-side comparison

The two implementations expose coefficient names differently. `mlogit` labels
alternative-specific intercepts as `(Intercept):keebler` etc., while
`torch-choice` indexes them positionally inside `intercept[item]_*`. The
mapping below is the only per-dataset adapter required for the comparison
helper.

Note the positional convention: with `nabisco` pinned as the reference at
item index 0, `torch-choice`'s `intercept[item]_0`, `_1`, `_2` correspond to
items at indices 1, 2, 3 in `BRAND_ORDER` — i.e., `keebler`, `private`,
`sunshine`.


```python
NAME_MAP = {
    'itemsession_price[constant]_0': 'price',
    'itemsession_disp[constant]_0':  'disp',
    'itemsession_feat[constant]_0':  'feat',
    'intercept[item]_0':             '(Intercept):keebler',
    'intercept[item]_1':             '(Intercept):private',
    'intercept[item]_2':             '(Intercept):sunshine',
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f'\nLL: mlogit={mlogit_ll:.4f}  '
    f'torch-choice={result.train_ll:.4f}  '
    f'abs_diff={abs(mlogit_ll - result.train_ll):.2e}'
)
diff
```

    [compare_coefs] estimates: max |diff| = 2.499e-04 (0.0178%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 5.282e-06 (0.0066%); tol 1e-03 -> PASS
    
    LL: mlogit=-3347.7133  torch-choice=-3347.7134  abs_diff=8.93e-05





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
      <th>coef</th>
      <th>mlogit_est</th>
      <th>tc_est</th>
      <th>est_abs_diff</th>
      <th>est_pct_diff</th>
      <th>mlogit_se</th>
      <th>tc_se</th>
      <th>se_abs_diff</th>
      <th>se_pct_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>itemsession_price[constant]_0</td>
      <td>-0.031247</td>
      <td>-0.031244</td>
      <td>0.000004</td>
      <td>0.012228</td>
      <td>0.002089</td>
      <td>0.002088</td>
      <td>4.877768e-08</td>
      <td>0.002336</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_disp[constant]_0</td>
      <td>0.091917</td>
      <td>0.091933</td>
      <td>0.000016</td>
      <td>0.017828</td>
      <td>0.062093</td>
      <td>0.062092</td>
      <td>8.402275e-07</td>
      <td>0.001353</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_feat[constant]_0</td>
      <td>0.496126</td>
      <td>0.496172</td>
      <td>0.000045</td>
      <td>0.009130</td>
      <td>0.095430</td>
      <td>0.095429</td>
      <td>1.156571e-06</td>
      <td>0.001212</td>
    </tr>
    <tr>
      <th>3</th>
      <td>intercept[item]_0</td>
      <td>-1.961608</td>
      <td>-1.961537</td>
      <td>0.000071</td>
      <td>0.003626</td>
      <td>0.072354</td>
      <td>0.072353</td>
      <td>1.488822e-06</td>
      <td>0.002058</td>
    </tr>
    <tr>
      <th>4</th>
      <td>intercept[item]_1</td>
      <td>-1.792814</td>
      <td>-1.792633</td>
      <td>0.000181</td>
      <td>0.010096</td>
      <td>0.100107</td>
      <td>0.100104</td>
      <td>2.489680e-06</td>
      <td>0.002487</td>
    </tr>
    <tr>
      <th>5</th>
      <td>intercept[item]_2</td>
      <td>-2.455213</td>
      <td>-2.454963</td>
      <td>0.000250</td>
      <td>0.010180</td>
      <td>0.080015</td>
      <td>0.080010</td>
      <td>5.281947e-06</td>
      <td>0.006601</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion of §4.** `torch-choice` reproduces `mlogit`'s MNL coefficient
estimates and log-likelihood to within float32 round-off. This validates the
package's optimizer and standard-error implementation against an established
reference on a second CPG dataset (Yogurt was the first).

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong assumption:
*every household responds identically to price, in-store displays, and
newspaper features*. The original Jain–Vilcassim–Chintagunta (1994) paper
questions exactly this assumption — its central contribution is a
**random-coefficients logit** that lets price- and promotional-sensitivity
vary across households. For Cracker the panel is even richer than for
Yogurt (~24 purchases per household on average across 136 households), and
the literature consistently reports substantial taste heterogeneity in this
category — particularly along the price axis, where private-label loyalists
and national-brand loyalists exhibit very different responsiveness.

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_price\|constant) + (itemsession_disp\|constant) + (itemsession_feat\|constant) + (intercept\|item)` | one price coef, one disp coef, one feat coef, three brand intercepts | 6 |
| **B** (HH price sensitivity) | `(itemsession_price\|user) + (itemsession_disp\|constant) + (itemsession_feat\|constant) + (intercept\|item)` | each of 136 households gets its own price coefficient; disp + feat shared | 136+1+1+3 = 141 |
| **C** (full HH heterogeneity) | `(itemsession_price\|user) + (itemsession_disp\|user) + (itemsession_feat\|user) + (intercept\|item)` | each household gets its own price, disp *and* feat coefficient | 136+136+136+3 = 411 |

Specs B and C are *fixed-effects* approximations to the JVC 1994
random-coefficients logit. They estimate one coefficient per household
(treating heterogeneity as known finite parameters) instead of integrating
over a continuous mixing distribution. With 136 households and ~24 purchases
each, the FE approach is feasible; with thousands of users it would not be.

### Fit Specs B and C


```python
def fit_spec(formula: str, label: str, *, num_epochs: int = 5000,
             learning_rate: float = 0.001,
             optimizer: str = 'LBFGS') -> tuple[float, int, 'EstimationOutput']:
    """Fit a torch-choice spec; return (train_ll, n_params, result)."""
    torch.manual_seed(0)
    m = ConditionalLogitModel(
        formula=formula, dataset=dataset, num_items=4, num_users=NUM_USERS,
    )
    r = m.fit(
        dataset,
        batch_size=-1,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_optimizer=optimizer,
        backend='lightning',
        print_summary=False,
    )
    n_params = int(r.coef_summary.shape[0])
    return float(r.train_ll), n_params, r

ll_a, k_a = float(result.train_ll), int(result.coef_summary.shape[0])

ll_b, k_b, result_b = fit_spec(
    '(itemsession_price|user) + (itemsession_disp|constant)'
    ' + (itemsession_feat|constant) + (intercept|item)',
    'B: HH price sensitivity',
)

ll_c, k_c, result_c = fit_spec(
    '(itemsession_price|user) + (itemsession_disp|user)'
    ' + (itemsession_feat|user) + (intercept|item)',
    'C: full HH heterogeneity',
)

print(f'Spec A: LL={ll_a:.2f}, n_params={k_a}')
print(f'Spec B: LL={ll_b:.2f}, n_params={k_b}')
print(f'Spec C: LL={ll_c:.2f}, n_params={k_c}')
```

    GPU available: True (mps), used: False


    TPU available: False, using: 0 TPU cores


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.


    Starting PyTorch Lightning training loop.


    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    
      | Name  | Type                  | Params | Mode  | FLOPs
    ----------------------------------------------------------------
    0 | model | ConditionalLogitModel | 141    | train | 0    
    ----------------------------------------------------------------
    141       Trainable params
    0         Non-trainable params
    141       Total params
    0.001     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=5000` reached.


    GPU available: True (mps), used: False


    TPU available: False, using: 0 TPU cores


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.


    Starting PyTorch Lightning training loop.


    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    
      | Name  | Type                  | Params | Mode  | FLOPs
    ----------------------------------------------------------------
    0 | model | ConditionalLogitModel | 411    | train | 0    
    ----------------------------------------------------------------
    411       Trainable params
    0         Non-trainable params
    411       Total params
    0.002     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=5000` reached.


    Spec A: LL=-3347.71, n_params=6
    Spec B: LL=-2220.63, n_params=141
    Spec C: LL=-1927.06, n_params=411


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively, so it tends to favor parsimonious models — appropriate when we
believe the simpler spec is closer to the truth. AIC penalizes less and
tends to favor richer models when sample size is large.


```python
specs = [
    ('A: pooled MNL',            ll_a, k_a),
    ('B: HH price sensitivity',  ll_b, k_b),
    ('C: full HH heterogeneity', ll_c, k_c),
]
table = pd.DataFrame([
    {
        'spec':     name,
        'LL':       ll,
        'n_params': k,
        'AIC':      2 * k - 2 * ll,
        'BIC':      k * math.log(N_OCC) - 2 * ll,
    }
    for name, ll, k in specs
])
table['delta_AIC vs A'] = table['AIC'] - table.iloc[0]['AIC']
table['delta_BIC vs A'] = table['BIC'] - table.iloc[0]['BIC']
table.set_index('spec')
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
      <th>LL</th>
      <th>n_params</th>
      <th>AIC</th>
      <th>BIC</th>
      <th>delta_AIC vs A</th>
      <th>delta_BIC vs A</th>
    </tr>
    <tr>
      <th>spec</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A: pooled MNL</th>
      <td>-3347.713379</td>
      <td>6</td>
      <td>6707.426758</td>
      <td>6744.022261</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: HH price sensitivity</th>
      <td>-2220.629883</td>
      <td>141</td>
      <td>4723.259766</td>
      <td>5583.254095</td>
      <td>-1984.166992</td>
      <td>-1160.768166</td>
    </tr>
    <tr>
      <th>C: full HH heterogeneity</th>
      <td>-1927.055908</td>
      <td>411</td>
      <td>4676.111816</td>
      <td>7182.903797</td>
      <td>-2031.314941</td>
      <td>438.881536</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **household
heterogeneity matters**, and the right model is one that lets price- and
promotional-sensitivity vary across households.

- **Jain, Vilcassim & Chintagunta (1994)** introduced random coefficients
  on price as the canonical Cracker spec. They reject pooled MNL on this
  exact dataset; their random-coefficients logit shows that ignoring
  heterogeneity biases the average price coefficient toward zero (a classic
  "attenuation through aggregation" result). They also find that allowing
  brand-specific state dependence (lagged-choice effects) substantially
  improves fit — Cracker exhibits stronger brand loyalty than Yogurt.
- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6** treats
  panel scanner data of this shape as the canonical motivating example for
  mixed logit, with random coefficients distributed normally or log-normally.
- **McFadden & Train (2000)** prove that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary.

For Cracker specifically, the recommended specification is therefore a
**mixed logit with random coefficients on price** (and optionally on disp /
feat), plus brand-specific state-dependence terms. `torch-choice` does not
yet ship a Monte-Carlo or quadrature-based mixed-logit estimator, so the
closest in-package approximations are Specs B and C, which estimate one
coefficient per household (fixed effects).

**Practical guidance for this notebook's reader:**

1. Use **Spec A** (pooled MNL) as a baseline, mainly to verify the workflow
   end-to-end and as a numerical reference (R/`mlogit` agreement, §4).
2. Use **Spec B** (per-household price coefficient) when the research
   question asks about *price elasticity heterogeneity* — e.g., does
   household X respond more to discounts than household Y? Spec B's
   identification is strongest where households face within-household price
   variation, which Cracker has plenty of (within-store week-to-week price
   variation across all four brands).
3. Use **Spec C** (per-household price *and* promotional coefficient) when
   all marketing instruments are believed to elicit heterogeneous response.
   With ~24 purchases per household, Spec C's per-household disp/feat
   coefficient is identified primarily off households that experience
   promotions for multiple brands; expect noisier individual estimates.
4. The published JVC random-coefficients spec remains the gold standard;
   reach for it when `torch-choice` adds mixed-logit support, or use Apollo
   / Stata's `mixlogit` in the meantime.

**Caveat.** §5's alternative specs are *not* cross-validated against
`mlogit` (R doesn't natively fit per-user fixed-effects MNL with this many
parameters in the same form). The §4 verification covers Spec A only. Specs
B and C are internal-to-`torch-choice` model-fit comparisons.
