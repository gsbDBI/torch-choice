# Yogurt brand choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of household yogurt brand choice
on the Jain–Vilcassim–Chintagunta (JBES, 1994) panel. We fit the same model
twice — once in R with `mlogit`, once in Python with `torch-choice` — and show
that the two implementations recover the same coefficients to numerical
precision. We then explore richer alternative specifications and discuss
which is theoretically preferred.


## 1. About this dataset

**Domain.** Consumer packaged goods / brand choice. 100 households make 2,412
yogurt purchases over time, choosing one of four brands per trip:

| Brand | Notes |
|---|---|
| `yoplait` | Reference (premium) |
| `dannon` | Mass-market |
| `hiland` | Local (regional) |
| `weight` | "Weight Watchers" |

Each row is one purchase occasion. For every occasion we observe the **price**
charged for all four brands and a **feature** indicator (`feat.*` = 1 if the
brand was on a newspaper feature/promotion that week). The chosen brand is in
the `choice` column.

**Reference.** Jain, D. C., Vilcassim, N. J., Chintagunta, P. K. (1994).
"A Random-Coefficients Logit Brand-Choice Model Applied to Panel Data."
*Journal of Business & Economic Statistics* 12(3), 317–328.
([doi:10.1080/07350015.1994.10524547](https://doi.org/10.1080/07350015.1994.10524547))

**License.** Distributed in the R package `Ecdat` under GPL-2, redistributable
with citation.

### How to download

The CSV in this folder was extracted from the cran/Ecdat GitHub mirror:

```python
import pyreadr
rda = pyreadr.read_r_url(
    "https://raw.githubusercontent.com/cran/Ecdat/master/data/Yogurt.rda"
)
yogurt = list(rda.values())[0]   # only one DataFrame in the .rda
```

Or, in R:

```r
data(Yogurt, package = "Ecdat")
write.csv(Yogurt, "yogurt.csv", row.names = FALSE)
```

The notebook reads `yogurt.csv` from this folder.



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
df = pd.read_csv(HERE / "yogurt.csv")
print(f"shape={df.shape}, n_households={df['id'].nunique()}, n_occasions={len(df)}")
df.head()

```

    shape=(2412, 10), n_households=100, n_occasions=2412





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
      <th>feat.yoplait</th>
      <th>feat.dannon</th>
      <th>feat.hiland</th>
      <th>feat.weight</th>
      <th>price.yoplait</th>
      <th>price.dannon</th>
      <th>price.hiland</th>
      <th>price.weight</th>
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
      <td>10.8</td>
      <td>8.1</td>
      <td>6.1</td>
      <td>7.9</td>
      <td>weight</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.8</td>
      <td>9.8</td>
      <td>6.4</td>
      <td>7.5</td>
      <td>dannon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.8</td>
      <td>9.8</td>
      <td>6.1</td>
      <td>8.6</td>
      <td>dannon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.8</td>
      <td>9.8</td>
      <td>6.1</td>
      <td>8.6</td>
      <td>dannon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.5</td>
      <td>9.8</td>
      <td>4.9</td>
      <td>7.9</td>
      <td>dannon</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook MNL on the wide-format CSV using `mlogit::mlogit`, with
`yoplait` as the reference alternative:

```r
suppressPackageStartupMessages(library(mlogit))
df   <- read.csv("yogurt.csv")
data <- dfidx(df, shape="wide", choice="choice", varying=2:9,
              sep=".", idnames=c("chid","alt"))
mod  <- mlogit(choice ~ price + feat, data=data, reflevel="yoplait")
summary(mod)
```

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to the
cached output in `mlogit_output.json` (also next to this notebook), so the
comparison still renders for readers without an R installation.



```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "yogurt.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df

```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -2656.8879





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
      <td>(Intercept):dannon</td>
      <td>-0.734571</td>
      <td>0.080644</td>
      <td>-9.108791</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Intercept):hiland</td>
      <td>-4.450166</td>
      <td>0.187118</td>
      <td>-23.782711</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Intercept):weight</td>
      <td>-1.375755</td>
      <td>0.088982</td>
      <td>-15.461097</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>price</td>
      <td>-0.366584</td>
      <td>0.024366</td>
      <td>-15.044876</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>feat</td>
      <td>0.491433</td>
      <td>0.120063</td>
      <td>4.093129</td>
      <td>0.000043</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_price|constant) + (itemsession_feat|constant) + (intercept|item)
```

— one shared coefficient on `price`, one shared coefficient on `feat`, and an
alt-specific intercept (the first item, `yoplait`, is the reference, with
intercept pinned to zero, which matches R's `reflevel="yoplait"`).

### Build a `ChoiceDataset`

The CSV is in *wide* format (one row per occasion, with brand-suffixed price /
feat columns). We reshape to long format and pivot into the 3D
`(num_sessions, num_items, num_features)` tensors `torch-choice` expects.



```python
# Brand encoding: index 0 is the reference (matches R's reflevel="yoplait").
BRAND_ORDER = ["yoplait", "dannon", "hiland", "weight"]
brand_to_idx = {b: i for i, b in enumerate(BRAND_ORDER)}

# Wide -> long: 2,412 occasions x 4 brands = 9,648 rows.
df = df.copy()
df["occasion"] = np.arange(len(df))
long_records = []
for brand, idx in brand_to_idx.items():
    long_records.append(
        df[["occasion", f"price.{brand}", f"feat.{brand}"]]
        .rename(columns={f"price.{brand}": "price", f"feat.{brand}": "feat"})
        .assign(brand=idx)
    )
long_df = pd.concat(long_records, ignore_index=True).sort_values(["occasion", "brand"])

# Pivot into (num_sessions, num_items, num_features) tensors.
itemsession_price = utils.pivot3d(long_df, dim0="occasion", dim1="brand", values="price")
itemsession_feat  = utils.pivot3d(long_df, dim0="occasion", dim1="brand", values="feat")

# Per-occasion chosen brand index, household id, and session id.
item_index    = torch.LongTensor(df["choice"].map(brand_to_idx).values)
user_index    = torch.LongTensor(df["id"].astype(int).values - 1)  # 0-indexed
session_index = torch.arange(len(df))
NUM_USERS     = int(user_index.max().item()) + 1   # 100 households
N_OCC         = len(item_index)                    # 2412 occasions

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=4,
    num_users=NUM_USERS,
    user_index=user_index,
    session_index=session_index,
    itemsession_price=itemsession_price,
    itemsession_feat=itemsession_feat,
)
print(dataset)

```

    ChoiceDataset(num_items=4, num_users=100, num_sessions=2412, label=[], item_index=[2412], user_index=[2412], session_index=[2412], item_availability=[], itemsession_price=[2412, 4, 1], itemsession_feat=[2412, 4, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


### Fit (Spec A — pooled MNL)

We use full-batch LBFGS for 1,000 epochs — the same recipe the package's
`paper_demo.py` uses for ModeCanada. `model.fit(...)` returns an
`EstimationOutput` whose `coef_summary` mirrors a standard regression table.



```python
model = ConditionalLogitModel(
    formula="(itemsession_price|constant) + (itemsession_feat|constant) + (intercept|item)",
    dataset=dataset,
    num_items=4,
)
print(model)

```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_price[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_feat[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_price[constant]] with 1 parameters, with constant level variation.
    X[itemsession_feat[constant]] with 1 parameters, with constant level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu



```python
result = model.fit(
    dataset,
    batch_size=-1,
    learning_rate=0.01,
    num_epochs=1000,
    model_optimizer="LBFGS",
    backend="lightning",
    print_summary=False,
)
print(result)
print(f"\ntorch-choice train log-likelihood: {result.train_ll:.4f}")

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
    0 | model | ConditionalLogitModel | 5      | train | 0    
    ----------------------------------------------------------------
    5         Trainable params
    0         Non-trainable params
    5         Total params
    0.000     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -2656.888, [Validation] None, [Test] None
    
    | Coefficient                   |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_price[constant]_0 |    -0.366582 |   0.0243659 |   -15.045 | < 2e-16    | ***            |
    | itemsession_feat[constant]_0  |     0.491423 |   0.120063  |     4.093 | 4.257e-05  | ***            |
    | intercept[item]_0             |    -0.734573 |   0.080644  |    -9.109 | < 2e-16    | ***            |
    | intercept[item]_1             |    -4.45013  |   0.187116  |   -23.783 | < 2e-16    | ***            |
    | intercept[item]_2             |    -1.37575  |   0.0889814 |   -15.461 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -2656.8879


## 4. Side-by-side comparison

The two implementations expose coefficient names differently. `mlogit`
labels alternative-specific intercepts as `(Intercept):dannon` etc., while
`torch-choice` indexes them positionally inside `intercept[item]_*`. The
mapping below is the only per-dataset adapter required for the comparison
helper.



```python
NAME_MAP = {
    "itemsession_price[constant]_0": "price",
    "itemsession_feat[constant]_0":  "feat",
    "intercept[item]_0":             "(Intercept):dannon",
    "intercept[item]_1":             "(Intercept):hiland",
    "intercept[item]_2":             "(Intercept):weight",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff

```

    [compare_coefs] estimates: max |diff| = 3.856e-05 (0.0022%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 1.777e-06 (0.0009%); tol 1e-03 -> PASS
    
    LL: mlogit=-2656.8879  torch-choice=-2656.8879  abs_diff=6.15e-05





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
      <td>-0.366584</td>
      <td>-0.366582</td>
      <td>0.000003</td>
      <td>0.000724</td>
      <td>0.024366</td>
      <td>0.024366</td>
      <td>1.176509e-07</td>
      <td>0.000483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_feat[constant]_0</td>
      <td>0.491433</td>
      <td>0.491423</td>
      <td>0.000011</td>
      <td>0.002172</td>
      <td>0.120063</td>
      <td>0.120063</td>
      <td>2.266068e-07</td>
      <td>0.000189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>intercept[item]_0</td>
      <td>-0.734571</td>
      <td>-0.734573</td>
      <td>0.000001</td>
      <td>0.000195</td>
      <td>0.080644</td>
      <td>0.080644</td>
      <td>2.345911e-07</td>
      <td>0.000291</td>
    </tr>
    <tr>
      <th>3</th>
      <td>intercept[item]_1</td>
      <td>-4.450166</td>
      <td>-4.450128</td>
      <td>0.000039</td>
      <td>0.000867</td>
      <td>0.187118</td>
      <td>0.187116</td>
      <td>1.777026e-06</td>
      <td>0.000950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>intercept[item]_2</td>
      <td>-1.375755</td>
      <td>-1.375749</td>
      <td>0.000007</td>
      <td>0.000495</td>
      <td>0.088982</td>
      <td>0.088981</td>
      <td>3.240716e-07</td>
      <td>0.000364</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion of §4.** `torch-choice` reproduces `mlogit`'s MNL coefficient
estimates and log-likelihood to within float32 round-off. This validates the
package's optimizer and standard-error implementation against an established
reference.


## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong assumption:
*every household responds identically to price changes and to newspaper
features*. The original Jain–Vilcassim–Chintagunta (1994) paper questions
exactly this assumption — its central contribution is a **random-coefficients
logit** that lets price- and feature-sensitivity vary across households. They
argue the panel structure (~24 purchases per household on average) makes
heterogeneity identifiable, and they find substantial taste variation: the
estimated standard deviation of the price coefficient is comparable in
magnitude to its mean.

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_price\|constant) + (itemsession_feat\|constant) + (intercept\|item)` | one price coef, one feat coef, three brand intercepts | 5 |
| **B** (household price sensitivity) | `(itemsession_price\|user) + (itemsession_feat\|constant) + (intercept\|item)` | each of 100 households gets its own price coefficient; feat shared | 100+1+3 = 104 |
| **C** (full household heterogeneity) | `(itemsession_price\|user) + (itemsession_feat\|user) + (intercept\|item)` | each household gets its own price *and* feat coefficient | 100+100+3 = 203 |

Specs B and C are *fixed-effects* approximations to the JVC 1994
random-coefficients logit. They estimate one coefficient per household
(treating heterogeneity as known finite parameters) instead of integrating
over a continuous mixing distribution. With 100 households and ~24 purchases
each, the FE approach is feasible; with thousands of users it would not be.

### Fit Specs B and C



```python
def fit_spec(formula: str, label: str, *, num_epochs: int = 1000,
             optimizer: str = "LBFGS") -> tuple[float, int, "EstimationOutput"]:
    """Fit a torch-choice spec; return (train_ll, n_params, result)."""
    torch.manual_seed(0)
    m = ConditionalLogitModel(
        formula=formula, dataset=dataset, num_items=4, num_users=NUM_USERS,
    )
    r = m.fit(
        dataset,
        batch_size=-1,
        learning_rate=0.01,
        num_epochs=num_epochs,
        model_optimizer=optimizer,
        backend="lightning",
        print_summary=False,
    )
    n_params = int(r.coef_summary.shape[0])
    return float(r.train_ll), n_params, r

ll_a, k_a = float(result.train_ll), int(result.coef_summary.shape[0])

ll_b, k_b, result_b = fit_spec(
    "(itemsession_price|user) + (itemsession_feat|constant) + (intercept|item)",
    "B: HH price sensitivity",
)

ll_c, k_c, result_c = fit_spec(
    "(itemsession_price|user) + (itemsession_feat|user) + (intercept|item)",
    "C: full HH heterogeneity",
)

print(f"Spec A: LL={ll_a:.2f}, n_params={k_a}")
print(f"Spec B: LL={ll_b:.2f}, n_params={k_b}")
print(f"Spec C: LL={ll_c:.2f}, n_params={k_c}")

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
    0 | model | ConditionalLogitModel | 104    | train | 0    
    ----------------------------------------------------------------
    104       Trainable params
    0         Non-trainable params
    104       Total params
    0.000     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    GPU available: True (mps), used: False


    TPU available: False, using: 0 TPU cores


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.


    Starting PyTorch Lightning training loop.


    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    
      | Name  | Type                  | Params | Mode  | FLOPs
    ----------------------------------------------------------------
    0 | model | ConditionalLogitModel | 203    | train | 0    
    ----------------------------------------------------------------
    203       Trainable params
    0         Non-trainable params
    203       Total params
    0.001     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Spec A: LL=-2656.89, n_params=5
    Spec B: LL=-2085.90, n_params=104
    Spec C: LL=-1980.28, n_params=203


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively, so it tends to favor parsimonious models — appropriate when we
believe the simpler spec is closer to the truth. AIC penalizes less and tends
to favor richer models when sample size is large.



```python
specs = [
    ("A: pooled MNL",         ll_a, k_a),
    ("B: HH price sensitivity",  ll_b, k_b),
    ("C: full HH heterogeneity", ll_c, k_c),
]
table = pd.DataFrame([
    {
        "spec":   name,
        "LL":     ll,
        "n_params": k,
        "AIC":    2 * k - 2 * ll,
        "BIC":    k * math.log(N_OCC) - 2 * ll,
    }
    for name, ll, k in specs
])
table["ΔAIC vs A"] = table["AIC"] - table.iloc[0]["AIC"]
table["ΔBIC vs A"] = table["BIC"] - table.iloc[0]["BIC"]
table.set_index("spec")

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
      <th>ΔAIC vs A</th>
      <th>ΔBIC vs A</th>
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
      <td>-2656.887939</td>
      <td>5</td>
      <td>5323.775879</td>
      <td>5352.716937</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: HH price sensitivity</th>
      <td>-2085.898682</td>
      <td>104</td>
      <td>4379.797363</td>
      <td>4981.771365</td>
      <td>-943.978516</td>
      <td>-370.945571</td>
    </tr>
    <tr>
      <th>C: full HH heterogeneity</th>
      <td>-1980.280518</td>
      <td>203</td>
      <td>4366.561035</td>
      <td>5541.567981</td>
      <td>-957.214844</td>
      <td>188.851045</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **household
heterogeneity matters**, and the right model is one that lets price- and
feature-sensitivity vary across households.

- **Jain, Vilcassim & Chintagunta (1994)** reject pooled MNL on this exact
  dataset. Their random-coefficients logit shows that ignoring heterogeneity
  biases the average price coefficient toward zero (a classic "attenuation
  through aggregation" result).
- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6** treats
  panel SP and RP data of this shape as the canonical motivating example for
  mixed logit, with random coefficients distributed normally or log-normally.
- **McFadden & Train (2000)** prove that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary.

For Yogurt specifically, the recommended specification is therefore a **mixed
logit with random coefficients on price and feat**. `torch-choice` does not
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
   variation.
3. Use **Spec C** (per-household price *and* feature coefficient) when both
   marketing instruments are believed to elicit heterogeneous response. With
   the small sample (≈24 purchases per household), Spec C's per-household
   feat coefficient is identified primarily off households that experience
   features for multiple brands; expect noisier individual estimates.
4. The published JVC random-coefficients spec remains the gold standard;
   reach for it when `torch-choice` adds mixed-logit support, or use Apollo /
   Stata's `mixlogit` in the meantime.

**Caveat.** §5's alternative specs are *not* cross-validated against `mlogit`
(R doesn't natively fit per-user fixed-effects MNL with this many parameters
in the same form). The §4 verification covers Spec A only. Specs B and C are
internal-to-`torch-choice` model-fit comparisons.

