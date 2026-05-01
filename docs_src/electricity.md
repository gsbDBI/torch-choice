# Residential electricity supplier choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a pooled multinomial logit model of residential
electricity-supplier choice on Kenneth Train's stated-preference panel
(361 U.S. households × ~12 hypothetical choice occasions each = 4,308
observations). We fit the same model twice — once in R with `mlogit`, once
in Python with `torch-choice` — and show that the two implementations
recover the same coefficients to numerical precision.

## 1. About this dataset

**Domain.** Energy retail / residential electricity-supplier choice. In a
stated-preference experiment, each of 361 U.S. households was shown ~12
scenarios. In every scenario the household picks one of 4 hypothetical
suppliers. Each supplier is described by 6 attributes:

| Attribute | Meaning |
|---|---|
| `pf` | Fixed price per kWh (cents) |
| `cl` | Contract length (years; 0 = no contract / cancel anytime) |
| `loc` | 1 if the supplier is a local company |
| `wk` | 1 if the supplier is a well-known company |
| `tod` | 1 if the supplier offers time-of-day pricing (11¢ day / 5¢ night) |
| `seas` | 1 if the supplier offers seasonal pricing (summer/winter/shoulder) |

The 4 supplier "labels" 1–4 are *hypothetical placeholders*, not real brands,
so the canonical mlogit spec uses `| 0` to suppress alt-specific intercepts.

**Reference.** Revelt, D., Train, K. (2001). "Customer-Specific Taste
Parameters and Mixed Logit: Households' Choice of Electricity Supplier."
*Econometrics* 0012001, University Library of Munich.

**License.** Distributed in the R package `mlogit` under GPL-2,
redistributable with citation.

**Caveat: pooled MNL vs mixed logit.** Train's textbook treatment
(*Discrete Choice Methods with Simulation*, ch. 6) fits a **mixed logit**
with random coefficients on every attribute, exploiting the panel structure
(multiple occasions per household). For v1 of this tutorial we replicate
the simpler **pooled MNL** from `?Electricity`, which treats the 4,308
occasions as independent. A mixed-logit replication is a v2 follow-up.

### How to download

The CSV in this folder was extracted from the cran/`mlogit` GitHub mirror:

```python
import pyreadr
rda = pyreadr.read_r_url(
    "https://raw.githubusercontent.com/cran/mlogit/master/data/Electricity.rda"
)
electricity = list(rda.values())[0]
```

Or, in R:

```r
data(Electricity, package = "mlogit")
write.csv(Electricity, "electricity.csv", row.names = FALSE)
```

The notebook reads `electricity.csv` from this folder.


```python
# Make `tutorials/_mlogit_compare.py` importable from this folder.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from _mlogit_compare import run_or_load_mlogit, compare_coefs

HERE = Path.cwd()
```


```python
df = pd.read_csv(HERE / "electricity.csv")
n_households = df["id"].nunique()
median_per_hh = int(df.groupby("id").size().median())
print(
    f"shape={df.shape}, n_households={n_households}, "
    f"n_occasions={len(df)} (median {median_per_hh} per household)"
)
df.head()
```

    shape=(4308, 26), n_households=361, n_occasions=4308 (median 12 per household)





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
      <th>choice</th>
      <th>id</th>
      <th>pf1</th>
      <th>pf2</th>
      <th>pf3</th>
      <th>pf4</th>
      <th>cl1</th>
      <th>cl2</th>
      <th>cl3</th>
      <th>cl4</th>
      <th>...</th>
      <th>wk3</th>
      <th>wk4</th>
      <th>tod1</th>
      <th>tod2</th>
      <th>tod3</th>
      <th>tod4</th>
      <th>seas1</th>
      <th>seas2</th>
      <th>seas3</th>
      <th>seas4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



## 2. R/`mlogit` reference fit

We fit a pooled MNL on the wide-per-occasion CSV using `mlogit::mlogit`,
with **no alternative-specific intercepts** (the `| 0` term suppresses them
because the four "suppliers" are interchangeable hypothetical placeholders
rather than real brands):

```r
suppressPackageStartupMessages(library(mlogit))
df       <- read.csv("electricity.csv")
df$chid  <- seq_len(nrow(df))
data <- mlogit.data(df, id.var="id", choice="choice", varying=3:26,
                    shape="wide", sep="")
mod  <- mlogit(choice ~ pf + cl + loc + wk + tod + seas | 0, data=data)
summary(mod)
```

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to
the cached output in `mlogit_output.json` (also next to this notebook), so
the comparison still renders for readers without an R installation.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "electricity.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -4958.6491





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
      <td>pf</td>
      <td>-0.625228</td>
      <td>0.023222</td>
      <td>-26.923575</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cl</td>
      <td>-0.108299</td>
      <td>0.008244</td>
      <td>-13.136373</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>loc</td>
      <td>1.442243</td>
      <td>0.050557</td>
      <td>28.526996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wk</td>
      <td>0.995504</td>
      <td>0.044780</td>
      <td>22.230958</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tod</td>
      <td>-5.462759</td>
      <td>0.183713</td>
      <td>-29.735366</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>seas</td>
      <td>-5.840031</td>
      <td>0.186678</td>
      <td>-31.283997</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_pf|constant) + (itemsession_cl|constant)
  + (itemsession_loc|constant) + (itemsession_wk|constant)
  + (itemsession_tod|constant) + (itemsession_seas|constant)
```

— one shared coefficient per attribute, **no** `(intercept|item)` term
(matching mlogit's `| 0`).

### Build a `ChoiceDataset`

The CSV is in *wide-per-occasion* format: each row already lists all four
suppliers' six attributes side-by-side (`pf1, pf2, pf3, pf4, cl1, ...`).
We reshape to long format and pivot into the 3D
`(num_sessions, num_items, num_features)` tensors `torch-choice` expects.


```python
# Supplier encoding: index 0 = supplier 1 (mlogit's default reference).
ATTRS = ["pf", "cl", "loc", "wk", "tod", "seas"]
SUPPLIERS = [1, 2, 3, 4]                    # mlogit's labels
supplier_to_idx = {s: i for i, s in enumerate(SUPPLIERS)}

# Wide -> long: 4,308 occasions x 4 suppliers = 17,232 rows.
df = df.copy()
df["occasion"] = np.arange(len(df))
long_records = []
for supplier, idx in supplier_to_idx.items():
    cols = {f"{a}{supplier}": a for a in ATTRS}
    long_records.append(
        df[["occasion", *cols.keys()]]
        .rename(columns=cols)
        .assign(supplier=idx)
    )
long_df = pd.concat(long_records, ignore_index=True).sort_values(
    ["occasion", "supplier"]
)

# Pivot each attribute into a (num_sessions, num_items, 1) tensor.
itemsession_tensors = {
    f"itemsession_{a}": utils.pivot3d(
        long_df, dim0="occasion", dim1="supplier", values=a
    )
    for a in ATTRS
}

# Per-occasion chosen supplier index, household id, and session id.
item_index    = torch.LongTensor(df["choice"].astype(int).map(supplier_to_idx).values)
user_index    = torch.LongTensor(df["id"].astype(int).values - 1)  # 0-indexed
session_index = torch.arange(len(df))

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=4,
    user_index=user_index,
    session_index=session_index,
    **itemsession_tensors,
)
print(dataset)
```

    ChoiceDataset(num_items=4, num_users=361, num_sessions=4308, label=[], item_index=[4308], user_index=[4308], session_index=[4308], item_availability=[], itemsession_pf=[4308, 4, 1], itemsession_cl=[4308, 4, 1], itemsession_loc=[4308, 4, 1], itemsession_wk=[4308, 4, 1], itemsession_tod=[4308, 4, 1], itemsession_seas=[4308, 4, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:241: UserWarning: The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.")
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


### Fit

We use full-batch LBFGS for 1,000 epochs — the same recipe the package's
`paper_demo.py` uses for ModeCanada. `model.fit(...)` returns an
`EstimationOutput` whose `coef_summary` mirrors a standard regression table.


```python
FORMULA = (
    "(itemsession_pf|constant) + (itemsession_cl|constant) "
    "+ (itemsession_loc|constant) + (itemsession_wk|constant) "
    "+ (itemsession_tod|constant) + (itemsession_seas|constant)"
)
model = ConditionalLogitModel(
    formula=FORMULA,
    dataset=dataset,
    num_items=4,
)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_pf[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_cl[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_loc[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_wk[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_tod[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_seas[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_pf[constant]] with 1 parameters, with constant level variation.
    X[itemsession_cl[constant]] with 1 parameters, with constant level variation.
    X[itemsession_loc[constant]] with 1 parameters, with constant level variation.
    X[itemsession_wk[constant]] with 1 parameters, with constant level variation.
    X[itemsession_tod[constant]] with 1 parameters, with constant level variation.
    X[itemsession_seas[constant]] with 1 parameters, with constant level variation.
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
    0 | model | ConditionalLogitModel | 6      | train | 0    
    ----------------------------------------------------------------
    6         Trainable params
    0         Non-trainable params
    6         Total params
    0.000     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -4958.649, [Validation] None, [Test] None
    
    | Coefficient                  |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:-----------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_pf[constant]_0   |    -0.625204 |  0.0232221  |   -26.923 | < 2e-16    | ***            |
    | itemsession_cl[constant]_0   |    -0.108299 |  0.00824419 |   -13.136 | < 2e-16    | ***            |
    | itemsession_loc[constant]_0  |     1.44224  |  0.0505569  |    28.527 | < 2e-16    | ***            |
    | itemsession_wk[constant]_0   |     0.995502 |  0.0447799  |    22.231 | < 2e-16    | ***            |
    | itemsession_tod[constant]_0  |    -5.46257  |  0.183711   |   -29.735 | < 2e-16    | ***            |
    | itemsession_seas[constant]_0 |    -5.83985  |  0.186677   |   -31.283 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -4958.6494


## 4. Side-by-side comparison

The two implementations expose coefficient names differently. `mlogit`
uses the bare attribute names (`pf`, `cl`, ...), while `torch-choice`
namespaces them as `itemsession_<attr>[constant]_0`. The mapping below is
the only per-dataset adapter required for the comparison helper.


```python
NAME_MAP = {
    "itemsession_pf[constant]_0":   "pf",
    "itemsession_cl[constant]_0":   "cl",
    "itemsession_loc[constant]_0":  "loc",
    "itemsession_wk[constant]_0":   "wk",
    "itemsession_tod[constant]_0":  "tod",
    "itemsession_seas[constant]_0": "seas",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 1.861e-04 (0.0037%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 1.258e-06 (0.0008%); tol 1e-03 -> PASS
    
    LL: mlogit=-4958.6491  torch-choice=-4958.6494  abs_diff=2.95e-04





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
      <td>itemsession_pf[constant]_0</td>
      <td>-0.625228</td>
      <td>-0.625204</td>
      <td>2.332132e-05</td>
      <td>0.003730</td>
      <td>0.023222</td>
      <td>0.023222</td>
      <td>1.933299e-07</td>
      <td>0.000833</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_cl[constant]_0</td>
      <td>-0.108299</td>
      <td>-0.108299</td>
      <td>1.850313e-07</td>
      <td>0.000171</td>
      <td>0.008244</td>
      <td>0.008244</td>
      <td>2.314026e-08</td>
      <td>0.000281</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_loc[constant]_0</td>
      <td>1.442243</td>
      <td>1.442240</td>
      <td>2.752147e-06</td>
      <td>0.000191</td>
      <td>0.050557</td>
      <td>0.050557</td>
      <td>1.999113e-07</td>
      <td>0.000395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>itemsession_wk[constant]_0</td>
      <td>0.995504</td>
      <td>0.995502</td>
      <td>2.307260e-06</td>
      <td>0.000232</td>
      <td>0.044780</td>
      <td>0.044780</td>
      <td>1.430920e-07</td>
      <td>0.000320</td>
    </tr>
    <tr>
      <th>4</th>
      <td>itemsession_tod[constant]_0</td>
      <td>-5.462759</td>
      <td>-5.462573</td>
      <td>1.860803e-04</td>
      <td>0.003406</td>
      <td>0.183713</td>
      <td>0.183711</td>
      <td>1.201802e-06</td>
      <td>0.000654</td>
    </tr>
    <tr>
      <th>5</th>
      <td>itemsession_seas[constant]_0</td>
      <td>-5.840031</td>
      <td>-5.839845</td>
      <td>1.856531e-04</td>
      <td>0.003179</td>
      <td>0.186678</td>
      <td>0.186677</td>
      <td>1.258492e-06</td>
      <td>0.000674</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

`torch-choice` reproduces `mlogit`'s pooled-MNL coefficient estimates and
log-likelihood to within float32 round-off on the Electricity SP panel. The
next step for this dataset is the **mixed-logit** specification from
Train (2009, ch. 6), which exploits the 4-occasions-per-household panel
structure with random coefficients on every attribute.

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong assumption:
*every household responds identically to price and contract length*. This is
exactly the assumption that Kenneth Train and co-authors challenge in the
canonical treatments of this dataset. **Train (2009),
*Discrete Choice Methods with Simulation*, Ch. 6** uses this very SP panel as
the textbook motivating example for **mixed logit** with random coefficients,
arguing that the panel design (~12 hypothetical scenarios per household) was
constructed precisely to identify *taste heterogeneity*. **Revelt & Train
(1998)** — the original SP-electricity application — show that household-
specific price and contract-length sensitivities are economically large and
statistically well-identified.

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_pf\|constant) + (itemsession_cl\|constant) + (itemsession_loc\|constant) + (itemsession_wk\|constant) + (itemsession_tod\|constant) + (itemsession_seas\|constant)` | one shared coefficient per attribute | 6 |
| **B** (HH price sensitivity) | swap `itemsession_pf\|constant` → `itemsession_pf\|user`; keep others constant | each of 361 households gets its own price (`pf`) coefficient; the rest are pooled | 361+5 = 366 |
| **C** (HH price + contract length sensitivity) | also swap `itemsession_cl\|constant` → `itemsession_cl\|user` | each household gets its own *price* and *contract length* coefficient | 361+361+4 = 726 |

Specs B and C are *fixed-effects* approximations to the Train (2009) /
Revelt & Train (1998) random-coefficients logit. They estimate one coefficient
per household (treating heterogeneity as known finite parameters) instead of
integrating over a continuous mixing distribution. With 361 households and
~12 scenarios each, the FE approach is feasible; with millions of users it
would not be.

### Fit Specs B and C


```python
import math

# Helper constants for Specs B / C and the AIC/BIC table.
NUM_USERS = int(user_index.max().item()) + 1   # 361 households
N_OCC     = len(item_index)                    # 4308 occasions


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
    "(itemsession_pf|user) + (itemsession_cl|constant) "
    "+ (itemsession_loc|constant) + (itemsession_wk|constant) "
    "+ (itemsession_tod|constant) + (itemsession_seas|constant)",
    "B: HH price sensitivity",
)

ll_c, k_c, result_c = fit_spec(
    "(itemsession_pf|user) + (itemsession_cl|user) "
    "+ (itemsession_loc|constant) + (itemsession_wk|constant) "
    "+ (itemsession_tod|constant) + (itemsession_seas|constant)",
    "C: HH price + contract length sensitivity",
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
    0 | model | ConditionalLogitModel | 366    | train | 0    
    ----------------------------------------------------------------
    366       Trainable params
    0         Non-trainable params
    366       Total params
    0.001     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:241: UserWarning: The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.")
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
    0 | model | ConditionalLogitModel | 726    | train | 0    
    ----------------------------------------------------------------
    726       Trainable params
    0         Non-trainable params
    726       Total params
    0.003     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:241: UserWarning: The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of users is inferred from the number of unique users in the user_index tensor. This might lead to unexpected behaviors if some users never appeared in the user_index tensor. For a safer behavior, please provide the number of users explicitly by using the num_users keyword while initializing the ChoiceDataset class.")
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Spec A: LL=-4958.65, n_params=6
    Spec B: LL=-4052.33, n_params=366
    Spec C: LL=-3377.33, n_params=726


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively, so it tends to favor parsimonious models — appropriate when we
believe the simpler spec is closer to the truth. AIC penalizes less and tends
to favor richer models when sample size is large.


```python
specs = [
    ("A: pooled MNL",                            ll_a, k_a),
    ("B: HH price sensitivity",                  ll_b, k_b),
    ("C: HH price + contract length sensitivity", ll_c, k_c),
]
table = pd.DataFrame([
    {
        "spec":     name,
        "LL":       ll,
        "n_params": k,
        "AIC":      2 * k - 2 * ll,
        "BIC":      k * math.log(N_OCC) - 2 * ll,
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
      <td>-4958.649414</td>
      <td>6</td>
      <td>9929.298828</td>
      <td>9967.508202</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: HH price sensitivity</th>
      <td>-4052.331299</td>
      <td>366</td>
      <td>8836.662598</td>
      <td>11167.434426</td>
      <td>-1092.636230</td>
      <td>1199.926223</td>
    </tr>
    <tr>
      <th>C: HH price + contract length sensitivity</th>
      <td>-3377.329590</td>
      <td>726</td>
      <td>8206.659180</td>
      <td>12829.993461</td>
      <td>-1722.639648</td>
      <td>2862.485259</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **household
heterogeneity matters**, and the right model is one that lets price and
contract-length sensitivity vary across households.

- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6** treats
  this exact SP-electricity panel as the canonical motivating example for
  mixed logit. Random coefficients on `pf` (price), `cl` (contract length),
  and the qualitative attributes are needed because households trade off
  cost vs. lock-in vs. supplier reputation in heterogeneous ways.
- **Revelt & Train (1998)** — the original SP-electricity paper — fit a
  mixed logit on this dataset and find sizable across-household standard
  deviations on price and contract length, with substantively different
  policy implications from a pooled MNL (e.g., aggregate price elasticities
  are no longer correctly estimated when heterogeneity is ignored).
- **McFadden & Train (2000)** prove that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary across the population
  in this way.

For the Train SP-electricity dataset specifically, the recommended
specification is therefore a **mixed logit with random coefficients on price
and contract length** (and, in many published variants, on the qualitative
attributes as well). `torch-choice` does not yet ship a Monte-Carlo or
quadrature-based mixed-logit estimator, so the closest in-package
approximations are Specs B and C, which estimate one coefficient per
household (fixed effects).

**Practical guidance for this notebook's reader:**

1. Use **Spec A** (pooled MNL) as a baseline, mainly to verify the workflow
   end-to-end and as a numerical reference (R/`mlogit` agreement, §4).
2. Use **Spec B** (per-household price coefficient) when the research
   question asks about *price elasticity heterogeneity* — e.g., does
   household X respond more to a 1¢/kWh increase than household Y? With
   ~12 SP scenarios per household and substantial within-household price
   variation by experimental design, Spec B's per-household price
   coefficients are well identified.
3. Use **Spec C** (per-household price *and* contract length coefficient)
   when both economically primary attributes are believed to elicit
   heterogeneous response — e.g., when modelling the disutility of being
   locked into a multi-year contract for households that move frequently
   versus those that don't.
4. The published Train / Revelt-Train mixed-logit spec remains the gold
   standard; reach for it when `torch-choice` adds mixed-logit support, or
   use Apollo / Stata's `mixlogit` in the meantime.

**Caveat.** §5's alternative specs are *not* cross-validated against
`mlogit` (R doesn't natively fit per-user fixed-effects MNL with this many
parameters in the same form). The §4 verification covers Spec A only.
Specs B and C are internal-to-`torch-choice` model-fit comparisons.
