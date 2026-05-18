# Brownstone-Train Car choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model on the Brownstone & Train
(JoE, 1999) stated-preference vehicle-choice study. We fit the same model
twice — once in R with `mlogit`, once in Python with `torch-choice` — and
show that the two implementations recover the same coefficients to numerical
precision.

## 1. About this dataset

**Domain.** Stated-preference (SP) study of consumer adoption of
alternative-fuel vehicles. 4,654 California respondents each rated **6**
hypothetical vehicle profiles; the chosen profile is recorded in the
`choice` column (values `choice1` … `choice6`).

Each of the 6 alternatives is described by 11 attributes:

| Attribute | Type | Description |
|---|---|---|
| `type` | categorical | Body type (regcar, sportuv, sportcar, stwagon, truck, van) |
| `fuel` | categorical | Fuel (gasoline, methanol, cng, electric) |
| `price` | numeric | Price / log(income) |
| `range` | numeric | Hundreds of miles between refuels/recharges |
| `acc` | numeric | Tens of seconds to 30 mph from stop (smaller is better) |
| `speed` | numeric | Top attainable speed (hundreds of mph) |
| `pollution` | numeric | Tailpipe emissions, fraction of new gas vehicle |
| `size` | numeric | 0=mini, 1=subcompact, 2=compact, 3=mid/large |
| `space` | numeric | Luggage space relative to a comparable new gas vehicle |
| `cost` | numeric | Per-mile cost (tens of cents) |
| `station` | numeric | Fraction of stations that can refuel/recharge |

Respondent-level covariates (`college`, `hsg2`, `coml5`) are also recorded
but are not used in this v1 specification.

**Reference.** Brownstone, D. and Train, K. (1999). "Forecasting new
product penetration with flexible substitution patterns." *Journal of
Econometrics* 89(1-2), 109–129.
([doi:10.1016/S0304-4076(98)00057-8](https://doi.org/10.1016/S0304-4076(98)00057-8))

**License.** Distributed in the R package `mlogit` under GPL-2,
redistributable with citation. Source: Journal of Applied Econometrics data
archive.

**Scope of this notebook.** v1 fits a simplified MNL using only the **9
numeric attributes** (the categorical `type` and `fuel` columns and the
published mixed-logit specification with random coefficients are out of
scope here — see the README).

### How to download

The CSV in this folder was extracted from the cran/mlogit GitHub mirror:

```python
import pyreadr
rda = pyreadr.read_r_url(
    "https://raw.githubusercontent.com/cran/mlogit/master/data/Car.rda"
)
car = list(rda.values())[0]
```

Or, in R:

```r
data(Car, package = "mlogit")
write.csv(Car, "car.csv", row.names = FALSE)
```

The notebook reads `car.csv` from this folder.


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

torch.manual_seed(0)
HERE = Path.cwd()
```


```python
df = pd.read_csv(HERE / "car.csv")
print(f"shape={df.shape}, n_respondents={len(df)}, n_alts=6")
print("choice value counts:")
print(df["choice"].value_counts().sort_index())
df.iloc[:3, :12]
```

    shape=(4654, 70), n_respondents=4654, n_alts=6
    choice value counts:
    choice
    choice1     887
    choice2     269
    choice3    1345
    choice4     349
    choice5    1499
    choice6     305
    Name: count, dtype: int64





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
      <th>college</th>
      <th>hsg2</th>
      <th>coml5</th>
      <th>type1</th>
      <th>type2</th>
      <th>type3</th>
      <th>type4</th>
      <th>type5</th>
      <th>type6</th>
      <th>fuel1</th>
      <th>fuel2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>choice1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>van</td>
      <td>regcar</td>
      <td>van</td>
      <td>stwagon</td>
      <td>van</td>
      <td>truck</td>
      <td>cng</td>
      <td>cng</td>
    </tr>
    <tr>
      <th>1</th>
      <td>choice2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>regcar</td>
      <td>van</td>
      <td>regcar</td>
      <td>stwagon</td>
      <td>regcar</td>
      <td>truck</td>
      <td>methanol</td>
      <td>methanol</td>
    </tr>
    <tr>
      <th>2</th>
      <td>choice5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>regcar</td>
      <td>truck</td>
      <td>regcar</td>
      <td>van</td>
      <td>regcar</td>
      <td>stwagon</td>
      <td>cng</td>
      <td>cng</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook MNL on the wide-format CSV using `mlogit::mlogit`. Two
subtleties make this dataset trickier than Yogurt:

1. **No separator in the wide column names.** Yogurt uses `price.dannon`,
   `feat.dannon`, etc. — a `.` separates attribute and alt label. Car uses
   `price1`, `range1`, ..., `station6` with **no separator**, so we pass
   `sep=""` to `dfidx`. With this setting `dfidx` strips the trailing
   numeric suffix from each attribute column to produce alt levels
   `"1"`, `"2"`, ..., `"6"`.
2. **`choice` is a string.** Its values (`"choice1"` … `"choice6"`) do not
   match the alt levels `"1"` … `"6"` that `dfidx` produced from the wide
   column suffixes. The R script strips the `"choice"` prefix before calling
   `dfidx`, which fixes the mismatch. Without this step `mlogit` silently
   marks every alternative as unchosen and the optimizer fails with
   "missing value where TRUE/FALSE needed."

The mlogit call is:

```r
suppressPackageStartupMessages(library(mlogit))
df          <- read.csv("car.csv")
df$choice   <- as.integer(sub("choice", "", as.character(df$choice)))
data        <- dfidx(df, shape="wide", choice="choice", varying=5:70, sep="")
mod         <- mlogit(
    choice ~ price + range + acc + speed + pollution + size + space + cost + station,
    data = data
)
summary(mod)
```

`varying=5:70` covers the 6 × 11 = 66 alt-specific columns (positions 5–70
in the CSV). Columns 1–4 are `choice`, `college`, `hsg2`, `coml5`. mlogit's
default reference is the first alt level, i.e. `"1"`, which corresponds to
the original `choice1` and matches what we'll pin in `torch-choice`.

The full R script is in `fit_mlogit.R` next to this notebook. The cell
below calls it via `Rscript`; if R isn't installed it transparently falls
back to the cached output in `mlogit_output.json`.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "car.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -7080.8290





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
      <td>(Intercept):2</td>
      <td>-1.193134</td>
      <td>0.069605</td>
      <td>-17.141472</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Intercept):3</td>
      <td>-0.016056</td>
      <td>0.062364</td>
      <td>-0.257454</td>
      <td>7.968283e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Intercept):4</td>
      <td>-1.365133</td>
      <td>0.077531</td>
      <td>-17.607507</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Intercept):5</td>
      <td>-0.325563</td>
      <td>0.092056</td>
      <td>-3.536585</td>
      <td>4.053360e-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Intercept):6</td>
      <td>-1.917805</td>
      <td>0.105289</td>
      <td>-18.214611</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>price</td>
      <td>-0.187130</td>
      <td>0.026997</td>
      <td>-6.931513</td>
      <td>4.163558e-12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>range</td>
      <td>0.004015</td>
      <td>0.000258</td>
      <td>15.573743</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>acc</td>
      <td>-0.072813</td>
      <td>0.010953</td>
      <td>-6.647573</td>
      <td>2.979639e-11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>speed</td>
      <td>0.003400</td>
      <td>0.000768</td>
      <td>4.429826</td>
      <td>9.430924e-06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pollution</td>
      <td>-0.181478</td>
      <td>0.095211</td>
      <td>-1.906054</td>
      <td>5.664323e-02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>size</td>
      <td>0.072032</td>
      <td>0.029170</td>
      <td>2.469442</td>
      <td>1.353239e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>space</td>
      <td>0.914812</td>
      <td>0.175387</td>
      <td>5.215956</td>
      <td>1.828720e-07</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cost</td>
      <td>-0.072335</td>
      <td>0.007442</td>
      <td>-9.719509</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>station</td>
      <td>0.250234</td>
      <td>0.070471</td>
      <td>3.550895</td>
      <td>3.839236e-04</td>
    </tr>
  </tbody>
</table>
</div>



Sanity check: `price` and `cost` should be **negative** (higher price/cost
reduces utility) and `range` should be **positive** (more range is better).
These hold, so the spec is sane and the comparison can proceed.

## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_price|constant) + (itemsession_range|constant) + (itemsession_acc|constant)
  + (itemsession_speed|constant) + (itemsession_pollution|constant)
  + (itemsession_size|constant) + (itemsession_space|constant)
  + (itemsession_cost|constant) + (itemsession_station|constant)
  + (intercept|item)
```

— one shared coefficient per numeric attribute and an alt-specific
intercept. Item 0 (= `choice1`) is the reference, with intercept pinned to
zero, matching mlogit's default `reflevel`.

### Wide-to-long reshape

The CSV has 4,654 rows × 70 columns. We need a long DataFrame with one row
per (occasion, alternative) pair — 27,924 rows × (occasion, alt, 9 numeric
attrs). The reshape is mechanical: for each `alt_idx` in `0..5`, slice out
the columns ending in `f"{alt_idx + 1}"`, rename them to attribute-only
names, attach `alt = alt_idx`, and concatenate. We then sort by
`(occasion, alt)` and pivot to 3D tensors of shape
`(num_sessions, num_items, 1)`.


```python
NUMERIC_ATTRS = [
    "price", "range", "acc", "speed", "pollution",
    "size", "space", "cost", "station",
]
N_ALTS = 6

df = df.copy()
df["occasion"] = np.arange(len(df))
# 'choice1'..'choice6' -> 0..5 (item index 0 is the reference).
df["choice_idx"] = df["choice"].str.replace("choice", "", regex=False).astype(int) - 1

# Wide -> long: 4,654 occasions x 6 alts = 27,924 rows.
long_records = []
for alt_idx in range(N_ALTS):
    suffix = alt_idx + 1  # CSV columns are 1-indexed (e.g. price1..price6)
    rename = {f"{a}{suffix}": a for a in NUMERIC_ATTRS}
    sub = df[["occasion"] + list(rename.keys())].rename(columns=rename)
    sub["alt"] = alt_idx
    long_records.append(sub)
long_df = (
    pd.concat(long_records, ignore_index=True)
    .sort_values(["occasion", "alt"])
    .reset_index(drop=True)
)
print(f"long_df shape: {long_df.shape}  (expected: {len(df) * N_ALTS} x {2 + len(NUMERIC_ATTRS)})")
long_df.head(8)
```

    long_df shape: (27924, 11)  (expected: 27924 x 11)





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
      <th>occasion</th>
      <th>price</th>
      <th>range</th>
      <th>acc</th>
      <th>speed</th>
      <th>pollution</th>
      <th>size</th>
      <th>space</th>
      <th>cost</th>
      <th>station</th>
      <th>alt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4.175345</td>
      <td>250.0</td>
      <td>4.0</td>
      <td>95.0</td>
      <td>0.60</td>
      <td>3.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>4.175345</td>
      <td>250.0</td>
      <td>4.0</td>
      <td>95.0</td>
      <td>0.60</td>
      <td>3.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4.817706</td>
      <td>400.0</td>
      <td>6.0</td>
      <td>110.0</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4.817706</td>
      <td>400.0</td>
      <td>6.0</td>
      <td>110.0</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5.138886</td>
      <td>250.0</td>
      <td>2.5</td>
      <td>140.0</td>
      <td>0.50</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5.138886</td>
      <td>250.0</td>
      <td>2.5</td>
      <td>140.0</td>
      <td>0.50</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3.310947</td>
      <td>125.0</td>
      <td>2.5</td>
      <td>85.0</td>
      <td>0.00</td>
      <td>3.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>3.310947</td>
      <td>125.0</td>
      <td>2.5</td>
      <td>85.0</td>
      <td>0.00</td>
      <td>3.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pivot each numeric attribute into a (num_sessions, num_items, 1) tensor.
tensor_kwargs = {
    f"itemsession_{attr}": utils.pivot3d(long_df, dim0="occasion", dim1="alt", values=attr)
    for attr in NUMERIC_ATTRS
}
for name, t in tensor_kwargs.items():
    print(f"{name}: {tuple(t.shape)}")
```

    itemsession_price: (4654, 6, 1)
    itemsession_range: (4654, 6, 1)
    itemsession_acc: (4654, 6, 1)
    itemsession_speed: (4654, 6, 1)
    itemsession_pollution: (4654, 6, 1)
    itemsession_size: (4654, 6, 1)
    itemsession_space: (4654, 6, 1)
    itemsession_cost: (4654, 6, 1)
    itemsession_station: (4654, 6, 1)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))



```python
item_index    = torch.LongTensor(df["choice_idx"].values)
session_index = torch.arange(len(df))

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=N_ALTS,
    num_sessions=len(df),
    session_index=session_index,
    **tensor_kwargs,
)
print(dataset)
```

    ChoiceDataset(num_items=6, num_users=1, num_sessions=4654, label=[], item_index=[4654], user_index=[], session_index=[4654], item_availability=[], itemsession_price=[4654, 6, 1], itemsession_range=[4654, 6, 1], itemsession_acc=[4654, 6, 1], itemsession_speed=[4654, 6, 1], itemsession_pollution=[4654, 6, 1], itemsession_size=[4654, 6, 1], itemsession_space=[4654, 6, 1], itemsession_cost=[4654, 6, 1], itemsession_station=[4654, 6, 1], device=cpu)


### Fit

Full-batch LBFGS for 1,000 epochs.


```python
formula = (
    "(itemsession_price|constant) + (itemsession_range|constant) + (itemsession_acc|constant)"
    " + (itemsession_speed|constant) + (itemsession_pollution|constant)"
    " + (itemsession_size|constant) + (itemsession_space|constant)"
    " + (itemsession_cost|constant) + (itemsession_station|constant)"
    " + (intercept|item)"
)
model = ConditionalLogitModel(formula=formula, dataset=dataset, num_items=N_ALTS)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_price[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_range[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_acc[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_speed[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_pollution[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_size[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_space[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_cost[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_station[constant]): Coefficient(variation=constant, num_items=6, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=6, num_users=None, num_params=1, 5 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_price[constant]] with 1 parameters, with constant level variation.
    X[itemsession_range[constant]] with 1 parameters, with constant level variation.
    X[itemsession_acc[constant]] with 1 parameters, with constant level variation.
    X[itemsession_speed[constant]] with 1 parameters, with constant level variation.
    X[itemsession_pollution[constant]] with 1 parameters, with constant level variation.
    X[itemsession_size[constant]] with 1 parameters, with constant level variation.
    X[itemsession_space[constant]] with 1 parameters, with constant level variation.
    X[itemsession_cost[constant]] with 1 parameters, with constant level variation.
    X[itemsession_station[constant]] with 1 parameters, with constant level variation.
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
    0 | model | ConditionalLogitModel | 14     | train | 0    
    ----------------------------------------------------------------
    14        Trainable params
    0         Non-trainable params
    14        Total params
    0.000     Total estimated model params size (MB)
    12        Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -7080.829, [Validation] None, [Test] None
    
    | Coefficient                       |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:----------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_price[constant]_0     |  -0.187128   | 0.026997    |    -6.931 | 4.166e-12  | ***            |
    | itemsession_range[constant]_0     |   0.00401534 | 0.000257826 |    15.574 | < 2e-16    | ***            |
    | itemsession_acc[constant]_0       |  -0.0728139  | 0.0109533   |    -6.648 | 2.978e-11  | ***            |
    | itemsession_speed[constant]_0     |   0.00340036 | 0.000767594 |     4.43  | 9.428e-06  | ***            |
    | itemsession_pollution[constant]_0 |  -0.181489   | 0.0952113   |    -1.906 | 0.057      | .              |
    | itemsession_size[constant]_0      |   0.0720375  | 0.0291695   |     2.47  | 0.014      | *              |
    | itemsession_space[constant]_0     |   0.914808   | 0.175387    |     5.216 | 1.829e-07  | ***            |
    | itemsession_cost[constant]_0      |  -0.0723348  | 0.00744226  |    -9.719 | < 2e-16    | ***            |
    | itemsession_station[constant]_0   |   0.250232   | 0.0704706   |     3.551 | 3.840e-04  | ***            |
    | intercept[item]_0                 |  -1.19314    | 0.0696053   |   -17.142 | < 2e-16    | ***            |
    | intercept[item]_1                 |  -0.0160558  | 0.062364    |    -0.257 | 0.797      |                |
    | intercept[item]_2                 |  -1.36514    | 0.0775314   |   -17.608 | < 2e-16    | ***            |
    | intercept[item]_3                 |  -0.325559   | 0.0920558   |    -3.537 | 4.054e-04  | ***            |
    | intercept[item]_4                 |  -1.9178     | 0.105289    |   -18.215 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -7080.8291


## 4. Side-by-side comparison

`torch-choice` indexes the alt-specific intercepts positionally
(`intercept[item]_0` is the first non-reference alt, item 1 = `choice2`),
while `mlogit` labels them by alt level (`(Intercept):2` ... `(Intercept):6`).
The mapping below is the only per-dataset adapter required.


```python
NAME_MAP = {
    "itemsession_price[constant]_0":     "price",
    "itemsession_range[constant]_0":     "range",
    "itemsession_acc[constant]_0":       "acc",
    "itemsession_speed[constant]_0":     "speed",
    "itemsession_pollution[constant]_0": "pollution",
    "itemsession_size[constant]_0":      "size",
    "itemsession_space[constant]_0":     "space",
    "itemsession_cost[constant]_0":      "cost",
    "itemsession_station[constant]_0":   "station",
    "intercept[item]_0":                 "(Intercept):2",  # item 1 = choice2
    "intercept[item]_1":                 "(Intercept):3",  # item 2 = choice3
    "intercept[item]_2":                 "(Intercept):4",  # item 3 = choice4
    "intercept[item]_3":                 "(Intercept):5",  # item 4 = choice5
    "intercept[item]_4":                 "(Intercept):6",  # item 5 = choice6
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 1.129e-05 (0.0071%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 2.025e-07 (0.0003%); tol 1e-03 -> PASS
    
    LL: mlogit=-7080.8290  torch-choice=-7080.8291  abs_diff=1.03e-04





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
      <td>-0.187130</td>
      <td>-0.187128</td>
      <td>1.913318e-06</td>
      <td>0.001022</td>
      <td>0.026997</td>
      <td>0.026997</td>
      <td>3.646654e-09</td>
      <td>0.000014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_range[constant]_0</td>
      <td>0.004015</td>
      <td>0.004015</td>
      <td>2.846170e-08</td>
      <td>0.000709</td>
      <td>0.000258</td>
      <td>0.000258</td>
      <td>3.111547e-10</td>
      <td>0.000121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_acc[constant]_0</td>
      <td>-0.072813</td>
      <td>-0.072814</td>
      <td>7.962995e-07</td>
      <td>0.001094</td>
      <td>0.010953</td>
      <td>0.010953</td>
      <td>3.922749e-09</td>
      <td>0.000036</td>
    </tr>
    <tr>
      <th>3</th>
      <td>itemsession_speed[constant]_0</td>
      <td>0.003400</td>
      <td>0.003400</td>
      <td>5.146042e-08</td>
      <td>0.001513</td>
      <td>0.000768</td>
      <td>0.000768</td>
      <td>1.015868e-10</td>
      <td>0.000013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>itemsession_pollution[constant]_0</td>
      <td>-0.181478</td>
      <td>-0.181489</td>
      <td>1.128819e-05</td>
      <td>0.006220</td>
      <td>0.095211</td>
      <td>0.095211</td>
      <td>4.491838e-08</td>
      <td>0.000047</td>
    </tr>
    <tr>
      <th>5</th>
      <td>itemsession_size[constant]_0</td>
      <td>0.072032</td>
      <td>0.072038</td>
      <td>5.110698e-06</td>
      <td>0.007094</td>
      <td>0.029170</td>
      <td>0.029170</td>
      <td>2.055074e-08</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <th>6</th>
      <td>itemsession_space[constant]_0</td>
      <td>0.914812</td>
      <td>0.914808</td>
      <td>4.138166e-06</td>
      <td>0.000452</td>
      <td>0.175387</td>
      <td>0.175387</td>
      <td>3.559424e-08</td>
      <td>0.000020</td>
    </tr>
    <tr>
      <th>7</th>
      <td>itemsession_cost[constant]_0</td>
      <td>-0.072335</td>
      <td>-0.072335</td>
      <td>2.605574e-07</td>
      <td>0.000360</td>
      <td>0.007442</td>
      <td>0.007442</td>
      <td>1.430494e-09</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <th>8</th>
      <td>itemsession_station[constant]_0</td>
      <td>0.250234</td>
      <td>0.250232</td>
      <td>1.490954e-06</td>
      <td>0.000596</td>
      <td>0.070471</td>
      <td>0.070471</td>
      <td>9.043141e-09</td>
      <td>0.000013</td>
    </tr>
    <tr>
      <th>9</th>
      <td>intercept[item]_0</td>
      <td>-1.193134</td>
      <td>-1.193142</td>
      <td>8.098121e-06</td>
      <td>0.000679</td>
      <td>0.069605</td>
      <td>0.069605</td>
      <td>2.025011e-07</td>
      <td>0.000291</td>
    </tr>
    <tr>
      <th>10</th>
      <td>intercept[item]_1</td>
      <td>-0.016056</td>
      <td>-0.016056</td>
      <td>8.511367e-08</td>
      <td>0.000530</td>
      <td>0.062364</td>
      <td>0.062364</td>
      <td>7.672855e-09</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>11</th>
      <td>intercept[item]_2</td>
      <td>-1.365133</td>
      <td>-1.365138</td>
      <td>4.344493e-06</td>
      <td>0.000318</td>
      <td>0.077531</td>
      <td>0.077531</td>
      <td>9.296053e-08</td>
      <td>0.000120</td>
    </tr>
    <tr>
      <th>12</th>
      <td>intercept[item]_3</td>
      <td>-0.325563</td>
      <td>-0.325559</td>
      <td>4.424440e-06</td>
      <td>0.001359</td>
      <td>0.092056</td>
      <td>0.092056</td>
      <td>2.175398e-08</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>13</th>
      <td>intercept[item]_4</td>
      <td>-1.917805</td>
      <td>-1.917801</td>
      <td>3.845221e-06</td>
      <td>0.000201</td>
      <td>0.105289</td>
      <td>0.105289</td>
      <td>3.869587e-08</td>
      <td>0.000037</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

`torch-choice` reproduces `mlogit`'s MNL coefficient estimates and
log-likelihood on the Brownstone-Train Car dataset to within numerical
precision. With 14 free parameters (9 shared coefficients on the numeric
attributes + 5 alt-specific intercepts) and 4,654 choice occasions over 6
alternatives, both packages converge to the same optimum.

Extending this to the full Brownstone-Train mixed-logit specification —
random coefficients on a subset of attributes plus interactions with
`college`, `hsg2`, `coml5` and the categorical `type`/`fuel` levels — is a
v2 follow-up that needs a different verification strategy (parameter
distributions rather than point estimates).

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong
assumption: *every respondent responds identically to price, range,
pollution, and the other vehicle attributes*. The original Brownstone &
Train (1999) paper questions exactly this assumption — its central
contribution is a **random-coefficients (mixed) logit** that lets the most
economically meaningful attributes vary in taste across respondents. They
argue that price/cost sensitivity is the most heterogeneous channel
(varying with income), and that range and pollution preferences are also
substantially heterogeneous because they trade off lifestyle convenience
and environmental concern in ways that differ sharply across consumers.

### Why fixed-effects approximations here

`torch-choice` does not yet ship a Monte-Carlo or quadrature-based
mixed-logit estimator, so we approximate Brownstone & Train's random
coefficients by switching the relevant attributes from a single shared
coefficient to **alt-specific** coefficients (`item-full` variation).
Each of the 6 vehicle alternatives gets its own slope on the heterogeneous
attribute, capturing item-level variation in marginal utility.

**Why `item-full`, not `user`?** The Brownstone–Train SP design records
exactly **one** vehicle-choice occasion per respondent (4,654 rows = 4,654
respondents = 4,654 sessions). Per-respondent random coefficients are
therefore *not identified* from this data alone — there is no within-
respondent variation to estimate them off of. (Yogurt has ~24 purchases
per household, which is what makes the per-user FE ladder feasible there.)
The next-best in-package approximation is to allow the slope to vary
across the 6 vehicle alternatives, which is what `item-full` does.

### Specs to compare

| Spec | Formula change | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | all attrs `\|constant` | one shared coef per attribute + 5 alt intercepts | 14 |
| **B** (alt-specific price) | `itemsession_price\|item-full` | each of 6 alts gets its own price slope; everything else shared | 14 + 5 = 19 |
| **C** (alt-specific price + range + pollution) | `price`, `range`, `pollution` all `\|item-full` | per-alt slopes on the three attributes BT 1999 flag as most heterogeneous | 14 + 5 + 5 + 5 = 29 |

Specs B and C are *fixed-effects* approximations to the BT 1999
random-coefficients logit. They estimate one coefficient per alt
(treating heterogeneity as known finite parameters indexed by the choice
alternative) instead of integrating over a continuous mixing distribution
across respondents. This trades some economic interpretability — the
attribute slope now picks up vehicle-specific average response rather
than respondent-specific taste — for a tractable in-package fit.

### Fit Specs B and C



```python
import math


def fit_spec(formula: str, label: str, *, num_epochs: int = 1000,
             optimizer: str = "LBFGS") -> tuple[float, int, "EstimationOutput"]:
    """Fit a torch-choice spec; return (train_ll, n_params, result)."""
    torch.manual_seed(0)
    m = ConditionalLogitModel(formula=formula, dataset=dataset, num_items=N_ALTS)
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


# Shared tail for all specs: 6 attrs that stay pooled + alt-specific intercept.
SHARED_TAIL = (
    " + (itemsession_acc|constant) + (itemsession_speed|constant)"
    " + (itemsession_size|constant) + (itemsession_space|constant)"
    " + (itemsession_cost|constant) + (itemsession_station|constant)"
    " + (intercept|item)"
)

ll_a, k_a = float(result.train_ll), int(result.coef_summary.shape[0])

# Spec B: alt-specific price slope; range/pollution stay pooled.
formula_b = (
    "(itemsession_price|item-full) + (itemsession_range|constant)"
    " + (itemsession_pollution|constant)"
    + SHARED_TAIL
)
ll_b, k_b, result_b = fit_spec(formula_b, "B: alt-specific price")

# Spec C: alt-specific slopes on price, range, and pollution.
formula_c = (
    "(itemsession_price|item-full) + (itemsession_range|item-full)"
    " + (itemsession_pollution|item-full)"
    + SHARED_TAIL
)
ll_c, k_c, result_c = fit_spec(formula_c, "C: alt-specific price+range+pollution")

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
    0 | model | ConditionalLogitModel | 19     | train | 0    
    ----------------------------------------------------------------
    19        Trainable params
    0         Non-trainable params
    19        Total params
    0.000     Total estimated model params size (MB)
    12        Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


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
    0 | model | ConditionalLogitModel | 29     | train | 0    
    ----------------------------------------------------------------
    29        Trainable params
    0         Non-trainable params
    29        Total params
    0.000     Total estimated model params size (MB)
    12        Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Spec A: LL=-7080.83, n_params=14
    Spec B: LL=-7064.62, n_params=19
    Spec C: LL=-7037.56, n_params=29


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively (its penalty grows with `log(N)`), so it tends to favor
parsimonious models — appropriate when we believe the simpler spec is
closer to the truth. AIC penalizes less (constant per-parameter penalty
of 2) and tends to favor richer models when sample size is large.



```python
N_OCC = len(df)  # 4,654 choice occasions

specs = [
    ("A: pooled MNL",                            ll_a, k_a),
    ("B: alt-specific price",                    ll_b, k_b),
    ("C: alt-specific price+range+pollution",    ll_c, k_c),
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
      <td>-7080.829102</td>
      <td>14</td>
      <td>14189.658203</td>
      <td>14279.894956</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: alt-specific price</th>
      <td>-7064.618164</td>
      <td>19</td>
      <td>14167.236328</td>
      <td>14289.700493</td>
      <td>-22.421875</td>
      <td>9.805537</td>
    </tr>
    <tr>
      <th>C: alt-specific price+range+pollution</th>
      <td>-7037.564941</td>
      <td>29</td>
      <td>14133.129883</td>
      <td>14320.048871</td>
      <td>-56.528320</td>
      <td>40.153915</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **taste
heterogeneity matters**, and the right model is one that lets price-,
range-, and pollution-sensitivity vary across consumers.

- **Brownstone & Train (1999), *Journal of Econometrics* 89:109–129** is
  the canonical reference. They reject pooled MNL on this exact dataset
  and argue that ignoring random coefficients on price/cost, range, and
  pollution biases the substitution patterns toward IIA — a flaw that
  matters most when forecasting penetration of *new* alternative-fuel
  vehicles, since IIA forces overly symmetric cross-elasticities. Their
  preferred specification puts random coefficients (normal or log-normal)
  on the most economically-motivated attributes.
- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6**
  treats Brownstone-Train Car as the canonical motivating example for
  mixed logit. The chapter walks through random-coefficient estimation
  via simulated maximum likelihood and explains why this dataset is
  particularly hard for pooled MNL: the alternatives differ on many
  attributes simultaneously, so attribute-level heterogeneity drives
  large amounts of unobserved choice variance.
- **McFadden & Train (2000)** prove that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary, especially in SP
  studies designed (as Brownstone–Train was) to elicit responses across a
  rich attribute grid.

For Brownstone-Train Car specifically, the recommended specification is
therefore a **mixed logit with random coefficients on price/cost, range,
and pollution**. `torch-choice` does not yet ship a Monte-Carlo or
quadrature-based mixed-logit estimator, so the closest in-package
approximations are Specs B and C, which estimate one coefficient per
vehicle alternative on the heterogeneous attributes (item-level fixed
effects rather than per-respondent random effects).

**Practical guidance for this notebook's reader:**

1. Use **Spec A** (pooled MNL) as a baseline, mainly to verify the
   workflow end-to-end and as a numerical reference (R/`mlogit`
   agreement, §4).
2. Use **Spec B** (alt-specific price) when the research question asks
   whether *price sensitivity differs across vehicle types* — e.g., do
   respondents react more strongly to a one-dollar price change on an
   electric than on a gas vehicle? Spec B's identification is strongest
   where the SP design induces meaningful price variation across the
   alternatives presented to each respondent.
3. Use **Spec C** (alt-specific price + range + pollution) when all three
   attributes are believed to elicit alt-specific average responses.
   With 4,654 occasions and 6 alternatives the per-alt slopes are
   well-identified; the cost is +15 parameters relative to Spec A.
4. The published BT 1999 random-coefficients spec remains the gold
   standard; reach for it when `torch-choice` adds mixed-logit support,
   or use Apollo / Stata's `mixlogit` in the meantime.

**Caveat.** §5's alternative specs are *not* cross-validated against
`mlogit` (R doesn't natively fit alt-specific-slope MNL in the same form
without rebuilding the design matrix). The §4 verification covers Spec A
only. Specs B and C are internal-to-`torch-choice` model-fit comparisons.

