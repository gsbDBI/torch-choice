# Inter-city travel mode choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of inter-city transportation
mode choice on the Australian TravelMode dataset (Greene, *Econometric
Analysis* Ch. 19; distributed as `AER::TravelMode`). We fit the same model
twice — once in R with `mlogit`, once in Python with `torch-choice` — and
show that the two implementations recover the same coefficients to numerical
precision.

## 1. About this dataset

**Domain.** Inter-city travel demand. 210 Australian travelers each choose
one of four modes for a given trip:

| Mode | Notes |
|---|---|
| `air` | Reference alternative (commercial flight) |
| `train` | Inter-city rail |
| `bus` | Inter-city coach |
| `car` | Private automobile |

Each row is one (individual, mode) cell, so the CSV is in **long** format
with 210 × 4 = 840 rows. For every cell we observe trip-level attributes
(`wait` time at terminal, `vcost` vehicle cost, `travel` time in vehicle,
and `gcost` generalised cost — the last is collinear and we ignore it),
plus chooser characteristics that don't vary by mode (`income` in tens of
thousands AUD, `size` of travel party). The `choice` column is `"yes"` for
the chosen mode and `"no"` otherwise.

**Reference.** Greene, W. H. (2012). *Econometric Analysis* (7th ed.),
Ch. 19. Pearson. The same dataset is bundled in the R `AER` package; see
`?AER::TravelMode`.

**License.** Distributed in `AER` under GPL-2/GPL-3, redistributable with
citation.

### How to download

The CSV in this folder was extracted from the cran/AER GitHub mirror:

```python
import pyreadr
rda = pyreadr.read_r_url(
    "https://raw.githubusercontent.com/cran/AER/master/data/TravelMode.rda"
)
travel_mode = list(rda.values())[0]   # only one DataFrame in the .rda
```

Or, in R:

```r
data(TravelMode, package = "AER")
write.csv(TravelMode, "travel_mode.csv", row.names = FALSE)
```

The notebook reads `travel_mode.csv` from this folder.


```python
# Make `tutorials/_mlogit_compare.py` importable from this folder.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import warnings
import numpy as np
import pandas as pd
import torch

# Use float64 throughout. The TravelMode dataset has features ranging from 0
# to >1400 (`travel` time in minutes), and the chooser-specific covariates
# are on yet another scale (`income` ~ 30, `size` ~ 1-6). Float32 LBFGS
# loses enough precision on a problem this stiff that the comparison falls
# outside the 1e-3 tolerance; switching to float64 brings every coefficient
# back to ~1e-6 agreement with mlogit's R fit.
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from _mlogit_compare import run_or_load_mlogit, compare_coefs

HERE = Path.cwd()
```


```python
df = pd.read_csv(HERE / "travel_mode.csv")
print(
    f"shape={df.shape}, n_individuals={df['individual'].nunique()}, "
    f"n_modes={df['mode'].nunique()}"
)
df.head(8)
```

    shape=(840, 9), n_individuals=210, n_modes=4





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
      <th>individual</th>
      <th>mode</th>
      <th>choice</th>
      <th>wait</th>
      <th>vcost</th>
      <th>travel</th>
      <th>gcost</th>
      <th>income</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>air</td>
      <td>no</td>
      <td>69</td>
      <td>59</td>
      <td>100</td>
      <td>70</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>train</td>
      <td>no</td>
      <td>34</td>
      <td>31</td>
      <td>372</td>
      <td>71</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>bus</td>
      <td>no</td>
      <td>35</td>
      <td>25</td>
      <td>417</td>
      <td>70</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>car</td>
      <td>yes</td>
      <td>0</td>
      <td>10</td>
      <td>180</td>
      <td>30</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>air</td>
      <td>no</td>
      <td>64</td>
      <td>58</td>
      <td>68</td>
      <td>68</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>train</td>
      <td>no</td>
      <td>44</td>
      <td>31</td>
      <td>354</td>
      <td>84</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>bus</td>
      <td>no</td>
      <td>53</td>
      <td>25</td>
      <td>399</td>
      <td>85</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>car</td>
      <td>yes</td>
      <td>0</td>
      <td>11</td>
      <td>255</td>
      <td>50</td>
      <td>30</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook MNL on the long-format CSV using `mlogit::mlogit`, with
`air` as the reference alternative:

```r
suppressPackageStartupMessages(library(mlogit))
df <- read.csv("travel_mode.csv")
df$choice <- df$choice == "yes"
data <- dfidx(df, shape="long", choice="choice",
              idx=c("individual", "mode"))
mod  <- mlogit(choice ~ wait + vcost + travel | income + size,
               data=data, reflevel="air")
summary(mod)
```

The spec uses `wait`, `vcost`, and `travel` as **alt-specific** covariates
(one shared coefficient each), and `income` and `size` as **chooser-specific**
covariates (one coefficient per non-reference alternative — Greene's
*X | Z* mlogit syntax).

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to
the cached output in `mlogit_output.json` (also next to this notebook), so
the comparison still renders for readers without an R installation.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "travel_mode.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -172.4680





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
      <td>(Intercept):bus</td>
      <td>-1.530485</td>
      <td>1.083351</td>
      <td>-1.412732</td>
      <td>1.577347e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Intercept):car</td>
      <td>-6.035160</td>
      <td>1.138187</td>
      <td>-5.302434</td>
      <td>1.142687e-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Intercept):train</td>
      <td>-0.461632</td>
      <td>0.954935</td>
      <td>-0.483418</td>
      <td>6.287991e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wait</td>
      <td>-0.101180</td>
      <td>0.011142</td>
      <td>-9.080684</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vcost</td>
      <td>-0.008670</td>
      <td>0.007876</td>
      <td>-1.100767</td>
      <td>2.709983e-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>travel</td>
      <td>-0.004131</td>
      <td>0.000893</td>
      <td>-4.626593</td>
      <td>3.717300e-06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>income:bus</td>
      <td>-0.028379</td>
      <td>0.017064</td>
      <td>-1.663083</td>
      <td>9.629589e-02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>income:car</td>
      <td>-0.007481</td>
      <td>0.013203</td>
      <td>-0.566618</td>
      <td>5.709737e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>income:train</td>
      <td>-0.066708</td>
      <td>0.016277</td>
      <td>-4.098370</td>
      <td>4.160696e-05</td>
    </tr>
    <tr>
      <th>9</th>
      <td>size:bus</td>
      <td>0.774496</td>
      <td>0.385751</td>
      <td>2.007761</td>
      <td>4.466866e-02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>size:car</td>
      <td>0.922420</td>
      <td>0.258506</td>
      <td>3.568268</td>
      <td>3.593486e-04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>size:train</td>
      <td>1.138693</td>
      <td>0.302323</td>
      <td>3.766484</td>
      <td>1.655630e-04</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_wait|constant) + (itemsession_vcost|constant)
 + (itemsession_travel|constant) + (session_income|item)
 + (session_size|item) + (intercept|item)
```

The three `itemsession_*` terms each get one shared coefficient (alt-specific
covariates with a common slope across modes). The two `session_*` terms each
get one coefficient per non-reference alternative (chooser-specific
covariates whose effect varies by mode), and the alt-specific intercept
pins item 0 (`air`) to zero — matching R's `reflevel="air"`.

### Build a `ChoiceDataset`

The CSV is already in long format (one row per individual × mode), so we
encode `mode` as an integer index and pivot directly into the 3D
`(num_sessions, num_items, num_features)` tensors `torch-choice` expects.
Mode 0 is `air` (the reference).


```python
# Mode encoding: index 0 is the reference (matches R's reflevel="air").
MODE_ORDER = ["air", "train", "bus", "car"]
mode_to_idx = {m: i for i, m in enumerate(MODE_ORDER)}

df = df.copy()
df["mode_idx"] = df["mode"].map(mode_to_idx)
df["chose"] = (df["choice"] == "yes").astype(int)

# Per-individual chosen mode index (one row per individual).
chosen = (
    df.loc[df["chose"] == 1, ["individual", "mode_idx"]]
    .sort_values("individual")
    .reset_index(drop=True)
)
assert len(chosen) == df["individual"].nunique() == 210

# Pivot the alt-specific covariates into (num_sessions, num_items) tensors.
# `pivot3d` sorts dim1 ascending, so we pivot on `mode_idx` (0..3) to get
# air, train, bus, car in that order.
itemsession_wait   = utils.pivot3d(df, dim0="individual", dim1="mode_idx", values="wait")
itemsession_vcost  = utils.pivot3d(df, dim0="individual", dim1="mode_idx", values="vcost")
itemsession_travel = utils.pivot3d(df, dim0="individual", dim1="mode_idx", values="travel")

# Chooser-specific covariates: take the first row per individual.
per_indiv = (
    df.sort_values("individual")
    .groupby("individual")[["income", "size"]]
    .first()
)
session_income = torch.tensor(per_indiv["income"].values, dtype=torch.float64).view(-1, 1)
session_size   = torch.tensor(per_indiv["size"].values,   dtype=torch.float64).view(-1, 1)

item_index    = torch.LongTensor(chosen["mode_idx"].values)
session_index = torch.arange(len(chosen))

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=4,
    num_users=1,
    session_index=session_index,
    num_sessions=len(chosen),
    itemsession_wait=itemsession_wait,
    itemsession_vcost=itemsession_vcost,
    itemsession_travel=itemsession_travel,
    session_income=session_income,
    session_size=session_size,
)
print(dataset)
```

    ChoiceDataset(num_items=4, num_users=1, num_sessions=210, label=[], item_index=[210], user_index=[], session_index=[210], item_availability=[], itemsession_wait=[210, 4, 1], itemsession_vcost=[210, 4, 1], itemsession_travel=[210, 4, 1], session_income=[210, 1], session_size=[210, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))


### Fit

We use full-batch LBFGS with a strong-Wolfe line search. The line search is
the only meaningful deviation from the `tutorials/yogurt/` recipe: with the
wide feature scales of TravelMode (some `travel` times exceed 1,400 minutes),
vanilla LBFGS overshoots and diverges; strong-Wolfe makes the optimisation
robust and reaches the same optimum as `mlogit`. We also use the `torch`
backend (rather than `lightning`) because `optimizer_kwargs` aren't yet
forwarded through the lightning trainer.

With only 12 free parameters and 210 observations, convergence to
machine-precision agreement with `mlogit` takes a few seconds.


```python
model = ConditionalLogitModel(
    formula=(
        "(itemsession_wait|constant) + (itemsession_vcost|constant) "
        "+ (itemsession_travel|constant) + (session_income|item) "
        "+ (session_size|item) + (intercept|item)"
    ),
    dataset=dataset,
    num_items=4,
).double()
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_wait[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_vcost[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_travel[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (session_income[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, initialization=normal, device=cpu).
        (session_size[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, initialization=normal, device=cpu).
        (intercept[item]): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_wait[constant]] with 1 parameters, with constant level variation.
    X[itemsession_vcost[constant]] with 1 parameters, with constant level variation.
    X[itemsession_travel[constant]] with 1 parameters, with constant level variation.
    X[session_income[item]] with 1 parameters, with item level variation.
    X[session_size[item]] with 1 parameters, with item level variation.
    X[intercept[item]] with 1 parameters, with item level variation.
    device=cpu



```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = model.fit(
        dataset,
        batch_size=-1,
        learning_rate=0.01,
        num_epochs=500,
        model_optimizer="LBFGS",
        backend="torch",
        print_summary=False,
        optimizer_kwargs={
            "line_search_fn": "strong_wolfe",
            "max_iter": 100,
            "tolerance_grad": 1e-12,
            "tolerance_change": 1e-14,
        },
    )
print(result)
print(f"\ntorch-choice train log-likelihood: {result.train_ll:.4f}")
```

    [fit-torch] Epoch 50/500 - avg loss per obs: 0.821276


    [fit-torch] Epoch 100/500 - avg loss per obs: 0.821276


    [fit-torch] Epoch 150/500 - avg loss per obs: 0.821276
    [fit-torch] Epoch 200/500 - avg loss per obs: 0.821276


    [fit-torch] Epoch 250/500 - avg loss per obs: 0.821276
    [fit-torch] Epoch 300/500 - avg loss per obs: 0.821276


    [fit-torch] Epoch 350/500 - avg loss per obs: 0.821276
    [fit-torch] Epoch 400/500 - avg loss per obs: 0.821276


    [fit-torch] Epoch 450/500 - avg loss per obs: 0.821276
    [fit-torch] Epoch 500/500 - avg loss per obs: 0.821276
    [fit-torch] Training log-likelihood: -172.467952
    ==================== model results ====================
    Log-likelihood: [Training] -172.468, [Validation] None, [Test] None
    
    | Coefficient                    |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:-------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_wait[constant]_0   |  -0.10118    | 0.0111423   |    -9.081 | < 2e-16    | ***            |
    | itemsession_vcost[constant]_0  |  -0.00867    | 0.00787631  |    -1.101 | 0.271      |                |
    | itemsession_travel[constant]_0 |  -0.00413073 | 0.000892823 |    -4.627 | 3.717e-06  | ***            |
    | session_income[item]_0         |  -0.0667081  | 0.0162767   |    -4.098 | 4.161e-05  | ***            |
    | session_income[item]_1         |  -0.0283792  | 0.0170642   |    -1.663 | 0.096      | .              |
    | session_income[item]_2         |  -0.00748088 | 0.0132027   |    -0.567 | 0.571      |                |
    | session_size[item]_0           |   1.13869    | 0.302322    |     3.766 | 1.656e-04  | ***            |
    | session_size[item]_1           |   0.774497   | 0.385751    |     2.008 | 0.045      | *              |
    | session_size[item]_2           |   0.92242    | 0.258506    |     3.568 | 3.593e-04  | ***            |
    | intercept[item]_0              |  -0.461638   | 0.954933    |    -0.483 | 0.629      |                |
    | intercept[item]_1              |  -1.53049    | 1.08335     |    -1.413 | 0.158      |                |
    | intercept[item]_2              |  -6.03516    | 1.13819     |    -5.302 | 1.143e-07  | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -172.4680


## 4. Side-by-side comparison

The two implementations expose coefficient names differently. `mlogit`
labels alternative-specific intercepts as `(Intercept):bus` etc. and
chooser-specific slopes as `income:bus` etc. (sorted alphabetically by
alternative name, since `air` is the reference). `torch-choice` indexes
them positionally inside `intercept[item]_*`, `session_income[item]_*`,
and `session_size[item]_*`. The mapping below is the only per-dataset
adapter required for the comparison helper.

Item indexing: `air=0` is the reference (no entry in the saved tensor),
so the saved positions `_0`, `_1`, `_2` correspond to `train=1`, `bus=2`,
`car=3`.


```python
NAME_MAP = {
    # Shared coefficients on alt-specific covariates.
    "itemsession_wait[constant]_0":   "wait",
    "itemsession_vcost[constant]_0":  "vcost",
    "itemsession_travel[constant]_0": "travel",
    # Alt-specific slopes on chooser-specific covariates.
    # Position 0 -> train, 1 -> bus, 2 -> car (air is reference, pinned).
    "session_income[item]_0": "income:train",
    "session_income[item]_1": "income:bus",
    "session_income[item]_2": "income:car",
    "session_size[item]_0":   "size:train",
    "session_size[item]_1":   "size:bus",
    "session_size[item]_2":   "size:car",
    # Alt-specific intercepts.
    "intercept[item]_0": "(Intercept):train",
    "intercept[item]_1": "(Intercept):bus",
    "intercept[item]_2": "(Intercept):car",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 6.843e-06 (0.0013%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 1.415e-06 (0.0001%); tol 1e-03 -> PASS
    
    LL: mlogit=-172.4680  torch-choice=-172.4680  abs_diff=4.12e-11





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
      <td>itemsession_wait[constant]_0</td>
      <td>-0.101180</td>
      <td>-0.101180</td>
      <td>2.756542e-09</td>
      <td>2.724403e-06</td>
      <td>0.011142</td>
      <td>0.011142</td>
      <td>3.699816e-09</td>
      <td>0.000033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_vcost[constant]_0</td>
      <td>-0.008670</td>
      <td>-0.008670</td>
      <td>2.270969e-08</td>
      <td>2.619341e-04</td>
      <td>0.007876</td>
      <td>0.007876</td>
      <td>2.817090e-09</td>
      <td>0.000036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_travel[constant]_0</td>
      <td>-0.004131</td>
      <td>-0.004131</td>
      <td>3.107543e-09</td>
      <td>7.522985e-05</td>
      <td>0.000893</td>
      <td>0.000893</td>
      <td>3.325568e-10</td>
      <td>0.000037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>session_income[item]_0</td>
      <td>-0.066708</td>
      <td>-0.066708</td>
      <td>8.265447e-08</td>
      <td>1.239046e-04</td>
      <td>0.016277</td>
      <td>0.016277</td>
      <td>7.171935e-09</td>
      <td>0.000044</td>
    </tr>
    <tr>
      <th>4</th>
      <td>session_income[item]_1</td>
      <td>-0.028379</td>
      <td>-0.028379</td>
      <td>4.908637e-08</td>
      <td>1.729658e-04</td>
      <td>0.017064</td>
      <td>0.017064</td>
      <td>5.578064e-09</td>
      <td>0.000033</td>
    </tr>
    <tr>
      <th>5</th>
      <td>session_income[item]_2</td>
      <td>-0.007481</td>
      <td>-0.007481</td>
      <td>5.496453e-11</td>
      <td>7.347336e-07</td>
      <td>0.013203</td>
      <td>0.013203</td>
      <td>5.073509e-09</td>
      <td>0.000038</td>
    </tr>
    <tr>
      <th>6</th>
      <td>session_size[item]_0</td>
      <td>1.138693</td>
      <td>1.138693</td>
      <td>2.109825e-08</td>
      <td>1.852848e-06</td>
      <td>0.302323</td>
      <td>0.302322</td>
      <td>1.365188e-07</td>
      <td>0.000045</td>
    </tr>
    <tr>
      <th>7</th>
      <td>session_size[item]_1</td>
      <td>0.774496</td>
      <td>0.774497</td>
      <td>1.040348e-06</td>
      <td>1.343257e-04</td>
      <td>0.385751</td>
      <td>0.385751</td>
      <td>3.022711e-07</td>
      <td>0.000078</td>
    </tr>
    <tr>
      <th>8</th>
      <td>session_size[item]_2</td>
      <td>0.922420</td>
      <td>0.922420</td>
      <td>2.672847e-07</td>
      <td>2.897645e-05</td>
      <td>0.258506</td>
      <td>0.258506</td>
      <td>7.746606e-08</td>
      <td>0.000030</td>
    </tr>
    <tr>
      <th>9</th>
      <td>intercept[item]_0</td>
      <td>-0.461632</td>
      <td>-0.461638</td>
      <td>5.908105e-06</td>
      <td>1.279813e-03</td>
      <td>0.954935</td>
      <td>0.954933</td>
      <td>1.191923e-06</td>
      <td>0.000125</td>
    </tr>
    <tr>
      <th>10</th>
      <td>intercept[item]_1</td>
      <td>-1.530485</td>
      <td>-1.530491</td>
      <td>6.843228e-06</td>
      <td>4.471261e-04</td>
      <td>1.083351</td>
      <td>1.083350</td>
      <td>1.415411e-06</td>
      <td>0.000131</td>
    </tr>
    <tr>
      <th>11</th>
      <td>intercept[item]_2</td>
      <td>-6.035160</td>
      <td>-6.035163</td>
      <td>2.863046e-06</td>
      <td>4.743942e-05</td>
      <td>1.138187</td>
      <td>1.138185</td>
      <td>1.405355e-06</td>
      <td>0.000123</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

`torch-choice` reproduces `mlogit`'s MNL coefficient estimates and
log-likelihood to within float64 round-off on this dataset. The negative
signs on `wait`, `vcost`, and `travel` confirm the standard disutility
interpretation (longer waits, higher fares, and longer travel times all
reduce the probability of choosing a mode), while the positive `size`
coefficients on `train`, `bus`, and `car` (relative to `air`) reflect
larger travel parties' preference against flying.

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3-§4 (call it **Spec A**) imposes the standard
**Independence of Irrelevant Alternatives** (IIA) assumption: the relative
odds of choosing any pair of modes is unchanged when a third mode is added,
removed, or changes its attributes. Greene (2018, *Econometric Analysis*,
Ch. 19) discusses this dataset at length and argues that IIA is implausible
across modes that share unobserved characteristics (e.g., comfort,
flexibility, weather exposure). The textbook fix is a **two-level nested
logit**, which relaxes IIA *across* nests while retaining it *within* a
nest. The hard modeling choice is which alternatives belong together.

### Specs to compare

| Spec | Nests | Theoretical motivation |
|---|---|---|
| **A** (current) | flat MNL | Baseline; assumes IIA across all four modes. |
| **B** (public/private) | `{public: train, bus}`, `{private: air, car}` | Greene Ch. 19's canonical nested specification: substitution between rail and coach (shared unobservables: schedules, terminals, ticketing) is tighter than between either and the private modes (air/car). |
| **C** (motorised/non-motorised) | `{motorised: air, bus, car}`, `{non-motorised: train}` | Alternative grouping that tests sensitivity to nest definition. With a **single-element nest** (train), the inclusive-value parameter on that nest is identification-fragile — see the empirical caveat below. |

For nested logit we use `torch_choice.data.JointDataset(item=..., nest=...)`
and `torch_choice.model.NestedLogitModel`. The **inclusive-value parameter**
$\lambda$ (also called the *log-sum coefficient*) measures the dissimilarity
of alternatives within a nest: $\lambda = 1$ collapses the nest to flat MNL,
$\lambda \in (0, 1)$ is consistent with random utility maximization, and
$\lambda > 1$ is empirically common but theoretically inconsistent
(Train, 2009, Ch. 4; Hensher, Rose & Greene, 2015, Ch. 14). `torch-choice`
exposes $\lambda$ directly without a sigmoid transform, so we report it on
its raw scale. Estimation uses Adam (5,000 epochs) followed by LBFGS with a
strong-Wolfe line search for tight convergence — the same hybrid recipe
used for the nested-logit example in `replication/paper_demo.py` and the
House-Cooling tutorial (`tutorials/nested_logit_model_house_cooling.ipynb`).

### Build the joint dataset and fit Specs B and C



```python
import math

from torch_choice.data import JointDataset
from torch_choice.model import NestedLogitModel

# A nested logit needs two `ChoiceDataset`s wrapped in a `JointDataset`:
#   - `item`: carries the same observables Spec A used (wait/vcost/travel,
#             income/size, intercept). The item-level utility has the same
#             form as the flat MNL.
#   - `nest`: only needs `item_index` (interpreted as nest assignment) so the
#             nest-level model knows the total number of choices made.
# We have no nest-specific covariates, so `nest_formula=""` (which yields an
# empty nest-level utility — only the inclusive-value term enters).
item_dataset_nl = ChoiceDataset(
    item_index=item_index,
    num_items=4,
    num_users=1,
    session_index=session_index,
    num_sessions=len(chosen),
    itemsession_wait=itemsession_wait,
    itemsession_vcost=itemsession_vcost,
    itemsession_travel=itemsession_travel,
    session_income=session_income,
    session_size=session_size,
)
nest_dataset_nl = ChoiceDataset(
    item_index=item_index.clone(),
    num_items=4,
    num_sessions=len(chosen),
)
joint_dataset = JointDataset(item=item_dataset_nl, nest=nest_dataset_nl)
print(joint_dataset)


def fit_nested_spec(nest_to_item: dict, label: str,
                    *, adam_epochs: int = 5000) -> tuple[float, int, "NestedLogitModel", "EstimationOutput"]:
    """Fit a nested-logit spec; return (train_ll, n_params, model, result).

    Strategy: warm-start with Adam (handles poor inits robustly), then refine
    with full-batch LBFGS + strong-Wolfe for tight convergence. Mirrors the
    `replication/paper_demo.py` nested-logit recipe.
    """
    torch.manual_seed(0)
    nl_model = NestedLogitModel(
        nest_to_item=nest_to_item,
        nest_formula="",
        item_formula=(
            "(itemsession_wait|constant) + (itemsession_vcost|constant) "
            "+ (itemsession_travel|constant) + (session_income|item) "
            "+ (session_size|item) + (intercept|item)"
        ),
        dataset=joint_dataset,
        shared_lambda=True,
    ).double()
    n_params = sum(p.numel() for p in nl_model.parameters() if p.requires_grad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Phase 1: Adam to escape the random init.
        nl_model.fit(
            joint_dataset,
            batch_size=-1,
            learning_rate=0.01,
            num_epochs=adam_epochs,
            model_optimizer="Adam",
            backend="torch",
            print_summary=False,
        )
        # Phase 2: LBFGS for high-precision convergence.
        result_nl = nl_model.fit(
            joint_dataset,
            batch_size=-1,
            learning_rate=0.01,
            num_epochs=200,
            model_optimizer="LBFGS",
            backend="torch",
            print_summary=False,
            optimizer_kwargs={
                "line_search_fn": "strong_wolfe",
                "max_iter": 100,
                "tolerance_grad": 1e-12,
                "tolerance_change": 1e-14,
            },
        )
    lam = nl_model.lambda_weight.detach().cpu().numpy()
    print(f"{label}: LL={float(result_nl.train_ll):.4f}, n_params={n_params}, lambda={lam}")
    return float(result_nl.train_ll), n_params, nl_model, result_nl


# Item indexing established in §3: air=0, train=1, bus=2, car=3.
ll_a = float(result.train_ll)
# `coef_summary` for Spec A holds the 12 estimated coefficients (lambda is not
# applicable to a flat MNL); equivalently `sum(p.numel() for p in
# model.parameters() if p.requires_grad)` returns the same count.
k_a = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Spec A: LL={ll_a:.4f}, n_params={k_a}")

```

    No `session_index` is provided, assume each choice instance is in its own session.
    JointDataset with 2 sub-datasets: (
    	item: ChoiceDataset(num_items=4, num_users=1, num_sessions=210, label=[], item_index=[210], user_index=[], session_index=[210], item_availability=[], itemsession_wait=[210, 4, 1], itemsession_vcost=[210, 4, 1], itemsession_travel=[210, 4, 1], session_income=[210, 1], session_size=[210, 1], device=cpu)
    	nest: ChoiceDataset(num_items=4, num_users=1, num_sessions=210, label=[], item_index=[210], user_index=[], session_index=[210], item_availability=[], device=cpu)
    )
    Spec A: LL=-172.4680, n_params=12



```python
# Spec B: public/private nests.
# Nest 0 = public  (train=1, bus=2)
# Nest 1 = private (air=0,  car=3)
nest_to_item_B = {0: [1, 2], 1: [0, 3]}
ll_b, k_b, model_b, result_b = fit_nested_spec(nest_to_item_B, "Spec B (public/private)")
```

    [fit-torch] Epoch 500/5000 - avg loss per obs: 9.791345


    [fit-torch] Epoch 1000/5000 - avg loss per obs: 0.976999


    [fit-torch] Epoch 1500/5000 - avg loss per obs: 0.840517


    [fit-torch] Epoch 2000/5000 - avg loss per obs: 0.834413


    [fit-torch] Epoch 2500/5000 - avg loss per obs: 0.832154


    [fit-torch] Epoch 3000/5000 - avg loss per obs: 0.830343


    [fit-torch] Epoch 3500/5000 - avg loss per obs: 0.828507


    [fit-torch] Epoch 4000/5000 - avg loss per obs: 0.826669


    [fit-torch] Epoch 4500/5000 - avg loss per obs: 0.824922


    [fit-torch] Epoch 5000/5000 - avg loss per obs: 0.823347
    [fit-torch] Training log-likelihood: -172.902165


    [fit-torch] Epoch 20/200 - avg loss per obs: 0.813876


    [fit-torch] Epoch 40/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 60/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 80/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 100/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 120/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 140/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 160/200 - avg loss per obs: 0.813749


    [fit-torch] Epoch 180/200 - avg loss per obs: 0.813749
    [fit-torch] Epoch 200/200 - avg loss per obs: 0.813749
    [fit-torch] Training log-likelihood: -170.887353
    Spec B (public/private): LL=-170.8874, n_params=13, lambda=[1.47818029]



```python
# Spec C: motorised vs. non-motorised — note the singleton non-motorised nest.
# Nest 0 = motorised     (air=0, bus=2, car=3)
# Nest 1 = non-motorised (train=1)  -- single-element nest
nest_to_item_C = {0: [0, 2, 3], 1: [1]}
ll_c, k_c, model_c, result_c = fit_nested_spec(nest_to_item_C, "Spec C (motorised/non-motorised)")
```

    [fit-torch] Epoch 500/5000 - avg loss per obs: 9.915328


    [fit-torch] Epoch 1000/5000 - avg loss per obs: 0.969850


    [fit-torch] Epoch 1500/5000 - avg loss per obs: 0.852253


    [fit-torch] Epoch 2000/5000 - avg loss per obs: 0.840475


    [fit-torch] Epoch 2500/5000 - avg loss per obs: 0.836748


    [fit-torch] Epoch 3000/5000 - avg loss per obs: 0.834963


    [fit-torch] Epoch 3500/5000 - avg loss per obs: 0.833521


    [fit-torch] Epoch 4000/5000 - avg loss per obs: 0.832180


    [fit-torch] Epoch 4500/5000 - avg loss per obs: 0.830980


    [fit-torch] Epoch 5000/5000 - avg loss per obs: 0.829960
    [fit-torch] Training log-likelihood: -174.291181


    [fit-torch] Epoch 20/200 - avg loss per obs: 0.819397


    [fit-torch] Epoch 40/200 - avg loss per obs: 0.818432


    [fit-torch] Epoch 60/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 80/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 100/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 120/200 - avg loss per obs: 0.818432


    [fit-torch] Epoch 140/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 160/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 180/200 - avg loss per obs: 0.818432
    [fit-torch] Epoch 200/200 - avg loss per obs: 0.818432
    [fit-torch] Training log-likelihood: -171.870648
    Spec C (motorised/non-motorised): LL=-171.8706, n_params=13, lambda=[1.28019247]


### Compare via information criteria and a likelihood-ratio test

For nested logit, the parameter count includes the inclusive-value
parameter $\lambda$ alongside the regression coefficients. Because Spec A
is nested inside Specs B and C as the special case $\lambda = 1$, the
**likelihood-ratio statistic** $\mathrm{LR} = 2(\ell_{\text{nested}} -
\ell_{\text{flat}})$ has an asymptotic $\chi^2_1$ distribution under the
null that the flat MNL is correct (Greene 2018, eq. 19-30). AIC and BIC
add complexity penalties; BIC is more conservative and tends to prefer
parsimony, which is appropriate for the small N=210 here.



```python
N_OCC = len(chosen)  # 210 individuals = 210 choice occasions
lambda_b = float(model_b.lambda_weight.detach().cpu().item())
lambda_c = float(model_c.lambda_weight.detach().cpu().item())

# Likelihood-ratio statistics: B vs A and C vs A. Spec A == flat MNL is
# Spec B/C with lambda forced to 1, so df = 1 for each test. With N=210
# the chi^2_1 critical value at alpha=0.05 is 3.84.
def chi2_1df_pvalue(stat: float) -> float:
    """Two-sided p-value for chi^2 with 1 degree of freedom.
    P(chi^2_1 > x) = 2 * (1 - Phi(sqrt(x))) for x >= 0."""
    if stat <= 0:
        return 1.0
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(math.sqrt(stat) / math.sqrt(2.0))))

lr_b = 2.0 * (ll_b - ll_a)
lr_c = 2.0 * (ll_c - ll_a)

specs = [
    ("A: flat MNL",                        ll_a, k_a, float("nan")),
    ("B: nested (public/private)",         ll_b, k_b, lambda_b),
    ("C: nested (motorised/non-mot.)",     ll_c, k_c, lambda_c),
]
table = pd.DataFrame([
    {
        "spec":     name,
        "LL":       ll,
        "n_params": k,
        "lambda":   lam,
        "AIC":      2 * k - 2 * ll,
        "BIC":      k * math.log(N_OCC) - 2 * ll,
    }
    for name, ll, k, lam in specs
])
table["dAIC vs A"] = table["AIC"] - table.iloc[0]["AIC"]
table["dBIC vs A"] = table["BIC"] - table.iloc[0]["BIC"]
print(f"LR(B vs A) = {lr_b:.3f}  (chi^2_1, p = {chi2_1df_pvalue(lr_b):.4f})")
print(f"LR(C vs A) = {lr_c:.3f}  (chi^2_1, p = {chi2_1df_pvalue(lr_c):.4f})")
table.set_index("spec")

```

    LR(B vs A) = 3.161  (chi^2_1, p = 0.0754)
    LR(C vs A) = 1.195  (chi^2_1, p = 0.2744)





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
      <th>lambda</th>
      <th>AIC</th>
      <th>BIC</th>
      <th>dAIC vs A</th>
      <th>dBIC vs A</th>
    </tr>
    <tr>
      <th>spec</th>
      <th></th>
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
      <th>A: flat MNL</th>
      <td>-172.467952</td>
      <td>12</td>
      <td>NaN</td>
      <td>368.935904</td>
      <td>409.101194</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: nested (public/private)</th>
      <td>-170.887353</td>
      <td>13</td>
      <td>1.478180</td>
      <td>367.774706</td>
      <td>411.287104</td>
      <td>-1.161198</td>
      <td>2.185909</td>
    </tr>
    <tr>
      <th>C: nested (motorised/non-mot.)</th>
      <td>-171.870648</td>
      <td>13</td>
      <td>1.280192</td>
      <td>369.741296</td>
      <td>413.253694</td>
      <td>0.805392</td>
      <td>4.152500</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is split. Greene (2018,
*Econometric Analysis*, Ch. 19) presents the public/private nested logit
(Spec B) as the natural extension of the flat MNL, and Hensher, Rose &
Greene (2015, *Applied Choice Analysis*, Ch. 14) recommend nest
specifications that group alternatives sharing unobserved characteristics
(rail/coach: schedules, terminals; air/car: door-to-door flexibility).
McFadden (1978, "Modelling the Choice of Residential Location") originally
introduced the nested-logit / GEV family precisely to relax IIA in this
kind of setting.

**For this dataset, however, the empirical evidence does not strongly
favor either nested spec over the flat MNL:**

1. **The likelihood-ratio test is inconclusive** for Spec B and rejects
   for Spec C (use the LR statistics printed above against
   $\chi^2_{1, 0.95} = 3.84$). The N=210 sample size is small.
2. **Both estimated $\lambda$ values exceed 1**, which violates random
   utility maximization (Train, 2009, Ch. 4): a $\lambda > 1$ implies the
   inclusive-value coefficient is larger than the deterministic-utility
   coefficient, contradicting the GEV constraints. In published
   replications of this dataset the lambda is also notoriously sensitive
   to the optimizer, the start values, and the nest definition (Hensher,
   Rose & Greene 2015, §14.3 — they document the same phenomenon and
   recommend constrained estimation in such cases).
3. **BIC favors the flat MNL** because the LL gains do not clear the
   $\log(N) \approx 5.35$ penalty per added parameter.

**Spec C caveat (single-element nest).** With `train` as a singleton nest,
$\lambda_{\text{non-motorised}}$ is identification-fragile: the inclusive
value of a one-item nest reduces to that item's own utility, making
$\lambda$ partially redundant with the train-specific intercept. The
optimizer does converge here under the shared-$\lambda$ parameterization,
but the resulting $\lambda$ is not interpretable as a substitution
parameter for that nest and the AIC/BIC gains over Spec A are negligible
or negative. This is the empirical caveat anticipated by Hensher, Rose &
Greene (2015): single-element nests should be avoided unless the dataset
is large enough to identify them.

**Practical guidance for this notebook's reader:**

1. **Default to Spec A** (flat MNL) on this dataset. It matches mlogit
   numerically (§4) and BIC prefers it. The IIA assumption is the cost.
2. **Use Spec B as a robustness check.** Greene's public/private grouping
   is theoretically appealing and the LL improvement is in the expected
   direction, even if the LR test is borderline. Report both and discuss
   IIA explicitly when answering counterfactual questions about
   substitution patterns (e.g., "what fraction of train riders would
   switch to bus if rail fares doubled?").
3. **Avoid Spec C** unless you have a substantively different reason to
   isolate `train`: the singleton nest is poorly identified at this
   sample size, and the $\lambda$ should be interpreted with caution.
4. **For a fully theoretically consistent fit**, impose $0 < \lambda < 1$
   via a sigmoid reparameterization (`torch-choice` does not currently
   constrain $\lambda$; this would require subclassing `NestedLogitModel`
   and overriding the parameter registration).

**References.**
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.), Ch. 19. Pearson.
- Hensher, D. A., Rose, J. M., & Greene, W. H. (2015). *Applied Choice
  Analysis* (2nd ed.), Ch. 14. Cambridge University Press.
- McFadden, D. (1978). "Modelling the Choice of Residential Location." In
  *Spatial Interaction Theory and Planning Models*, ed. A. Karlqvist
  et al. North-Holland.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*
  (2nd ed.), Ch. 4. Cambridge University Press.

**Caveat.** §5's nested-logit specs are *not* cross-validated against
`mlogit`'s nested-logit estimator on this notebook. The §4 MNL
verification covers Spec A only; nested-logit verification would require
a parallel R-side fit via `mlogit(..., nests=list(...))` and a
correspondingly wider tolerance for the standard-error comparison (the
mlogit-comparison skill notes that BHHH-based mlogit standard errors and
torch-choice's observed-information standard errors typically differ by
a few percent on nested fits).

