# Swiss Route Choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of inter-urban rail route choice
on the Apollo Swiss Route Choice stated-preference panel (Axhausen et al.,
*Transport Policy*, 2008). We fit the same model twice — once in R with
`mlogit`, once in Python with `torch-choice` — and show that the two
implementations recover the same coefficients to numerical precision. We then
explore richer alternative specifications and discuss which is theoretically
preferred.

## 1. About this dataset

**Domain.** Inter-urban rail route choice in Switzerland. The Apollo
`apollo_swissRouteChoiceData` dataset is a stated-preference (SP) panel of
388 Swiss travellers, each shown exactly 9 binary choice scenarios (3,492
total observations). In every scenario the respondent picks one of two
hypothetical rail routes between the same origin–destination pair. Each
route is described by four attributes:

| Attribute | Meaning | Units |
|---|---|---|
| `tt` | Travel time | minutes |
| `tc` | Travel cost | CHF |
| `hw` | Headway (time between trains) | minutes |
| `ch` | Number of interchanges | count |

The CSV also carries chooser-level covariates (`hh_inc_abs`,
`car_availability`, four trip-purpose indicators) that this v1 tutorial does
not use; the canonical Apollo MNL spec uses only the four route attributes.

**Reference.** Axhausen, K. W., Hess, S., König, A., Abay, G., Bates, J. J.,
Bierlaire, M. (2008). "Income and distance elasticities of values of travel
time savings: New Swiss results." *Transport Policy* 15(3), 173–185.
([doi:10.1016/j.tranpol.2008.02.001](https://doi.org/10.1016/j.tranpol.2008.02.001))
Apollo software paper: Hess, S., Palma, D. (2019), *Journal of Choice
Modelling* 32, 100170.

**License.** Distributed with the Apollo R package
(Hess & Palma, JoCM 2019) under GPL-2; redistributable with citation.

**Caveat: pooled MNL vs. mixed logit.** This dataset is the workhorse of
Apollo's mixed-logit examples (`MMNL_*`); the original Axhausen 2008 paper
fits a **mixed logit** with random coefficients on travel time and travel
cost so that values-of-travel-time (VTTS) can vary across travellers. For v1
of this tutorial we replicate the simpler **pooled MNL** baseline; §5 fits a
torch-choice in-package approximation and discusses why the published
literature prefers mixed logit on this data.

### How to download

The CSV in this folder is a verbatim copy of the file distributed by the
Apollo project:

```bash
curl -L -o apollo_swissRouteChoiceData.csv \
  http://www.apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData.csv
```

The notebook reads `apollo_swissRouteChoiceData.csv` from this folder.


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
df = pd.read_csv(HERE / "apollo_swissRouteChoiceData.csv")
n_individuals = df["ID"].nunique()
median_per_id = int(df.groupby("ID").size().median())
print(
    f"shape={df.shape}, n_individuals={n_individuals}, "
    f"n_tasks={len(df)} (median {median_per_id} SC tasks per individual)"
)
df.head()
```

    shape=(3492, 16), n_individuals=388, n_tasks=3492 (median 9 SC tasks per individual)





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
      <th>ID</th>
      <th>choice</th>
      <th>tt1</th>
      <th>tc1</th>
      <th>hw1</th>
      <th>ch1</th>
      <th>tt2</th>
      <th>tc2</th>
      <th>hw2</th>
      <th>ch2</th>
      <th>hh_inc_abs</th>
      <th>car_availability</th>
      <th>commute</th>
      <th>shopping</th>
      <th>business</th>
      <th>leisure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2439</td>
      <td>2</td>
      <td>58</td>
      <td>7</td>
      <td>30</td>
      <td>1</td>
      <td>50</td>
      <td>8</td>
      <td>30</td>
      <td>0</td>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2439</td>
      <td>1</td>
      <td>30</td>
      <td>8</td>
      <td>60</td>
      <td>0</td>
      <td>41</td>
      <td>7</td>
      <td>15</td>
      <td>2</td>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2439</td>
      <td>1</td>
      <td>41</td>
      <td>7</td>
      <td>30</td>
      <td>0</td>
      <td>34</td>
      <td>8</td>
      <td>15</td>
      <td>2</td>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2439</td>
      <td>1</td>
      <td>44</td>
      <td>10</td>
      <td>60</td>
      <td>1</td>
      <td>52</td>
      <td>9</td>
      <td>60</td>
      <td>2</td>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2439</td>
      <td>2</td>
      <td>43</td>
      <td>9</td>
      <td>60</td>
      <td>0</td>
      <td>34</td>
      <td>10</td>
      <td>30</td>
      <td>0</td>
      <td>50000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a pooled MNL on the wide-per-occasion CSV using `mlogit::mlogit`,
with **no alternative-specific intercepts** (the `| 0` term suppresses them
because the two "routes" are interchangeable hypothetical labels rather
than real branded alternatives — this matches the canonical Apollo
`MNL_SP_RouteChoice` example):

```r
suppressPackageStartupMessages(library(mlogit))
df   <- read.csv("apollo_swissRouteChoiceData.csv")
data <- dfidx(df, shape="wide", choice="choice", varying=3:10,
              sep="", idnames=c("chid","alt"))
mod  <- mlogit(choice ~ tt + tc + hw + ch | 0, data=data, reflevel="1")
summary(mod)
```

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to
the cached output in `mlogit_output.json` (also next to this notebook), so
the comparison still renders for readers without an R installation.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "apollo_swissRouteChoiceData.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -1665.6885





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
      <td>tt</td>
      <td>-0.059771</td>
      <td>0.004257</td>
      <td>-14.040029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tc</td>
      <td>-0.131815</td>
      <td>0.013506</td>
      <td>-9.760068</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hw</td>
      <td>-0.037451</td>
      <td>0.001848</td>
      <td>-20.268687</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ch</td>
      <td>-1.152070</td>
      <td>0.043419</td>
      <td>-26.533653</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_tt|constant) + (itemsession_tc|constant)
  + (itemsession_hw|constant) + (itemsession_ch|constant)
```

— one shared coefficient per attribute, **no** `(intercept|item)` term
(matching mlogit's `| 0`, since the routes are interchangeable).

### Build a `ChoiceDataset`

The CSV is in *wide-per-occasion* format: each row already lists both
routes' four attributes side-by-side (`tt1, tc1, hw1, ch1, tt2, tc2, hw2,
ch2`). We reshape to long format and pivot into the 3D
`(num_sessions, num_items, num_features)` tensors `torch-choice` expects.


```python
# Route encoding: index 0 = route 1 (mlogit's reflevel="1"), index 1 = route 2.
ATTRS  = ["tt", "tc", "hw", "ch"]
ROUTES = [1, 2]
route_to_idx = {r: i for i, r in enumerate(ROUTES)}

# Wide -> long: 3,492 occasions x 2 routes = 6,984 rows.
df = df.copy()
df["occasion"] = np.arange(len(df))
long_records = []
for route, idx in route_to_idx.items():
    cols = {f"{a}{route}": a for a in ATTRS}
    long_records.append(
        df[["occasion", *cols.keys()]]
        .rename(columns=cols)
        .assign(route=idx)
    )
long_df = pd.concat(long_records, ignore_index=True).sort_values(
    ["occasion", "route"]
)

# Pivot each attribute into a (num_sessions, num_items, 1) tensor.
itemsession_tensors = {
    f"itemsession_{a}": utils.pivot3d(
        long_df, dim0="occasion", dim1="route", values=a
    )
    for a in ATTRS
}

# Per-occasion chosen route index, individual id, and session id.
item_index = torch.LongTensor(df["choice"].astype(int).map(route_to_idx).values)
unique_ids = sorted(df["ID"].unique())
id_to_idx  = {i: k for k, i in enumerate(unique_ids)}
user_index = torch.LongTensor(df["ID"].map(id_to_idx).values)
session_index = torch.arange(len(df))

NUM_USERS = len(unique_ids)              # 388 individuals
N_OCC     = len(item_index)              # 3,492 SC tasks

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=2,
    num_users=NUM_USERS,
    user_index=user_index,
    session_index=session_index,
    **itemsession_tensors,
)
print(dataset)
```

    ChoiceDataset(num_items=2, num_users=388, num_sessions=3492, label=[], item_index=[3492], user_index=[3492], session_index=[3492], item_availability=[], itemsession_tt=[3492, 2, 1], itemsession_tc=[3492, 2, 1], itemsession_hw=[3492, 2, 1], itemsession_ch=[3492, 2, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))
    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


### Fit (Spec A — pooled MNL)

We use full-batch LBFGS for 1,000 epochs — the same recipe the package's
`paper_demo.py` uses for ModeCanada. `model.fit(...)` returns an
`EstimationOutput` whose `coef_summary` mirrors a standard regression table.


```python
FORMULA = (
    "(itemsession_tt|constant) + (itemsession_tc|constant) "
    "+ (itemsession_hw|constant) + (itemsession_ch|constant)"
)
model = ConditionalLogitModel(
    formula=FORMULA,
    dataset=dataset,
    num_items=2,
)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_tt[constant]): Coefficient(variation=constant, num_items=2, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_tc[constant]): Coefficient(variation=constant, num_items=2, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_hw[constant]): Coefficient(variation=constant, num_items=2, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_ch[constant]): Coefficient(variation=constant, num_items=2, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_tt[constant]] with 1 parameters, with constant level variation.
    X[itemsession_tc[constant]] with 1 parameters, with constant level variation.
    X[itemsession_hw[constant]] with 1 parameters, with constant level variation.
    X[itemsession_ch[constant]] with 1 parameters, with constant level variation.
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
    0 | model | ConditionalLogitModel | 4      | train | 0    
    ----------------------------------------------------------------
    4         Trainable params
    0         Non-trainable params
    4         Total params
    0.000     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -1665.688, [Validation] None, [Test] None
    
    | Coefficient                |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:---------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_tt[constant]_0 |   -0.0597703 |  0.00425715 |   -14.04  | < 2e-16    | ***            |
    | itemsession_tc[constant]_0 |   -0.131815  |  0.0135056  |    -9.76  | < 2e-16    | ***            |
    | itemsession_hw[constant]_0 |   -0.0374505 |  0.00184771 |   -20.269 | < 2e-16    | ***            |
    | itemsession_ch[constant]_0 |   -1.15206   |  0.043419   |   -26.534 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -1665.6885


## 4. Side-by-side comparison

The two implementations expose coefficient names differently. `mlogit` uses
the bare attribute names (`tt`, `tc`, `hw`, `ch`), while `torch-choice`
namespaces them as `itemsession_<attr>[constant]_0`. The mapping below is
the only per-dataset adapter required for the comparison helper.


```python
NAME_MAP = {
    "itemsession_tt[constant]_0": "tt",
    "itemsession_tc[constant]_0": "tc",
    "itemsession_hw[constant]_0": "hw",
    "itemsession_ch[constant]_0": "ch",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 7.579e-06 (0.0008%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 1.647e-07 (0.0004%); tol 1e-03 -> PASS
    
    LL: mlogit=-1665.6885  torch-choice=-1665.6885  abs_diff=2.03e-05





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
      <td>itemsession_tt[constant]_0</td>
      <td>-0.059771</td>
      <td>-0.059770</td>
      <td>2.723971e-07</td>
      <td>0.000456</td>
      <td>0.004257</td>
      <td>0.004257</td>
      <td>2.991838e-09</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_tc[constant]_0</td>
      <td>-0.131815</td>
      <td>-0.131815</td>
      <td>6.696898e-07</td>
      <td>0.000508</td>
      <td>0.013506</td>
      <td>0.013506</td>
      <td>4.408414e-09</td>
      <td>0.000033</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_hw[constant]_0</td>
      <td>-0.037451</td>
      <td>-0.037450</td>
      <td>3.053090e-07</td>
      <td>0.000815</td>
      <td>0.001848</td>
      <td>0.001848</td>
      <td>7.970746e-09</td>
      <td>0.000431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>itemsession_ch[constant]_0</td>
      <td>-1.152070</td>
      <td>-1.152062</td>
      <td>7.578969e-06</td>
      <td>0.000658</td>
      <td>0.043419</td>
      <td>0.043419</td>
      <td>1.647030e-07</td>
      <td>0.000379</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion of §4.** `torch-choice` reproduces `mlogit`'s pooled-MNL
coefficient estimates and log-likelihood to within float32 round-off on the
Swiss Route Choice SP panel. All four disutility coefficients are negative
and statistically significant: respondents dislike longer travel time,
higher fares, longer headways, and more interchanges — as expected for a
rail-route-choice model.

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong
assumption: *every traveller responds identically to time, cost, headway,
and interchanges*. The published literature on this exact dataset rejects
that assumption flatly. **Axhausen et al. (2008)** introduced this SP panel
specifically to estimate **values of travel time savings (VTTS)** and
their distribution across the population, and they fit a **mixed logit**
with random coefficients on `tt` and `tc`. The Apollo software paper
(**Hess & Palma, JoCM 2019**) then ships *this exact dataset* as the
headline example for several `MMNL_*` (mixed-MNL) tutorials, including a
WTP-space parameterisation that estimates value-of-time directly.

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_tt\|constant) + (itemsession_tc\|constant) + (itemsession_hw\|constant) + (itemsession_ch\|constant)` | one shared coefficient per attribute | 4 |
| **B** (respondent-specific time) | swap `itemsession_tt\|constant` → `itemsession_tt\|user`; keep others constant | each of 388 respondents gets their own travel-time coefficient; `tc`, `hw`, `ch` remain pooled | 388+3 = 391 |
| **C** (WTP-space mixed logit) | not fit here — see recommendation below | random coefficients on `tt` and `tc` reparameterised so VTTS = β_tt / β_tc has its own distribution | n/a |

Spec B is a *fixed-effects* approximation to the mixed-logit time-coefficient
specification: it estimates one travel-time coefficient per respondent
(treating heterogeneity as known finite parameters) instead of integrating
over a continuous mixing distribution. With 388 respondents and exactly 9
SC tasks each, the FE approach is feasible.

Spec C — the WTP-space mixed logit that the Apollo team actually publishes
on this data — is **not currently supported by `torch-choice`** (no built-in
Monte-Carlo or quadrature-based mixed-logit estimator). For that
specification, use the Apollo R package directly; its repository ships
`apollo_example_19_*.r` etc. on this same CSV. The recommendation block
below covers when each spec is appropriate.

### Fit Specs A and B


```python
def fit_spec(formula: str, label: str, *, num_epochs: int = 1000,
             optimizer: str = "LBFGS") -> tuple[float, int, "EstimationOutput"]:
    """Fit a torch-choice spec; return (train_ll, n_params, result)."""
    torch.manual_seed(0)
    m = ConditionalLogitModel(
        formula=formula, dataset=dataset, num_items=2, num_users=NUM_USERS,
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
    "(itemsession_tt|user) + (itemsession_tc|constant) "
    "+ (itemsession_hw|constant) + (itemsession_ch|constant)",
    "B: respondent-specific time coef",
)

print(f"Spec A: LL={ll_a:.2f}, n_params={k_a}")
print(f"Spec B: LL={ll_b:.2f}, n_params={k_b}")
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
    0 | model | ConditionalLogitModel | 391    | train | 0    
    ----------------------------------------------------------------
    391       Trainable params
    0         Non-trainable params
    391       Total params
    0.002     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    /Users/tianyudu/Development/torch-choice/torch_choice/data/choice_dataset.py:286: UserWarning: The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.
      warnings.warn(f"The number of sessions is inferred from the number of unique sessions in the session_index tensor. This might lead to unexpected behaviors if some sessions never appeared in the session_index tensor. For a safer behavior, please provide the number of sessions explicitly by using the num_sessions keyword while initializing the ChoiceDataset class.")


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Spec A: LL=-1665.69, n_params=4
    Spec B: LL=-1173.42, n_params=391


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively, so it tends to favor parsimonious models — appropriate when
we believe the simpler spec is closer to the truth. AIC penalizes less and
tends to favor richer models when sample size is large.


```python
specs = [
    ("A: pooled MNL",                    ll_a, k_a),
    ("B: respondent-specific time coef", ll_b, k_b),
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
      <td>-1665.688477</td>
      <td>4</td>
      <td>3339.376953</td>
      <td>3364.009873</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: respondent-specific time coef</th>
      <td>-1173.416504</td>
      <td>391</td>
      <td>3128.833008</td>
      <td>5536.700905</td>
      <td>-210.543945</td>
      <td>2172.691033</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **traveller
heterogeneity matters**, and the right model is a mixed logit with random
coefficients on (at least) travel time and travel cost.

- **Axhausen, Hess, König, Abay, Bates & Bierlaire (2008),
  *Transport Policy* 15(3):173–185.** The original paper using this exact
  SP panel. They estimate VTTS distributions via mixed logit and document
  how mean willingness-to-pay for time savings rises with income and trip
  distance — results that pooled MNL cannot recover by construction.
- **Hess & Palma (2019), *Journal of Choice Modelling* 32:100170** — the
  Apollo software paper. The Apollo repository ships *this exact dataset*
  as the headline example for several mixed-logit tutorials
  (`apollo_example_19_*` etc.), including a WTP-space parameterisation
  that estimates value-of-time directly. This dataset is, in effect, the
  workhorse of Apollo's mixed-logit examples — it was specifically
  designed for VTTS estimation with random coefficients.
- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6** is
  the canonical textbook on mixed logit and treats panel SP data of this
  shape as the motivating example. Random coefficients can be normal,
  log-normal, or — in WTP-space parameterisations — calibrated so the
  ratio β_tt / β_tc itself has a tractable distribution.
- **McFadden & Train (2000)** prove that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary across the
  population.

For Swiss Route Choice specifically, the recommended specification is
therefore a **WTP-space mixed logit with random coefficients on travel
time and travel cost**. `torch-choice` does not yet ship a Monte-Carlo or
quadrature-based mixed-logit estimator, so the closest in-package
approximation is **Spec B**, which estimates one travel-time coefficient
per respondent (fixed effects). If you need the published WTP-space
specification, use the Apollo R package on this same CSV.

**Practical guidance for this notebook's reader:**

1. Use **Spec A** (pooled MNL) as a baseline — it's the most parsimonious
   spec, runs in seconds, and gives the R/`mlogit` agreement check in §4.
   BIC actually prefers Spec A over Spec B here (the BIC penalty for the
   extra 387 parameters outweighs the within-sample LL gain).
2. Use **Spec B** (per-respondent travel-time coefficient) when the
   research question asks about *VTTS heterogeneity* — e.g., does
   respondent X have a higher value of time than respondent Y? With
   exactly 9 SC tasks per respondent and substantial within-respondent
   time variation by experimental design, Spec B's per-respondent time
   coefficients are well identified. AIC prefers Spec B over Spec A,
   consistent with the literature's finding that taste heterogeneity is
   real and economically large.
3. Use **Spec C** (the published WTP-space mixed logit) when reporting
   policy numbers — VTTS distributions, income / distance elasticities of
   VTTS, etc. — that need to be comparable to Axhausen et al. 2008.
   `torch-choice` does not currently support this; use Apollo's
   `apollo_estimate(..., model="MMNL")` on the same CSV.

**Caveat.** §5's alternative spec (B) is *not* cross-validated against
`mlogit` (R doesn't natively fit per-user fixed-effects MNL with this many
parameters in the same form). The §4 verification covers Spec A only.
Spec B is an internal-to-`torch-choice` model-fit comparison.
