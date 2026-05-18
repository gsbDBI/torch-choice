# NOx pollution-control technology choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of power-plant pollution-control
technology adoption on the Fowlie (AER, 2010) panel. We fit the same model
twice — once in R with `mlogit`, once in Python with `torch-choice` — and show
that the two implementations recover the same coefficients to numerical
precision.

The key wrinkle relative to the Yogurt tutorial is **unbalanced choice sets**:
each plant faces 15 candidate technologies on paper, but only some are
engineering / regulatory feasible. We line up the two packages by passing
`subset = available == 1` to mlogit's `dfidx` (which drops infeasible rows
from the likelihood) and an explicit `(num_sessions, num_items)` availability
mask to `torch-choice` (which sets the utility of infeasible alts to
$-\infty$). The two formulations are mathematically identical.

## 1. About this dataset

**Domain.** Environmental policy / industrial pollution-control investment.
632 American electricity-generating units choose one of up to 15 NOx-emission
abatement technologies. Each row is one (plant, candidate technology) pair.

| Column | Meaning |
|---|---|
| `chid` | Plant id (1..632); 632 unique plants. |
| `alt` | Technology id (1..15); reference is alt 1. |
| `id` | Owner / firm id. |
| `choice` | Boolean, `True` for the adopted technology. |
| `available` | 1.0 if the technology is feasible for the plant, else 0.0. |
| `env` | Regulatory regime (`regulated` / `deregulated` / `public`). |
| `post`, `cm`, `lnb` | Technology-class dummies (post-combustion, combustion modification, low-NOx burner). |
| `age` | Plant age (centered). |
| `vcost`, `kcost` | Variable and capital cost of the alternative for the plant. |

The `chid * alt` long table has $632 \times 15 = 9{,}480$ rows; only
$\sum \text{available} = 4{,}213$ are actually feasible.

**Reference.** Fowlie, M. (2010). "Emissions Trading, Electricity Restructuring,
and Investment in Pollution Abatement."
*American Economic Review* 100(3), 837-869.
([doi:10.1257/aer.100.3.837](https://doi.org/10.1257/aer.100.3.837))

**License.** Distributed in the R package `mlogit` under GPL-2.

### How to download

The CSV in this folder was extracted from the cran/mlogit GitHub mirror via
`pyreadr`:

```python
import pyreadr
rda = pyreadr.read_r_url(
    "https://raw.githubusercontent.com/cran/mlogit/master/data/NOx.rda"
)
nox = list(rda.values())[0]   # only one DataFrame in the .rda
```

Or, in R:

```r
data(NOx, package = "mlogit")
write.csv(NOx, "nox.csv", row.names = FALSE)
```

The notebook reads `nox.csv` from this folder.

### Model spec

We fit the simple alt-generic specification used in mlogit's `NOx` example —
utility depends on variable cost, capital cost, and the combustion-modification
dummy, with no alternative-specific intercepts:

$$
V_{ij} \;=\; \beta_v\,\mathrm{vcost}_{ij}
            \;+\; \beta_k\,\mathrm{kcost}_{ij}
            \;+\; \beta_c\,\mathrm{cm}_{ij}, \qquad
\Pr(\text{choose } j \mid \mathcal{A}_i)
   \;=\; \frac{\exp(V_{ij})}{\sum_{k\in\mathcal{A}_i} \exp(V_{ik})},
$$

where $\mathcal{A}_i$ is the available choice set for plant $i$.
Fowlie's full Table 4 spec adds regulatory-regime interactions; that's a
separate exercise.


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

torch.manual_seed(0)

HERE = Path.cwd()
```


```python
df = pd.read_csv(HERE / "nox.csv")
print(
    f"shape={df.shape}, "
    f"n_plants={df['chid'].nunique()}, "
    f"n_alts={df['alt'].nunique()}, "
    f"n_available_rows={int(df['available'].sum())}"
)
df.head()
```

    shape=(9480, 12), n_plants=632, n_alts=15, n_available_rows=4213





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
      <th>chid</th>
      <th>alt</th>
      <th>id</th>
      <th>choice</th>
      <th>available</th>
      <th>env</th>
      <th>post</th>
      <th>cm</th>
      <th>lnb</th>
      <th>age</th>
      <th>vcost</th>
      <th>kcost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>False</td>
      <td>1.0</td>
      <td>regulated</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.369104</td>
      <td>1.0944</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>1</td>
      <td>False</td>
      <td>1.0</td>
      <td>regulated</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>2.107275</td>
      <td>0.6720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3.0</td>
      <td>1</td>
      <td>True</td>
      <td>1.0</td>
      <td>regulated</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.697158</td>
      <td>0.4224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
      <td>False</td>
      <td>1.0</td>
      <td>regulated</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.604005</td>
      <td>0.9152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5.0</td>
      <td>1</td>
      <td>False</td>
      <td>1.0</td>
      <td>regulated</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.472123</td>
      <td>1.2096</td>
    </tr>
  </tbody>
</table>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook unbalanced-MNL via `mlogit::mlogit` on the long-format CSV.
The `subset = available == 1` argument to `dfidx` drops infeasible rows from
each plant's likelihood term — this is mlogit's way of handling unbalanced
choice sets.

```r
suppressPackageStartupMessages(library(mlogit))
df   <- read.csv("nox.csv")
data <- dfidx(df, shape="long", choice="choice",
              idx=c("chid","alt"), subset = available == 1)
mod  <- mlogit(choice ~ vcost + kcost + cm | 0, data=data, reflevel=1)
summary(mod)
```

The `| 0` after the formula suppresses alt-specific intercepts; only the
three slope coefficients are estimated. The full R script is in `fit_mlogit.R`
next to this notebook. The cell below calls it via `Rscript`; if R isn't
installed it transparently falls back to the cached output in
`mlogit_output.json`.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "nox.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -1064.8147





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
      <td>vcost</td>
      <td>-0.166479</td>
      <td>0.031341</td>
      <td>-5.311813</td>
      <td>1.085400e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kcost</td>
      <td>-0.035716</td>
      <td>0.009731</td>
      <td>-3.670189</td>
      <td>2.423712e-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cm</td>
      <td>-1.452331</td>
      <td>0.121085</td>
      <td>-11.994298</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_vcost|constant) + (itemsession_kcost|constant) + (itemsession_cm|constant)
```

— three shared (alt-generic) coefficients on the cost / dummy regressors, and
no alternative-specific intercepts. Because the data are unbalanced we must
also pass an `item_availability` mask to the dataset; without it, `torch-choice`
would treat infeasible alternatives as feasible and give the wrong answer
(with `vcost=0` and `kcost=0` rows in the data, infeasible alts look like
free options).

### Build a `ChoiceDataset`

The CSV is already in *long* format. We pivot the per-row covariates into
$(N_{\text{sessions}},\,N_{\text{items}},\,1)$ tensors with `pivot3d`,
extract the chosen-alt index from the `choice==True` rows, and build the
$(N_{\text{sessions}},\,N_{\text{items}})$ availability mask from the
`available` column.


```python
N_ALTS = 15
N_SESSIONS = df["chid"].nunique()

# Each chid has exactly 15 alt rows; encode 0-indexed session/alt for pivoting.
df = df.copy()
df["session"] = df["chid"] - 1
df["alt0"]    = df["alt"].astype(int) - 1
df = df.sort_values(["session", "alt0"]).reset_index(drop=True)

# Per-(session, alt) covariate tensors of shape (N_SESSIONS, N_ALTS, 1).
itemsession_vcost = utils.pivot3d(df, dim0="session", dim1="alt0", values="vcost")
itemsession_kcost = utils.pivot3d(df, dim0="session", dim1="alt0", values="kcost")
itemsession_cm    = utils.pivot3d(df, dim0="session", dim1="alt0", values="cm")

# Availability mask, shape (N_SESSIONS, N_ALTS).
item_availability = (
    utils.pivot3d(df, dim0="session", dim1="alt0", values="available")
    .squeeze(-1)
    .bool()
)

# Per-session chosen alt (0-indexed). Choice is a bool flag in the CSV; we pick
# the row with choice==True per chid.
chosen_rows = df.loc[df["choice"] == True].sort_values("session")
assert len(chosen_rows) == N_SESSIONS, "every plant should have exactly one chosen tech"
item_index = torch.LongTensor(chosen_rows["alt0"].values)

# Sanity: every chosen alt must be available.
assert item_availability[torch.arange(N_SESSIONS), item_index].all().item()

session_index = torch.arange(N_SESSIONS)

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=N_ALTS,
    num_users=1,
    num_sessions=N_SESSIONS,
    session_index=session_index,
    item_availability=item_availability,
    itemsession_vcost=itemsession_vcost,
    itemsession_kcost=itemsession_kcost,
    itemsession_cm=itemsession_cm,
)
print(dataset)
print(
    f"\navailability mask: shape={tuple(item_availability.shape)}, "
    f"sum={int(item_availability.sum())} (matches mlogit's {int(df['available'].sum())} "
    f"available rows)"
)
```

    ChoiceDataset(num_items=15, num_users=1, num_sessions=632, label=[], item_index=[632], user_index=[], session_index=[632], item_availability=[632, 15], itemsession_vcost=[632, 15, 1], itemsession_kcost=[632, 15, 1], itemsession_cm=[632, 15, 1], device=cpu)
    
    availability mask: shape=(632, 15), sum=4213 (matches mlogit's 4213 available rows)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))


### Fit

We use full-batch LBFGS for 1,000 epochs — the same recipe the package's
`paper_demo.py` uses for ModeCanada. `model.fit(...)` returns an
`EstimationOutput` whose `coef_summary` mirrors a standard regression table.


```python
model = ConditionalLogitModel(
    formula="(itemsession_vcost|constant) + (itemsession_kcost|constant) + (itemsession_cm|constant)",
    dataset=dataset,
    num_items=N_ALTS,
)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_vcost[constant]): Coefficient(variation=constant, num_items=15, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_kcost[constant]): Coefficient(variation=constant, num_items=15, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_cm[constant]): Coefficient(variation=constant, num_items=15, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_vcost[constant]] with 1 parameters, with constant level variation.
    X[itemsession_kcost[constant]] with 1 parameters, with constant level variation.
    X[itemsession_cm[constant]] with 1 parameters, with constant level variation.
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
    0 | model | ConditionalLogitModel | 3      | train | 0    
    ----------------------------------------------------------------
    3         Trainable params
    0         Non-trainable params
    3         Total params
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
    Log-likelihood: [Training] -1064.815, [Validation] None, [Test] None
    
    | Coefficient                   |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_vcost[constant]_0 |   -0.16648   |  0.0313414  |    -5.312 | 1.085e-07  | ***            |
    | itemsession_kcost[constant]_0 |   -0.0357165 |  0.00973147 |    -3.67  | 2.424e-04  | ***            |
    | itemsession_cm[constant]_0    |   -1.45234   |  0.121085   |   -11.994 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -1064.8147


## 4. Side-by-side comparison

`mlogit` exposes the three coefficients as bare names (`vcost`, `kcost`, `cm`).
`torch-choice` indexes them positionally inside `itemsession_<x>[constant]_0`.
The mapping below is the only per-dataset adapter required for the comparison
helper.


```python
NAME_MAP = {
    "itemsession_vcost[constant]_0": "vcost",
    "itemsession_kcost[constant]_0": "kcost",
    "itemsession_cm[constant]_0":    "cm",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 5.949e-06 (0.0007%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 2.176e-07 (0.0002%); tol 1e-03 -> PASS
    
    LL: mlogit=-1064.8147  torch-choice=-1064.8147  abs_diff=2.16e-05





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
      <td>itemsession_vcost[constant]_0</td>
      <td>-0.166479</td>
      <td>-0.166480</td>
      <td>7.586951e-07</td>
      <td>0.000456</td>
      <td>0.031341</td>
      <td>0.031341</td>
      <td>3.035920e-08</td>
      <td>0.000097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_kcost[constant]_0</td>
      <td>-0.035716</td>
      <td>-0.035717</td>
      <td>2.339528e-07</td>
      <td>0.000655</td>
      <td>0.009731</td>
      <td>0.009731</td>
      <td>5.742934e-09</td>
      <td>0.000059</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_cm[constant]_0</td>
      <td>-1.452331</td>
      <td>-1.452337</td>
      <td>5.948567e-06</td>
      <td>0.000410</td>
      <td>0.121085</td>
      <td>0.121085</td>
      <td>2.175956e-07</td>
      <td>0.000180</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

`torch-choice` reproduces `mlogit`'s unbalanced-MNL coefficient estimates,
standard errors, and log-likelihood to within float-precision round-off on the
NOx dataset. Both `vcost` and `kcost` come out **negative** — consistent with
cost as disutility. The same workflow extends to other unbalanced choice
datasets: the only per-dataset work is constructing the `(num_sessions,
num_items)` availability mask from the source's `available` column.

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong assumption:
*every plant responds identically to variable cost, capital cost, and the
combustion-modification class, regardless of how it is regulated.* The whole
point of Fowlie (2010) is that this is not true. Her central claim is that
**rate-of-return regulated plants over-invest in capital** relative to plants
exposed to market discipline, because regulated plants can roll capital
expenditure into their rate base while passing variable cost through directly.
Operationally, this means rate-of-return regulated plants should look *less
sensitive to* `kcost` (capital cost) — capital is partly free for them — and
**more sensitive to** `vcost` (variable cost) — they bear all of it.

The cleanest in-package way to put this hypothesis to the data is to interact
the cost regressors with a regulatory-regime dummy. This is a pooled-MNL
estimator with regime-specific cost coefficients; it is a coarser cousin of
Fowlie's full Table 4 specification (which adds `post`, `env`, `age`, and
firm-level controls).

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_vcost\|constant) + (itemsession_kcost\|constant) + (itemsession_cm\|constant)` | one shared `vcost`, `kcost`, `cm` coefficient | 3 |
| **B** (regime × cost) | regime-specific `vcost` and `kcost` coefficients via interaction observables | separate `vcost` and `kcost` coefs for regulated vs. unregulated plants; shared `cm` | 5 |
| **C** (per-plant cost sensitivity) | `(itemsession_vcost\|user) + ...` with `user_index = chid` | one `vcost` coef per plant (≈632 params) | not identified — see below |

**Why we drop Spec C.** The NOx panel is a *cross-section*: each of the 632
plants is observed making a single technology choice. A per-plant `vcost`
coefficient would require within-plant variation across multiple choice
occasions to identify; with a single observation per plant the per-plant
likelihood is maximized by sending that plant's coefficient to whichever sign
makes its observed alt the unique argmax of its choice-set utility — i.e., the
parameter is not point-identified. We document Spec C here for completeness
but do **not** fit it.

For Spec B we collapse `env` to a 2-level dummy: `regulated` (rate-of-return
regulated) vs. `unregulated` (deregulated + public). This matches the
treatment-control contrast Fowlie emphasizes: rate-of-return regulation
versus everything else.

### Build Spec B — regime-interacted cost observables

`torch-choice`'s formula syntax doesn't have a `*` interaction operator, so we
build the four interaction features by hand and pass each as its own
`itemsession_*` observable. With $D_i \in \{0, 1\}$ a regulated dummy:

$$
\begin{aligned}
\text{vcost\_reg}_{ij}   &= D_i \cdot \text{vcost}_{ij}, & \text{vcost\_unreg}_{ij}   &= (1 - D_i) \cdot \text{vcost}_{ij}, \\
\text{kcost\_reg}_{ij}   &= D_i \cdot \text{kcost}_{ij}, & \text{kcost\_unreg}_{ij}   &= (1 - D_i) \cdot \text{kcost}_{ij}.
\end{aligned}
$$

Stacking these four columns, plus the unchanged `cm` regressor, gives a model
that is mathematically identical to fitting MNL twice — once on regulated
plants, once on unregulated — with a *shared* `cm` coefficient.


```python
# Per-plant 0/1 indicator: 1 if rate-of-return regulated, 0 otherwise.
plant_env = df.groupby("session")["env"].first()
is_regulated = (plant_env == "regulated").astype(float).values   # shape (N_SESSIONS,)
print(
    f"regulated plants:   {int(is_regulated.sum())}, "
    f"unregulated plants: {int((1 - is_regulated).sum())}"
)

# Broadcast the (N_SESSIONS,) plant-level dummy over the (N_SESSIONS, N_ALTS, 1)
# cost tensors to build the four interaction features.
D_reg   = torch.tensor(is_regulated, dtype=torch.float32).view(-1, 1, 1)   # (N, 1, 1)
D_unreg = 1.0 - D_reg

itemsession_vcost_reg   = itemsession_vcost * D_reg
itemsession_vcost_unreg = itemsession_vcost * D_unreg
itemsession_kcost_reg   = itemsession_kcost * D_reg
itemsession_kcost_unreg = itemsession_kcost * D_unreg

# A new dataset that carries the original tensors plus the four interactions.
# `cm` keeps a single shared coefficient.
dataset_B = ChoiceDataset(
    item_index=item_index,
    num_items=N_ALTS,
    num_users=1,
    num_sessions=N_SESSIONS,
    session_index=session_index,
    item_availability=item_availability,
    itemsession_vcost_reg=itemsession_vcost_reg,
    itemsession_vcost_unreg=itemsession_vcost_unreg,
    itemsession_kcost_reg=itemsession_kcost_reg,
    itemsession_kcost_unreg=itemsession_kcost_unreg,
    itemsession_cm=itemsession_cm,
)
print(dataset_B)
```

    regulated plants:   292, unregulated plants: 340
    ChoiceDataset(num_items=15, num_users=1, num_sessions=632, label=[], item_index=[632], user_index=[], session_index=[632], item_availability=[632, 15], itemsession_vcost_reg=[632, 15, 1], itemsession_vcost_unreg=[632, 15, 1], itemsession_kcost_reg=[632, 15, 1], itemsession_kcost_unreg=[632, 15, 1], itemsession_cm=[632, 15, 1], device=cpu)


### Fit Spec B


```python
torch.manual_seed(0)
model_B = ConditionalLogitModel(
    formula=(
        "(itemsession_vcost_reg|constant)"
        " + (itemsession_vcost_unreg|constant)"
        " + (itemsession_kcost_reg|constant)"
        " + (itemsession_kcost_unreg|constant)"
        " + (itemsession_cm|constant)"
    ),
    dataset=dataset_B,
    num_items=N_ALTS,
)
result_B = model_B.fit(
    dataset_B,
    batch_size=-1,
    learning_rate=0.01,
    num_epochs=1000,
    model_optimizer="LBFGS",
    backend="lightning",
    print_summary=False,
)
print(result_B)
print(f"\nSpec B train log-likelihood: {result_B.train_ll:.4f}")
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
    7         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    ==================== model results ====================
    Log-likelihood: [Training] -1059.534, [Validation] None, [Test] None
    
    | Coefficient                         |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:------------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_vcost_reg[constant]_0   |   -0.140127  |   0.0448601 |    -3.124 | 0.002      | **             |
    | itemsession_vcost_unreg[constant]_0 |   -0.167513  |   0.0409194 |    -4.094 | 4.245e-05  | ***            |
    | itemsession_kcost_reg[constant]_0   |   -0.0156633 |   0.0141741 |    -1.105 | 0.269      |                |
    | itemsession_kcost_unreg[constant]_0 |   -0.0526103 |   0.0128475 |    -4.095 | 4.222e-05  | ***            |
    | itemsession_cm[constant]_0          |   -1.4648    |   0.121813  |   -12.025 | < 2e-16    | ***            |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Spec B train log-likelihood: -1059.5339


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalizes complexity more
aggressively, so it tends to favor parsimonious models — appropriate when we
believe the simpler spec is closer to the truth. AIC penalizes less and tends
to favor richer models when sample size is large.


```python
ll_a = float(result.train_ll)
k_a  = int(result.coef_summary.shape[0])

ll_b = float(result_B.train_ll)
k_b  = int(result_B.coef_summary.shape[0])

specs = [
    ("A: pooled MNL",        ll_a, k_a),
    ("B: regime x cost",     ll_b, k_b),
]
table = pd.DataFrame([
    {
        "spec":     name,
        "LL":       ll,
        "n_params": k,
        "AIC":      2 * k - 2 * ll,
        "BIC":      k * math.log(N_SESSIONS) - 2 * ll,
    }
    for name, ll, k in specs
])
table["dAIC vs A"] = table["AIC"] - table.iloc[0]["AIC"]
table["dBIC vs A"] = table["BIC"] - table.iloc[0]["BIC"]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A: pooled MNL</th>
      <td>-1064.814697</td>
      <td>3</td>
      <td>2135.629395</td>
      <td>2148.976063</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: regime x cost</th>
      <td>-1059.533936</td>
      <td>5</td>
      <td>2129.067871</td>
      <td>2151.312318</td>
      <td>-6.561523</td>
      <td>2.336255</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **regulatory regime
matters for technology choice**, and a pooled MNL that ignores it is
mis-specified.

- **Fowlie, M. (2010)**, "Emissions Trading, Electricity Restructuring, and
  Investment in Pollution Abatement." *American Economic Review* 100(3),
  837–857. The paper's headline result is that rate-of-return regulated
  plants — whose capital expenditure is partially recoverable through the
  rate base — choose more capital-intensive abatement technologies than
  otherwise-similar deregulated plants. Fowlie's full Table 4 estimates a
  conditional logit with regime × technology-class interactions; our Spec B
  is a stripped-down version of the same idea, interacting `vcost` and
  `kcost` with a single regulated/unregulated dummy.
- **Train, K. (2009),** *Discrete Choice Methods with Simulation*, Ch. 6.
  The general theory of taste heterogeneity in random-utility models. When
  heterogeneity has an observable cause (here: regulatory regime), interaction
  terms are the textbook fixed-effects approach; when it does not, mixed
  logit with random coefficients is preferred. NOx has an observable cause,
  so interactions are the right tool.

**Practical guidance.**

1. Use **Spec A** (pooled MNL) as a baseline and as the numerical reference
   the §4 R/`mlogit` cross-check is built on.
2. Use **Spec B** (regime × cost) when the research question is about
   *regime-driven capital bias* — the Fowlie (2010) headline. Compare the
   `vcost_reg` vs. `vcost_unreg` and `kcost_reg` vs. `kcost_unreg`
   coefficients in `result_B.coef_summary` directly: if the regulated `kcost`
   coefficient is closer to zero (less negative) than the unregulated one,
   regulated plants weigh capital cost less heavily — Fowlie's prediction.
3. Specification **C** (per-plant random coefficients) is **not identified**
   on this cross-section. To approximate it you would need to either (a)
   pool plants within a firm via `id` (220 firms × ~3 plants each — still
   thin), or (b) bring in a separate panel of plant-year decisions. Both
   are out of scope for this tutorial.

**Caveat.** §5's alternative specs are **not** cross-validated against
`mlogit`. The R `fit_mlogit.R` script in this folder runs Spec A only, and
the `mlogit_output.json` cache stores Spec A's reference. Spec B is an
internal-to-`torch-choice` model-fit comparison; reproducing it in R would
require building the four interaction columns server-side and re-running
`mlogit::mlogit` with the new formula.
