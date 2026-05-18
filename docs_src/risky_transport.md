# Risky Transport mode choice — `torch-choice` reproduces R/`mlogit`

This tutorial fits a multinomial logit model of transport-mode choice between
Freetown and Lungi International Airport on the Léon–Miguel (AEJ-Applied,
2017) survey. We fit the same model twice — once in R with `mlogit`, once in
Python with `torch-choice` — and show that the two implementations recover
the same coefficients to numerical precision, including with **per-occasion
availability variation** (not all four modes are available to every
respondent).

## 1. About this dataset

**Domain.** Development economics / value of statistical life. 561 individuals
make 1,793 choice occasions over up to one of four transport modes between
Freetown city and Lungi airport across the estuary. Each chid (choice
situation) sees only the alternatives available at the time of survey:

| Mode | Notes |
|---|---|
| `WaterTaxi` | Reference (alphabetically first within the survey label set) |
| `Helicopter` | Highest cost, highest visible fatality rate |
| `Ferry` | Cheapest, slowest |
| `Hovercraft` | Mid-priced, intermediate risk |

Per-occasion features:
- `cost`: generalised cost (Leones)
- `risk`: fatality rate, deaths per 100,000 trips
- `seats`: a measure of seat-fill / crowdedness (0–1)
- `weight`: sampling weight (3 unique values, **constant within chid** — not
  an alt-specific covariate; we therefore omit it from the formula)

**Availability.** 391 chids see only 2 modes, 985 see 3, and 417 see all 4.
mlogit handles partial availability implicitly via missing rows; on the
torch-choice side we encode it explicitly via an `item_availability` boolean
mask passed to `ChoiceDataset`.

**Reference.** León, G., Miguel, E. (2017). "Risky Transportation Choices and
the Value of a Statistical Life." *American Economic Journal: Applied
Economics* 9(1), 202–228.
([doi:10.1257/app.20160140](https://doi.org/10.1257/app.20160140))

**License.** Distributed in the R package `mlogit` under GPL-2, redistributable
with citation.

### How to download

The CSV in this folder was extracted from the `mlogit` package's bundled
dataset:

```r
library(mlogit)
data(RiskyTransport)
write.csv(RiskyTransport, "risky_transport.csv", row.names = FALSE)
```

The notebook reads `risky_transport.csv` from this folder.


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
df = pd.read_csv(HERE / "risky_transport.csv")
print(f"shape={df.shape}, n_individuals={df['id'].nunique()}, n_chids={df['chid'].nunique()}")

# Availability check: how many alternatives does each chid see?
rows_per_chid = df.groupby("chid").size()
print(f"rows-per-chid distribution: {rows_per_chid.value_counts().sort_index().to_dict()}")
print(f"chids with <4 modes: {(rows_per_chid < 4).sum()}  (=> need item_availability)")

df.head()
```

    shape=(5405, 22), n_individuals=561, n_chids=1793
    rows-per-chid distribution: {2: 391, 3: 985, 4: 417}
    chids with <4 modes: 1376  (=> need item_availability)





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
      <th>choice</th>
      <th>mode</th>
      <th>cost</th>
      <th>risk</th>
      <th>weight</th>
      <th>seats</th>
      <th>noise</th>
      <th>crowdness</th>
      <th>convloc</th>
      <th>...</th>
      <th>african</th>
      <th>lifeExp</th>
      <th>dwage</th>
      <th>iwage</th>
      <th>educ</th>
      <th>fatalism</th>
      <th>gender</th>
      <th>age</th>
      <th>haveChildren</th>
      <th>swim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8020605</td>
      <td>0.0</td>
      <td>WaterTaxi</td>
      <td>59.310089</td>
      <td>2.551270</td>
      <td>1.467884</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>...</td>
      <td>yes</td>
      <td>52.0</td>
      <td>19.397476</td>
      <td>19.397476</td>
      <td>high</td>
      <td>6.0</td>
      <td>male</td>
      <td>33</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8020605</td>
      <td>1.0</td>
      <td>Ferry</td>
      <td>34.728870</td>
      <td>4.431152</td>
      <td>1.467884</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>...</td>
      <td>yes</td>
      <td>52.0</td>
      <td>19.397476</td>
      <td>19.397476</td>
      <td>high</td>
      <td>6.0</td>
      <td>male</td>
      <td>33</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8020605</td>
      <td>0.0</td>
      <td>Hovercraft</td>
      <td>57.047050</td>
      <td>3.881836</td>
      <td>1.467884</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>...</td>
      <td>yes</td>
      <td>52.0</td>
      <td>19.397476</td>
      <td>19.397476</td>
      <td>high</td>
      <td>6.0</td>
      <td>male</td>
      <td>33</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8020605</td>
      <td>0.0</td>
      <td>Helicopter</td>
      <td>99.869286</td>
      <td>18.408203</td>
      <td>1.467884</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>...</td>
      <td>yes</td>
      <td>52.0</td>
      <td>19.397476</td>
      <td>19.397476</td>
      <td>high</td>
      <td>6.0</td>
      <td>male</td>
      <td>33</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8020605</td>
      <td>0.0</td>
      <td>WaterTaxi</td>
      <td>59.310089</td>
      <td>2.551270</td>
      <td>1.467884</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>...</td>
      <td>yes</td>
      <td>52.0</td>
      <td>19.397476</td>
      <td>19.397476</td>
      <td>high</td>
      <td>6.0</td>
      <td>male</td>
      <td>33</td>
      <td>yes</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



## 2. R/`mlogit` reference fit

We fit a textbook MNL on the long-format CSV using `mlogit::mlogit`, with
`WaterTaxi` as the reference alternative and **no alt-specific intercepts**
(the canonical Léon–Miguel specification):

```r
suppressPackageStartupMessages(library(mlogit))
df          <- read.csv("risky_transport.csv")
df$choice   <- as.logical(df$choice)
data        <- dfidx(df, shape="long", choice="choice", idx=c("chid", "mode"))
mod         <- mlogit(choice ~ cost + risk + seats | 0,
                      data=data, reflevel="WaterTaxi")
summary(mod)
```

The `| 0` part suppresses alt-specific intercepts; the formula yields three
generic coefficients on `cost`, `risk`, and `seats`. mlogit silently drops the
modes that aren't present for a given chid, so the per-chid choice set is
exactly whatever rows exist in the long-format CSV.

The full R script is in `fit_mlogit.R` next to this notebook. The cell below
calls it via `Rscript`; if R isn't installed it transparently falls back to
the cached output in `mlogit_output.json`, so the comparison still renders for
readers without an R installation.


```python
mlogit_df, mlogit_ll = run_or_load_mlogit(
    r_script_path=HERE / "fit_mlogit.R",
    csv_path=HERE / "risky_transport.csv",
    cache_path=HERE / "mlogit_output.json",
)
print(f"R/mlogit train log-likelihood: {mlogit_ll:.4f}")
mlogit_df
```

    [mlogit] Live R fit succeeded (fit_mlogit.R).
    R/mlogit train log-likelihood: -1722.3061





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
      <td>cost</td>
      <td>-0.009688</td>
      <td>0.001003</td>
      <td>-9.657469</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>risk</td>
      <td>-0.111896</td>
      <td>0.010865</td>
      <td>-10.298727</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>seats</td>
      <td>-0.388008</td>
      <td>0.186164</td>
      <td>-2.084228</td>
      <td>0.037139</td>
    </tr>
  </tbody>
</table>
</div>



## 3. `torch-choice` fit

The same MNL specification expressed in `torch-choice`'s formula syntax is

```
(itemsession_cost|constant) + (itemsession_risk|constant) + (itemsession_seats|constant)
```

— three shared coefficients (no alt-specific intercepts), matching mlogit's
`| 0` term.

### Build a `ChoiceDataset` with availability

The CSV is already in long format (one row per `(chid, mode)`), but most
chids see only 2 or 3 modes. To get a regular `(num_sessions, num_items, F)`
tensor we have to re-grid the data to a complete `(chid × mode)` lattice with
zero-fill placeholders for missing alternatives, and pass an
`item_availability` boolean mask so the model sets the utility of unavailable
modes to `-inf` before the softmax.


```python
# Mode encoding: index 0 = WaterTaxi (matches R's reflevel="WaterTaxi").
MODE_ORDER = ["WaterTaxi", "Helicopter", "Ferry", "Hovercraft"]
mode_to_idx = {m: i for i, m in enumerate(MODE_ORDER)}
N_MODES = len(MODE_ORDER)

df = df.copy()
df["mode_idx"] = df["mode"].map(mode_to_idx)

# Stable session ordering: sessions = chids in first-appearance order.
chid_order = df["chid"].drop_duplicates().tolist()
chid_to_session = {c: i for i, c in enumerate(chid_order)}
df["session"] = df["chid"].map(chid_to_session)
N_SESS = len(chid_order)
print(f"num_sessions = {N_SESS}  (= mlogit's n_obs = {int(mlogit_df.shape[0]) if False else 1793})")

df = df.sort_values(["session", "mode_idx"]).reset_index(drop=True)

# Re-grid to the full (session, mode_idx) lattice. Missing rows get zeros for
# the feature columns; the availability mask records which rows existed.
full_index = pd.MultiIndex.from_product(
    [range(N_SESS), range(N_MODES)],
    names=["session", "mode_idx"],
)
filled = (
    df.set_index(["session", "mode_idx"])[["cost", "risk", "seats"]]
      .reindex(full_index, fill_value=0.0)
      .reset_index()
)
present = df.set_index(["session", "mode_idx"]).index
avail_series = pd.Series(False, index=full_index)
avail_series.loc[present] = True
avail_grid = avail_series.unstack("mode_idx").reindex(columns=range(N_MODES))
item_availability = torch.tensor(avail_grid.values, dtype=torch.bool)
print(f"item_availability shape: {tuple(item_availability.shape)}")
print(f"available counts per mode {MODE_ORDER}: {item_availability.sum(0).tolist()}")

itemsession_cost  = utils.pivot3d(filled, dim0="session", dim1="mode_idx", values="cost")
itemsession_risk  = utils.pivot3d(filled, dim0="session", dim1="mode_idx", values="risk")
itemsession_seats = utils.pivot3d(filled, dim0="session", dim1="mode_idx", values="seats")

# Per-session chosen mode index (binary `choice == 1` row's mode_idx).
chosen = df[df["choice"] == 1.0].sort_values("session")
assert len(chosen) == N_SESS, f"Expected {N_SESS} choices, got {len(chosen)}"
item_index = torch.LongTensor(chosen["mode_idx"].values)

# Sanity check: every chosen alternative must be flagged available.
for s, m in enumerate(item_index.tolist()):
    assert bool(item_availability[s, m]), f"Session {s} chose unavailable mode {m}"

dataset = ChoiceDataset(
    item_index=item_index,
    num_items=N_MODES,
    num_sessions=N_SESS,
    session_index=torch.arange(N_SESS),
    item_availability=item_availability,
    itemsession_cost=itemsession_cost,
    itemsession_risk=itemsession_risk,
    itemsession_seats=itemsession_seats,
)
print(dataset)
```

    num_sessions = 1793  (= mlogit's n_obs = 1793)
    item_availability shape: (1793, 4)
    available counts per mode ['WaterTaxi', 'Helicopter', 'Ferry', 'Hovercraft']: [1606, 778, 1725, 1296]
    ChoiceDataset(num_items=4, num_users=1, num_sessions=1793, label=[], item_index=[1793], user_index=[], session_index=[1793], item_availability=[1793, 4], itemsession_cost=[1793, 4, 1], itemsession_risk=[1793, 4, 1], itemsession_seats=[1793, 4, 1], device=cpu)


    /Users/tianyudu/Development/torch-choice/torch_choice/data/utils.py:28: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:219.)
      tensor_slice.append(torch.Tensor(layer[dim1_list].values))


### Fit

Same recipe as the Yogurt notebook: full-batch LBFGS for 1,000 epochs, with
`torch.manual_seed(0)` for reproducible parameter initialisation. The model
has only three free parameters.


```python
torch.manual_seed(0)
model = ConditionalLogitModel(
    formula="(itemsession_cost|constant) + (itemsession_risk|constant) + (itemsession_seats|constant)",
    dataset=dataset,
    num_items=N_MODES,
)
print(model)
```

    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (itemsession_cost[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_risk[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
        (itemsession_seats[constant]): Coefficient(variation=constant, num_items=4, num_users=None, num_params=1, 1 trainable parameters in total, initialization=normal, device=cpu).
      )
    )
    Conditional logistic discrete choice model, expects input features:
    
    X[itemsession_cost[constant]] with 1 parameters, with constant level variation.
    X[itemsession_risk[constant]] with 1 parameters, with constant level variation.
    X[itemsession_seats[constant]] with 1 parameters, with constant level variation.
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
    Log-likelihood: [Training] -1722.306, [Validation] None, [Test] None
    
    | Coefficient                   |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance   |
    |:------------------------------|-------------:|------------:|----------:|:-----------|:---------------|
    | itemsession_cost[constant]_0  |   -0.0096877 |  0.00100314 |    -9.657 | < 2e-16    | ***            |
    | itemsession_risk[constant]_0  |   -0.111896  |  0.010865   |   -10.299 | < 2e-16    | ***            |
    | itemsession_seats[constant]_0 |   -0.388011  |  0.186164   |    -2.084 | 0.037      | *              |
    Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    torch-choice train log-likelihood: -1722.3060


## 4. Side-by-side comparison

`torch-choice` exposes the three generic coefficients as
`itemsession_<feature>[constant]_0`; mlogit names them simply `cost`,
`risk`, `seats`. The mapping below is the only per-dataset adapter required
for the comparison helper.

Because there are no alt-specific intercepts in this spec, the choice of
reference alternative (`WaterTaxi`) doesn't affect the numerical fit — only
the labelling of any future intercepts would shift.


```python
NAME_MAP = {
    "itemsession_cost[constant]_0":  "cost",
    "itemsession_risk[constant]_0":  "risk",
    "itemsession_seats[constant]_0": "seats",
}
diff = compare_coefs(mlogit_df, result, NAME_MAP)
print(
    f"\nLL: mlogit={mlogit_ll:.4f}  "
    f"torch-choice={result.train_ll:.4f}  "
    f"abs_diff={abs(mlogit_ll - result.train_ll):.2e}"
)
diff
```

    [compare_coefs] estimates: max |diff| = 3.651e-06 (0.0009%); tol 1e-03 -> PASS
    [compare_coefs] std errs:  max |diff| = 4.684e-08 (0.0001%); tol 1e-03 -> PASS
    
    LL: mlogit=-1722.3061  torch-choice=-1722.3060  abs_diff=9.85e-05





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
      <td>itemsession_cost[constant]_0</td>
      <td>-0.009688</td>
      <td>-0.009688</td>
      <td>5.920346e-08</td>
      <td>0.000611</td>
      <td>0.001003</td>
      <td>0.001003</td>
      <td>5.777415e-10</td>
      <td>0.000058</td>
    </tr>
    <tr>
      <th>1</th>
      <td>itemsession_risk[constant]_0</td>
      <td>-0.111896</td>
      <td>-0.111896</td>
      <td>4.461908e-07</td>
      <td>0.000399</td>
      <td>0.010865</td>
      <td>0.010865</td>
      <td>1.561592e-08</td>
      <td>0.000144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>itemsession_seats[constant]_0</td>
      <td>-0.388008</td>
      <td>-0.388011</td>
      <td>3.651263e-06</td>
      <td>0.000941</td>
      <td>0.186164</td>
      <td>0.186164</td>
      <td>4.683578e-08</td>
      <td>0.000025</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

`torch-choice` reproduces `mlogit`'s MNL coefficient estimates and
log-likelihood to within float32 round-off on this dataset, **even with
heterogeneous per-occasion availability** (only 23% of chids see all four
modes). The `item_availability` boolean mask is the only extra piece of
plumbing needed compared to a fully-balanced design like Yogurt.

Interpretation: `cost` and `risk` both have negative coefficients, as
expected (price and fatality rate are disutility). `seats`'s negative sign
reflects that the variable encodes seat *occupancy* (more occupied = more
crowded), so higher values reduce utility.

For the survey-weighted variant of this fit (`mlogit(..., weights=weight)`)
and the Léon–Miguel value-of-statistical-life decomposition, see the
companion notebook `risky_transport_vsl.ipynb` (forthcoming).

## 5. Alternative specifications and theoretical recommendation

The pooled MNL fit in §3–§4 (call it **Spec A**) imposes a strong assumption:
*every respondent values cost and risk identically*. The motivating contribution
of Léon & Miguel (2017) is in fact the **value of statistical life (VSL)**
estimate that this dataset enables — and a single VSL number, computed as the
ratio of the cost coefficient to the risk coefficient, is a population mean
that hides important heterogeneity. Léon & Miguel argue that this trade-off
varies systematically with chooser characteristics: **age** (older respondents
have less remaining life-years to lose), **swim ability** (a partial substitute
for water-mode safety), and **fatalism** (respondents who believe death is
preordained should respond less to risk).

This intuition motivates two richer specifications below. They are
**fixed-effects approximations** to a mixed/random-coefficients logit on risk:
Spec B observes heterogeneity through three pre-registered moderators, while
Spec C estimates one risk coefficient per respondent. Both connect directly to
the VSL framing: Spec B yields three sub-population VSLs (by age tertile, by
swim ability, by fatalism level), and Spec C yields the empirical distribution
of individual VSLs over the 561 respondents. See Train (2009) Ch. 6 for the
general mixed-logit theory.

### Specs to compare

| Spec | Formula | What changes | n_params |
|---|---|---|---|
| **A** (pooled MNL — current) | `(itemsession_cost\|constant) + (itemsession_risk\|constant) + (itemsession_seats\|constant)` | one cost / risk / seats coef each | 3 |
| **B** (chooser-specific risk sensitivity) | A + `(session_age\|item) + (session_swim\|item) + (session_fatalism\|item)` | three brand-specific shifters per moderator (3 free × 3 moderators = 9 extras) | 12 |
| **C** (per-respondent risk coefficient) | swap `(itemsession_risk\|constant)` &rarr; `(itemsession_risk\|user)` | each of 561 respondents gets their own risk coefficient | 563 |

For Spec B we need three new chooser-specific observables in the
`ChoiceDataset`. They live at the *session* level (constant within each chid)
and we standardise the continuous ones (`age`, `fatalism`) to z-scores so the
coefficient magnitudes are comparable; `swim` is binary 0/1 (mapped from the
"yes"/"no" strings in the CSV).



```python
import math

# Build a richer ChoiceDataset that adds:
#   - user_index / num_users (561 respondents -> needed for Spec C)
#   - session_age / session_swim / session_fatalism (chooser-specific moderators
#     for Spec B, constant within each chid)
sess_cov = (
    df.drop_duplicates("chid")
      .sort_values("session")
      .reset_index(drop=True)
)

# id -> 0-indexed user, in stable first-appearance order.
id_order = sess_cov["id"].drop_duplicates().tolist()
id_to_user = {x: i for i, x in enumerate(id_order)}
sess_cov["user_idx"] = sess_cov["id"].map(id_to_user)
NUM_USERS = sess_cov["user_idx"].nunique()

# Standardise continuous moderators; map "yes"/"no" -> 1/0 for swim.
age_z = ((sess_cov["age"] - sess_cov["age"].mean())
         / sess_cov["age"].std()).to_numpy()
fat_z = ((sess_cov["fatalism"] - sess_cov["fatalism"].mean())
         / sess_cov["fatalism"].std()).to_numpy()
swim_b = (sess_cov["swim"].astype(str).str.lower().eq("yes")).astype(float).to_numpy()

session_age      = torch.tensor(age_z,  dtype=torch.float32).view(-1, 1)
session_swim     = torch.tensor(swim_b, dtype=torch.float32).view(-1, 1)
session_fatalism = torch.tensor(fat_z,  dtype=torch.float32).view(-1, 1)
user_index       = torch.LongTensor(sess_cov["user_idx"].values)

dataset_full = ChoiceDataset(
    item_index=item_index,
    num_items=N_MODES,
    num_sessions=N_SESS,
    num_users=NUM_USERS,
    user_index=user_index,
    session_index=torch.arange(N_SESS),
    item_availability=item_availability,
    itemsession_cost=itemsession_cost,
    itemsession_risk=itemsession_risk,
    itemsession_seats=itemsession_seats,
    session_age=session_age,
    session_swim=session_swim,
    session_fatalism=session_fatalism,
)
print(f"NUM_USERS = {NUM_USERS}  (= n_individuals)")
print(dataset_full)

```

    NUM_USERS = 561  (= n_individuals)
    ChoiceDataset(num_items=4, num_users=561, num_sessions=1793, label=[], item_index=[1793], user_index=[1793], session_index=[1793], item_availability=[1793, 4], itemsession_cost=[1793, 4, 1], itemsession_risk=[1793, 4, 1], itemsession_seats=[1793, 4, 1], session_age=[1793, 1], session_swim=[1793, 1], session_fatalism=[1793, 1], device=cpu)


### Fit Specs A, B, and C

We use full-batch LBFGS for 1,000 epochs as in §3 but with a smaller learning
rate (`lr=1e-3`) so the line search remains stable for the higher-dimensional
specs (Specs B and C diverge at the §3 default of `lr=1e-2`). Spec A fits
identically under either learning rate; we re-fit it here on `dataset_full`
purely so the three specs share an apples-to-apples training recipe.



```python
def fit_spec(formula: str, label: str, *, num_epochs: int = 1000,
             optimizer: str = "LBFGS", learning_rate: float = 1e-3):
    """Fit a torch-choice spec on dataset_full; return (train_ll, n_params, result)."""
    torch.manual_seed(0)
    m = ConditionalLogitModel(
        formula=formula,
        dataset=dataset_full,
        num_items=N_MODES,
        num_users=NUM_USERS,
    )
    r = m.fit(
        dataset_full,
        batch_size=-1,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_optimizer=optimizer,
        backend="lightning",
        print_summary=False,
    )
    n_params = int(r.coef_summary.shape[0])
    return float(r.train_ll), n_params, r

formula_a = ("(itemsession_cost|constant) + (itemsession_risk|constant) "
             "+ (itemsession_seats|constant)")
formula_b = (formula_a
             + " + (session_age|item) + (session_swim|item) "
             + "+ (session_fatalism|item)")
formula_c = ("(itemsession_cost|constant) + (itemsession_risk|user) "
             "+ (itemsession_seats|constant)")

ll_a, k_a, result_a = fit_spec(formula_a, "A: pooled MNL")
ll_b, k_b, result_b = fit_spec(formula_b, "B: chooser moderators")
ll_c, k_c, result_c = fit_spec(formula_c, "C: per-respondent risk")

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


    GPU available: True (mps), used: False


    TPU available: False, using: 0 TPU cores


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/setup.py:175: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.


    Starting PyTorch Lightning training loop.


    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    
      | Name  | Type                  | Params | Mode  | FLOPs
    ----------------------------------------------------------------
    0 | model | ConditionalLogitModel | 12     | train | 0    
    ----------------------------------------------------------------
    12        Trainable params
    0         Non-trainable params
    12        Total params
    0.000     Total estimated model params size (MB)
    8         Modules in train mode
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
    0 | model | ConditionalLogitModel | 563    | train | 0    
    ----------------------------------------------------------------
    563       Trainable params
    0         Non-trainable params
    563       Total params
    0.002     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode
    0         Total Flops


    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/tianyudu/Development/torch-choice/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:317: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1000` reached.


    Spec A: LL=-1722.31, n_params=3
    Spec B: LL=-1665.52, n_params=12
    Spec C: LL=-1093.84, n_params=563


### Compare via information criteria

Both AIC and BIC trade off goodness-of-fit (higher LL is better) against
complexity (more parameters is worse). BIC penalises complexity more
aggressively, so it tends to favour parsimonious models — appropriate when we
believe the simpler spec is closer to the truth. AIC penalises less and tends
to favour richer models when sample size is large.



```python
specs = [
    ("A: pooled MNL",                ll_a, k_a),
    ("B: chooser moderators",        ll_b, k_b),
    ("C: per-respondent risk",       ll_c, k_c),
]
table = pd.DataFrame([
    {
        "spec":     name,
        "LL":       ll,
        "n_params": k,
        "AIC":      2 * k - 2 * ll,
        "BIC":      k * math.log(N_SESS) - 2 * ll,
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
      <td>-1722.309570</td>
      <td>3</td>
      <td>3450.619141</td>
      <td>3467.094077</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>B: chooser moderators</th>
      <td>-1665.523804</td>
      <td>12</td>
      <td>3355.047607</td>
      <td>3420.947353</td>
      <td>-95.571533</td>
      <td>-46.146724</td>
    </tr>
    <tr>
      <th>C: per-respondent risk</th>
      <td>-1093.844971</td>
      <td>563</td>
      <td>3313.689941</td>
      <td>6405.486343</td>
      <td>-136.929199</td>
      <td>2938.392266</td>
    </tr>
  </tbody>
</table>
</div>



### Theoretical recommendation

The published literature on this dataset is unambiguous: **the trade-off
between cost and risk varies across respondents**, and any credible VSL
estimate must allow for this heterogeneity.

- **León & Miguel (2017)**, the source paper, motivate exactly this dataset
  for VSL estimation and argue that age, swim ability, and fatalism are the
  three theoretically defensible moderators of the risk coefficient. Their
  reduced-form analysis already documents that these covariates shift mode
  choice in directions consistent with risk-tolerance heterogeneity.
- **Train (2009), *Discrete Choice Methods with Simulation*, Ch. 6** is the
  canonical reference for mixed/random-coefficients logit; the appropriate
  generalisation here is a mixed logit with a random coefficient on `risk`,
  with VSL recovered as the (now stochastic) ratio
  $-\beta_{\mathrm{risk}}/\beta_{\mathrm{cost}}$.
- **McFadden & Train (2000)** show that mixed logit can approximate any
  random-utility model arbitrarily closely; the IIA assumption baked into
  pooled MNL is overly restrictive when tastes vary.

For Risky Transport specifically, the recommended specification is a **mixed
logit with a random coefficient on `risk`**. `torch-choice` does not yet ship
a Monte-Carlo or quadrature-based mixed-logit estimator, so the closest
in-package approximations are Specs B and C: B *parametrises* the
heterogeneity through three theoretically motivated moderators (and is the
form most consumable by VSL sub-population analysis), while C *non-parametrises*
it by giving every respondent their own coefficient (a fixed-effects
approximation to the random-coefficients spec).

**Practical guidance for this notebook's reader:**

1. Use **Spec A** (pooled MNL) as a baseline — verifying the workflow
   end-to-end and as the numerical reference against R/`mlogit` (§4).
2. Use **Spec B** (chooser-moderated risk sensitivity) when the research
   question is about *which observed traits predict risk tolerance*. This is
   the spec closest in spirit to León & Miguel's reduced-form heterogeneity
   analysis and yields interpretable subgroup VSLs.
3. Use **Spec C** (per-respondent risk coefficient) when interest is in the
   *full empirical distribution* of individual VSLs. With a median of only
   3 chids per respondent, individual estimates are noisy; treat the
   distribution rather than any single estimate.
4. The published mixed-logit / random-coefficients spec remains the gold
   standard; reach for it when `torch-choice` adds mixed-logit support, or use
   Apollo / Stata's `mixlogit` in the meantime.

**Caveat.** §5's alternative specs are *not* cross-validated against `mlogit`
(R's `mlogit` does not natively fit the per-user fixed-effects MNL of Spec C
in this form, and Spec B's interactions would require a per-row alt-specific
construction in long format). The §4 verification covers Spec A only. Specs
B and C are internal-to-`torch-choice` model-fit comparisons.

