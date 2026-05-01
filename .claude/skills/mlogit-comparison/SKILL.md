---
name: mlogit-comparison
description: Use when verifying that a `torch-choice` fit on a new dataset reproduces an R/`mlogit` reference fit, for either MNL/conditional logit or nested logit (mixed/random-coefficients logit is out of scope). The skill is the runbook for adding a new `tutorials/<dataset>/` folder following the established 4-section template (intro → R fit → torch-choice fit → side-by-side comparison) with both point estimates and standard errors checked. Trigger when the user says things like "let's add an mlogit tutorial for X", "verify torch-choice matches R on dataset Y", "replicate this paper's MNL or nested logit", or when adding a new entry alongside `tutorials/yogurt/`.
---

# `torch-choice` ↔ R/`mlogit` cross-package replication

This skill codifies the procedure used to build `tutorials/yogurt/` so the next
six R-package datasets (and any future references) get the same shape and the
same verification rigor.

## When to use this skill

Use it when:
- Adding a new `tutorials/<dataset>/` folder for a discrete-choice dataset that
  also has a published R/`mlogit` (or other reference) fit to verify against.
- Asked to "show torch-choice matches R on X."
- Replicating a paper's MNL/CL/nested-logit numbers from a public dataset.

Do **not** use it for:
- Tutorials that don't have a numerical reference to match (those are just
  feature demonstrations, not replications).
- Datasets whose reference fit uses random-coefficients / mixed logit — those
  need a different verification strategy (parameter distributions, not point
  estimates).

## Reference example

`tutorials/yogurt/` is the canonical worked example. When in doubt, mimic
its file layout and code shape exactly. Files:

```
tutorials/yogurt/
  README.md              # 1-paragraph dataset description + citation + license
  yogurt.csv             # data file (only if license + size permit redistribution)
  fit_mlogit.R           # R script: fits MNL, emits JSON to stdout
  mlogit_output.json     # cached R output (committed; readers without R use this)
  yogurt.ipynb           # 4-section notebook (intro / R fit / tc fit / compare)
```

Shared infrastructure that already exists (do not re-implement):
- `tutorials/_mlogit_compare.py` — `run_or_load_mlogit()` and
  `compare_coefs()`. Imported by every dataset notebook.

## Preflight checks (before you start)

0. **R + mlogit are reachable.** Run `Rscript -e 'cat(R.version.string,
   "\n"); cat("mlogit ", as.character(packageVersion("mlogit")))'`. Both the
   R version and the mlogit version should print without error. If R isn't
   installed, install via `brew install --cask r`; if mlogit is missing,
   `Rscript -e 'install.packages("mlogit", repos="https://cloud.r-project.org")'`.
1. **Data is available** at `tutorials/public_datasets/downloads/<name>/<name>.csv`
   (extracted from the cran/<package> mirror via `pyreadr`, or downloaded from
   Dataverse/Zenodo). If not, run the extraction first.
2. **Reference numbers exist** somewhere citable: a paper's coefficient table,
   a textbook (Train 2009, Greene's *Econometric Analysis*), an mlogit vignette,
   or an Apollo case study. Without a reference there's nothing to verify.
3. **License permits redistribution** of the CSV inside the tutorial folder.
   GPL-2 (mlogit/Ecdat/AER), CC0, CC-BY are fine. If not, the notebook should
   read from `tutorials/public_datasets/downloads/<name>/` instead of copying.
4. **Model spec is alignable** between mlogit and torch-choice. If the
   reference fits a mixed/random-coefficients logit, fall back to a vanilla MNL
   for v1 and document the difference. Nested-logit models are in scope; see
   the dedicated subsection in step 5.

## Step-by-step procedure

### 1. Create the per-dataset folder

```bash
mkdir -p tutorials/<name>/
cp tutorials/public_datasets/downloads/<name>/<name>.csv tutorials/<name>/
```

### 2. Identify the reference fit

Pin down the **exact** mlogit call you'll replicate. Write it as a code block
in the notebook's §2 markdown so readers can read it without running R.
**Decide all of the following up front** — every later step depends on them:

- **Reference alternative** (`reflevel="..."` in mlogit). torch-choice's item
  index 0 *must* equal this alternative — the formula `(intercept|item)` pins
  `intercept[0]` to zero, which is mlogit's reference. Plan the encoding
  before you write the R script *or* the Python loader; they have to agree.
- **Wide vs long format.** mlogit can ingest both via
  `dfidx(..., shape="wide", varying=..., sep=".")` or `shape="long"`.
- **Formula type.** Alt-specific covariates use mlogit's `X | Z` syntax;
  chooser-specific covariates use `X || Y`.
- **Nested vs flat.** If the reference fits a nested logit
  (`mlogit(..., nests=list(...))`), see the nested-logit subsection in
  step 5; you'll need `NestedLogitModel` and `JointDataset` instead of
  the plain `ConditionalLogitModel` path.

### 3. Write `tutorials/<name>/fit_mlogit.R`

Contract the script must satisfy:
- Reads the CSV path from `commandArgs(trailingOnly=TRUE)[1]`.
- Fits the model with `mlogit::mlogit(...)`.
- Emits a single JSON object on **stdout** (mlogit's diagnostics go to stderr).
- No extra dependencies beyond `mlogit` (do not pull in `jsonlite` — write JSON
  by hand the way `tutorials/yogurt/fit_mlogit.R` does).

JSON schema (every dataset must follow this exact shape so the helper works
unchanged):

```json
{
  "log_likelihood": -2656.89,
  "n_obs": 2412,
  "n_alts": 4,
  "coefficients": [
    {"name": "price", "estimate": -0.3666, "std_err": 0.0244, "z_value": -15.04, "p_value": 0.0},
    ...
  ]
}
```

Use `summary(model)$CoefTable` to harvest (Estimate, Std. Error, z-value,
Pr(>|z|)) per coefficient.

### 4. Generate the cached JSON

```bash
Rscript tutorials/<name>/fit_mlogit.R tutorials/<name>/<name>.csv \
    > tutorials/<name>/mlogit_output.json
```

Validate it parses: `python -m json.tool < tutorials/<name>/mlogit_output.json`.

**Sanity-check coefficients against the reference** before going further. If
key coefficients don't match the published numbers (e.g., the well-known
mlogit-Yogurt `price ≈ -0.36`, `feat ≈ +0.49`), stop and debug the spec
*before* writing the torch-choice side. Saves an hour of debugging the wrong
side later.

### 5. Translate the spec into a torch-choice formula

| mlogit syntax | torch-choice formula term | Notes |
|---|---|---|
| Alt-specific intercepts | `(intercept\|item)` | Item 0 is the reference (pinned to 0). |
| Shared coefficient on alt-specific X | `(itemsession_X\|constant)` | One scalar coefficient. |
| Alt-specific coefficient on chooser-specific Z | `(session_Z\|item)` | One coefficient per non-reference item. |
| Alt-specific coefficient on alt-specific X | `(itemsession_X\|item-full)` | One per item (no reference pinned). |
| Random / user-specific coefficient on X | `(X\|user)` | Requires `user_index`. |
| Nested logit (`nests=list(g1=c(...), g2=c(...))`) | `NestedLogitModel` + `JointDataset(item=..., nest=...)` | Different model class; see subsection below. |

#### If the model is nested logit

The flat MNL pipeline (one `ChoiceDataset` + `ConditionalLogitModel`) does
not apply. Instead:

1. Build **two** `ChoiceDataset`s — one for the item-level choice and one for
   the nest-level choice — and bundle them via
   `torch_choice.data.JointDataset(item=ds_item, nest=ds_nest)`.
2. Use `torch_choice.model.NestedLogitModel(...)` with separate `item_formula`
   and `nest_formula` arguments. See
   `tutorials/nested_logit_model_house_cooling.ipynb` for the canonical usage.
3. mlogit's nested-logit output adds a `iv` (or `iv:<group>` for non-shared
   nest elasticities) row per nest — the inclusive-value / lambda parameter.
   Map these in `NAME_MAP` to torch-choice's `lambda_weight_<n>` entries.
4. **Lambda parameterization (verified 2026-04 via `torch_choice/model/nested_logit_model.py`)**:
   `NestedLogitModel` exposes the lambda directly via `self.lambdas =
   self.lambda_weight`. There is NO sigmoid transform in the current source.
   So `lambda_weight_<n>` and mlogit's `iv` value are on the same raw scale
   and can be diffed directly — no transform needed before comparison.
5. **SE tolerance is wider for nested logit.** torch-choice computes
   coefficient standard errors from `sqrt(diag(H^{-1}))` (observed
   information), while mlogit's nested-logit fits use BHHH / outer-product-
   of-gradients. They routinely differ by a few percent on nested fits.
   Relax `compare_coefs(..., se_abs_tol=1e-1)` (≈ 10% absolute) and document
   the relaxation in the notebook. Estimate tolerance can stay at 1e-2.
6. See `replication/paper_demo.py`'s nested-logit section for the intended
   training recipe (Adam, 5000 epochs).

### 6. Build `ChoiceDataset`

For wide-format CSVs, reshape to long with a per-brand `pd.concat` and use
`torch_choice.data.utils.pivot3d` to build the `(num_sessions, num_items, F)`
tensors. Mirror the pattern in
`torch_choice/data/example_datasets.py::load_mode_canada_dataset`:

```python
itemsession_price = utils.pivot3d(long_df, dim0="occasion", dim1="brand", values="price")
item_index = torch.LongTensor(df["choice"].map(brand_to_idx).values)
dataset = ChoiceDataset(
    item_index=item_index, num_items=N,
    user_index=..., session_index=...,
    itemsession_price=itemsession_price,
    ...
)
```

**Brand encoding rule**: `brand_to_idx[reference_brand] == 0`. If you encode
otherwise, the alt-specific intercepts will line up with the wrong mlogit
labels and the comparison will fail with order-of-magnitude differences.

### 7. Fit the model

Recipe (same as `replication/paper_demo.py`'s CLM):

```python
model = ConditionalLogitModel(formula="...", dataset=dataset, num_items=N)
result = model.fit(
    dataset, batch_size=-1, learning_rate=0.01, num_epochs=1000,
    model_optimizer="LBFGS", backend="lightning", print_summary=False,
)
```

### 8. Build the `NAME_MAP`

Every coefficient name in `result.coef_summary.index` must appear as a key.
The helper raises `KeyError` if any key is missing — this is intentional, no
silent skips allowed.

```python
NAME_MAP = {
    "itemsession_price[constant]_0": "price",
    "intercept[item]_0":             "(Intercept):dannon",   # 1st non-reference item
    "intercept[item]_1":             "(Intercept):hiland",   # 2nd
    "intercept[item]_2":             "(Intercept):weight",   # 3rd
    ...
}
```

The `_<n>` suffix in `intercept[item]_<n>` is the **position in the saved
coefficient tensor**, not the item index. Item 0 is pinned to zero so it
isn't stored; `intercept[item]_0` is therefore the intercept for *item 1*
(the first non-reference brand in your `BRAND_ORDER`).

### 9. Run `compare_coefs`

```python
diff = compare_coefs(mlogit_df, result, NAME_MAP)
```

Reports two PASS/FAIL lines: estimates and standard errors. Both use a 1e-3
absolute tolerance by default.

## Verification criteria (must hold before marking the tutorial done)

1. Coefficient **estimates** match within 1e-3 absolute. Typical observed:
   1e-5 to 1e-4.
2. Coefficient **standard errors** match within 1e-3 absolute. Typical
   observed: 1e-6. (SE match is a stronger consistency test than estimates;
   if estimates match but SEs don't, suspect a parameterization mismatch.)
3. Log-likelihood matches to **≥ 4 significant figures**.
4. Notebook runs end-to-end via `jupyter nbconvert --to notebook --execute`
   without exceptions.
5. **R-disabled fallback** works: `PATH="" python -c "from
   _mlogit_compare import run_or_load_mlogit; ..."` returns the cached JSON
   with the same numbers.
6. Notebook ships **with cell outputs** (matches the convention of the other
   `tutorials/*.ipynb`).
7. `lightning_logs/` side-effect cleaned up (it's globally gitignored, but
   leave the working tree clean).

## Common pitfalls

- **Reference category mismatch.** mlogit's `reflevel="X"` must equal
  torch-choice's item index 0. If they disagree, all the alt-specific
  intercepts shift by the same constant and the comparison fails by huge
  margins — *not* by a small constant per coefficient.
- **`pivot3d` ordering surprise.** It sorts `dim1` ascending, so brand index
  0 is at axis position 0, brand index 1 at position 1, etc. If your
  `brand_to_idx` doesn't match the encoding you used in the R `dfidx` call,
  the per-brand prices/features will get attached to the wrong alternatives.
- **`int` vs `float` ids.** Some R-extracted CSVs store household ids as
  `1.0, 2.0, ...`. Cast to `int` before subtracting 1 for `user_index`.
- **Choice column dtype.** mlogit can take a logical (`TRUE`/`FALSE`) choice
  indicator OR a factor naming the chosen alternative. Match the script's
  `dfidx`/`mlogit.data` call to whatever the source CSV provides.
- **Wide-to-long reshape errors.** Build `long_df` deliberately (per-brand
  slice + concat + sort by `(occasion, brand)`); don't trust
  `pd.wide_to_long` blindly with mlogit-style `price.brand` separators.
- **NA / missing values.** mlogit silently drops rows with NA in any used
  variable. torch-choice will either crash on NaN tensors or silently train
  on garbage. Before fitting, run `df.isna().sum()` and either drop NA-rows
  *before* the `pivot3d` reshape (matching mlogit's behavior) or document
  the divergence. If you drop rows, also drop them from the R-side input so
  both fits use the same N.
- **Notebook cwd.** When `jupyter nbconvert --execute` runs the notebook, cwd
  is the notebook's directory. The `import _mlogit_compare` line uses
  `sys.path.insert(0, str(Path.cwd().parent))` — that requires running with
  cwd = `tutorials/<name>/`. Don't break this when refactoring.
- **Stale cache.** If you change the R script's spec, regenerate
  `mlogit_output.json` *and* re-run the notebook to update its cells. A stale
  cache + fresh torch-choice fit = silent FAIL.

## Determinism / re-run noise

torch-choice initializes parameters randomly and LBFGS uses finite-precision
line search, so identical fits typically diverge by **1e-5 to 1e-4** absolute
across runs even on the same machine. That's well below the 1e-3 PASS
tolerance, but it does mean the cached `mlogit_output.json` and the notebook's
shipped cell outputs won't be byte-identical the next time someone executes
the notebook.

If you want byte-identical reruns, set `torch.manual_seed(0)` at the top of
the notebook *before* constructing the model. Don't seed mlogit — it's a
deterministic optimizer, no seed needed.

## Speed budget

For Yogurt-shape MNL workloads (~2,400 occasions, 4 alternatives, ~5 free
parameters) the full pipeline (`Rscript` + 1000-epoch LBFGS + comparison)
finishes in tens of seconds on CPU. Larger datasets are not benchmarked here
— time the first end-to-end run before extrapolating. If your LBFGS run
takes more than a couple of minutes, suspect non-convergence (inspect the
loss curve in TensorBoard / `result.train_ll`'s trajectory) rather than just
raising `num_epochs`.

## What's *not* in scope here

- **Mixed / random-coefficients logit replication.** mlogit supports it via
  `mlogit(..., rpar=...)`, and torch-choice can express it via user-specific
  coefficients, but the verification approach is different (compare
  parameter distributions, not point estimates) and warrants a separate
  skill.

- **Stata-based reference fits.** Same general pattern, different runner.
  Should live as a sibling skill at
  `.claude/skills/stata-comparison/SKILL.md`. Concrete pointers for that
  sibling, since the differences are non-trivial:
  - The runner is `stata -b do <script>.do` (or `pystata.run` if available),
    not `Rscript`. Most Stata installs are licensed; expect users without
    Stata, so the cache fallback is even more important than for R.
  - Stata has no native JSON. Either `ssc install jsonio` (third-party) or
    have the do-file write tab-separated `<name>\t<est>\t<se>\t<z>\t<p>`
    lines and parse them in the Python helper.
  - Coefficient names use Stata's factor-variable convention (`1.brand`,
    `2.brand`, `_cons`) rather than mlogit's `(Intercept):dannon` style;
    `NAME_MAP`s for Stata datasets will look noticeably different.
  - Choice command: prefer `asclogit` (or `cmclogit` on Stata 17+) over
    barebones `clogit`, since they handle alt-specific covariates more
    cleanly — closer to mlogit's syntax.
  - `_compare.py`'s `compare_coefs` is reusable as-is; only the
    `run_or_load_*` runner and the per-dataset reference script change.

- **Python-side reference fits (`pylogit`, `xlogit`, `choice-learn`).** No
  subprocess layer needed — fit in-process and call `compare_coefs`
  directly. Same alignment principles; mostly a packaging detail.
