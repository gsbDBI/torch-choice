# NOx pollution-control technology choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of power-plant
pollution-control technology adoption (Fowlie 2010, *AER*) and verifies that
`torch-choice` reproduces R/`mlogit`'s coefficients and log-likelihood.

The data are *unbalanced*: every plant faces 15 candidate technologies on
paper, but only a subset of them is *available* (engineering / regulatory
feasibility) to a given unit. We use the `available` column directly as
torch-choice's `item_availability` mask, which is mathematically equivalent
to mlogit's `dfidx(..., subset = available == 1)` filtering.

## Files

| File | What it is |
|---|---|
| `nox.ipynb` | The 4-section tutorial notebook (intro -> R fit -> torch-choice fit -> comparison). |
| `nox.csv` | Long-format data, 632 plants x 15 technologies = 9,480 rows. Extracted from `mlogit::NOx` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Reference

Fowlie, M. (2010). "Emissions Trading, Electricity Restructuring, and Investment
in Pollution Abatement." *American Economic Review* 100(3), 837-869.
[doi:10.1257/aer.100.3.837](https://doi.org/10.1257/aer.100.3.837)

The dataset is distributed in the R package `mlogit` under GPL-2,
redistributable with citation. Source: AEA data archive
(<https://www.aeaweb.org/aer/>).

## Note on model spec

This tutorial fits the simpler MNL specification used in mlogit's `NOx`
example: utility depends on variable cost (`vcost`), capital cost (`kcost`),
and the combustion-modification dummy (`cm`), with no alternative-specific
intercepts (`| 0`). The original Fowlie (2010) paper uses a richer
specification with regulatory-regime interactions; this is the canonical
"verify the math" benchmark, not a full replication of Table 4.
