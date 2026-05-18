# Yogurt brand choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of household yogurt brand
choice on the Jain–Vilcassim–Chintagunta panel (JBES, 1994) and verifies that
`torch-choice` reproduces R/`mlogit`'s coefficients and log-likelihood.

## Files

| File | What it is |
|---|---|
| `yogurt.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `yogurt.csv` | Wide-format data, 2,412 occasions × 4 brands × 100 households. Extracted from `Ecdat::Yogurt` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers. Future
`tutorials/<dataset>/` folders follow the same shape.

## Reference

Jain, D. C., Vilcassim, N. J., Chintagunta, P. K. (1994). "A Random-Coefficients
Logit Brand-Choice Model Applied to Panel Data." *Journal of Business &
Economic Statistics* 12(3), 317–328.
[doi:10.1080/07350015.1994.10524547](https://doi.org/10.1080/07350015.1994.10524547)
