# Cracker brand choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of household cracker
brand choice on the Jain–Vilcassim–Chintagunta panel-scanner data (JBES,
1994) and verifies that `torch-choice` reproduces R/`mlogit`'s coefficients
and log-likelihood. This is the **brand-loyalty / state-dependence companion
to the Yogurt tutorial** — same source paper, different CPG category, same
wide 4-brand panel structure.

## Files

| File | What it is |
|---|---|
| `cracker.ipynb` | The 5-section tutorial notebook (intro → R fit → torch-choice fit → comparison → alternative specs). |
| `cracker.csv` | Wide-format data, 3,292 occasions × 4 brands × 136 households. Extracted from `mlogit::Cracker` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers. Sister datasets in
`tutorials/yogurt/` and other `tutorials/<dataset>/` folders follow the same
shape.

## Reference

Jain, D. C., Vilcassim, N. J., Chintagunta, P. K. (1994). "A Random-Coefficients
Logit Brand-Choice Model Applied to Panel Data." *Journal of Business &
Economic Statistics* 12(3), 317–328.
[doi:10.1080/07350015.1994.10524547](https://doi.org/10.1080/07350015.1994.10524547)

The Cracker dataset is the brand-loyalty / state-dependence companion to
Yogurt: both come out of the same JBES 1994 paper and serve as canonical
testbeds for random-coefficients / mixed-logit models in the marketing
literature.

## License

GPL-2 (inherited from the `mlogit` R package, which redistributes the data).
