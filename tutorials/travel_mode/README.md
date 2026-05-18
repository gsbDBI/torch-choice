# Inter-city travel mode choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of inter-city
transportation mode choice on the Australian TravelMode dataset (Greene's
*Econometric Analysis*, Ch. 19) and verifies that `torch-choice` reproduces
R/`mlogit`'s coefficients and log-likelihood.

## Files

| File | What it is |
|---|---|
| `travel_mode.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `travel_mode.csv` | Long-format data, 210 individuals × 4 modes = 840 rows. Mirror of `AER::TravelMode` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Reference

Greene, W. H. (2012). *Econometric Analysis* (7th ed.), Ch. 19. Pearson.
The dataset is the canonical example distributed in the R package `AER` as
`AER::TravelMode` (also in `mlogit` as `mlogit::ModeChoice`); see
`?AER::TravelMode`.

## License

Distributed in `AER` under GPL-2/GPL-3, redistributable with citation.
