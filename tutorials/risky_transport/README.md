# Risky Transport — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of transport-mode
choice between Freetown city and Lungi airport (Léon & Miguel,
*AEJ-Applied* 2017) and verifies that `torch-choice` reproduces R/`mlogit`'s
coefficients and log-likelihood.

## Files

| File | What it is |
|---|---|
| `risky_transport.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `risky_transport.csv` | Long-format data, 5,405 rows = 1,793 chooser-occasions × {2, 3, or 4} available modes. Extracted from `mlogit::RiskyTransport` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Reference

León, G., Miguel, E. (2017). "Risky Transportation Choices and the Value of a
Statistical Life." *American Economic Journal: Applied Economics* 9(1),
202–228. [doi:10.1257/app.20160140](https://doi.org/10.1257/app.20160140)

## Substantive angle

Travelers between Freetown and the international airport across an estuary
choose among up to four modes — `WaterTaxi`, `Ferry`, `Hovercraft`,
`Helicopter` — that vary substantially in fatality rate (`risk`, deaths per
100,000 trips) and price (`cost`). Léon and Miguel use this discrete choice
to estimate a Value of a Statistical Life from revealed risk-money trade-offs;
the MNL fit here reproduces their core risk-and-cost coefficients (without the
extended VSL-by-covariate decomposition).

## License

Distributed in the R package `mlogit` under GPL-2, redistributable with
citation.
