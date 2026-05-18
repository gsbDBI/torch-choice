# Brownstone-Train Car choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model on the Brownstone &
Train (1999) stated-preference vehicle-choice study and verifies that
`torch-choice` reproduces R/`mlogit`'s coefficients and log-likelihood. Each
of 4,654 respondents rated 6 hypothetical vehicle profiles described by 11
attributes (body type, fuel, price, range, acceleration, top speed, tailpipe
pollution, size, luggage space, per-mile cost, station availability).

> **Scope.** v1 fits a simplified MNL with the **9 numeric attributes** only
> (`price + range + acc + speed + pollution + size + space + cost + station`)
> and alt-specific intercepts. The categorical `type` and `fuel` attributes
> and the published mixed-logit specification (random coefficients on a
> subset of attributes, plus interactions with chooser-specific covariates
> `college`, `hsg2`, `coml5`) are out of scope here — that is a v2 follow-up
> better suited to a separate skill (random-coefficients comparison
> validates parameter distributions, not point estimates).

## Files

| File | What it is |
|---|---|
| `brownstone_train_car.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `car.csv` | Wide-format data, 4,654 occasions × 6 alternatives × 11 attrs. Extracted from `mlogit::Car` (GPL-2). |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Reference

Brownstone, D., Train, K. (1999). "Forecasting new product penetration with
flexible substitution patterns." *Journal of Econometrics* 89(1-2), 109–129.
[doi:10.1016/S0304-4076(98)00057-8](https://doi.org/10.1016/S0304-4076(98)00057-8)

McFadden, D., Train, K. (2000). "Mixed MNL models for discrete response."
*Journal of Applied Econometrics* 15(5), 447–470.

**License.** Distributed in the R package `mlogit` under GPL-2,
redistributable with citation. Source: Journal of Applied Econometrics data
archive.
