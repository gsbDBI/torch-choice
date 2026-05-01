# Residential electricity supplier choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a pooled multinomial logit model of residential
electricity-supplier choice on Kenneth Train's stated-preference panel
(361 U.S. households × ~12 hypothetical choice occasions each = 4,308
observations, each with 4 supplier alternatives) and verifies that
`torch-choice` reproduces R/`mlogit`'s coefficients and log-likelihood.

## Files

| File | What it is |
|---|---|
| `electricity.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `electricity.csv` | Wide-per-occasion data, 4,308 rows × 26 cols. Extracted from `mlogit::Electricity` (GPL-2). |
| `fit_mlogit.R` | R script that fits the pooled MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Caveat: pooled MNL vs mixed logit

Train's textbook treatment of this dataset (Train 2009, *Discrete Choice
Methods with Simulation*, ch. 6) fits a **mixed logit** with random
coefficients on every attribute, exploiting the panel structure (multiple
occasions per household). For v1 of this tutorial we replicate the simpler
**pooled MNL** (`mlogit(... | 0)` from the `?Electricity` help), which
treats the 4,308 occasions as independent. This matches the stripped-down
MNL spec used to introduce the dataset before random-coefficients
estimation. A mixed-logit replication is a v2 follow-up and would require a
different verification strategy (parameter distributions, not point
estimates).

## Reference

Revelt, D., Train, K. (2001). "Customer-Specific Taste Parameters and Mixed
Logit: Households' Choice of Electricity Supplier." *Econometrics* 0012001,
University Library of Munich.
[ideas.repec.org/p/wpa/wuwpem/0012001](https://ideas.repec.org/p/wpa/wuwpem/0012001.html)

Huber, J., Train, K. (2000). "On the Similarity of Classical and Bayesian
Estimates of Individual Mean Partworths." *Marketing Letters* 12, 259–269.

Train, K. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
Cambridge University Press.

## License

The CSV is distributed in the R package `mlogit` under GPL-2, redistributable
with citation.
