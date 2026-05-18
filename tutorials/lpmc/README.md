# London Passenger Mode Choice (LPMC) — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of urban trip mode choice
on the London Passenger Mode Choice dataset (Hillel et al., 2018) and verifies
that `torch-choice` reproduces R/`mlogit`'s coefficients and standard errors.
This is the largest dataset in the `tutorials/` mlogit-comparison suite —
**81,086 trips × 4 modes** — and so doubles as a small "real workload" stress
test for both packages.

## Files

| File | What it is |
|---|---|
| `lpmc.ipynb` | The 5-section tutorial notebook (intro → R fit → torch-choice fit → comparison → alternative specs). |
| `load_lpmc.py` | Fetches the raw `lpmc.dat` from the canonical EPFL course-asset URL on first use and caches it locally as `lpmc.csv`. The dataset is gitignored — we do not redistribute it. |
| `fit_mlogit.R` | R script that fits the canonical Bierlaire-MOOC MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers, the same shared
infrastructure that backs `tutorials/yogurt/`.

## Reference

- Hillel, T., Elshafie, M. Z. E. B., Jin, Y. (2018). "Recreating passenger
  mode choice-sets for transport simulation: A case study of London, UK."
  *Proceedings of the Institution of Civil Engineers - Smart Infrastructure
  and Construction* 171(1), 29–42.
  [doi:10.1680/jsmic.17.00018](https://doi.org/10.1680/jsmic.17.00018)
- Hillel, T. (2019). "London Passenger Mode Choice (LPMC)."
  Technical report. EPFL.
  [transp-or.epfl.ch/documents/technicalReports/CS_LPMC.pdf](https://transp-or.epfl.ch/documents/technicalReports/CS_LPMC.pdf)
- The MNL specification fit in §2–§4 follows Bierlaire's MOOC reference notebook
  [`mooc-discrete-choice/LPMC_DCM_ML.ipynb`](https://github.com/michelbierlaire/mooc-discrete-choice/blob/master/LPMC_DCM_ML.ipynb).

## License & redistribution

The Hillel et al. (2018) paper is published under a Creative Commons
Attribution license, but the LPMC technical report only states "please cite
this paper in any work using this dataset" and does not explicitly authorize
redistribution of the raw CSV. To avoid that ambiguity, this tutorial does
**not** ship `lpmc.dat`/`lpmc.csv`; instead `load_lpmc.py` fetches it on
demand from the EPFL ChoiceModels MOOC's public course-asset URL (the same
URL used by Bierlaire's reference notebook). Both files are gitignored.

## Dataset shape

- 81,086 trips (one row each — the dataset is single-day travel diary data).
- 31,954 individuals across 17,616 households.
- April 2012 – March 2015 (London Travel Demand Survey).
- 4 modes: walking, cycling, public transport, driving — all four assumed
  available for every trip (the choice-set construction in Hillel et al. 2018
  produces level-of-service variables for all four modes regardless of
  individual car ownership / driver's license / age, so we follow that
  convention; the resulting MNL therefore needs no availability mask).

## Runtime budget

Both fits are CPU-bound and finish in seconds:

| Step | Duration |
|---|---|
| `Rscript fit_mlogit.R lpmc.csv`                  | ≈ 3 s  |
| §3 Spec A — `torch-choice` LBFGS, 500 epochs    | ≈ 18 s |
| §5 Spec B — `torch-choice` LBFGS, 500 epochs    | ≈ 20 s |
| §5 Spec C — nested logit, LBFGS, 1000 epochs    | ≈ 60 s |

Full notebook (including data download on first run) executes end-to-end in
≈ 2 min on a 2024 M-series Mac mini, well under the 5-min "tutorial-friendly"
budget.
