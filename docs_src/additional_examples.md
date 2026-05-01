# Additional Examples

This page indexes a set of additional tutorial notebooks that fit a
discrete-choice model on a public dataset twice — once in R using
`mlogit`, once in Python using `torch-choice` — and verify the two
implementations recover the same estimates, standard errors, and
log-likelihood.

Each tutorial lives in its own folder under
[`tutorials/<dataset>/`](https://github.com/gsbDBI/torch-choice/tree/main/tutorials)
and contains:

- a short `README.md`,
- the dataset CSV,
- a `fit_mlogit.R` script (called via `Rscript`),
- a cached `mlogit_output.json` (so readers without R can still see the numbers),
- the executed notebook with cell outputs.

## Verification summary across 8 datasets

For each dataset we report:
- **Log-likelihood (LL)** match between the two packages
- **Coefficient estimate** disagreement, expressed as both absolute and
  percent of the larger of `|mlogit_est|` and `|tc_est|`
- **Standard error** disagreement, same units

| Dataset | mlogit LL | tc LL | LL diff | Est max abs | Est max % | SE max abs | SE max % |
|---|---:|---:|---:|---:|---:|---:|---:|
| [yogurt](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/yogurt/yogurt.ipynb) | -2656.8879 | -2656.8879 | 6.2e-5 | 3.86e-5 | 0.0022% | 1.78e-6 | 0.0009% |
| [electricity](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/electricity/electricity.ipynb) | -4958.6491 | -4958.6494 | 2.9e-4 | 1.86e-4 | 0.0037% | 1.26e-6 | 0.0008% |
| [nox](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/nox/nox.ipynb) | -1064.8147 | -1064.8147 | 2.2e-5 | 5.95e-6 | 0.0007% | 2.18e-7 | 0.0002% |
| [risky_transport](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/risky_transport/risky_transport.ipynb) | -1722.3061 | -1722.3060 | 9.9e-5 | 3.65e-6 | 0.0009% | 4.68e-8 | 0.0001% |
| [brownstone_train_car](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/brownstone_train_car/brownstone_train_car.ipynb) | -7080.8290 | -7080.8291 | 1.0e-4 | 1.13e-5 | 0.0071% | 2.03e-7 | 0.0003% |
| [travel_mode](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/travel_mode/travel_mode.ipynb) | -172.4680 | -172.4680 | **4.1e-11** | 6.84e-6 | 0.0013% | 1.42e-6 | 0.0001% |
| [japanese_fdi](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/japanese_fdi/japanese_fdi.ipynb) (nested) | -1726.9810 | -1726.9810 | 3.5e-5 | 2.67e-4 | 0.0315% | 8.75e-2 | 9.43% |
| [swiss_route_choice](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/swiss_route_choice/swiss_route_choice.ipynb) | -1665.6885 | -1665.6885 | 2.0e-5 | 7.58e-6 | 0.0008% | 1.65e-7 | 0.0004% |

**One-line takeaway.** On flat MNL / conditional-logit fits, `torch-choice`
agrees with `mlogit` at the float-precision noise floor (~1e-5 absolute,
~1e-3 % relative) for both estimates and standard errors. On nested logit
(`japanese_fdi`), the estimates still match at this floor, but the standard
errors differ by ~9 % because `torch-choice` computes SEs from the
inverse observed information `sqrt(diag(H⁻¹))` while `mlogit` uses BHHH
(outer-product-of-gradients). The two are first-order asymptotically
equivalent but differ measurably in finite samples on nested-logit
inclusive-value parameters.

## Per-dataset overview

| Dataset | Domain | Reference | Notable feature |
|---|---|---|---|
| yogurt | CPG / brand choice (panel of 100 households × ~24 occasions) | Jain–Vilcassim–Chintagunta, JBES 1994 | Wide-format brand prices and feature dummies; classic mixed-logit motivating example |
| electricity | Energy retail / SP supplier choice (panel) | Train, *Discrete Choice Methods* Ch. 6 | Wide-per-row SP design with 4 hypothetical suppliers per task |
| nox | Environmental policy / pollution-control technology | Fowlie, AER 2010 | Explicit `available` mask; regime × cost interactions reproduce Fowlie's identification |
| risky_transport | Development econ / transport mode + risk | Léon & Miguel, AEJ-Applied 2017 | Availability variation; framed for value-of-statistical-life estimation |
| brownstone_train_car | Vehicle adoption SP (6 alternatives × 11 attributes) | Brownstone & Train, JoE 1999 | Wide-format reshape with 70 columns; high-dimensional attribute space |
| travel_mode | Inter-city transportation (Greene textbook) | Greene, *Econometric Analysis* Ch. 19 | Tiny (210 individuals × 4 modes); float64 + LBFGS strong-Wolfe needed for stable fit |
| japanese_fdi | Firm location / FDI (Europe) | Head & Mayer, RES 2004 | **Nested logit**: 50 NUTS1 regions within 9 country nests |
| swiss_route_choice | Inter-urban rail SP route choice | Axhausen et al., Transp Policy 2008 | Apollo workhorse for VTTS; binary route choice, panel SP |

## Beyond the cross-package check

Each tutorial also includes a **§5 Alternative specifications** section
that fits 2–3 plausible specifications (e.g., pooled MNL, fixed-effects
heterogeneity, nested logit), compares them via AIC and BIC, and
recommends one as canonical based on what the original paper or discipline
norms argue. The recommendations cite the primary literature directly.

## Reproducing the comparisons yourself

The tutorials assume `mlogit` is installed in R:

```r
install.packages("mlogit")
```

The Python side relies on `torch-choice` plus the shared helper module
[`tutorials/_mlogit_compare.py`](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/_mlogit_compare.py),
which transparently calls `Rscript` and falls back to the cached JSON if
R is unavailable. So the comparison cells render even on machines without
R installed — the live R fit is opt-in.

The methodology is codified as a reusable runbook in
[`mlogit-comparison`](https://github.com/gsbDBI/torch-choice/blob/main/.claude/skills/mlogit-comparison/SKILL.md)
under `.claude/skills/`, including verification criteria, common pitfalls
(reference category alignment, NA handling, wide-to-long reshape), and
a Stata-comparison extension breadcrumb for future tutorials.
