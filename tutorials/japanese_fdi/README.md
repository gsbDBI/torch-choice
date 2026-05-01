# Japanese FDI in Europe — `torch-choice` ↔ R/`mlogit` (nested logit)

Tutorial notebook that fits a **nested logit** model of Japanese firm
location choice in European NUTS1 regions and verifies that
`torch-choice` reproduces the R/`mlogit` reference fit. Choosers (Japanese
firms) first pick a country, then a NUTS1 region inside that country, so the
specification has one nest per country (BE, DE, ES, FR, IE, IT, NL, PT, UK).
This is the first tutorial in `tutorials/<dataset>/` to use a non-flat
specification — it relies on `JointDataset` and `NestedLogitModel` rather
than `ChoiceDataset` and `ConditionalLogitModel`.

## Files

| File | What it is |
|---|---|
| `japanese_fdi.ipynb` | The 4-section tutorial notebook (intro → R fit → torch-choice fit → comparison). |
| `japanese_fdi.csv` | Long-format data, 25,764 rows = 452 firms × 57 regions (9 countries). Extracted from `mlogit::JapaneseFDI` (GPL-2). |
| `fit_mlogit.R` | R script that fits the nested logit with `mlogit` and emits coefficients (incl. `iv`) + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers.

## Specification (v1)

This tutorial fits the 6-covariate spec from the `mlogit` vignette
`c4.relaxiid` (a simplified subset of Head & Mayer 2004's full specification,
which adds market-potential and industry/network covariates):

```
choice ~ log(wage) + unemp + elig + log(area) + scrate + ctaxrate | 0
nests = TRUE          # auto-detected from idx=list("firm", c("region","country"))
un.nest.el = TRUE     # single shared inclusive-value coefficient `iv`
```

In `torch-choice` this becomes a `NestedLogitModel` with
`item_formula="(itemsession_x|constant)"`, `nest_formula=""`, and
`shared_lambda=True`, where `itemsession_x` packs the six covariates into a
`(num_firms, num_regions, 6)` tensor.

A "v2" tutorial that adds `harris`, `krugman`, `domind`, `japind`, and
`network` would be a straightforward extension — bump `K` from 6 to 11 and
update the `NAME_MAP`.

## Reference

Head, K., Mayer, T. (2004). "Market Potential and the Location of Japanese
Investment in the European Union." *Review of Economics and Statistics*
86(4), 959–972.
[doi:10.1162/0034653043125257](https://doi.org/10.1162/0034653043125257)

## License

The dataset is redistributed in this folder under the terms of GPL-2, the
license of its source R package `mlogit`. Please cite Head & Mayer (2004)
when using this data.
