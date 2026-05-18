# Swiss Route Choice — `torch-choice` ↔ R/`mlogit`

Tutorial notebook that fits a multinomial logit model of inter-urban rail
route choice on the Apollo Swiss Route Choice stated-preference panel
(Axhausen et al., 2008) and verifies that `torch-choice` reproduces
R/`mlogit`'s coefficients and log-likelihood.

## Files

| File | What it is |
|---|---|
| `swiss_route_choice.ipynb` | The 5-section tutorial notebook (intro → R fit → torch-choice fit → comparison → alternative specs + recommendation). |
| `apollo_swissRouteChoiceData.csv` | Wide-format data, 3,492 SP tasks × 2 routes × 388 individuals (~9 SC tasks each). Distributed with the Apollo R package under GPL-2. |
| `fit_mlogit.R` | R script that fits MNL with `mlogit` and emits coefficients + LL as JSON to stdout. |
| `mlogit_output.json` | Cached R fit, used as a fallback when R is unavailable so the notebook still renders. |

The notebook imports `tutorials/_mlogit_compare.py` (one level up) for the
shared `run_or_load_mlogit` and `compare_coefs` helpers. Future
`tutorials/<dataset>/` folders follow the same shape.

## Dataset

A panel of 388 Swiss travellers, each completing 9 stated-choice tasks where
they pick one of two hypothetical inter-urban rail routes. Each route is
described by four attributes:

| Attribute | Meaning | Units |
|---|---|---|
| `tt` | Travel time | minutes |
| `tc` | Travel cost | CHF |
| `hw` | Headway (time between trains) | minutes |
| `ch` | Number of interchanges | count |

The CSV also carries chooser-level covariates (household income, car
availability, four trip-purpose indicators) that this v1 tutorial does not
use; the canonical MNL spec uses only the four route attributes.

## References

- Axhausen, K. W., Hess, S., König, A., Abay, G., Bates, J. J., Bierlaire, M.
  (2008). "Income and distance elasticities of values of travel time savings:
  New Swiss results." *Transport Policy* 15(3), 173–185.
  [doi:10.1016/j.tranpol.2008.02.001](https://doi.org/10.1016/j.tranpol.2008.02.001)

- Hess, S., Palma, D. (2019). "Apollo: A flexible, powerful and customisable
  freeware package for choice model estimation and application." *Journal of
  Choice Modelling* 32, 100170.
  [doi:10.1016/j.jocm.2019.100170](https://doi.org/10.1016/j.jocm.2019.100170)

## License

The data is redistributed with the Apollo R package under the GPL-2 license.
The CSV in this folder is a verbatim copy of
`http://www.apollochoicemodelling.com/files/examples/data/apollo_swissRouteChoiceData.csv`.
