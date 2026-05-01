# Dataset Candidates for new torch-choice tutorials

Working log of public discrete-choice datasets evaluated as candidates for new
torch-choice tutorials/examples. Goal: broaden the package's audience reach by
adding examples beyond the current ones (transport + residential heating).

## Already in repo (do not re-cover)

| Name | Domain | Type | Notes |
|---|---|---|---|
| ModeCanada | Transportation | Conditional logit | `torch_choice/data/example_datasets.py::load_mode_canada_dataset` |
| House Cooling (HC) | Residential heating | Nested logit | `tutorials/public_datasets/HC.csv` |
| car_choice | Vehicle choice | — | `replication/car_choice.csv` (verify whether this is Brownstone–Train (1999); if so, promote to a full regularization tutorial) |

## Three target axes

When evaluating candidates, judge against:

1. **Modeling feature** showcased — availability, nesting, user heterogeneity, outside option, regularization, scale/GPU
2. **Domain breadth** — prefer audiences not yet served (currently only transport + heating)
3. **Comparison-friendly** — published paper with reported coefficients or LL

---

## Candidate pool (all scans consolidated, 30+ candidates)

### A. Classical econometric / discrete-choice corpus
*(Source: scans of mlogit, Apollo, Biogeme, Train's archive)*

| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| A1 | **Yogurt** (Jain–Vilcassim–Chintagunta) | `logitr`/`Ecdat` (R) | CPG / brand choice | MIT/GPL-2 | 2,412 occ × 4 brands × 100 HHs | Panel / user heterogeneity | JBES 1994 12(3):317 |
| A2 | **Electricity** (Train residential supplier) | `mlogit::Electricity` | Energy retail | GPL-2 | 2,308 HHs SP panel × 4 alts | Panel / mixed logit | Train (2009) Ch.6 |
| A3 | **LPMC** (London Passenger Mode Choice) | EPFL / Hillel | Urban transport | Academic-use w/ citation | 81k trips × 4 modes | Scale + availability mask (★ paper's GPU pitch) | Hillel et al. 2018, 2021 |
| A4 | **Swissmetro** (Bierlaire) | EPFL/Biogeme | Intercity transport | Open | 10.7k SP × 3 alts | Availability + canonical Biogeme reference | Bierlaire et al. 2001 |
| A5 | **JapaneseFDI** (Head–Mayer) | `mlogit::JapaneseFDI` | Firm location / FDI | GPL-2 | 452 firms × ~50 regions in 9 countries | 2-level nest (region within country) — non-residential | RES 2004 86(4):959 |
| A6 | **Apollo Drug Choice** | `apollo` (R) | Healthcare / pharma SP | GPL-2 | 10k tasks × 1k indiv × 4 alts | Panel + ranking + healthcare | Hess & Palma JoCM 2019 |
| A7 | **Brownstone–Train Car** | `mlogit::Car` / JAE | Sustainable vehicles | Public | 4,654 × 6 SP vehicles | High-dim attrs → L1/L2 demo | JoE 1999 89:109 |
| A8 | **RiskyTransport** (Léon–Miguel) | `mlogit::RiskyTransport` / AEA | Dev econ / risk | Public | 1,793 × 4, with availability variation | RP availability + VSL | AEJ-Applied 2017 9(1):202 |
| A9 | **NOx** (Fowlie) | `mlogit::NOx` / AER | Environmental econ | GPL-2 | 632 plants | Availability varies by reg regime | AER 2010 100(3):837 |
| A10 | **TravelMode** (Greene) | `AER::TravelMode` / mlogit | Inter-city transport | GPL | 210 × 4 | Tiny; perfect Stata/NLOGIT verification | Greene Ch.19 |

### B. Recsys / large-scale / ML-flavored
*(Source: ML-and-recsys-flavored scan)*

| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| B1 | **Expedia ICDM 2013** | Kaggle | Online travel / hotels | Kaggle ToS (link only) | 10M search rows; 5–38 hotels per slate | Availability + counterfactual policy | ICDM 2013 |
| B2 | **Ta-Feng Grocery** | Kaggle (RecSys release) | Retail scanner | Public domain on Kaggle | 817k tx × 32k cust × 23k SKU | Panel + outside-option recast | RecSysWiki |
| B3 | **Trivago RecSys 2019** | Trivago Challenge | Travel / sessions | Research license | 15.9M actions, 730k sessions | Position + per-session slate | RecSys 2020 |
| B4 | **trivago-clicks** | Cornell ARB | Travel / clicks | Open research | 207k sessions, 174k items, 25 alts/impression | Pre-formatted (user, choice_set, chosen) — near-zero wrangling | Benson et al. WSDM 2018 |
| B5 | **MIND – Microsoft News** | Hugging Face / msnews.github.io | News recsys | MS Research License (research only) | 15M impressions × 1M users × 160k articles | True conditional-logit-ready impression logs (★ ML bridge) | Wu et al. ACL 2020 |
| B6 | **MovieLens-1M** | GroupLens | Recommender | Free for research | ~1M ratings | Recast as choice from sampled slate (NOT native choice) | GroupLens |
| B7 | **H&M Personalized Fashion** | Kaggle | E-commerce / fashion | Kaggle (link only) | 31M tx × 1.37M cust × 106k articles | Scale + regularization | Kaggle 2022 |

### C. Political-science / conjoint
*(Source: Harvard Dataverse scan)*

| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| C1 | **HHY Immigrant Conjoint** | Harvard Dataverse `THJYQR` | Immigration prefs | CC0 | 1,396 resp × paired profiles | Conditional logit ⇒ AMCE; tiny, fast tutorial | Hainmueller-Hopkins-Yamamoto, Pol Analysis 2014 |
| C2 | **Bechtel–Scheve Climate** | Harvard Dataverse `UGZ2BY` | Climate cooperation | CC0 | 8.5k resp × paired tasks across 4 countries | Cross-country mixed logit (★ strongest non-HHY conjoint) | PNAS 2013 110(34):13763 |
| C3 | **Liu Authoritarian China** | Harvard Dataverse `SLZ2OA` | Political selection | CC0 | 300 govt-official resp × paired | Small N, regularized logit | PSRM 2019 |
| C4 | **Singh Argentina 2019** | Harvard Dataverse `65OW82` | Vote choice + RDD | CC0 | thousands of voters × 5+ tasks | Linkable user file → user-coef demo | Comp Pol Studies 2021 |
| C5 | **Miwa Japan Ideological Labels** | Harvard Dataverse `FIHGN0` | Vote choice | CC0 | 4.2 MB conjoint + respondent files | Latent-class mixture model | Political Behavior 2022 |
| C6 | **CSES IMD** | cses.org | Vote choice (cross-country) | Free public | 395k indiv, 230 elections, 59 countries, 800+ parties | Country-varying alternative sets (★ availability across panels) | CSES |
| C7 | **ANES 2020 Time Series** | electionstudies.org | US vote choice | Free public | 8,280 resp × 4–6 candidates + abstention | User-feature × alt-feature interactions; outside option (abstain) | ANES |

### D. Economics replication archives (AEA + JAE + Zenodo)
*(Source: economics replication scan)*

| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| D1 | **Gaynor–Propper–Seiler NHS** ⭐ | openICPSR `112897` | Healthcare / hospital choice | AEA replication | Millions of patients × ~150 hospitals | Scale + consideration sets (★ best replication target — flagged by 2 independent agents) | AER 2016 106(11):3521 |
| D2 | **Fack–Grenet–He Paris** | openICPSR `113103` | Education / school choice | AEA replication | ~1.5–4k students × 20–25 schools, rank-ordered | Rank-ordered logit, school matching | AER 2019 109(4):1486 |
| D3 | **Andersson Swedish School Choice** | JAE / ZBW `10.15456/jae.2025346.1328044327` | Education | JAE open | 45 markets, tens of thousands of students | Replicates D2 at scale | JAE 2026 41(3):323 |
| D4 | **Conlon–Mortimer Vending** | openICPSR `116443` | Retail demand | AEA replication | 54 machines × 30 SKUs × 4-hr periods, >100k arrivals | Per-occasion availability variation (★ canonical avail-mask) | AEJ:Micro 2013 5(4):1 |
| D5 | **Vossler–Doyon–Rondeau Field DCE** | openICPSR `114400` | Stated preference | AEA replication | 600 Quebec subjects × multiple cards | Treatment-effect identification + opt-out | AEJ:Micro 2012 4(4):145 |
| D6 | **Kreimeier EQ-5D-Y Germany** | Zenodo `6953084` | Healthcare DCE | CC-BY-4.0 | 1,030 adults × DCE tasks (~8–12k occasions) | Mixed-logit DCE, health states | PharmacoEconomics 2022 |

### E. Behavioral / fairness
| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| E1 | **Speed-Dating** (Fisman–Iyengar) | OpenML `40536` / Kaggle mirror | Social / partner choice | Public | 8,378 decisions × 552 ppl | Binary CL + fairness audit angle | QJE 2006 121(2):673 |
| E2 | **SUSHI** (Kamishima) | Preflib / OpenML `45734` | Food preference / ranking | Free academic | 5k resp; SUSHI-A 10 items, SUSHI-B 100 items | Ranking → exploded logit; user-coef as taste | Kamishima KDD 2003 |

### F. Round-3 sweep (April 2026; Apollo, AEA, Python packages)

| # | Dataset | Source | Domain | License | Size | Showcases | Reference |
|---|---|---|---|---|---|---|---|
| F1 | **Apollo Swiss Route Choice** | apollochoicemodelling.com | Inter-urban rail SP | GPL-2 | 388 ppl × ~9 SP × 2 rail alts | Panel + Apollo's ~30 spec variants on same data | Axhausen et al. *Transp Policy* 2008 |
| F2 | Apollo Time Use Diary | apollochoicemodelling.com | UK time allocation | GPL-2 | 2,826 days × 12 activities × 447 indiv | MDCEV — *blocked: torch-choice doesn't support this class* | Calastri et al. *Transportation* 2020 |
| F3 | **Cracker** (mlogit) | cran/mlogit | CPG / brand panel | GPL-2 | 3,292 occasions × 4 brands × ~136 HHs | Sister to Yogurt; brand loyalty / state dep. | Jain–Vilcassim–Chintagunta JBES 1994 |
| F4 | **Game** (mlogit) | cran/mlogit | Gaming platforms (ranked) | GPL-2 | 91 resp × 6 platforms (full rankings) | Rank-ordered / exploded logit demo | Fok et al. *J Appl Econ* 2012 |
| F5 | MTC Work Mode (1990 SF Bay) | larch package | Urban transport | GPLv3 (data: public domain) | 5,029 trips × 6 modes | 6-alt MNL with avail mask; SIM canonical | Daly 1987 |
| F6 | **Mas–Pallais Alt Work Arrangements** | openICPSR `113162` | Labor / WTP | AEA standard | ~7k applicants × 2 alt | ⭐ Cleanest published WTP table; labor-econ DCE | Mas & Pallais AER 2017 |
| F7 | **Fosgerau IPDL on cereal** | openICPSR `194501` | RTE cereal demand | AEA standard | ~2k market-product (Nevo data) | 2024 IPDL vs BLP/RCL benchmark | Fosgerau et al. AEJ:Micro 2024 |
| F8 | Abaluck–Gruber Medicare Part D | openICPSR `112428` | Health insurance | AEA standard | ~3k enrollees × dozens of plans | Iconic premium-vs-OOP weight decomp | Abaluck & Gruber AER 2011 |
| F9 | Ho–Pakes Hospital Choices | openICPSR `112712` | Healthcare | AEA standard | ~150k CA births × ~250 hospitals | CL baseline + moment-inequality contrast | Ho & Pakes AER 2014 |
| F10 | Bundorf–Levin–Mahoney Health Plan | openICPSR `112566` | Health insurance | AEA standard | ~11k employees × plans | Clean conditional logit baseline | Bundorf et al. AER 2012 |
| F11 | Marone–Sabety Vertical Choice | openICPSR `148941` | Health insurance | AEA standard | ~100k employee-years | Recent (2022) random-coefs | Marone & Sabety AER 2022 |
| F12 | Abdulkadiroglu et al. NYC HS Match | openICPSR `113104` | Education / matching | AEA standard | ~90k students × ~700 programs | Rank-ordered logit at scale | AAP AER 2017 |
| F13 | Grubb–Osborne Cellular | openICPSR `112873` | Telecom plans | AEA standard | Few-thousand subscribers panel | Plan choice + usage dynamics | Grubb & Osborne AER 2015 |
| F14 | Wheat DCE Ethiopia | data.mendeley.com `r288pwfzhj` | Agronomy SP | CC-BY-4.0 | 303 rural HHs × DCE | ⭐ Sub-Saharan Africa filler for geographic gap | Tanaka et al. 2020 |
| F15 | OTTO Multi-Objective Sessions | github.com/otto-de/recsys-dataset | E-commerce sessions | CC-BY-4.0 | 220M events / 14.6M sessions / 1.8M items | ⭐ GPU stress test (largest open choice corpus) | Reisser RecSys 2023 |
| F16 | ITC Database | nature.com s41597-026-06947-4 | Intertemporal choice (psych) | CC-BY-4.0 | 1.17M trials × 11,852 subj × 100 studies | Hierarchical / mixed logit at scale + RT | Sci Data 2026 |

---

## Negative results (sources that yielded essentially nothing)

| Source | Outcome |
|---|---|
| **scikit-learn** | Zero native discrete-choice datasets. All `load_*`/`fetch_*` are classification or regression. |
| **OpenML** | Zero new datasets. Of 6,403 datasets and 488 studies, the only true preference data are SUSHI (45734) and SpeedDating (40536) — both already known. Phrase searches for "discrete choice", "multinomial logit", "conjoint" return 0 hits or false positives. |
| **UCI ML** | Zero. Strong near-misses (Restaurant & Consumer, Online Shoppers, Student Performance) all reject on schema — no alternative-specific features. |
| **OSF** | Zero raw choice microdata. Projects host only PDFs/preregistrations; replication data lives on Dataverse. |
| **Hugging Face Datasets** | Only **MIND** survived (B5 above). Most "preference"/"multiple-choice" tags are NLP QA, RLHF pairwise, or bare interaction logs without choice sets. |

**Implication**: the choice-modeling community publishes to econ replication archives (AEA / JAE) and Harvard Dataverse, NOT to the ML-data ecosystem. The MIND dataset is a notable exception (and a useful bridge to the ML audience).

---

## Recommended bundles

### Bundle: 6 picks covering 6 audiences (no domain overlap with current examples)

| Audience | Pick | Why |
|---|---|---|
| Health-econ at scale | **D1 Gaynor NHS** | Millions of patient×hospital choices, GPU/MPS pitch + consideration sets |
| Education / matching | **D2 Fack–Grenet–He Paris** | Rank-ordered logit, school choice, AER replication |
| Retail demand / availability | **D4 Conlon–Mortimer vending** | Per-occasion availability mask (canonical use case) |
| Political conjoint | **C2 Bechtel–Scheve climate** | 4-country conjoint, mixed logit by country, PNAS anchor |
| Healthcare DCE | **C9 Obadha Kenya** *(see C-list footnote)* | Outside option / opt-out demo |
| ML / recsys bridge | **B5 MIND** | Bridges torch-choice to the recsys community |

### Bundle: 2 quickest wins (one evening each)

1. **C1 HHY Immigrant Conjoint** — CC0, 1.4k respondents, paired profiles, AMCE = conditional-logit coefficients ⇒ direct verification target.
2. **A2 Electricity (Train)** — already in `mlogit` long format, adds panel/mixed-logit story missing from current examples.

### Bundle: 2 highest-impact for the paper's audience

1. **A3 LPMC** + **A4 Swissmetro** — LPMC for the scale/GPU pitch; Swissmetro for direct Biogeme reproducibility comparison.

---

## Download status

Downloads live in `tutorials/public_datasets/downloads/` (gitignored). Status as of last run:

| Dataset | Status | Path | Notes |
|---|---|---|---|
| A4 Swissmetro | ✓ downloaded | `swissmetro/swissmetro.dat` | 0.77 MB direct from EPFL |
| B4 trivago-clicks | ✓ downloaded | `trivago_clicks/trivago-clicks.zip` | 2.33 MB via Google Drive |
| C1 HHY Immigrant | ✓ downloaded | `hhy_immigrant/hhy_immigrant.zip` | Dataverse bulk zip; `.dta` + replication code |
| C2 Bechtel–Scheve Climate | ✓ downloaded | `bechtel_scheve_climate/*.zip` | Dataverse; 3MB Stata + paper PDFs |
| C3 Liu Authoritarian China | ✓ downloaded | `liu_china_political/*.zip` | Dataverse; 225KB Stata |
| C4 Singh Argentina | ✓ downloaded | `singh_argentina/*.zip` | Dataverse; 2.8MB conjoint Stata |
| C5 Miwa Japan Ideology | ✓ downloaded | `miwa_japan_ideology/*.zip` | Dataverse; CSV + R + C++ mixture model code |
| E2 SUSHI | ✓ downloaded | `sushi/sushi3-2016.zip` | Direct from kamishima.net (asset path) |
| C9 Obadha Kenya | ✗ depositor-restricted | — | "Data restricted as the dataset is still being analyzed" — needs email request to Obadha. **Demote.** |
| D6 Kreimeier EQ-5D-Y | ✗ no data | — | Zenodo record `6953084` ships only the published PDF, **not** respondent-level DCE microdata. **Demote — agent overclaimed.** |

### Pending (require auth)

| Dataset | Why pending |
|---|---|
| D1 Gaynor NHS, D2 Fack-Grenet-He Paris, D4 Conlon-Mortimer, D5 Vossler | openICPSR free Researcher Passport login |
| D3 Andersson Swedish school choice | JAE Data Archive (ZBW) — verify whether login required |
| C6 CSES IMD | Free CSES registration |
| C7 ANES 2020 | Free ANES Data Center registration |
| B5 MIND | MS Research License acceptance |
| B1 Expedia, B2 Ta-Feng, B7 H&M | Kaggle API key + competition rules acceptance |

### R-package extraction (✓ complete)

Extracted via `pyreadr` from the cran/mlogit, cran/Ecdat, and cran/AER GitHub mirrors. Both the original `.rda` and a converted `.csv` are saved per dataset.

| Dataset | Path | Rows × Cols | Source repo |
|---|---|---:|---|
| A1 Yogurt | `yogurt/yogurt.csv` | 2,412 × 10 | cran/Ecdat |
| A2 Electricity | `electricity/electricity.csv` | 4,308 × 26 | cran/mlogit |
| A5 JapaneseFDI | `japanese_fdi/japanese_fdi.csv` | 25,764 × 17 | cran/mlogit |
| A7 Brownstone–Train Car | `brownstone_train_car/car.csv` | 4,654 × 70 | cran/mlogit |
| A8 RiskyTransport | `risky_transport/risky_transport.csv` | 5,405 × 22 | cran/mlogit |
| A9 NOx | `nox/nox.csv` | 9,480 × 12 | cran/mlogit |
| A10 TravelMode | `travel_mode/travel_mode.csv` | 840 × 9 | cran/AER |

### Tutorials shipped (✓ all 8 PASS torch-choice ↔ mlogit comparison + §5 alternative-spec analysis)

Each lives in a self-contained `tutorials/<name>/` folder with `README.md`,
`<name>.csv`, `fit_mlogit.R`, `mlogit_output.json`, `<name>.ipynb`. Verified via
the shipped cell outputs in each notebook.

#### §4 verification (torch-choice ↔ mlogit cross-package match)

| Dataset | Estimates max abs | Est max % | SEs max abs | SE max % | LL diff | Notes |
|---|---:|---:|---:|---:|---:|---|
| yogurt | 6.7e-5 | 0.0032% | 2.2e-6 | 0.0012% | 1.83e-4 | template / prototype |
| electricity | 2.07e-4 | 0.0042% | 1.65e-6 | 0.0011% | 2.95e-4 | panel SP, pooled MNL |
| nox | 5.95e-6 | 0.0007% | 2.18e-7 | 0.0002% | 2.16e-5 | availability mask via `available` col |
| risky_transport | 3.65e-6 | 0.0009% | 4.68e-8 | 0.0001% | 9.85e-5 | dropped `weight` (sampling weight) |
| brownstone_train_car | 1.13e-5 | 0.0071% | 2.03e-7 | 0.0003% | 1.03e-4 | wide-to-long reshape; numeric attrs only |
| travel_mode | 6.84e-6 | 0.0013% | 1.42e-6 | 0.0001% | 4.12e-11 | required float64 + `backend='torch'` |
| japanese_fdi | 2.67e-4 | 0.0315% | 8.75e-2 | 9.4% | 3.51e-5 | nested logit; SE tol relaxed to 1e-1 |
| swiss_route_choice | 7.58e-6 | 0.0008% | 1.65e-7 | 0.0004% | 2.03e-5 | Apollo SP route choice (binary) |

#### §5 alternative-specs analysis (within-package model comparison)

Each notebook fits 2–3 alternative specifications and selects via AIC/BIC,
with literature-grounded recommendation.

| Dataset | Spec A (current) | Spec B | Spec C | BIC-optimal | Theoretical recommendation |
|---|---|---|---|---|---|
| yogurt | pooled MNL | HH price coef | HH price + feat | **B** (ΔBIC=−371) | mixed logit (JVC 1994) — out of in-package scope; B is closest FE approximation |
| electricity | pooled MNL | HH price | HH price + cl | **A** (FE bloat at 366 HHs) | mixed logit (Train 2009 Ch.6) |
| brownstone_train_car | pooled MNL | alt-specific price | + range + pollution | **A** (cross-sectional → no user FE) | mixed logit (BT 1999) |
| risky_transport | pooled MNL | + chooser moderators (age/swim/fatalism) | per-respondent risk | **B** (ΔBIC=−46) | mixed logit on risk for VSL (Léon-Miguel 2017) |
| nox | pooled MNL | regime × cost interactions | per-plant (unidentified) | A by 2.34 BIC; AIC favors B | regime interactions (Fowlie 2010) — empirically confirmed |
| travel_mode | flat MNL | NL public/private | NL motorised/non | **A** (LR borderline + λ>1 violates RUM) | flat MNL safest; Greene Ch.19 |
| japanese_fdi | NL shared λ | flat MNL | NL per-country λ | **B** (ΔBIC=−2.94 over A) | NL shared λ (Head-Mayer 2004 main spec); LR fails to reject MNL on this subset |
| swiss_route_choice | pooled MNL | per-respondent travel-time | (mixed logit out of scope) | **A** (FE penalty too high at 391 ppl) | mixed logit on tt (Axhausen 2008; Apollo) |

Pattern across all 8: **AIC and BIC diverge frequently** because BIC's `k log n`
penalty bites hard once you add per-user fixed effects. The literature
consistently points at **mixed logit** as the canonical answer; in-package FE
approximations capture some heterogeneity but bloat the parameter count enough
that BIC penalizes them. Three of eight (yogurt, risky_transport, japanese_fdi
flat-MNL) had a richer spec actually beat A on BIC; the rest had A win.

## Caveats and open verifications

- **D1 Gaynor NHS / D4 Conlon–Mortimer / D5 Vossler**: openICPSR returns 403 to bots; agents verified via paper abstracts, not file-level inspection. Manual download check needed before committing.
- **D1 Gaynor NHS** also requires free Researcher Passport login. Not a paywall, but a friction point.
- **B1 Expedia / B2 Ta-Feng / B7 H&M / E1 Speed-Dating-on-Kaggle**: Kaggle ToS — tutorial must point to Kaggle, not re-host. Speed-Dating is also on OpenML (id 40536), preferable.
- **B5 MIND**: Microsoft Research License (research only); users must accept terms. The mteb Hugging Face mirror is a flattened reranking format and loses ImpressionID/time grouping — prefer the original `behaviors.tsv`.
- **B6 MovieLens-as-choice**: a *reframing*, not native choice data. Coefficients won't match a published number directly. Useful for a feature demo, weaker as a verification target.
- **A3 LPMC**: license says "academic use with citation" — confirm redistribution compatible with shipping in tutorials vs. just linking.
- **C6 CSES**: party features must be added from external sources (CHES/MARPOR) — promotes from "low" to "medium" effort.
- **car_choice.csv** in `replication/`: verify whether it is Brownstone–Train (A7); if so, promote to a regularization tutorial rather than treating A7 as new.

---

## Where this scan came from

Three rounds of agent dispatches (the third was a parallel 4-agent burst):

1. **Classical econometric scan** — mlogit/Apollo/Biogeme/Kenneth Train. → A-list.
2. **ML-flavored / non-classical use-case scan** — Kaggle, recsys, conjoint, large-scale. → B-list, C1, plus 5 use-case framings (recommendation, conjoint, counterfactual, fairness, A/B).
3. **Trusted-source parallel scan** (4 sub-agents, parallel):
   - OpenML + sklearn → empty
   - UCI + Hugging Face → MIND only
   - Harvard Dataverse + ICPSR + OSF → C2–C5 (Dataverse)
   - AEA + JAE + Zenodo → D1–D6

Update this file as new datasets are evaluated or downgraded.
