#!/usr/bin/env Rscript
# Fit a nested logit on the JapaneseFDI dataset (Head & Mayer 2004) and emit
# coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-japanese_fdi.csv>
# Spec:   choice ~ log(wage) + unemp + elig + log(area) + scrate + ctaxrate | 0
#         nests = TRUE (auto: one nest per `country`),
#         un.nest.el = TRUE (single shared inclusive-value coefficient `iv`).
#
# This is the spec from the mlogit vignette (e2nlogit / c4.relaxiid), which
# reproduces the nested-logit model used in Head & Mayer (RES, 2004) up to the
# 6-covariate v1 subset: log(wage), unemp, elig, log(area), scrate, ctaxrate.
#
# JSON schema is the same flat list used by tutorials/yogurt etc.; the lambda
# (inclusive-value) coefficient appears as a regular row named "iv".

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-japanese_fdi.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path, stringsAsFactors = FALSE)

# CSV is long format. dfidx with idx=list("firm", c("region","country")) tells
# mlogit that "firm" is the chooser id and "region" is the alternative id, with
# "country" being the nest containing each region. The nest structure is then
# auto-detected when we pass `nests = TRUE`.
data <- dfidx(
    df,
    idx = list("firm", c("region", "country")),
    idnames = c("chid", "alt")
)

model <- mlogit(
    choice ~ log(wage) + unemp + elig + log(area) + scrate + ctaxrate | 0,
    data = data,
    nests = TRUE,
    un.nest.el = TRUE   # shared lambda across nests; matches torch-choice shared_lambda=True.
)

ct <- summary(model)$CoefTable
coef_records <- lapply(rownames(ct), function(nm) {
    row <- ct[nm, ]
    list(
        name      = nm,
        estimate  = unname(row["Estimate"]),
        std_err   = unname(row["Std. Error"]),
        z_value   = unname(row["z-value"]),
        p_value   = unname(row["Pr(>|z|)"])
    )
})

ll <- as.numeric(logLik(model))
n_obs <- length(unique(df$firm))
n_alts <- length(model$freq)

# Hand-rolled JSON writer (no jsonlite dependency).
escape_str <- function(s) gsub('"', '\\"', s, fixed = TRUE)
fmt_num <- function(x) {
    if (is.na(x)) "null" else formatC(x, format = "g", digits = 17)
}
coef_json <- paste0(
    "[",
    paste(vapply(coef_records, function(r) {
        sprintf(
            '{"name":"%s","estimate":%s,"std_err":%s,"z_value":%s,"p_value":%s}',
            escape_str(r$name),
            fmt_num(r$estimate),
            fmt_num(r$std_err),
            fmt_num(r$z_value),
            fmt_num(r$p_value)
        )
    }, character(1)), collapse = ","),
    "]"
)
out <- sprintf(
    '{"log_likelihood":%s,"n_obs":%d,"n_alts":%d,"coefficients":%s}',
    fmt_num(ll), n_obs, n_alts, coef_json
)
cat(out, "\n", sep = "")
