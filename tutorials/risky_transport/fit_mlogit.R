#!/usr/bin/env Rscript
# Fit a multinomial logit on the RiskyTransport dataset (Léon & Miguel,
# AEJ-Applied 2017) and emit coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-risky_transport.csv>
# Spec:   choice ~ cost + risk + seats | 0
#         (alt-specific intercepts SUPPRESSED via |0; WaterTaxi = reference)
#
# Note: the `weight` column in this dataset is a *sampling weight* (3 unique
# values, constant within chid), not an alt-specific covariate. The standard
# unweighted MNL is reproduced here so torch-choice's ConditionalLogitModel
# can replicate the fit exactly. mlogit's `weights=weight` survey-weighted
# variant is left for a follow-up tutorial.

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-risky_transport.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path)

# Convert the 0/1 binary choice column to a logical so dfidx interprets it as
# the chosen-alternative indicator within each chid.
df$choice <- as.logical(df$choice)

# CSV is in mlogit long format: one row per (chid, mode). Availability varies
# by chid (391 chids have 2 alts, 985 have 3, 417 have 4) — mlogit handles
# that implicitly via the missing rows.
data <- dfidx(
    df,
    shape = "long",
    choice = "choice",
    idx = c("chid", "mode")
)

# Generic coefficients on alt-specific cost, risk, seats; no alt-specific
# intercepts (the "| 0" part suppresses them). WaterTaxi is the reference
# alternative.
model <- mlogit(
    choice ~ cost + risk + seats | 0,
    data = data,
    reflevel = "WaterTaxi"
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
n_obs <- length(unique(df$chid))   # one choice per chid
n_alts <- length(model$freq)

# Emit JSON.  Tiny hand-rolled writer so we don't add a jsonlite dep.
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
