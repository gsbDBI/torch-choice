#!/usr/bin/env Rscript
# Fit a multinomial logit on the NOx pollution-control technology dataset
# (Fowlie 2010, AER) and emit coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-nox.csv>
# Spec:   choice ~ vcost + kcost + cm | 0   (no intercepts; alt 1 reference)
#
# Note on availability: each plant (chid) sees 15 candidate technologies on
# paper but only ~3-9 are actually feasible. The CSV's `available` column
# encodes this. We pass `subset = available == 1` to dfidx so mlogit drops
# the infeasible alts from the likelihood sum -- the unbalanced-MNL
# formulation. torch-choice represents the same constraint via an explicit
# (num_sessions, num_items) availability mask.

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-nox.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path)

# Long format. dfidx with subset filters infeasible (available=0) rows so the
# resulting model is the standard unbalanced MNL described in Train (2009, ch. 3).
data <- dfidx(
    df,
    shape   = "long",
    choice  = "choice",
    idx     = c("chid", "alt"),
    subset  = available == 1
)

model <- mlogit(choice ~ vcost + kcost + cm | 0, data = data, reflevel = 1)

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
n_obs <- length(unique(df$chid))    # number of plants (choice occasions)
n_alts <- length(model$freq)

# Emit JSON.  Use a tiny hand-rolled writer so we don't add a jsonlite dep.
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
