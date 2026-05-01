#!/usr/bin/env Rscript
# Fit a multinomial logit on the Brownstone-Train (1999) Car dataset
# (stated-preference vehicle choice with 6 hypothetical alternatives) and
# emit coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-car.csv>
# Spec:   choice ~ price + range + acc + speed + pollution + size + space
#                  + cost + station (alt-specific intercepts; choice1 = ref)
#
# v1 fits the simplified MNL with numeric attributes only. The published
# specification is a mixed logit with random coefficients + the categorical
# `type`/`fuel` features; that is a v2 follow-up (see README).

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-car.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path)

# The CSV has a string `choice` column ("choice1".."choice6"). dfidx with
# `sep=""` infers alt levels by stripping the numeric suffix from columns
# like `price1` -> alt level "1". The `choice` column must therefore use
# matching levels, so we strip the "choice" prefix to leave plain integers.
df$choice <- as.integer(sub("choice", "", as.character(df$choice)))

# Wide -> long. Columns 5..70 = 6 alts x 11 attrs (type, fuel, price, range,
# acc, speed, pollution, size, space, cost, station). `sep=""` because the
# wide columns use `<attr><alt_index>` with no separator (e.g. `price1`).
data <- dfidx(
    df,
    shape   = "wide",
    choice  = "choice",
    varying = 5:70,
    sep     = ""
)

# v1 spec: numeric attributes only with alt-specific intercepts.
# Reference alternative is "1" (= "choice1" in the original CSV); mlogit's
# default reference is the first level so we don't pass `reflevel`.
model <- mlogit(
    choice ~ price + range + acc + speed + pollution + size + space + cost + station,
    data = data
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
n_obs <- nrow(df)        # one row per choice occasion in wide format
n_alts <- length(model$freq)

# Emit JSON. Hand-rolled writer (no jsonlite dependency).
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
