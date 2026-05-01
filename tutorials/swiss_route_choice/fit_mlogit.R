#!/usr/bin/env Rscript
# Fit a multinomial logit on the Apollo Swiss Route Choice SP dataset
# (Axhausen et al. 2008, Transport Policy 15(3):173-185) and emit
# coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-apollo_swissRouteChoiceData.csv>
# Spec:   choice ~ tt + tc + hw + ch | 0
#         (no alt-specific intercepts; the two "routes" are interchangeable
#          experimental labels, so |0 suppresses them, matching the canonical
#          Apollo MNL_SP_RouteChoice example.)

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-apollo_swissRouteChoiceData.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path)

# CSV is wide-per-occasion: each row lists both routes' four attributes
# side-by-side (tt1, tc1, hw1, ch1, tt2, tc2, hw2, ch2). Columns 3:10 are the
# alt-varying attributes (cols 1-2 are ID and choice).
data <- dfidx(
    df,
    shape = "wide",
    choice = "choice",
    varying = 3:10,
    sep = "",
    idnames = c("chid", "alt")
)

model <- mlogit(choice ~ tt + tc + hw + ch | 0, data = data, reflevel = "1")

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
