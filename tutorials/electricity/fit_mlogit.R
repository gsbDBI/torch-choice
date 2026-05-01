#!/usr/bin/env Rscript
# Fit a pooled multinomial logit on the Electricity (Train SP) dataset and emit
# coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-electricity.csv>
# Spec:   choice ~ pf + cl + loc + wk + tod + seas | 0
#         (no alt-specific intercepts; the four "suppliers" are just hypothetical
#          labels so |0 suppresses them, matching the mlogit Electricity vignette.)

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-electricity.csv>")
}
csv_path <- args[1]

df <- read.csv(csv_path)

# CSV is wide-per-occasion (4,308 rows: 1,077 households x 4 occasions). Each
# row already lists all 4 suppliers' attributes side-by-side. Reshape to mlogit
# long format with mlogit.data (cols 3:26 = the 6 attributes x 4 suppliers).
df$chid <- seq_len(nrow(df))
data <- mlogit.data(
    df,
    id.var = "id",
    choice = "choice",
    varying = 3:26,
    shape = "wide",
    sep = ""
)

model <- mlogit(choice ~ pf + cl + loc + wk + tod + seas | 0, data = data)

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
