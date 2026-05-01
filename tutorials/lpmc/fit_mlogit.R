#!/usr/bin/env Rscript
# Fit the canonical Bierlaire-MOOC MNL on the London Passenger Mode Choice
# (LPMC) dataset and emit coefficients + log-likelihood as JSON on stdout.
#
# Usage:  Rscript fit_mlogit.R <path-to-lpmc.csv>
#
# Spec (matches the LPMC_DCM_ML.ipynb notebook in
#   https://github.com/michelbierlaire/mooc-discrete-choice ):
#   V_walk  = B_TIME_WALKING * dur_walking
#   V_cycle = ASC_CYCLING + B_TIME_CYCLING * dur_cycling
#   V_pt    = ASC_PT + B_COST_PT * cost_transit + B_TIME_PT_ACCESS * dur_pt_access
#                    + B_TIME_PT_RAIL  * dur_pt_rail  + B_TIME_PT_BUS * dur_pt_bus
#                    + B_TIME_PT_INT   * dur_pt_int
#   V_drive = ASC_DRIVING + B_TIME_DRIVING * dur_driving
#                         + B_COST_DRIVING * (cost_driving_fuel + cost_driving_ccharge)
#                         + B_TRAFFIC_DRIVING * driving_traffic_percent
# Walk (mode = 1) is the reference alternative (ASC_WALK pinned to 0).
# All four modes are always available (matches Hillel/Bierlaire convention).

suppressPackageStartupMessages({
    library(mlogit)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript fit_mlogit.R <path-to-lpmc.csv>")
}
csv_path <- args[1]

wide <- read.csv(csv_path)

# Reshape wide -> long. One row per (trip, mode), with utility components
# pre-multiplied so each variable is non-zero only on the alternative whose
# utility uses it. With `mlogit(... | 0 | 0)` this gives alt-specific
# coefficients without having to spell out a `|` formula by hand.
N <- nrow(wide)
modes <- c("walk", "cycle", "pt", "drive")
mode_id <- c(walk = 1, cycle = 2, pt = 3, drive = 4)

build_alt <- function(mode_label) {
    is_walk  <- mode_label == "walk"
    is_cycle <- mode_label == "cycle"
    is_pt    <- mode_label == "pt"
    is_drive <- mode_label == "drive"
    data.frame(
        trip_id            = wide$trip_id,
        mode               = mode_label,
        choice             = (wide$travel_mode == mode_id[[mode_label]]),
        # Mode-specific time coefficients (non-zero only on the owning alt).
        x_dur_walking      = if (is_walk)  wide$dur_walking      else 0,
        x_dur_cycling      = if (is_cycle) wide$dur_cycling      else 0,
        x_dur_pt_access    = if (is_pt)    wide$dur_pt_access    else 0,
        x_dur_pt_rail      = if (is_pt)    wide$dur_pt_rail      else 0,
        x_dur_pt_bus       = if (is_pt)    wide$dur_pt_bus       else 0,
        x_dur_pt_int       = if (is_pt)    wide$dur_pt_int       else 0,
        x_cost_transit     = if (is_pt)    wide$cost_transit     else 0,
        x_dur_driving      = if (is_drive) wide$dur_driving      else 0,
        x_cost_driving     = if (is_drive) wide$cost_driving_fuel + wide$cost_driving_ccharge else 0,
        x_traffic_driving  = if (is_drive) wide$driving_traffic_percent                       else 0,
        # Alt-specific intercepts (walk pinned to 0 = reference).
        asc_cycle          = as.integer(is_cycle),
        asc_pt             = as.integer(is_pt),
        asc_drive          = as.integer(is_drive)
    )
}

long <- do.call(rbind, lapply(modes, build_alt))
long <- long[order(long$trip_id, match(long$mode, modes)), ]

data <- dfidx(
    long,
    shape   = "long",
    choice  = "choice",
    idx     = c("trip_id", "mode")
)

# All coefficients are alt-generic in this formulation, since we already
# pre-multiplied each variable by its mode indicator. The `| 0 | 0` part
# disables mlogit's automatic alt-specific intercepts and chooser-specific
# variables (we built our own intercept dummies above).
model <- mlogit(
    choice ~ x_dur_walking + x_dur_cycling
           + x_dur_pt_access + x_dur_pt_rail + x_dur_pt_bus + x_dur_pt_int
           + x_cost_transit
           + x_dur_driving + x_cost_driving + x_traffic_driving
           + asc_cycle + asc_pt + asc_drive
           | 0 | 0,
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
n_obs <- N
n_alts <- length(modes)

# Hand-rolled JSON writer (no jsonlite dep; mirrors tutorials/yogurt/fit_mlogit.R).
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
