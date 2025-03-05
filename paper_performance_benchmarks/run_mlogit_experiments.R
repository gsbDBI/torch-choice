# run_mlogit_experiments.R
#
# This script performs performance benchmarking for the mlogit package across
# different experimental conditions. It measures execution time, parameter counts,
# and model likelihoods while systematically varying key factors:
#
# 1. Number of items in choice sets.
# 2. Number of records/observations in datasets.
# 3. Number of parameters in models.
#
# Each experiment is repeated multiple times (seeds) to ensure reliable measurements.
# The script uses a unified experiment runner function that handles all experiment
# types to maximize code reuse. Results are saved as CSV files with comprehensive
# system and package version information for reproducibility.
#
# Usage: Rscript run_mlogit_experiments.R <experiment_type> <input_path> <output_path> <num_seeds>
# where experiment_type is one of: "items", "records", "params", or "all"
#
# Author: Tianyu Du
# Date: 2025-03-05

library(mlogit)
library(tictoc)
library(stringr)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_mlogit_experiments.R <experiment_type> <input_path> <output_path> <num_seeds>")
}
experiment_type <- args[1]  # Options: "items", "records", "params", "all"
input_path <- args[2]
# Check if the input path exists
if (!dir.exists(input_path)) {
  stop("Invalid input path. Please provide a valid path to the input data.")
}

# Get the output path; it's okay if the output path does not exist, it will be created.
output_path <- args[3]
# Create output directory if it doesn't exist
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

num_seeds <- as.numeric(args[4])

# Define the complete set of all latent variables, this will be subsetted for each experiment if necessary.
user_latent_columns <- c('user_latent_0', 'user_latent_1', 'user_latent_2', 'user_latent_3', 'user_latent_4',
                         'user_latent_5', 'user_latent_6', 'user_latent_7', 'user_latent_8', 'user_latent_9',
                         'user_latent_10', 'user_latent_11', 'user_latent_12', 'user_latent_13', 'user_latent_14',
                         'user_latent_15', 'user_latent_16', 'user_latent_17', 'user_latent_18', 'user_latent_19',
                         'user_latent_20', 'user_latent_21', 'user_latent_22', 'user_latent_23', 'user_latent_24',
                         'user_latent_25', 'user_latent_26', 'user_latent_27', 'user_latent_28', 'user_latent_29')

# Define the complete set of all item latent variables, this will be subsetted for each experiment if necessary.
item_latent_columns <- c('item_latent_0', 'item_latent_1', 'item_latent_2', 'item_latent_3', 'item_latent_4',
                         'item_latent_5', 'item_latent_6', 'item_latent_7', 'item_latent_8', 'item_latent_9',
                         'item_latent_10', 'item_latent_11', 'item_latent_12', 'item_latent_13', 'item_latent_14',
                         'item_latent_15', 'item_latent_16', 'item_latent_17', 'item_latent_18', 'item_latent_19',
                         'item_latent_20', 'item_latent_21', 'item_latent_22', 'item_latent_23', 'item_latent_24',
                         'item_latent_25', 'item_latent_26', 'item_latent_27', 'item_latent_28', 'item_latent_29')

# Helper function to build formula strings
# Build the three kinds of formula strings for the benchmark experiments.
build_formula_strings <- function(num_user_dims, num_item_dims) {
  U <- paste(user_latent_columns[1:num_user_dims], collapse = " + ")
  I <- paste(item_latent_columns[1:num_item_dims], collapse = " + ")

  return(c(
    str_glue("choice ~ 0 | {U} - 1 | 0"),
    str_glue("choice ~ {I} - 1 | 0 | 0"),
    str_glue("choice ~ {I} - 1 | {U} - 1 | 0")
  ))
}

# Helper function to add system and package information to results
add_system_info <- function(df) {
  df$R_version <- R.version.string
  df$mlogit_version <- as.character(packageVersion("mlogit"))
  df$tictoc_version <- as.character(packageVersion("tictoc"))
  df$stringr_version <- as.character(packageVersion("stringr"))
  df$sysname <- Sys.info()[["sysname"]]
  df$nodename <- Sys.info()[["nodename"]]
  df$release <- Sys.info()[["release"]]
  df$machine <- Sys.info()[["machine"]]

  return(df)
}

# Unified experiment runner function
run_experiment <- function(experiment_type, input_path, output_path, num_seeds) {
  # Initialize lists for storing results
  t.list <- c()                # time taken for the model estimation.
  f.list <- c()                # the formula (specification) of the model.
  seed.list <- c()             # the random seed.
  parameter.count.list <- c()  # the number of parameters in the model.
  final.likelihood.list <- c() # the (log) likelihood of the final model.
  n.list <- c()                # the number of observations in the dataset.
  var.list <- c()              # the varying dimension of the experiment.

  # Set experiment-specific variables
  if (experiment_type == "items") {
    var_name <- "num_items"
    # 200 is the limit that we can run on a 128GiB machine.
    var_values <- c(10, 30, 50, 100, 150, 200)
  } else if (experiment_type == "records") {
    var_name <- "num_records"
    var_values <- c(1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000)
  } else if (experiment_type == "params") {
    var_name <- "num_params"
    var_values <- c(1, 5, 10, 15, 20, 30)
  } else {
    stop("Invalid experiment type. Use 'items', 'records', 'params', or 'all'.")
  }

  # Run experiment loops
  for (seed in 1:num_seeds) {
    # Set the seed for reproducibility for this iteration
    set.seed(seed)
    tic()  # Start timer for the entire seed iteration
    print(paste0('Seed=', seed))

    for (var_value in var_values) {
      # Loop over the varying dimension of the experiment

      if (experiment_type == "items") {
        # Read dataset for items experiment
        df <- read.csv(file.path(input_path, str_glue("simulated_choice_data_num_items_experiment_small_{var_value}_items_seed_42.csv")))
        df$item_id <- as.factor(df$item_id)
        data <- mlogit.data(df, choice="choice", shape="long", alt.var="item_id", chid.var="session_id", id.var="user_id")
        # Experiment with 5 user and 5 item latent dimensions.
        formula_str_list <- build_formula_strings(5, 5)

      } else if (experiment_type == "records") {
        # Load the full dataset.
        df_full <- read.csv(file.path(input_path, "simulated_choice_data_num_records_experiment_small.csv"))
        df_full$item_id <- as.factor(df_full$item_id)
        # Subset data for records experiment (with a limited number of records)
        data <- mlogit.data(subset(df_full, session_id < var_value),
                            choice="choice", shape="long", alt.var="item_id",
                            chid.var="session_id", id.var="user_id")
        # Run with 10 user and 10 item latent dimensions.
        formula_str_list <- build_formula_strings(10, 10)

      } else if (experiment_type == "params") {
        df <- read.csv(file.path(input_path, "simulated_choice_data_num_params_experiment_small.csv"))
        df$item_id <- as.factor(df$item_id)
        data <- mlogit.data(df, choice="choice", shape="long", alt.var="item_id",
                            chid.var="session_id", id.var="user_id")
        # Generate formulas with different numbers of parameters
        formula_str_list <- build_formula_strings(var_value, var_value)
      }

      # Run mlogit for each generated formula
      for (f in formula_str_list) {
        print(paste(var_name, "=", var_value, "Formula=", f))

        # Fit model and record time
        t <- system.time({ model <- mlogit(as.formula(f), data) })

        # Store results
        t.list <- append(t.list, as.numeric(t[3]))
        f.list <- append(f.list, f)
        var.list <- append(var.list, var_value)
        seed.list <- append(seed.list, seed)
        n.list <- append(n.list, nrow(data))

        # Record the number of trainable parameters and final likelihood
        parameter.count.list <- append(parameter.count.list, length(coef(model)))
        final.likelihood.list <- append(final.likelihood.list, as.numeric(logLik(model)))
      }
    }

    print(paste0('Time taken for completing seed ', seed))
    toc()  # End timer for this seed iteration
  }

  # Prepare results dataframe
  records <- data.frame(time = t.list, formula = f.list, seed = seed.list,
                        parameter_count = parameter.count.list,
                        final_likelihood = final.likelihood.list,
                        num_observations = n.list)

  # Add experiment-specific column
  records[[var_name]] <- var.list

  # Add system information
  records <- add_system_info(records)

  # Save results to the output path using a full file path
  output_file <- file.path(output_path, str_glue("R_performance_{experiment_type}.csv"))
  write.csv(records, output_file, row.names = FALSE)

  return(records)
}

# Main execution
if (experiment_type == "all") {
  # Run all experiment types
  print("Running number of items experiment...")
  run_experiment("items", input_path, output_path, num_seeds)

  print("Running number of records experiment...")
  run_experiment("records", input_path, output_path, num_seeds)

  print("Running number of parameters experiment...")
  run_experiment("params", input_path, output_path, num_seeds)

  print("All experiments completed successfully!")
} else if (experiment_type %in% c("items", "records", "params")) {
  # Run the specified experiment
  run_experiment(experiment_type, input_path, output_path, num_seeds)
} else {
  stop("Invalid experiment type. Use 'items', 'records', 'params', or 'all'.")
}