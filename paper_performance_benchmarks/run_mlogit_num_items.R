library(mlogit)
library(tictoc)
library(stringr)

# read output path from command line.
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]
num.seeds <- as.numeric(args[3])
setwd(input_path)

user_latent_columns <- c('user_latent_0', 'user_latent_1', 'user_latent_2', 'user_latent_3', 'user_latent_4', 'user_latent_5', 'user_latent_6', 'user_latent_7', 'user_latent_8', 'user_latent_9', 'user_latent_10', 'user_latent_11', 'user_latent_12', 'user_latent_13', 'user_latent_14', 'user_latent_15', 'user_latent_16', 'user_latent_17', 'user_latent_18', 'user_latent_19', 'user_latent_20', 'user_latent_21', 'user_latent_22', 'user_latent_23', 'user_latent_24', 'user_latent_25', 'user_latent_26', 'user_latent_27', 'user_latent_28', 'user_latent_29')
item_latent_columns <- c('item_latent_0', 'item_latent_1', 'item_latent_2', 'item_latent_3', 'item_latent_4', 'item_latent_5', 'item_latent_6', 'item_latent_7', 'item_latent_8', 'item_latent_9', 'item_latent_10', 'item_latent_11', 'item_latent_12', 'item_latent_13', 'item_latent_14', 'item_latent_15', 'item_latent_16', 'item_latent_17', 'item_latent_18', 'item_latent_19', 'item_latent_20', 'item_latent_21', 'item_latent_22', 'item_latent_23', 'item_latent_24', 'item_latent_25', 'item_latent_26', 'item_latent_27', 'item_latent_28', 'item_latent_29')

t.list <- c()
f.list <- c()
num.items.list <- c()
seed.list <- c()
parameter_count.list <- c()
final_likelihood.list <- c()

for (seed in 1:num.seeds) {
  tic()
  print(paste0('Seed=', seed))
  for (num_items in c(10, 30, 50, 100, 150, 200)) {
    # Read in the simulated dataset.
    df <- read.csv(str_glue("simulated_choice_data_num_items_experiment_{num_items}_items_seed_42.csv"))
    df$item_id <- as.factor(df$item_id)
    data <- mlogit.data(df, choice="choice", shape="long", alt.var="item_id", chid.var="session_id", id.var="user_id")

    # Construct latent variable substrings (using 5 dimensions).
    U <- paste(user_latent_columns[1:5], collapse = " + ")
    I <- paste(item_latent_columns[1:5], collapse = " + ")
    formula_str_list <- c(
      str_glue("choice ~ 0 | {U} - 1 | 0"),
      str_glue("choice ~ {I} - 1 | 0 | 0"),
      str_glue("choice ~ {I} - 1 | {U} - 1 | 0")
    )
    for (f in formula_str_list) {
      print(paste("Num Items=", num_items, "Formula=", f))
      # record the time taken to fit the model.
      t <- system.time({ model <- mlogit(as.formula(f), data) })
      # Record the elapsed time.
      t.list <- append(t.list, as.numeric(t[3]))
      f.list <- append(f.list, f)
      num.items.list <- append(num.items.list, num_items)
      seed.list <- append(seed.list, seed)

      # Record the number of trainable parameters and final likelihood.
      param_count <- length(coef(model))
      final_like <- as.numeric(logLik(model))
      parameter_count.list <- append(parameter_count.list, param_count)
      final_likelihood.list <- append(final_likelihood.list, final_like)
    }
  }
  print(paste0('Time taken for completing seed ', seed))
  toc()
}

# Wrap up the results.
records <- data.frame(time = t.list, formula = f.list, num_items = num.items.list, seed = seed.list,
                      parameter_count = parameter_count.list,
                      final_likelihood = final_likelihood.list)

# Record R and package version information.
records$R_version <- R.version.string
records$mlogit_version <- as.character(packageVersion("mlogit"))
records$tictoc_version <- as.character(packageVersion("tictoc"))
records$stringr_version <- as.character(packageVersion("stringr"))
records$sysname <- Sys.info()[["sysname"]]
records$nodename <- Sys.info()[["nodename"]]
records$release <- Sys.info()[["release"]]
records$machine <- Sys.info()[["machine"]]

# make the output directory if it doesn't exist.
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)
write.csv(records, str_glue("{output_path}/R_performance_num_items.csv"), row.names = FALSE)
