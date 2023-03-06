library(mlogit)
library(tictoc)
library(stringr)

# setwd("/home/tianyudu/Development/torch-choice/tutorials/performance_benchmark/benchmark_data")
setwd("/oak/stanford/groups/athey/tianyudu/Data/torch_choice_benchmark")

user_latent_columns <- c('user_latent_0', 'user_latent_1', 'user_latent_2', 'user_latent_3', 'user_latent_4', 'user_latent_5', 'user_latent_6', 'user_latent_7', 'user_latent_8', 'user_latent_9', 'user_latent_10', 'user_latent_11', 'user_latent_12', 'user_latent_13', 'user_latent_14', 'user_latent_15', 'user_latent_16', 'user_latent_17', 'user_latent_18', 'user_latent_19', 'user_latent_20', 'user_latent_21', 'user_latent_22', 'user_latent_23', 'user_latent_24', 'user_latent_25', 'user_latent_26', 'user_latent_27', 'user_latent_28', 'user_latent_29')

item_latent_columns <- c('item_latent_0', 'item_latent_1', 'item_latent_2', 'item_latent_3', 'item_latent_4', 'item_latent_5', 'item_latent_6', 'item_latent_7', 'item_latent_8', 'item_latent_9', 'item_latent_10', 'item_latent_11', 'item_latent_12', 'item_latent_13', 'item_latent_14', 'item_latent_15', 'item_latent_16', 'item_latent_17', 'item_latent_18', 'item_latent_19', 'item_latent_20', 'item_latent_21', 'item_latent_22', 'item_latent_23', 'item_latent_24', 'item_latent_25', 'item_latent_26', 'item_latent_27', 'item_latent_28', 'item_latent_29')


t.list = c()
f.list = c()
num.items.list = c()
seed.list = c()
# run for 3 times and take average.
num.seeds <- 5
for (seed in 1:num.seeds) {
  tic()
  print(paste0('Seed=', seed))
  for (num_items in c(200)) {
    # get substring to construct formula. Only 5 dimensions of latents are used.
    df <- read.csv(str_glue("simulated_choice_data_num_items_experiment_{num_items}.csv"))
    df$item_id <- as.factor(df$item_id)
    data <- mlogit.data(df, choice="choice", shape="long", alt.var="item_id", chid.var="session_id", id.var="user_id")
    U <- paste(user_latent_columns[1:5], collapse = " + ")
    I <- paste(item_latent_columns[1:5], collapse = " + ")
    formula_str_list = c(
      str_glue("choice ~ 0 | {U} - 1 | 0"),
      str_glue("choice ~ {I} - 1 | 0 | 0"),
      str_glue("choice ~ {I} - 1 | {U} - 1 | 0")
    )
    for (f in formula_str_list) {
      print(paste("Num Items=", num_items, "Formula=", f))
      t <- system.time({model <- mlogit(as.formula(f), data)})
      # record elapsed time.
      t.list <- append(t.list, as.numeric(t[3]))
      f.list <- append(f.list, f)
      num.items.list <- append(num.items.list, num_items)
      seed.list <- append(seed.list, seed)
    }
  }
  print(paste0('Time taken for completing seed ', seed))
  toc()
}
records <- data.frame('time'=t.list, 'formula'=f.list, 'num_items'=num.items.list, 'seed'=seed.list)
write.csv(records, "/home/tianyudu/Development/torch-choice/tutorials/performance_benchmark/R_performance_num_items_larger.csv", row.names=FALSE)
