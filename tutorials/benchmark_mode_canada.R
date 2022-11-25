library(mlogit)
library(tictoc)

# setwd('/Users/tianyudu/Development/torch-choice/tutorials/benchmark_data')
setwd('/oak/stanford/groups/athey/tianyudu/mode_canada_benchmark_data')
num.seeds <- 3
k.range <- c(1, 5, 10, 50, 100, 500, 1000, 5000, 10000)
# k.range <- c(1, 5, 10)

t.list = c()
ll.list = c()
k.list = c()
seed.list = c()
# run for 3 times and take average.
for (seed in 1:num.seeds) {
  tic()
  print(paste0('Seed=', seed))
  for (k in k.range){
    print(paste0('k=', k))
    ModeCanada <- read.csv(paste0('./mode_canada_', k, '.csv'))
    ModeCanada$alt <- as.factor(ModeCanada$alt)
    MC <- dfidx(ModeCanada, subset = noalt == 4)
      t <- system.time({model <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')})
      # record elapsed time.
      t.list <- append(t.list, as.numeric(t[3]))
      ll.list <- append(ll.list, as.numeric(model$logLik))
      k.list <- append(k.list, k)
      seed.list <- append(seed.list, seed)
  }
  print(paste0('Time taken for completing seed ', seed))
  toc()
}
records <- data.frame('k'=k.list, 'seed'=seed.list, 'time'=t.list, 'log-likelihood'=ll.list)
write.csv(records, "R_performances.csv", row.names=FALSE)
