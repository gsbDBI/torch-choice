install.packages("mlogit")
install.packages("tictoc")
library("mlogit")
library("tictoc")

# data("ModeCanada", package = "mlogit")

ModeCanada = read.csv('./public_datasets/ModeCanada.csv')
MC <- dfidx(ModeCanada, subset = noalt == 4)

tic()
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')
toc()

summary(ml.MC1)
