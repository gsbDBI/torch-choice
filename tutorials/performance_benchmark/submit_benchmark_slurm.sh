#!/bin/bash
#
#SBATCH --job-name=R_benchmark
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=athey
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --mail-user=tianyudu@stanford.edu
#SBATCH --mail-type=END

ml R/4.2

cd /home/users/tianyudu/Development/torch-choice/tutorials/performance_benchmark
Rscript run_mlogit_num_items.R
Rscript run_mlogit_num_records.R
Rscript run_mlogit_num_params.R