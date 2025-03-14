#! /bin/bash
# python ./run_torch_choice.py num_records_experiment_large
# python ./run_torch_choice.py num_records_experiment_small

# python ./run_torch_choice.py num_params_experiment_small
# python ./run_torch_choice.py num_params_experiment_large

# python ./run_torch_choice.py num_items_experiment_small
# python ./run_torch_choice.py num_items_experiment_large



# python3.9 ./run_torch_choice.py \
#     --data_path="/Volumes/HS_SSD/torch_choice_benchmark_data" \
#     --output_path="./benchmark_results_20250225" \
#     --device="cuda"

# python3.9 ./run_torch_choice.py \
#     --data_path="/oak/stanford/groups/athey/tianyudu/torch_choice_benchmark_data" \
#     --output_path="./benchmark_results_20250226" \
#     --device="cuda"

# run Sherlock version.
# cd /oak/stanford/groups/athey/tianyudu/Development/torch-choice/paper_performance_benchmarks
# ml py-pytorch/2.4.1_py312
# python3.12 ./run_torch_choice.py \
#     --data_path="/oak/stanford/groups/athey/tianyudu/Data/torch_choice_benchmark_data" \
#     --output_path="./torch_choice_benchmark_results_Sherlock_20250306_v2" \
#     --device="cuda"

# python3.9 ./run_torch_choice.py \
#     --data_path="/media/sata_drive/torch_choice_benchmark_data" \
#     --output_path="./torch_choice_benchmark_results_aurora_20250306" \
#     --device="cuda"

# # ================================
# # R benchmarks.
# # ================================
# cd /oak/stanford/groups/athey/tianyudu/Development/torch-choice/paper_performance_benchmarks

# ml R
# Rscript run_mlogit_experiments.R \
#     params \
#     /oak/stanford/groups/athey/tianyudu/Data/torch_choice_benchmark_data \
#     ./R_benchmark_results_Sherlock_20250306 \
#     1


# Rscript run_mlogit_num_items.R \
#     /oak/stanford/groups/athey/tianyudu/Data/torch_choice_benchmark_data \
#     ./R_benchmark_results_Sherlock_20250306 \
#     5


# Rscript run_mlogit_num_items.R \
#     /media/sata_drive/torch_choice_benchmark_data \
#     ./R_benchmark_results_20250305 \
#     5


# # ================================
# # R benchmarks current.
# # ================================
# Rscript run_mlogit_experiments.R \
#     all \
#     /media/sata_drive/torch_choice_benchmark_data \
#     ./R_benchmark_results_aurora_20250310 \
#     5


# # ================================
# # currently running.
# # ================================
# python3.9 ./run_torch_choice.py \
#     --data_path="/media/sata_drive/torch_choice_benchmark_data_totally_synthetic" \
#     --output_path="./benchmark_results_aurora_20250314" \
#     --device="cuda"

conda activate dev
for experiment_name in "num_records_experiment_small" "num_params_experiment_small" "num_items_experiment_small"
do
    python3.9 ./run_torch_choice.py \
        --data_path="/media/sata_drive/torch_choice_benchmark_data_totally_synthetic" \
        --output_path="./benchmark_results_aurora_20250314" \
        --device="cuda"
        --experiment_name=$experiment_name
done

conda deactivate
conda activate r-dev

Rscript run_mlogit_experiments.R \
    all \
    /media/sata_drive/torch_choice_benchmark_data_totally_synthetic \
    ./benchmark_results_aurora_20250314 \
    5