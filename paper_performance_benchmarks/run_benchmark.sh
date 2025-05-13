#! /bin/bash

DATA_PATH="/home/tianyudu/Development/torch-choice/paper_performance_benchmarks/torch_choice_benchmark_data_totally_synthetic"
OUTPUT_PATH="./benchmark_results_aurora_20250428"

conda activate dev

python3.9 ./run_torch_choice.py \
    --data_path="$DATA_PATH" \
    --output_path="$OUTPUT_PATH" \
    --device="cuda"

conda deactivate
conda activate r-dev

Rscript run_mlogit_experiments.R \
    all \
    "$DATA_PATH" \
    "$OUTPUT_PATH" \
    5
