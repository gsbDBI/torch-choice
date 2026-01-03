#! /bin/bash

DATA_PATH="./synthetic_data"
OUTPUT_PATH="./benchmark_results"

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
