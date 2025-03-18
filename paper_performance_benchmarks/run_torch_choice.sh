#! /bin/bash
conda activate dev

python3.9 ./run_torch_choice.py \
    --data_path="/media/sata_drive/torch_choice_benchmark_data_totally_synthetic" \
    --output_path="./benchmark_results_aurora_20250314" \
    --device="cuda"

conda deactivate
conda activate r-dev

Rscript run_mlogit_experiments.R \
    all \
    /media/sata_drive/torch_choice_benchmark_data_totally_synthetic \
    ./benchmark_results_aurora_20250314 \
    5
