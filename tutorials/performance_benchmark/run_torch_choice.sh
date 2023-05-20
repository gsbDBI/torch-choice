#! /bin/bash
# for task in "num_records_experiment_large" "num_params_experiment_large" "num_items_experiment_large"
for task in "num_items_experiment_small"
    do
    for optimizer in "LBFGS"
    do
        echo $task
        echo $optimizer
        python3.10 ./run_torch_choice.py --task=$task --model_optimizer=$optimizer
    done
done