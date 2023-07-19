#! /bin/bash
python ./run_torch_choice.py num_records_experiment_large
python ./run_torch_choice.py num_records_experiment_small

python ./run_torch_choice.py num_params_experiment_small
python ./run_torch_choice.py num_params_experiment_large

python ./run_torch_choice.py num_items_experiment_small
python ./run_torch_choice.py num_items_experiment_large
