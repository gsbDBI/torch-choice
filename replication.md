# Replication Material
The replication material of the paper consists of two parts, the paper demo code and the code for performance benchmark.

## Installation
Please refer to our installation guide to set up the package.

## Demo Code in the Paper
Please refer to the `./tutorials/paper_demo.ipynb` notebook for all code demos in the paper.

## Code for Performance Benchmark
The `paper_performance_benchmarks` includes code for generating synthetic data, running performance benchmarks, and visualizing the results.

### Synthetic Data Generation
The `simulate_datasets_synthetic.ipynb` notebook generates synthetic datasets of various sizes and complexities.
The generated synthetic datasets will be saved locally in a specified directory, the total size of the generated datasets is around 5.7GiB.

### Performance Benchmarks
The `run_benchmark.sh` script estimates several different model specifications on these synthetic datasets using `torch-choice` (Python) and `mlogit` (R) separately.
Please note that you would need to update the environment activation commands (e.g., `conda activate ...`) in the script to run it on your own machine.
You would also need to change data paths (`DATA_PATH` and `OUTPUT_PATH`) to the location where you save the synthetic datasets and the directory where you want to save the benchmarking results, respectively.
These scripts will save the benchmarking results (e.g., size of datasets, model specifications, and runtime) in the directory specified in by `OUTPUT_PATH` in the script.

### Visualization
The `visualize_performance_benchmarks_v2.ipynb` notebook visualizes the benchmarking results by reading the benchmarking results saved in the directory specified in by `OUTPUT_PATH` in the `run_benchmark.sh` script.
