# Replication Material
Thank you for your interest in our paper and for taking the time to replicate the results.

The replication material consists of two parts: demonstrations from the paper and code for running performance benchmarks.

## Setup the Environment
You can use either `conda` or a Python virtual environment with `pip` to run the code. You will need to install the software listed in the `requirements.txt` file to fulfill the dependencies required by the `torch-choice` package.

## Installing the `torch-choice` Package
Please refer to our installation guide to set up the package. The package can be installed via two methods: from PyPI or from source code.

Since we continuously update the package, if you are replicating the code presented in the paper and have received the source code in the replication package, we strongly recommend installing from source to ensure consistency and avoid potential discrepancies from newer package versions.

You can install from source by running:
```bash
python setup.py install
```

## Demo Code from the Paper
Use the console-friendly script `paper_demo.py` in this `replication` directory to reproduce the demonstrations from the paper. Example:
```bash
python paper_demo.py --skip-training  # add --num-epochs N to train
```
The original notebook (`paper_demo.ipynb`) is still available for interactive exploration.

## Code for Performance Benchmarks
This section describes the code for generating synthetic data to stress test the software package and replicate results presented in the paper. All relevant materials are located in the `replication/paper_performance_benchmarks` directory.

### Synthetic Data Generation
Run the script `paper_performance_benchmarks/simulate_datasets_synthetic.py` to generate synthetic datasets of varying sizes and feature counts:
```bash
cd replication/paper_performance_benchmarks
python simulate_datasets_synthetic.py --output-path ./synthetic_data --skip-plots
```
Key flags:
- `--experiments ...` (default `all`) to generate a subset of datasets.
- `--reference-path /path/to/reference` to compare outputs with a provided reference set.
- `--skip-plots` avoids t-SNE visualizations for faster, headless runs.

The generated datasets are roughly 5.7 GiB in total. Data generation is memory intensive; on low-memory machines the process can take up to an hour if the OS starts swapping to disk.

**Optional**: To compare against our published artifacts, download the reference data from [Google Drive](https://drive.google.com/drive/folders/1qPkDjbGiItfH-KBAK7jCEUlWT49E_t88?usp=sharing) and pass its location via `--reference-path`.

### Performance Benchmarks
The `run_benchmark.sh` script estimates several model specifications on these synthetic datasets using both `torch-choice` (Python) and `mlogit` (R).

Before running the script, you will need to:
1. Update the environment activation commands (e.g., `conda activate ...`) to match your Python and R environments
2. Set `DATA_PATH` to the location of your synthetic datasets (the `--output-path` you used with `simulate_datasets_synthetic.py`)
3. Set `OUTPUT_PATH` to the directory where you want to save the benchmarking results

The script will save benchmarking results (including dataset sizes, model specifications, and runtime) in the directory specified by `OUTPUT_PATH`.

#### Reference hardware and software
We report wall-clock time in seconds. Our reference runs (the files in `benchmark_results_aurora_20250428`) were executed on a Linux workstation (`6.8.0-58-generic`, host `tianyu-Alienware-Aurora`) with a 16-core x86_64 CPU, an NVIDIA GeForce RTX 3090 GPU, Python 3.9.15, and PyTorch 2.5.1+cu124 (CUDA 12.4). R baselines used R 3.6.1 with `mlogit` 1.1.1. Each CSV in that directory records the CPU/GPU identifiers, core count, Python/R versions, and run dates for transparency.

#### Interpreting runtime numbers
Because wall-clock time depends on hardware (CPU clock/architecture, GPU model and memory bandwidth), system load/temperature, and low-level libraries (e.g., BLAS or cuDNN), your absolute runtimes will differ from ours even when you use the same scripts and random seeds. The benchmarking procedure and code remain fully reproducible; see the README note we added alongside this hardware disclosure for more context.

### Visualization
The `visualize_performance_benchmarks_v2.ipynb` notebook visualizes the benchmarking results by reading from the directory specified by `OUTPUT_PATH` in the `run_benchmark.sh` script.

As mentioned above, your benchmark results will differ from ours because of hardware and low-level software differences, so the figures you generate will be slightly different from those in the paper. However, if you set `R_RECORD_PATH` and `TORCH_RECORD_PATH` to point to our benchmarking results in the `benchmark_results_aurora_20250428` directory, you will generate figures identical to those in the paper.

We have also included the figures generated from our benchmarking results in the `benchmark_figures_20250428` directory for your reference, these are figures we used in the paper draft.