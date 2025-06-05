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
Please refer to the `paper_demo.ipynb` notebook in this `replication` directory for all code demonstrations from the paper and their corresponding output figures.

## Code for Performance Benchmarks
This section describes the code for generating synthetic data to stress test the software package and replicate results presented in the paper. All relevant materials are located in the `replication/paper_performance_benchmarks` directory.

### Synthetic Data Generation
The `simulated_datasets_synthetic.ipynb` notebook generates synthetic datasets of various sizes and complexities in terms of number of records, items, and features.
The generated synthetic datasets will be saved locally in a directory specified in the notebook, the total size of the generated datasets is around 5.7GiB.
Please note that the process of generating synthetic datasets is memory intensive given the large number of records and features. It could take up to 1 hour to generate the datasets if you are running short on memory (e.g., suppose you are on a Mac, it uses the hard drive as the swap memory if you are running out of physical memory on your machine. These disk-based swap memory are much slower than the physical memory, leading to a slower data generation process).

**Optional**: the last section in the data generation notebook provides the chance to compare the data you generated with the version we used while writing the paper. You can download *our data* from [Google Drive](https://drive.google.com/drive/folders/1qPkDjbGiItfH-KBAK7jCEUlWT49E_t88?usp=sharing) and set the `REFERENCE_DATA_PATH` variable in the notebook to the location where you save the downloaded data. The notebook will then compare the data you generated with the version we used while writing the paper.

### Performance Benchmarks
The `run_benchmark.sh` script estimates several model specifications on these synthetic datasets using both `torch-choice` (Python) and `mlogit` (R).

Before running the script, you will need to:
1. Update the environment activation commands (e.g., `conda activate ...`) to match your Python and R environments
2. Set `DATA_PATH` to the location of your synthetic datasets (as specified in `simulated_datasets_synthetic.ipynb`)
3. Set `OUTPUT_PATH` to the directory where you want to save the benchmarking results

The script will save benchmarking results (including dataset sizes, model specifications, and runtime) in the directory specified by `OUTPUT_PATH`.

We have included our benchmarking results in the `benchmark_results_aurora_20250428` directory for reference. Note that these files record the runtime in seconds for each algorithm on the synthetic datasets, so your results will vary depending on your hardware.

### Visualization
The `visualize_performance_benchmarks_v2.ipynb` notebook visualizes the benchmarking results by reading from the directory specified by `OUTPUT_PATH` in the `run_benchmark.sh` script.

As mentioned previously, your benchmark results will differ from ours due to hardware differences, so the figures you generate will be slightly different from those in the paper. However, if you set `R_RECORD_PATH` and `TORCH_RECORD_PATH` to point to our benchmarking results in the `benchmark_results_aurora_20250428` directory, you will generate figures identical to those in the paper.

We have also included the figures generated from our benchmarking results in the `benchmark_figures_20250428` directory for your reference, these are figures we used in the paper draft.