# Replication Material
Thank you for your interest in our paper and spending time to replicate the results.

The replication material of the paper consists of two parts, demonstration in the paper and code for running performance benchmarks.

## Setup the Environment
You can choose your own Python environment to run the code, either using `conda` or pure Python virutal environment with `pip`. You would need to install the softwares listed in the `requirements.txt` file to fullfill the dependencies required by the `torch-choice` package.

## Installing `torch-choice` Package
Please refer to our installation guide to set up the package. The package can be installed via two methods: from PyPI or from the source code.
As we continuously update the package, if you are replicating the code presented in the paper and have received the source code in the replication package, we strongly recommend installing the package from the source code to ensure consistency and avoid potential discrepancies that may arise from newer versions of the package.

You can do so by running the following command:
```bash
python setup.py install
```

## Demo Code in the Paper
Please refer to the `paper_demo.ipynb` notebook in this `replication` directory for all code demos in the paper and corresponding output figures.

## Code for Performance Benchmark
The following section describes the code for generating synthetic data for stress testing the software package, replicating results presented in the paper. All relevant materials are included in the `replication/paper_performance_benchmarks` directory.

### Synthetic Data Generation
The `simulated_datasets_synthetic.ipynb` notebook generates synthetic datasets of various sizes and complexities in terms of number of records, items, and features.
The generated synthetic datasets will be saved locally in a directory specified in the notebook, the total size of the generated datasets is around 5.7GiB.
Please note that the process of generating synthetic datasets is memory intensive given the large number of records and features. It could take up to 1 hour to generate the datasets if you are running short on memory (e.g., suppose you are on a Mac, it uses the hard drive as the swap memory if you are running out of physical memory on your machine. These disk-based swap memory are much slower than the physical memory, leading to a slower data generation process).

**Optional**: the last section in the data generation notebook provides the chance to compare the data you generated with the version we used while writing the paper. You can download *our data* from [Google Drive](https://drive.google.com/drive/folders/1qPkDjbGiItfH-KBAK7jCEUlWT49E_t88?usp=sharing) and set the `REFERENCE_DATA_PATH` variable in the notebook to the location where you save the downloaded data. The notebook will then compare the data you generated with the version we used while writing the paper.

### Performance Benchmarks
The `run_benchmark.sh` script estimates several different model specifications on these synthetic datasets using `torch-choice` (Python) and `mlogit` (R) separately.
Please note that you would need to update the environment activation commands (e.g., `conda activate ...`) in the script to run it on your own machine to properly activate the Python and R environments.
You would also need to change data paths (`DATA_PATH` and `OUTPUT_PATH`) to the location where you save the synthetic datasets (set in `simulated_datasets_synthetic.ipynb`) and the directory where you want to save the benchmarking results, respectively.
These scripts will save the benchmarking results (e.g., size of datasets, model specifications, and runtime) in the directory specified in by `OUTPUT_PATH` in the script.

We have also included *our benchmarking results* in the `benchmark_results_aurora_20250428` directory for your reference. Please note that these files tracks the number of seconds taken for the algorithm to run on the synthetic datasets, so your results won't be exactly the same as ours, depending on the hardware you are using.

### Visualization
The `visualize_performance_benchmarks_v2.ipynb` notebook visualizes the benchmarking results by reading the benchmarking results saved in the directory specified in by `OUTPUT_PATH` in the `run_benchmark.sh` script.
As mentioned prevoiusly, the benchmark results you have will not be exactly the same as ours, these figures you generated will be slightly different from the ones in the paper. However, if you set `R_RECORD_PATH` and `TORCH_RECORD_PATH` to the location where you save *our benchmarking results* in the `benchmark_results_aurora_20250428` directory, the figures you generate will be exactly the same as the ones in the paper.
