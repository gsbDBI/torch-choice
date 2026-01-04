# Replication Material
Thank you for your interest in our paper and for taking the time to replicate the results.

The replication material consists of two parts: demonstrations from the paper and code for running performance benchmarks.

## Setup the Environment
You can use either `conda` or a Python virtual environment with `pip` to run the code. You will need to install the software listed in the `requirements.txt` file to fulfill the dependencies required by the `torch-choice` package.

## Installing the `torch-choice` Package
Please refer to our installation guide to set up the package. The package can be installed via two methods: from PyPI or from source code.

Since we continuously update the package, if you are replicating the code presented in the paper and have received the source code in the replication package, we strongly recommend installing from source to ensure consistency and avoid potential discrepancies from newer package versions.

Preferred (uv):
```bash
# From the repo root, run the helper, which will automatically install everything you would need to run the replication material.
bash ./scripts/setup_uv.sh complete
```
Notes:
- We would highly recommend using uv to set up the environment, it is the preferred method of installation and it is the method we used to run the scripts.
- On Windows use `.\scripts\setup_uv.sh`.
- The script installs uv, creates `.venv`, and installs `torch-choice` in editable mode with dependencies.
- If you do not want to use uv, follow the package installation instructions in `install.md` (conda/pip) and then return here; ensure the environment you activate matches the one used to run the scripts.

## Demo Code from the Paper
Use the `run_paper_demo.sh` script in this `replication` directory to reproduce the demonstrations from the paper and automatically launch TensorBoard afterward:
```bash
bash replication/run_paper_demo.sh
```

The script runs the demo and then starts TensorBoard to visualize training logs. Available options:
- `--skip-training` — Quick smoke test without model fitting
- `--num-epochs N` — Number of training epochs (default: 1000)
- `--tensorboard-logdir <dir>` — Directory for logs (default: `lightning_logs`)
- `--tensorboard-port <port>` — TensorBoard port (default: 6006)
- `--no-tensorboard` — Skip launching TensorBoard after the demo

Training uses LBFGS on the Mode Canada dataset and can take several minutes on CPU.

Alternatively, you can run the Python script directly (TensorBoard will not auto-launch):
```bash
uv run python replication/paper_demo.py
```

The original notebook (`paper_demo.ipynb`) is still available for interactive exploration.

## Code for Performance Benchmarks

This section covers the benchmarking pipeline used in the paper. All relevant materials are located in the `replication/paper_performance_benchmarks` directory. The pipeline is simplified to one Python CLI (`paper_performance_benchmark.py`), one R CLI (`run_mlogit_experiments.R`), and a single wrapper script you can run end-to-end.

> Timing is hardware-dependent (CPU/GPU/cores/RAM). The paper's reported wall-clock numbers will not reproduce exactly on different machines; relative trends should still match.

### Runtime expectations

A full end-to-end run of the benchmarking pipeline can take **around ~20 hours** to finish on a typical workstation. The runtime is usually dominated by the **R/mlogit** benchmarks (step 3), which are substantially slower than the Torch-Choice runs.

### Required R packages
The R runtime and packages are not included in the installation process (i.e., the uv setup script does not install them), the user is expected to install them manually. The following benchmarking script assumes that you have already setup the R environment with the required packages.

Install the R dependencies yourself before running (the script will not auto-install them). Launch R, then run:

```r
install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")
```


### Recommended Python setup (uv)
Then run the benchmarking script via uv (uv is required for the Python steps):

```bash
cd replication/paper_performance_benchmarks
chmod +x run_benchmarking.sh
./run_benchmarking.sh
```

By default, this creates a new timestamped run directory under `./replication/paper_performance_benchmarks/runs/` and writes all outputs there (data, CSVs, figures). The script prints the chosen run directory at the start.

### Quick smoke-test configuration (run this first)

Run this first to verify your environment before any full benchmark. Set `SMOKE_TEST=1` to run a minimal configuration that still **executes** and writes output CSVs, so you can inspect them:
- Generates only a single representative dataset/value per experiment (items/records/params).
- Runs only the corresponding small benchmark tasks.
- The run directory is tagged with `smoke_` (e.g., `runs/smoke_<timestamp>`).

```bash
cd replication/paper_performance_benchmarks
SMOKE_TEST=1 \
./run_benchmarking.sh
```

> Note: uv is required; the script uses `uv run python` for all Python steps.

Once the smoke test completes, rerun without `SMOKE_TEST=1` to execute the full benchmark grids.

### Sequence of operations (what runs, in order)

The benchmarking pipeline is **four sequential steps**. Each step consumes the outputs of earlier steps:

1) **Generate synthetic data** (`paper_performance_benchmark.py generate`)
   - **Writes**: datasets under `${RUN_PATH}/synthetic_data/` (default: `./runs/<timestamp>/synthetic_data/`)
     - `simulated_choice_data_{experiment_tag}_seed_42.pt` (Torch-Choice input)
     - `simulated_choice_data_{experiment_tag}_seed_42.csv` (R/mlogit input for the relevant experiments)

2) **Benchmark Torch-Choice** (`paper_performance_benchmark.py torch-choice`)
   - **Reads**: `${RUN_PATH}/synthetic_data/*.pt`
   - **Writes**: Torch-Choice results into `${RUN_PATH}/benchmark_results/`:
     - `torch_choice_performance_{task}.csv`, where `{task}` is one of:
       - `num_records_experiment_small`, `num_records_experiment_large`
       - `num_params_experiment_small`, `num_params_experiment_large`
       - `num_items_experiment_small`, `num_items_experiment_large`

3) **Benchmark R (mlogit)** (`Rscript run_mlogit_experiments.R ...`)
   - **Reads**: `${RUN_PATH}/synthetic_data/*.csv` produced by step 1
   - **Writes**: R/mlogit results into `${RUN_PATH}/benchmark_results/`:
     - `R_performance_items.csv`
     - `R_performance_records.csv`
     - `R_performance_params.csv`

4) **Visualize (v2)** (`paper_performance_benchmark.py visualize`)
   - **Reads**: the CSVs produced by steps 2 and 3
   - **Writes**: PDF figures into `${RUN_PATH}/benchmark_figures/`:
     - `absolute_time_{parameter}_time_cost_benchmark.pdf` and `time_ratio_{parameter}_time_cost_benchmark.pdf` (3-panel figures: Torch-Choice small / R / Torch-Choice large)
     - `absolute_time_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`
     - `time_ratio_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`
     - `debug_model_size_{parameter}.pdf`
   - **Also writes** likelihood-alignment tables comparing R `final_likelihood` vs Torch `best_loss`:
     - `likelihood_alignment_num_items.csv`, `likelihood_alignment_num_params.csv`, `likelihood_alignment_num_records.csv`
     - `likelihood_alignment.csv` (combined summary)

To keep runs organized, prefer setting `RUN_PATH` (and optionally device/seeds/epochs) without editing the script:

This benchmarking script will generate the synthetic datasets with around 3.1G on disk.

```bash
./run_benchmarking.sh
```

### What's in the benchmarking directory

#### Top-level entrypoints

- `run_benchmarking.sh`
  - **What it does**: A convenience wrapper that runs steps 1→4 end-to-end (generate → torch-choice → R → visualize).
  - **What you can change**: via env vars (paths, device, seeds, epochs, learning rate).

- `paper_performance_benchmark.py`
  - **What it does**: Master Python CLI providing subcommands:
    - `generate`: calls `steps/step01_generate_synthetic_data.py`
    - `torch-choice`: calls `steps/step02_torch_choice_benchmark.py`
    - `visualize`: calls `steps/step03_performance_visualization_v2.py`
  - **Why it exists**: keeps a single user-facing Python entrypoint while maintaining separate implementation files.

- `run_mlogit_experiments.R`
  - **What it does**: Runs the mlogit benchmarks and writes `R_performance_*.csv`.
  - **CLI**: `Rscript run_mlogit_experiments.R <experiment_type> <input_path> <output_path> <num_seeds>`
    - `experiment_type ∈ {items, records, params, all}`

#### Implementation files (review in order)

- `steps/step01_generate_synthetic_data.py`
  - **Role**: generate synthetic user/item latent variables, simulate choices, then export benchmark datasets.
  - **Outputs** (under `--output-path`):
    - `*.pt` datasets for Torch-Choice (ChoiceDataset saved via `torch.save`)
    - `*.csv` long-format datasets for R/mlogit where applicable

- `steps/step02_torch_choice_benchmark.py`
  - **Role**: load `.pt` datasets, fit Torch-Choice models across experiment grids, record runtime/loss/epochs.
  - **Outputs** (under `--output-path`):
    - `torch_choice_performance_{task}.csv` (one CSV per experiment task)

- `steps/step03_performance_visualization_v2.py`
  - **Role**: read Torch-Choice + R CSVs and generate the paper-style PDFs (v2).
  - **Outputs** (under `--output-path`):
    - `absolute_time_{parameter}_time_cost_benchmark.pdf` and `time_ratio_{parameter}_time_cost_benchmark.pdf` (3-panel figures)
    - `absolute_time_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`
    - `time_ratio_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`
    - `debug_model_size_{parameter}.pdf`
    - Likelihood alignment CSVs: `likelihood_alignment_{parameter}.csv` and a combined `likelihood_alignment.csv`

#### Paper artifacts included

- `benchmark_results_aurora_20250428/`: reference CSVs produced for the paper on a specific machine/run date.
- `benchmark_figures_20250428/`: reference PDFs produced for the paper on a specific machine/run date.

### Reference hardware and software

We report wall-clock time in seconds. Our reference runs (the files in `benchmark_results_aurora_20250428`) were executed on a Linux workstation (`6.8.0-58-generic`, host `tianyu-Alienware-Aurora`) with a 16-core x86_64 CPU, an NVIDIA GeForce RTX 3090 GPU, Python 3.9.15, and PyTorch 2.5.1+cu124 (CUDA 12.4). R baselines used R 3.6.1 with `mlogit` 1.1.1. Each CSV in that directory records the CPU/GPU identifiers, core count, Python/R versions, and run dates for transparency.

### Interpreting runtime numbers

Because wall-clock time depends on hardware (CPU clock/architecture, GPU model and memory bandwidth), system load/temperature, and low-level libraries (e.g., BLAS or cuDNN), your absolute runtimes will differ from ours even when you use the same scripts and random seeds. The benchmarking procedure and code remain fully reproducible.

### Notes

- Visualization expects CSVs like `R_performance_{items,params,records}.csv` and `torch_choice_performance_{num_items,num_params,num_records}_experiment_{small,large}.csv`.
- Outputs go to the paths you pass (`--output-path`), defaulting to `./benchmark_figures` for visualizations.
- Torch device defaults to auto-select (`cuda` if available, else `cpu`); timings vary by hardware.
