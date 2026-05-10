# Replication Material

The replication material consists of two parts: demonstrations from the paper and code for running performance benchmarks.

## Overview (what to run)

The replication material is driven by three entrypoint scripts (run from the repo root):

- `scripts/setup_uv.sh`: creates a local Python environment in `.venv` and installs `torch-choice` from source with the replication dependencies.
- `replication/run_paper_demo.sh`: runs the paper demo (`replication/paper_demo.py`) and (optionally) launches TensorBoard to visualize training logs.
- `replication/paper_performance_benchmarks/run_benchmarking.sh`: runs the computational speed benchmarks end-to-end (generate synthetic data → Torch-Choice → R/mlogit → visualize).

## Setup the Environment

Starting December 2025, we support and prefer using [uv](https://docs.astral.sh/uv/) for environment management. It is fast, reliable, and handles Python version management automatically.

Preferred (uv):
```bash
# From the repo root, run the helper to create `.venv` and install the Python dependencies needed for the replication material.
bash ./scripts/setup_uv.sh complete
```
The script checks that uv is installed (and prints install instructions if it is missing), creates `.venv`, and installs `torch-choice` in editable mode with dependencies.

**Traditional Method**: You can also use either `conda` or a Python virtual environment with `pip` to run the code. You will need to install the software listed in the `requirements.txt` file to fulfill the dependencies required by the `torch-choice` package.
Please refer to our installation guide to set up the package. The package can be installed via two methods: from PyPI or from source code.
Since we continuously update the package, if you are replicating the code presented in the paper and have received the source code in the replication package, we strongly recommend installing from source to ensure consistency and avoid potential discrepancies from newer package versions.

## Demo Code from the Paper
Use the `run_paper_demo.sh` script in this `replication` directory to reproduce the demonstrations from the paper and automatically launch TensorBoard afterward:
```bash
uv run bash ./replication/run_paper_demo.sh
```

The script runs the demo Python script `replication/paper_demo.py` and then starts TensorBoard to visualize training logs. Available options:
- `--skip-training` — Quick smoke test without model fitting
- `--num-epochs N` — Number of training epochs (default: 1000)
- `--tensorboard-logdir <dir>` — Directory for logs (default: `lightning_logs`)
- `--tensorboard-port <port>` — TensorBoard port (default: 6006)
- `--no-tensorboard` — Skip launching TensorBoard after the demo

We encourage readers to look into the `replication/paper_demo.py` with detailed in-line comments for a better understanding of the code.


## Performance (Computational Speed) Benchmarks

This section covers the benchmarking pipeline used in the paper. All relevant materials are located in the `replication/paper_performance_benchmarks` directory. The pipeline is simplified to one Python CLI (`paper_performance_benchmark.py`), one R CLI (`run_mlogit_experiments.R`), and a single wrapper script you can run end-to-end.

> The computational speed benchmarks are hardware-dependent (CPU/GPU/cores/RAM). The paper's reported wall-clock numbers will not reproduce exactly on different machines; relative trends should still match.
In our benchmark experiments, we focused on computational speed and report **wall-clock time in seconds** as our main performance measure. Wall-clock time naturally depends on the computing environment (e.g., CPU clock speed and architecture, GPU model and memory bandwidth, system load, and even temperature), so your absolute runtime values may differ from the paper even when you use the same scripts and random seeds.

For transparency, we ship the paper's reference benchmark artifacts in `replication/paper_performance_benchmarks/benchmark_results_aurora_20250428/` (CSVs) and `replication/paper_performance_benchmarks/benchmark_figures_20250428/` (PDFs). The reference CSVs record hardware/software metadata (CPU/GPU identifiers, core count, Python/PyTorch versions, R/package versions, and run date). We also include checksums for the reference synthetic datasets in `replication/paper_performance_benchmarks/reference_synthetic_dataset_md5sum.txt`.

### Runtime expectations

A full end-to-end run of the benchmarking pipeline can take **around ~20 hours** to finish on a typical workstation (16 cores, 128GiB RAM, NVIDIA GeForce RTX 3090 GPU). The runtime is usually dominated by the **R/mlogit** benchmarks (step 3), which are substantially slower than the Torch-Choice runs.

### GPU memory considerations

The benchmark automatically selects a batch size based on your GPU's available VRAM. These thresholds were empirically determined by stress testing with the largest experiment configuration (100K records × 500 items):

| GPU VRAM | Auto Batch Size | Est. Memory Usage |
|----------|-----------------|-------------------|
| < 8 GB   | 8,192           | ~600 MB           |
| 8–12 GB  | 16,384          | ~1,100 MB         |
| 12–16 GB | 32,768          | ~2,200 MB         |
| 16–22 GB | 65,536          | ~4,100 MB         |
| ≥ 22 GB  | 131,072         | ~8,100 MB         |

**Empirical limits found:**
- OOM at batch_size=262,144 on 24GB GPU (using ~17GB)
- Max safe batch: 131,072 on 24GB GPU (using ~8.1GB, 70% safety margin)

**Overriding auto-detection:**

```bash
# Force a specific batch size
export BATCH_SIZE=32768

# Force full-batch training (use with caution)
export BATCH_SIZE=-1

# Limit GPU memory usage (value in GB)
# Useful when running other GPU workloads concurrently
# This makes the benchmark behave as if your GPU has less memory than it actually does
# export GPU_MEM_LIMIT=10  # Use only 10GB of GPU memory
```

**If you still encounter OOM errors**, you can:
1. Set a smaller `BATCH_SIZE` (e.g., 4096)
2. Ensure the expandable-segments allocator is enabled (set by default in the wrapper):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
3. Fall back to CPU: `export DEVICE=cpu`

### Required R packages
The R runtime and packages are not included in the installation process (i.e., the uv setup script does not install them), the user is expected to install them manually. The following benchmarking script assumes that you have already setup the R environment with the required packages.

Install the R dependencies yourself before running (the script will not auto-install them). Launch R, then run:

```r
install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")
```

**Quick smoke-test configuration**
After setting up the Python and R environments, you can run a quick smoke-test to verify your environment before any full benchmark. Set `SMOKE_TEST=1` to run a minimal configuration that still **executes** and writes output CSVs, so you can inspect them:
- Generates only a single representative dataset/value per experiment (items/records/params).
- Runs only the corresponding small benchmark tasks.
- The run directory is tagged with `smoke_` (e.g., `runs/smoke_<timestamp>`).

```bash
export SMOKE_TEST=1
uv run bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```
The smoke test should only take a few minutes to complete on a typical workstation.

After verifying the setup with the smoke test, you can run the full benchmark.

```bash
# Enable the full benchmark by disabling the smoke test
export SMOKE_TEST=0
uv run bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

By default, the benchmarking script creates a new timestamped run directory under `./replication/paper_performance_benchmarks/runs/` and writes all outputs there (data, CSVs, figures). The script prints the chosen run directory at the start.

#### Overview of the benchmarking pipeline
The following is a high-level overview of our benchmarking pipeline, which you can find in the `replication/paper_performance_benchmarks` directory.

- `run_benchmarking.sh`: a convenience wrapper that runs the following steps end-to-end:
  - `generate`: generate synthetic user/item latent variables, simulate choices, then export benchmark datasets.
  - `torch-choice`: load `.pt` datasets, fit Torch-Choice models across experiment grids, record runtime/loss/epochs.
  - `R`: runs the mlogit benchmarks and writes `R_performance_*.csv`.
  - `visualize`: read Torch-Choice + R CSVs and generate the paper-style PDFs.
- `paper_performance_benchmark.py`
  - Master Python CLI providing subcommands:
    - `generate`: calls `steps/step01_generate_synthetic_data.py`
    - `torch-choice`: calls `steps/step02_torch_choice_benchmark.py`
    - `visualize`: calls `steps/step03_performance_visualization_v2.py`
- `run_mlogit_experiments.R`: runs the mlogit benchmarks and writes `R_performance_*.csv`.
  - CLI: `Rscript run_mlogit_experiments.R <experiment_type> <input_path> <output_path> <num_seeds>`, where `experiment_type in {items, records, params, all}`.

#### The benchmarking script (`run_benchmarking.sh`) will create the following directories and files in the `--output-path` directory (which defaults to `./replication/paper_performance_benchmarks/runs/<timestamp>`):

| Step | Script | Role | Outputs |
|------|--------|------|--------------------------------|
| 1 | `steps/step01_generate_synthetic_data.py` | Generate synthetic user/item latent variables, simulate choices, then export benchmark datasets. | `*.pt` datasets for Torch-Choice (ChoiceDataset saved via `torch.save`); `*.csv` long-format datasets for R/mlogit where applicable |
| 2 | `steps/step02_torch_choice_benchmark.py` | Load `.pt` datasets, fit Torch-Choice models across experiment grids, record runtime/loss/epochs. | `torch_choice_performance_{task}.csv` (one CSV per experiment task) |
| 3 | `run_mlogit_experiments.R` | Run mlogit benchmarks in R. | `R_performance_*.csv` |
| 4 | `steps/step03_performance_visualization_v2.py` | Read Torch-Choice + R CSVs and generate the paper-style PDFs. | `absolute_time_{parameter}_time_cost_benchmark.pdf` and `time_ratio_{parameter}_time_cost_benchmark.pdf` (3-panel figures); `absolute_time_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`; `time_ratio_{torch_choice_small_scale|r_small_scale|torch_choice_large_scale}_{parameter}_{loss|epochs}.pdf`; `debug_model_size_{parameter}.pdf`; Likelihood alignment CSVs: `likelihood_alignment_{parameter}.csv` and a combined `likelihood_alignment.csv` |
