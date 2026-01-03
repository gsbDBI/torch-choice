# Paper Performance Benchmarks

This folder lets you replay the benchmarking pipeline used in the paper. It is simplified to one Python CLI (`paper_performance_benchmark.py`), one R CLI (`run_mlogit_experiments.R`), and a single wrapper script you can run end-to-end.

> Timing is hardware-dependent (CPU/GPU/cores/RAM). The paper’s reported wall-clock numbers will not reproduce exactly on different machines; relative trends should still match.

## Recommended Python setup (uv)

From the **repo root**, create the Python environment with uv and install dependencies:

```bash
./scripts/setup_uv.sh complete
```

Then run the benchmarking script via uv (uv is required for the Python steps):

```bash
uv run bash replication/paper_performance_benchmarks/run_benchmarking.sh
```

## Quickstart (one command)

```bash
cd replication/paper_performance_benchmarks
chmod +x run_benchmarking.sh
./run_benchmarking.sh
```

By default, this creates a new timestamped run directory under `./runs/` and writes all outputs there (data, CSVs, figures). The script prints the chosen run directory at the start.

### Quick smoke-test configuration (run this first; runs fast, verify the computational environment, and produces CSV outputs)

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

## Required R packages (manual install)

Install the R dependencies yourself before running (the script will not auto-install them). Launch R, then run:

```r
install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")
```

## Sequence of operations (what runs, in order)

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

```bash
RUN_PATH=/tmp/bench_run \
DEVICE=cuda \
NUM_SEEDS=3 \
./run_benchmarking.sh
```

## What’s in here

### Top-level entrypoints

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

### Implementation files (review in order)

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

### Paper artifacts included

- `benchmark_results_aurora_20250428/`: reference CSVs produced for the paper on a specific machine/run date.
- `benchmark_figures_20250428/`: reference PDFs produced for the paper on a specific machine/run date.

## Notes

- Visualization expects CSVs like `R_performance_{items,params,records}.csv` and `torch_choice_performance_{num_items,num_params,num_records}_experiment_{small,large}.csv`.
- Outputs go to the paths you pass (`--output-path`), defaulting to `./benchmark_figures` for visualizations.
- Torch device defaults to auto-select (`cuda` if available, else `cpu`); timings vary by hardware.