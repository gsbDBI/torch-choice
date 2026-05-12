# GPU Stress Test Guide for torch-choice Benchmarks

This guide walks you through running the torch-choice PyTorch performance benchmarks under a *simulated* GPU memory cap, and verifying that peak memory consumption matches the empirical numbers documented in `replication/paper_replication_guideline.md`.

The benchmark pipeline consists of three stages:

1. **Generate synthetic data** — creates `.pt` and `.csv` datasets with configurable numbers of records, parameters, and items.
2. **Benchmark torch-choice** — trains conditional logit models on the synthetic data across multiple seeds, measuring runtime and memory usage.
3. **Visualize results** — produces PDF figures comparing performance across experiment configurations.

This guide focuses on stages 1 and 2 only. R/mlogit and the visualization stage are skipped (`SKIP_R=1`).

## Prerequisites

- A Linux machine with an NVIDIA GPU and working CUDA drivers
- A working `nvidia-smi` on PATH (used by the monitoring script)
- Internet connection for downloading packages

Any GPU size works — `GPU_MEM_LIMIT` (used in Step 5) simulates a smaller card, so you can verify the small-GPU path on, say, a 24 GB workstation without finding a real 10 GB box.

---

## Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## Step 2: Clone the repository and build the minimal replication archive

This guide tests the *replication archive* scenario: torch-choice installed from PyPI, no local package source. The `create_minimal_replication_release.sh` helper builds that archive from a fresh clone.

```bash
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice
bash ./scripts/create_minimal_replication_release.sh /tmp/torch-choice-replication
cd /tmp/torch-choice-replication
```

You should now be in a directory containing only the replication essentials: `replication/`, `scripts/`, `pyproject.toml`, the docs, and `LICENSE`. No `torch_choice/` source — that comes from PyPI in Step 3.

## Step 3: Install torch-choice from PyPI

`setup_uv_pypi.sh` creates a `.venv`, installs the dependency tree from `pyproject.toml` (`uv sync --no-install-project`), and then installs torch-choice itself from PyPI on top (`uv pip install --python .venv/bin/python --no-deps "torch-choice==1.0.7"`).

```bash
bash ./scripts/setup_uv_pypi.sh 1.0.7

uv run python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name}  total mem: {p.total_memory/2**30:.1f} GB')
"
```

**Expected:** The setup script ends with `[SUCCESS] Setup complete!` and a smoke-test model fit. The CUDA verification then prints `CUDA available: True` plus the card's name and total memory.

---

## Step 4: Sanity-check the tier selection (5 seconds, no training)

Before launching the real benchmark, confirm that `_auto_batch_size()` picks the expected batch size for the simulated GPU. This catches plumbing regressions in seconds, instead of after a multi-minute training run.

```bash
source .venv/bin/activate
GPU_MEM_LIMIT=10 python -c "
import sys; sys.path.insert(0, 'replication/paper_performance_benchmarks/steps')
import step02_torch_choice_benchmark as m
m._set_device('cuda')
print('selected:', m._auto_batch_size())
"
deactivate
```

**Expected:**
```
[Auto batch size] GPU: <name> (actual: <real_gb> GB, limit: 10.0 GB via GPU_MEM_LIMIT)
[Auto batch size] GPU: <name> (10.0 GB) -> batch_size=16,384
selected: 16384
```

Tier table — change `GPU_MEM_LIMIT` to test other tiers:

| `GPU_MEM_LIMIT` | Expected `batch_size` |
|-----------------|-----------------------|
| 6  (< 8 GB)     | 8,192                 |
| 10 (8–12 GB)    | 16,384                |
| 14 (12–16 GB)   | 32,768                |
| 18 (16–22 GB)   | 65,536                |
| 24 (≥ 22 GB)    | 131,072               |

If the printed `batch_size` doesn't match the row for your `GPU_MEM_LIMIT`, the tier logic has regressed — stop and investigate before continuing.

---

## Step 5: Run the PyTorch benchmark with GPU monitoring

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SMOKE_TEST` | `0` | Set to `1` for a quick run (few seeds, few epochs, small grid). |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, or `cuda`. |
| `SKIP_R` | `0` | Set to `1` to skip R and visualization (PyTorch only). |
| `GPU_MEM_LIMIT` | *(unset)* | Simulated GPU memory cap in GB (e.g. `10` to simulate a 10 GB GPU). |
| `BATCH_SIZE` | *(auto)* | Batch size for training. Leave unset for VRAM-based auto-detection. |
| `NUM_SEEDS` | `5` (`2` in smoke) | Number of random seeds per experiment. For a stress test, `1` is sufficient. |
| `NUM_EPOCHS` | `50000` (`5` in smoke) | Max training epochs (early stopping applies). |
| `GENERATE_EXPERIMENTS` | `all` | Which experiments the data-generation step (step01) should create. Use `full_dataset` to generate only the large-grid file. |
| `TORCH_EXPERIMENT_NAME` | `all` | Which benchmark experiment(s) (step02) to run. Use `num_records_experiment_large` to sweep the large-grid records range. |

### Recommended pattern: start the monitor first

The monitor must run during the benchmark — start it before, kill it after. The portable pattern (works in both interactive shells and SLURM job scripts):

```bash
# Background the monitor (writes one row/sec to gpu_metrics.csv)
bash ./scripts/monitor_gpu.sh ./gpu_metrics_10gb.csv &
MONITOR_PID=$!
```

### Option A: smoke test (recommended first run)

A few-minute end-to-end check of the full pipeline.

```bash
GPU_MEM_LIMIT=10 SMOKE_TEST=1 SKIP_R=1 DEVICE=cuda \
  bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

**Runtime:** a few minutes
**Output:** `replication/paper_performance_benchmarks/runs/smoke_<timestamp>/`

Note: the smoke test uses tiny grids (3K–7K records × 10–30 items × 3–5 params, 5 epochs), so peak memory will be much smaller than the headline numbers in Step 6's table (a few hundred MiB). It verifies the pipeline runs, not the empirical peak.

### Option B: targeted large-grid run (the headline-number test)

This is what produces the **~1,100 MiB peak** number that backs the small-GPU claim. It runs only the `num_records_experiment_large` benchmark (which sweeps 8 sample sizes from 3K to 100K records on the full 500-item / 30-param simulated dataset), with one seed, on the simulated 10 GB cap.

```bash
GPU_MEM_LIMIT=10 SMOKE_TEST=0 SKIP_R=1 DEVICE=cuda \
NUM_SEEDS=1 \
GENERATE_EXPERIMENTS=full_dataset \
TORCH_EXPERIMENT_NAME=num_records_experiment_large \
  bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

**Runtime:** roughly 20–30 minutes on a typical workstation (2–3 min generating the 3M-record synthetic dataset, then 24 training runs across 8 sample sizes × 3 formulas, most early-stopping in seconds).
**Output:** `replication/paper_performance_benchmarks/runs/<timestamp>/`

> **Why `GENERATE_EXPERIMENTS=full_dataset` rather than `num_records_experiment_large`?** The benchmark step and the generate step use different names for the same dataset: step01 calls it `full_dataset` (it's the 3M-record source file), step02 calls it `num_records_experiment_large` (it's the records sweep that reads that file). You set the generator to produce the dataset (`full_dataset`) and the benchmark to consume it (`num_records_experiment_large`).

### Option C: full benchmark (every experiment, every sample size)

Same as Option B but without the experiment restrictions — runs the entire small + large grid plus the params and items sweeps. Takes 1–2 hours with `SKIP_R=1` and `NUM_SEEDS=1`.

```bash
GPU_MEM_LIMIT=10 SMOKE_TEST=0 SKIP_R=1 DEVICE=cuda \
NUM_SEEDS=1 \
  bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

For interactive use where you prefer two terminals: run `bash ./scripts/monitor_gpu.sh ./gpu_metrics_10gb.csv` in terminal 1 (Ctrl+C to stop) and the benchmark in terminal 2 — same result.

The monitor CSV columns are: `timestamp`, `gpu_name`, `memory_used_mb`, `memory_total_mb`, `utilization_percent`.

---

## Step 6: Verify peak GPU memory against the expected number

Extract the peak memory observed during the run with one `awk` line:

```bash
awk -F',' 'NR>1 { gsub(/ MiB/, "", $3); if ($3+0 > max) max=$3+0 } END { print "peak GPU memory used:", max, "MiB" }' gpu_metrics_10gb.csv
```

### Expected peak by simulated GPU size (Option B / Option C runs only)

| `GPU_MEM_LIMIT` | Auto `batch_size` | Expected peak memory |
|-----------------|-------------------|----------------------|
| 6  (< 8 GB)     | 8,192             | ~600 MiB             |
| 10 (8–12 GB)    | 16,384            | ~1,100 MiB           |
| 14 (12–16 GB)   | 32,768            | ~2,200 MiB           |
| 18 (16–22 GB)   | 65,536            | ~4,100 MiB           |
| 24 (≥ 22 GB)    | 131,072           | ~8,100 MiB           |

A reading within roughly ±20 % of the expected value is normal — `nvidia-smi` samples once per second and may miss exact peaks, and other processes on the card add small offsets.

- **Peak much higher than expected (e.g. > 2×):** auto-detection probably didn't fire — check that the run log starts with a `[Auto batch size] ... -> batch_size=<N>` line and that `<N>` matches the tier table above. Also check that no other process is sharing the GPU.
- **Peak lower than expected:** fine — PyTorch's allocator was more conservative than the empirical worst-case used to derive the table. The Option A smoke-test peak will be much lower than the table values because smoke-test grids are tiny.

---

## Common pitfalls (and how to spot them)

The replication-archive scenario has two sharp edges worth knowing about, both of which manifest as confusing failures during Step 5.

### 1. Do not invoke the run wrapper with `uv run bash`

`bash ./replication/paper_performance_benchmarks/run_benchmarking.sh` — **correct**.
`uv run bash ./replication/paper_performance_benchmarks/run_benchmarking.sh` — **wrong**.

The wrapper internally calls `uv run python` for its actual work and exports `UV_NO_SYNC=1` to keep those calls from auto-syncing the local project. But an *outer* `uv run` runs *before* the wrapper has a chance to export anything, and that outer `uv run` will detect `pyproject.toml`'s `[project] name = "torch-choice"`, build an empty wheel from the metadata alone (since the replication archive has no `torch_choice/` source), and **overwrite the PyPI-installed torch-choice in `.venv`** with the empty wheel.

If you see these three lines at the start of the wrapper output, the outer `uv run` clobbered the install:

```
Built torch-choice @ file:///<archive-dir>
Uninstalled 1 package in <N>ms
Installed 1 package in <N>ms
```

Recovery: re-install torch-choice from PyPI before retrying.

```bash
uv pip install --python .venv/bin/python --no-deps "torch-choice==1.0.7" --force-reinstall
rm -rf torch_choice.egg-info
```

### 2. Generator and benchmark use different names for the same data

If you restrict the generate step to a benchmark-style name like `GENERATE_EXPERIMENTS=num_records_experiment_large`, the generator will silently produce nothing — its `ALL_EXPERIMENTS` set uses different names. The mapping you need:

| Benchmark experiment (`TORCH_EXPERIMENT_NAME`) | Generate experiment (`GENERATE_EXPERIMENTS`) |
|---|---|
| `num_records_experiment_small`  | `num_records_experiment_small` |
| `num_records_experiment_large`  | **`full_dataset`** |
| `num_params_experiment_small`   | `num_params_experiment_small` |
| `num_params_experiment_large`   | **`full_dataset`** |
| `num_items_experiment_small`    | `num_items_experiment_small` |
| `num_items_experiment_large`    | `num_items_experiment_large` |

You can spot a naming mismatch in two ways:
- The generate step prints `[Done] Finished generating synthetic datasets.` but no `[Saved] ...` lines before it.
- The benchmark step crashes with `FileNotFoundError: [Errno 2] No such file or directory: '.../simulated_choice_data_full_dataset_seed_42.pt'`.

---

## Two artifacts worth archiving

If you're using this guide to back a peer-review response or to validate a release, save:

1. The `[Auto batch size] ...` log line printed at the start of the benchmark (proves auto-detection fired with the right tier).
2. The full `gpu_metrics_10gb.csv` file (lets the reading be re-verified later, not just the single peak number).
