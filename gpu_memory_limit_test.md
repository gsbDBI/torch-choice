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

## Step 2: Clone the repository

```bash
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice
```

## Step 3: Set up the Python environment and verify GPU visibility

```bash
./scripts/setup_uv.sh complete

uv run python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name}  total mem: {p.total_memory/2**30:.1f} GB')
"
```

**Expected:** `CUDA available: True` plus the card's name and total memory.

---

## Step 4: Sanity-check the tier selection (5 seconds, no training)

Before launching the real benchmark, confirm that `_auto_batch_size()` picks the expected batch size for the simulated GPU. This catches plumbing regressions in seconds, instead of after a multi-minute training run.

```bash
GPU_MEM_LIMIT=10 uv run python -c "
import sys; sys.path.insert(0, 'replication/paper_performance_benchmarks/steps')
import step02_torch_choice_benchmark as m
m._set_device('cuda')
print('selected:', m._auto_batch_size())
"
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

### Recommended pattern: start the monitor first

The monitor must run during the benchmark — start it before, kill it after. The portable pattern (works in both interactive shells and SLURM job scripts):

```bash
# Background the monitor (writes one row/sec to gpu_metrics.csv)
bash ./scripts/monitor_gpu.sh ./gpu_metrics.csv &
MONITOR_PID=$!
```

### Option A: smoke test (recommended first run)

A few-minute end-to-end check of the full pipeline.

```bash
export SMOKE_TEST=1   # short run with reduced grids and epochs
export DEVICE=cuda    # run on GPU
export SKIP_R=1       # GPU stress test only needs PyTorch, no R
export GPU_MEM_LIMIT=10  # behave as if the GPU has 10 GB

bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

**Runtime:** a few minutes
**Output:** `replication/paper_performance_benchmarks/runs/smoke_<timestamp>/`

### Option B: full PyTorch benchmark

Complete benchmark with the full grids under the simulated small-GPU configuration. For the GPU memory test only, a single seed is enough — `NUM_SEEDS=1` cuts runtime by 5× without losing any information about peak memory.

```bash
export SMOKE_TEST=0   # run the full benchmark
export DEVICE=cuda    # run on GPU
export SKIP_R=1       # GPU stress test only needs PyTorch, no R
export NUM_SEEDS=1    # one seed is enough for a memory stress test
export GPU_MEM_LIMIT=10  # behave as if the GPU has 10 GB

bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

**Runtime (with `SKIP_R=1` and `NUM_SEEDS=1`):** roughly 20–30 minutes on a typical workstation. (The ~20 hr figure quoted in the main replication guideline is for the *with-R*, all-seeds pipeline; R/mlogit dominates that runtime, and `SKIP_R=1` removes it entirely.)
**Output:** `replication/paper_performance_benchmarks/runs/<timestamp>/`

For interactive use where you prefer two terminals: run `bash ./scripts/monitor_gpu.sh ./gpu_metrics.csv` in terminal 1 (Ctrl+C to stop) and the benchmark in terminal 2 — same result.

The monitor CSV columns are: `timestamp`, `gpu_name`, `memory_used_mb`, `memory_total_mb`, `utilization_percent`.

---

## Step 6: Verify peak GPU memory against the expected number

Extract the peak memory observed during the run with one `awk` line:

```bash
awk -F',' 'NR>1 { gsub(/ MiB/, "", $3); if ($3+0 > max) max=$3+0 } END { print "peak GPU memory used:", max, "MiB" }' gpu_metrics.csv
```

### Expected peak by simulated GPU size

| `GPU_MEM_LIMIT` | Auto `batch_size` | Expected peak memory |
|-----------------|-------------------|----------------------|
| 6  (< 8 GB)     | 8,192             | ~600 MiB             |
| 10 (8–12 GB)    | 16,384            | ~1,100 MiB           |
| 14 (12–16 GB)   | 32,768            | ~2,200 MiB           |
| 18 (16–22 GB)   | 65,536            | ~4,100 MiB           |
| 24 (≥ 22 GB)    | 131,072           | ~8,100 MiB           |

A reading within roughly ±20 % of the expected value is normal — `nvidia-smi` samples once per second and may miss exact peaks, and other processes on the card add small offsets.

- **Peak much higher than expected (e.g. > 2×):** auto-detection probably didn't fire — check that the run log starts with a `[Auto batch size] ... -> batch_size=<N>` line and that `<N>` matches the tier table above. Also check that no other process is sharing the GPU.
- **Peak lower than expected:** fine — PyTorch's allocator was more conservative than the empirical worst-case used to derive the table.

---

## Two artifacts worth archiving

If you're using this guide to back a peer-review response or to validate a release, save:

1. The `[Auto batch size] ...` log line printed at the start of the benchmark (proves auto-detection fired with the right tier).
2. The full `gpu_metrics.csv` file (lets the reading be re-verified later, not just the single peak number).
