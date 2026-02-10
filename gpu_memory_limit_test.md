# GPU Stress Test Guide for torch-choice Benchmarks

This guide walks you through setting up a fresh VM to run the torch-choice PyTorch performance benchmarks with GPU memory testing.

The benchmark pipeline consists of three stages:
1. **Generate synthetic data** — creates `.pt` and `.csv` datasets with configurable numbers of records, parameters, and items.
2. **Benchmark torch-choice** — trains conditional logit models on the synthetic data across multiple seeds, measuring runtime and memory usage.
3. **Visualize results** — produces PDF figures comparing performance across experiment configurations.

This guide focuses on the PyTorch steps only (stages 1 and 2). R and the visualization stage are not required.

## Prerequisites

- A Linux VM with an NVIDIA GPU (≥20 GB VRAM recommended for full-batch testing)
- CUDA drivers installed
- Internet connection for downloading packages

---

## Step 1: Install uv (Python Package Manager)

[uv](https://github.com/astral-sh/uv) is used to manage the Python environment and run all Python steps.

```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

---

## Step 2: Clone the Repository

```bash
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice
```

---

## Step 3: Set Up Python Environment

The setup script installs all Python dependencies (including PyTorch and benchmark tooling) into a uv-managed virtual environment.

```bash
# Install all dependencies including benchmark tools
./scripts/setup_uv.sh complete

# Verify PyTorch can see the GPU
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Step 4: Run the PyTorch Benchmark

> **Note:** This guide covers the PyTorch benchmarks only. R is not required. Set `SKIP_R=1` to skip the R and visualization steps.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SMOKE_TEST` | `0` | Set to `1` for a quick run with reduced data sizes and fewer seeds. |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, or `cuda`. |
| `SKIP_R` | `0` | Set to `1` to skip R benchmarks and visualization (PyTorch only). |
| `GPU_MEM_LIMIT` | *(unset)* | Manually forced GPU memory cap in GB (e.g., `10` to simulate a 10 GB GPU). |
| `NUM_SEEDS` | `5` (`2` in smoke) | Number of random seeds per experiment for reproducibility. |
| `NUM_EPOCHS` | `50000` (`5` in smoke) | Training epochs per run. |
| `BATCH_SIZE` | *(auto)* | Batch size for training. Left empty for automatic detection based on available GPU memory. |

### Option A: Quick Smoke Test (Recommended First)

A smoke test runs a minimal configuration (fewer data points, fewer seeds, fewer epochs) to verify the full pipeline works end-to-end before committing to a long run.

```bash
export SMOKE_TEST=1
export DEVICE=cuda
export SKIP_R=1
uv run bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

**Runtime:** A few minutes
**Output location:** `replication/paper_performance_benchmarks/runs/smoke_<timestamp>/`

### Option B: Full Benchmark

This runs the complete benchmarking suite with the full data grids (up to 3 M records, 5 seeds, 50 K epochs). Use `GPU_MEM_LIMIT` to simulate a smaller GPU if your hardware has more memory than the target configuration.

```bash
export SMOKE_TEST=0  # run the full benchmark.
export DEVICE=cuda  # run on GPU.
export SKIP_R=1  # GPU stress test only needs PyTorch, no R.
export NUM_SEEDS=1  # only need to set 1 seed for the GPU stress test.
export GPU_MEM_LIMIT=10  # Behave as if GPU has 10GB
uv run bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

**Runtime:** ~20 hours on a typical workstation
**Output location:** `replication/paper_performance_benchmarks/runs/<timestamp>/`

---

## Step 5: Monitor GPU Memory Usage

While the benchmark is running, you can monitor real-time GPU memory consumption from a **separate terminal**. The monitoring script polls `nvidia-smi` every second and appends readings to a CSV file.

```bash
bash ./scripts/monitor_gpu.sh ./gpu_metrics.csv
```

The CSV contains the following columns: `timestamp`, `gpu_name`, `memory_used_mb`, `memory_total_mb`, `utilization_percent`. Press `Ctrl+C` to stop monitoring.
