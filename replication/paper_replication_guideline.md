# Replication Material — Step-by-Step Walkthrough

This guide walks you through reproducing the empirical results in *Torch-Choice: A PyTorch Package for Large-Scale Choice Modeling with Python* end-to-end on a fresh Linux machine with an NVIDIA GPU. The procedure has two parts:

1. **Paper demo replication** (Sections 4.1.4 and 4.2.3 of the manuscript) — fits a conditional logit model on ModeCanada and a nested logit model on House Cooling, and reproduces the coefficient tables.
2. **GPU memory test** — verifies that the package runs cleanly on a small (10 GB) GPU, the empirical claim documented in our response to reviewer comments about memory requirements.

Total wall-clock: **~30 minutes** end-to-end.

---

## Quick start (one command)

If you just want to verify everything works, the entire procedure below is bundled in a single helper script:

```bash
# On the GPU machine, no prior clone needed:
bash <(curl -sSL https://raw.githubusercontent.com/gsbDBI/torch-choice/main/scripts/run_full_replication_test.sh)
```

It bootstraps from a fresh clone, installs the package from PyPI, runs both the paper demo and the GPU memory test, and archives logs + a GPU metrics CSV under `/tmp/torch-choice-full-replication-test/evidence/`. Skip ahead to the [Reference numbers](#reference-numbers) section to see what the artifacts should contain.

The rest of this document walks through the same procedure manually, one step at a time, for users who want fine-grained control or are diagnosing a problem.

---

## Prerequisites

- A Linux machine with an NVIDIA GPU and working CUDA drivers (any modern GPU works; the GPU memory test uses `GPU_MEM_LIMIT` to simulate a smaller card).
- `nvidia-smi` on `PATH` (used for memory monitoring).
- Internet access (for downloading the package and dependencies).

You do **not** need R installed unless you want to run the full Section 5 benchmark grid (Step 8 below, optional).

---

## Step 1: Install uv

[uv](https://docs.astral.sh/uv/) is the Python package manager we use for the replication environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

**Expected output:** `uv 0.x.y (...)` printed to the terminal.

---

## Step 2: Clone the repository

```bash
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice
```

**Expected output:** `Cloning into 'torch-choice'...` followed by the clone progress. The current directory should be `.../torch-choice` after running these commands.

---

## Step 3: Build the minimal replication archive

The replication material is meant to be self-contained: it should not include a copy of the `torch_choice` package source (that comes from PyPI in the next step). The `create_minimal_replication_release.sh` helper builds a stripped archive containing only what a replicator needs.

```bash
bash ./scripts/create_minimal_replication_release.sh /tmp/torch-choice-replication
cd /tmp/torch-choice-replication
```

**Expected output:**

- `[DONE] Minimal replication release built at: /tmp/torch-choice-replication`
- After `cd`, `ls` should show: `gpu_memory_limit_test.md`, `LICENSE`, `pyproject.toml`, `README.md`, `replication/`, `scripts/`, `UV_SETUP.md` (no `torch_choice/` directory — that's what we want).

---

## Step 4: Install torch-choice from PyPI

This is the load-bearing step. The `setup_uv_pypi.sh` helper does three things: creates a fresh `.venv`, installs all dependencies via `uv sync` (which reads `pyproject.toml`), and installs `torch-choice==1.0.7` from PyPI on top.

```bash
bash ./scripts/setup_uv_pypi.sh 1.0.7
```

**Wall-clock:** ~5 minutes (most of it is `torch-scatter` building its C++ extension).

**Expected output (success signals):**

- `[SUCCESS] Virtual environment activated (VIRTUAL_ENV=/tmp/torch-choice-replication/.venv)`
- `[SUCCESS] Prerequisites installed.`
- `[SUCCESS] torch-choice==1.0.7 installed from PyPI into .venv.`
- `[SUCCESS] Quick model run completed.` (a smoke fit on ModeCanada to verify the install)
- `[SUCCESS] All done.`

**Verify the install location:**

```bash
source .venv/bin/activate
python -c "
import torch_choice
print('version :', torch_choice.__version__)
print('location:', torch_choice.__file__)
"
deactivate
```

The reported `location` must be under `/tmp/torch-choice-replication/.venv/lib/python3.12/site-packages/torch_choice/` (not the local repo or `/usr/...`). If the location is wrong, the install used the wrong Python environment.

**Alternative install without uv:** If you prefer plain pip/conda, run `pip install "torch-choice[complete]==1.0.7"` inside any Python 3.10+ environment. All subsequent steps work identically.

---

## Step 5: Reproduce the paper demo

This step reproduces the coefficient tables in Sections 4.1.4 (Mode Canada CLM) and 4.2.3 (House Cooling NLM) of the manuscript.

```bash
bash ./replication/run_paper_demo.sh --no-tensorboard
```

**Wall-clock:** ~5–10 minutes (50,000 Adam epochs on the small ModeCanada dataset).

**Expected output:** at the end of the run, two regression tables.

For the **conditional logit model (Section 4.1.4)**:

```
Log-likelihood: [Training] -1874.343, [Validation] None, [Test] None

| Coefficient                           |   Estimation |   Std. Err. |   z-value | Pr(>|z|)   | Significance |
|:--------------------------------------|-------------:|------------:|----------:|:-----------|:-------------|
| itemsession_cost_freq_ovt[constant]_0 |  -0.0334045  |  0.00709508 |    -4.708 | 2.500e-06  | ***          |
| itemsession_cost_freq_ovt[constant]_1 |   0.0925443  |  0.00509738 |    18.155 | < 2e-16    | ***          |
| itemsession_cost_freq_ovt[constant]_2 |  -0.0430018  |  0.00322449 |   -13.336 | < 2e-16    | ***          |
| session_income[item]_0                |  -0.0890672  |  0.0183458  |    -4.855 | 1.205e-06  | ***          |
| session_income[item]_1                |  -0.0279754  |  0.00387223 |    -7.225 | 5.025e-13  | ***          |
| session_income[item]_2                |  -0.0381287  |  0.00408253 |    -9.339 | < 2e-16    | ***          |
| itemsession_ivt[item-full]_0          |   0.059507   |  0.0100726  |     5.908 | 3.466e-09  | ***          |
| itemsession_ivt[item-full]_2          |  -0.00645034 |  0.00189833 |    -3.398 | 6.790e-04  | ***          |
| ... (remaining rows match the manuscript Section 4.1.4 table) |
```

For the **nested logit model (Section 4.2.3)**:

```
Log-likelihood: [Training] -182.492, ...
| lambda_weight_0   |  0.332282 | ...
| item_price_obs[constant]_0 | -0.317104 | ...
| ... (matches Section 4.2.3 table)
```

**What to verify:**

- **CLM training log-likelihood ≈ −1874.34** (paper: −1874.343)
- **NLM training log-likelihood ≈ −182.49** (paper: −182.492)
- All `***`-significant coefficients match the manuscript's signs and magnitudes within ~5%
- Same significance pattern (i.e., what is `***` in the paper is `***` in your run)

Small numerical differences (typically < 5% on the highly significant rows) come from Adam being a first-order optimizer; the conditional logit intercepts are identified up to a constant, so they may differ by a normalization shift while still implying the same choice probabilities.

---

## Step 6: Run the GPU memory test

This step verifies the small-GPU claim. We use `GPU_MEM_LIMIT=10` to make the auto-batch-size logic behave as if the GPU has only 10 GB, regardless of the actual hardware.

### 6a. Sanity-check the tier selection (5 seconds)

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

**Expected output:**

```
[Auto batch size] GPU: <name> (actual: <real> GB, limit: 10.0 GB via GPU_MEM_LIMIT)
[Auto batch size] GPU: <name> (10.0 GB) -> batch_size=16,384
selected: 16384
```

If `selected` is anything other than `16384`, the tier logic regressed — stop and investigate before continuing.

### 6b. Run the targeted large-grid memory test (~20 min)

In one shell:

```bash
# Background a GPU memory monitor that writes one row/sec to a CSV
bash ./scripts/monitor_gpu.sh ./gpu_metrics_10gb.csv &
MONITOR_PID=$!

# Run the benchmark with the simulated 10 GB cap. NUM_SEEDS=1 cuts runtime
# without losing memory information. GENERATE_EXPERIMENTS=full_dataset
# tells the data-generation step to produce the file the large grid reads.
GPU_MEM_LIMIT=10 SMOKE_TEST=0 SKIP_R=1 DEVICE=cuda \
NUM_SEEDS=1 \
GENERATE_EXPERIMENTS=full_dataset \
TORCH_EXPERIMENT_NAME=num_records_experiment_large \
  bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

kill $MONITOR_PID
```

**Wall-clock:** ~20 minutes (3 min generating the 3M-record synthetic dataset, then training across 8 sample sizes × 3 formulas × 1 seed).

**Important pitfalls:**

- **Use plain `bash <wrapper>`, NOT `uv run bash <wrapper>`.** An outer `uv run` triggers a project auto-sync that overwrites the PyPI-installed `torch-choice` in `.venv` with an empty wheel built from `pyproject.toml`. The wrapper handles its own `uv run python` internally and exports `UV_NO_SYNC=1` for those.
- **Use `GENERATE_EXPERIMENTS=full_dataset`, NOT `num_records_experiment_large`.** The data-generation step and the benchmark step use different names for the same dataset — the generator's name is `full_dataset`, the benchmark's sweep name is `num_records_experiment_large`. Mismatched names silently produce zero output files, then the benchmark crashes with a missing-file error.

### 6c. Extract the peak GPU memory

```bash
awk -F',' 'NR>1 { gsub(/ MiB/, "", $3); if ($3+0 > max) max=$3+0 } END { print "peak GPU memory used:", max, "MiB" }' gpu_metrics_10gb.csv
```

**Expected output:** `peak GPU memory used: ~1,100 MiB` on RTX-3090-class hardware (somewhat higher on newer architectures — see the [Reference numbers](#reference-numbers) section for details). The headline takeaway is that the peak stays well under the 10 GB simulated ceiling, even with auto-batch=16,384.

---

## Step 7: Capture evidence

If you're using this run to back a peer-review response or to validate a release, archive these four artifacts:

```bash
mkdir -p ~/replication_evidence
# Demo training table (Step 5)
cp /tmp/torch-choice-replication/lightning_logs ~/replication_evidence/  # or copy the printed table from the terminal
# Auto-batch line (Step 6a/6b)
grep -m1 -- "-> batch_size=" /path/to/your/benchmark.log > ~/replication_evidence/auto_batch.txt
# Peak memory output (Step 6c)
awk -F',' 'NR>1 { gsub(/ MiB/, "", $3); if ($3+0 > max) max=$3+0 } END { print "peak GPU memory used:", max, "MiB" }' gpu_metrics_10gb.csv > ~/replication_evidence/peak_memory.txt
# Full per-second metrics CSV
cp gpu_metrics_10gb.csv ~/replication_evidence/
```

These are the artifacts a reviewer (or future-you) can re-inspect to verify the replication.

---

## Step 8: (Optional) Run the full Section 5 performance benchmarks

The replication archive also includes the full performance-benchmark pipeline used to generate Figures 1–3 of the manuscript. This pipeline compares `torch-choice` against R's `mlogit` across sweeps over records / parameters / items. **It is not required to verify the paper's empirical claims** — the steps above already do that — but it is here if you want to reproduce the figures.

**Prerequisites for this step:**

```r
# Inside R:
install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")
```

**Runtime:** ~20 hours on a typical workstation (16 cores, 128 GB RAM, RTX 3090). The runtime is dominated by R/mlogit step (3), not by `torch-choice`.

**Quick smoke test first** (verifies the pipeline runs end-to-end with reduced grids; few minutes):

```bash
export SMOKE_TEST=1
bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

**Full run:**

```bash
export SMOKE_TEST=0
bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

By default, each run writes to a timestamped directory under `./replication/paper_performance_benchmarks/runs/<timestamp>/` containing the synthetic datasets (`synthetic_data/`), benchmark CSVs (`benchmark_results/`), and figures (`benchmark_figures/`).

---

## Reference numbers

### Paper demo (Step 5)

| Quantity | Value |
|---|---:|
| CLM training log-likelihood (ModeCanada, 50,000 Adam epochs) | **−1874.343** |
| NLM training log-likelihood (House Cooling, 1,000 Adam epochs) | **−182.492** |
| CLM number of coefficients | 13 |
| NLM `lambda_weight_0` | 0.332 |

Coefficient signs and significance levels should match the manuscript tables exactly. Magnitudes should match to within ~5% on `***`-significant rows.

### GPU memory test (Step 6)

Empirical table (RTX 3090 + PyTorch 2.5.x baseline; newer architectures may consume somewhat more):

| `GPU_MEM_LIMIT` | Auto `batch_size` | Expected peak |
|-----------------|-------------------|---------------|
| 6  (< 8 GB)     | 8,192             | ~600 MiB      |
| **10 (8–12 GB)**| **16,384**        | **~1,100 MiB**|
| 14 (12–16 GB)   | 32,768            | ~2,200 MiB    |
| 18 (16–22 GB)   | 65,536            | ~4,100 MiB    |
| 24 (≥ 22 GB)    | 131,072           | ~8,100 MiB    |

A reading within ±20% of the table value is normal (`nvidia-smi` samples once per second and may miss exact peaks). On Blackwell-class GPUs with CUDA 13, we observe peaks ~1,800 MiB at the 10 GB tier — higher than the RTX 3090 baseline but still well under the 10 GB ceiling.

---

## Common pitfalls

These are the three sharp edges the test procedure has hit in practice:

### 1. `uv run bash <wrapper>` clobbers the PyPI install

When you invoke the run wrappers, use **plain `bash`**, not `uv run bash`. The wrapper internally uses `uv run python` with `UV_NO_SYNC=1` exported, which protects it from auto-sync. But an outer `uv run` runs *before* the wrapper's environment is set up, and that outer call will detect `pyproject.toml`'s `[project] name = "torch-choice"` and build an empty wheel from the metadata (since the replication archive has no `torch_choice/` source) — installing it over the PyPI version we want.

**Diagnostic signal:** if you see this triplet at the start of a wrapper run, the trap was sprung:

```
Built torch-choice @ file:///...
Uninstalled 1 package in <N>ms
Installed 1 package in <N>ms
```

**Recovery:** re-install torch-choice from PyPI:

```bash
uv pip install --python .venv/bin/python --no-deps "torch-choice==1.0.7" --force-reinstall
rm -rf torch_choice.egg-info
```

### 2. Generator and benchmark use different names for the same dataset

If you restrict the generate step to a benchmark-style name (e.g. `GENERATE_EXPERIMENTS=num_records_experiment_large`), the generator silently produces nothing — its `ALL_EXPERIMENTS` set uses different names. Mapping:

| Benchmark experiment (`TORCH_EXPERIMENT_NAME`) | Generate experiment (`GENERATE_EXPERIMENTS`) |
|---|---|
| `num_records_experiment_small`  | `num_records_experiment_small` |
| `num_records_experiment_large`  | **`full_dataset`** |
| `num_params_experiment_small`   | `num_params_experiment_small` |
| `num_params_experiment_large`   | **`full_dataset`** (shared) |
| `num_items_experiment_small`    | `num_items_experiment_small` |
| `num_items_experiment_large`    | `num_items_experiment_large` |

**Diagnostic signal:** the generate step finishes without printing any `[Saved] ...` lines, then the benchmark crashes with `FileNotFoundError: ... simulated_choice_data_<name>_seed_42.pt`.

### 3. LBFGS produces NaN on PyTorch ≥ 2.11

Earlier drafts of the demo used `model_optimizer="LBFGS"`, which is numerically unstable on this small dataset starting in PyTorch 2.11 (produces NaN values on both CPU and GPU). The current demo uses `model_optimizer="Adam"` with `learning_rate=0.0005` for 50,000 epochs, which is stable across all PyTorch versions and reaches a marginally tighter training fit than the original LBFGS recipe.

**Diagnostic signal:** if you somehow run the LBFGS path and see a coefficient table full of `nan` and `inf`, switch to Adam.

---

## File layout in the replication archive

After Step 3, the replication archive contains:

```
torch-choice-replication/
├── LICENSE
├── README.md
├── UV_SETUP.md
├── gpu_memory_limit_test.md            # detailed GPU memory test guide
├── pyproject.toml                       # dependency manifest (used by setup_uv_pypi.sh)
├── replication/
│   ├── car_choice.csv                   # sample dataset used in Section 3
│   ├── paper_demo.py                    # demo script reproducing Sections 4.1.4 + 4.2.3
│   ├── paper_demo.ipynb                 # same demo as a Jupyter notebook
│   ├── paper_replication_guideline.md   # this file
│   ├── paper_performance_benchmarks/    # Section 5 benchmark pipeline
│   │   ├── run_benchmarking.sh
│   │   ├── paper_performance_benchmark.py
│   │   ├── run_mlogit_experiments.R
│   │   ├── steps/                       # step01 (generate) / step02 (torch-choice) / step03 (visualize)
│   │   ├── benchmark_results_aurora_20250428/  # reference CSVs from the paper run
│   │   └── benchmark_figures_20250428/         # reference PDFs
│   ├── run_paper_demo.sh                # Section 4 demo wrapper
│   └── run_paper_demo_output.txt        # reference output for cross-checking
└── scripts/
    ├── setup_uv_pypi.sh                 # Step 4 (install torch-choice from PyPI)
    ├── monitor_gpu.sh                   # GPU memory monitor used in Step 6b
    └── (run_full_replication_test.sh)   # the all-in-one entry point (in the source clone, not this archive)
```

`torch_choice/` is *not* present — that's by design. The package source comes from PyPI in Step 4.
