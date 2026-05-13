# Replication Material: Step-by-Step Walkthrough

This guide walks you through reproducing the empirical results in *Torch-Choice: A PyTorch Package for Large-Scale Choice Modeling with Python* end-to-end on a Linux machine with an NVIDIA GPU. You can easily obtain one from a cloud platform such as Google Cloud, AWS, or Azure.

The procedure has two parts:

1. **Paper demo replication** (Sections 4.1.4 and 4.2.3 of the manuscript): fits a conditional logit model on ModeCanada and a nested logit model on House Cooling, and reproduces the coefficient tables.
2. **Performance benchmarks** (Section 5 of the manuscript): runs the full `torch-choice` vs `mlogit` (R) benchmark grid across sweeps in records, parameters, and items, and regenerates Figures 1–3.

Total wall-clock: **paper demo ≈ 10 minutes**, **full benchmark ≈ 20 hours** (dominated by R/mlogit). A smoke-test configuration of the benchmark runs in a few minutes if you just want to verify the pipeline before committing to the full run.

---

## Prerequisites

- A Linux machine with an NVIDIA GPU and working CUDA drivers (any modern GPU works).
- `nvidia-smi` on `PATH`.
- Internet access for downloading the package and dependencies.
- **R** with the `mlogit`, `tictoc`, and `stringr` packages installed (needed for Step 5, the Section 5 benchmark). Install them once with:

  ```r
  # Inside an R session:
  install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")
  ```

---

## Step 1: Install uv

[uv](https://docs.astral.sh/uv/) is the Python package manager we use for the replication environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

**Expected output:** `uv 0.x.y (...)` printed to the terminal.

---

## Step 2: Get the replication material

There are two ways to get the replication material onto your machine. Pick whichever matches your situation.

### Path A (default): you already have the replication archive

If you received the replication material directly, for example as a JSS-style submission attachment, a paper's supplementary materials zip, or a download link from the authors, simply extract it and `cd` into it. You can skip ahead to Step 3.

```bash
# Adjust the filename/path to whatever you received:
unzip torch-choice-replication.zip
cd torch-choice-replication
```

After `cd`, `ls` should show: `gpu_memory_limit_test.md`, `LICENSE`, `pyproject.toml`, `README.md`, `replication/`, `scripts/`, `UV_SETUP.md`. There is **no** `torch_choice/` directory. That's by design (the package itself comes from PyPI in Step 3).

### Path B (alternative): build the archive from the public GitHub repository

If you do not have the archive, you can build it yourself from the public source repository. This produces an identical archive to what Path A would have given you.

```bash
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice
bash ./scripts/create_minimal_replication_release.sh /tmp/torch-choice-replication
cd /tmp/torch-choice-replication
```

**Expected output:** `Cloning into 'torch-choice'...`, then `[DONE] Minimal replication release built at: /tmp/torch-choice-replication`. After the final `cd`, the listing should match Path A above.

The `create_minimal_replication_release.sh` helper copies the replication scripts, docs, and `pyproject.toml` (for dependency resolution) into the target directory, deliberately excluding the package source (`torch_choice/`), build artifacts, and developer-only material (`tests/`, `docs/`, `tutorials/`, etc.). See the [File layout](#file-layout-in-the-replication-archive) section at the end of this document for the full kept/excluded list.

---

## Step 3: Install torch-choice from PyPI

This is the load-bearing setup step. The `setup_uv_pypi.sh` helper does three things: creates a fresh `.venv`, installs all dependencies via `uv sync` (which reads `pyproject.toml`), and installs `torch-choice==1.0.7` from PyPI on top.

```bash
bash ./scripts/setup_uv_pypi.sh 1.0.7
```

**Wall-clock:** ~5 minutes (most of it is `torch-scatter` building its C++ extension).

**Expected output (success signals):**

- `[SUCCESS] Virtual environment activated (VIRTUAL_ENV=/tmp/torch-choice-replication/.venv)`
- `[SUCCESS] Prerequisites installed.`
- `[SUCCESS] torch-choice==1.0.7 installed from PyPI into .venv.`
- `[SUCCESS] Quick model run completed.`
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

**Alternative install without uv:** if you prefer plain pip/conda, run `pip install "torch-choice[complete]==1.0.7"` inside any Python 3.10+ environment. All subsequent steps work identically.

---

## Step 4: Reproduce the paper demo

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

- **CLM training log-likelihood ≈ −1874.34** (manuscript: −1874.343)
- **NLM training log-likelihood ≈ −182.49** (manuscript: −182.492)
- All `***`-significant coefficients match the manuscript's signs and magnitudes within ~5%.
- Same significance pattern (what is `***` in the manuscript is `***` in your run).

Small numerical differences (typically < 5% on the highly significant rows) come from Adam being a first-order optimizer; the conditional logit intercepts are identified only up to a constant, so they may differ by a normalization shift while still implying the same choice probabilities.

---

## Step 5: Reproduce the Section 5 performance benchmarks

This step runs the full performance-benchmark pipeline that produced Figures 1–3 of the manuscript. The pipeline:

1. Generates synthetic user/item latents and simulates choices (~3M-record full dataset plus smaller grids).
2. Times `torch-choice` across experiment grids (records / parameters / items).
3. Times R's `mlogit` on equivalent CSV inputs.
4. Renders the side-by-side comparison PDFs.

### 5a. Smoke test (recommended first, ~3 minutes)

Run a reduced-grid version end-to-end to verify the pipeline before committing to the full run:

```bash
export SMOKE_TEST=1
bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

**Expected output:** the script prints `[1/4] Generate synthetic data ...`, `[2/4] Benchmark Torch-Choice ...`, `[3/4] Benchmark R (mlogit) ...`, `[4/4] Visualize results ...`, then ends with `Done.` and the three output paths. A `runs/smoke_<timestamp>/` directory should now exist with CSV outputs in `benchmark_results/`.

### 5b. Full run (~20 hours)

```bash
export SMOKE_TEST=0
bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

**Wall-clock:** ~20 hours on a typical workstation (16 cores, 128 GB RAM, RTX 3090). The runtime is dominated by R/mlogit's step (3), not by `torch-choice`. We measured this on the reference hardware; you will see different absolute numbers on different machines but the *relative* trends (the actual claim of the paper's Section 5) should be the same.

**Output:** a timestamped directory under `./replication/paper_performance_benchmarks/runs/<timestamp>/` containing:

- `synthetic_data/`: `.pt` datasets for `torch-choice` and `.csv` long-format datasets for R/mlogit
- `benchmark_results/`: one CSV per task: `torch_choice_performance_{task}.csv` and `R_performance_{type}.csv`
- `benchmark_figures/`: the PDFs that correspond to Figures 1–3 of the manuscript

For comparison, the reference outputs from the run we conducted while writing the paper are shipped in `replication/paper_performance_benchmarks/benchmark_results_aurora_20250428/` (CSVs) and `benchmark_figures_20250428/` (PDFs). Each reference CSV records the hardware/software metadata of that run (CPU, GPU, Python, PyTorch, R, package versions, date) for transparency.

### 5c. GPU memory & batch size: auto-detection and manual tuning

The benchmark trains hundreds of models. To keep peak GPU memory predictable across heterogeneous hardware, `torch-choice` selects an effective `batch_size` automatically based on the GPU's available VRAM:

| Detected VRAM | Auto `batch_size` | Peak GPU memory (RTX-3090 reference) |
|---------------|-------------------|--------------------------------------|
| < 8 GB        | 8,192             | ~600 MiB                             |
| 8–12 GB       | 16,384            | ~1,100 MiB                           |
| 12–16 GB      | 32,768            | ~2,200 MiB                           |
| 16–22 GB      | 65,536            | ~4,100 MiB                           |
| ≥ 22 GB       | 131,072           | ~8,100 MiB                           |

**Default behavior: no action needed.** The script reads the actual GPU memory via `torch.cuda.get_device_properties(0).total_memory` at startup and picks the appropriate tier. On a standalone GPU you can run the full benchmark with no extra configuration.

**Manual tuning via `GPU_MEM_LIMIT`.** Export `GPU_MEM_LIMIT=<N>` (in GB) to make the auto-tier logic behave as if the GPU had only `<N>` GB. Three common reasons to use this:

- **Sharing the GPU with another process.** If you have a 24 GB card but another job is using ~14 GB of it, set `GPU_MEM_LIMIT=10` so the benchmark picks the 10 GB tier (`batch_size=16,384`) and stays out of the way.
- **Reproducing the small-GPU code path on a larger card.** Useful for CI validation or when verifying the package's claim that it runs cleanly on a 10 GB GPU (the empirical claim documented in our response to reviewer comments; see `gpu_memory_limit_test.md` for the detailed verification recipe).
- **Forcing a specific batch size for runtime comparisons.** `BATCH_SIZE=<N>` is also accepted as a direct override and bypasses the tier table entirely.

Examples:

```bash
# Default: auto-detect from actual GPU size (recommended for most users).
bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

# Reserve memory for another process: behave as if the GPU has 10 GB.
GPU_MEM_LIMIT=10 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh

# Force a specific batch size regardless of GPU memory.
BATCH_SIZE=32768 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

If you ever see an out-of-memory error during the benchmark, drop the effective tier by one notch: either `GPU_MEM_LIMIT=<smaller>` or `BATCH_SIZE=<smaller>`. The full set of memory-related knobs (allocator config, `expandable_segments`, CPU fallback) is documented in `gpu_memory_limit_test.md`.

### 5d. Skipping R (Python-only benchmarks)

If you don't have R installed and only want to time `torch-choice`, set `SKIP_R=1`:

```bash
SKIP_R=1 SMOKE_TEST=0 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

This runs steps 1 (generate) and 2 (`torch-choice` timing) only, skipping the R benchmarks and the visualization step (which requires R output). Useful for sanity-checking the `torch-choice` half of Section 5 without the 20-hour R run. **Note**: doing this means you cannot regenerate the comparison figures in Section 5.

---

## Step 6: Capture evidence

If you're using this run to back a peer-review response or to validate a release, archive these artifacts:

```bash
mkdir -p ~/replication_evidence

# Paper demo: the regression tables (Step 4)
# Copy the printed tables from the terminal, or grep them out of the run log
# if you redirected output to a file.

# Section 5 benchmark: the timestamped run directory (Step 5b)
cp -a ./replication/paper_performance_benchmarks/runs/<your_timestamp>/ \
       ~/replication_evidence/benchmark_run/
```

The contents of `runs/<timestamp>/benchmark_results/` are the source-of-truth CSVs that produced the manuscript figures; keeping them alongside the rendered PDFs and the printed demo tables gives a future reader everything they need to verify the empirical claims.

---

## Reference numbers (paper demo)

The benchmark wall-clock numbers depend on your hardware and are not expected to match the manuscript values exactly; the *relative* speed-ups (`torch-choice` versus R `mlogit`) are the actual claim. The paper demo's coefficient table, in contrast, should match closely:

| Quantity                                                       | Value         |
| -------------------------------------------------------------- | ------------- |
| CLM training log-likelihood (ModeCanada, 50,000 Adam epochs)   | **−1874.343** |
| NLM training log-likelihood (House Cooling, 1,000 Adam epochs) | **−182.492**  |
| CLM number of coefficients                                     | 13            |
| NLM `lambda_weight_0`                                          | 0.332         |

Coefficient signs and significance levels should match the manuscript tables exactly. Magnitudes should match to within ~5% on `***`-significant rows.

---

## Common pitfalls

Three sharp edges the test procedure has hit in practice:

### 1. `uv run bash <wrapper>` clobbers the PyPI install

When you invoke the run wrappers, use **plain `bash`**, not `uv run bash`. The wrapper internally uses `uv run python` with `UV_NO_SYNC=1` exported, which protects it from auto-sync. But an outer `uv run` runs *before* the wrapper's environment is set up, and that outer call will detect `pyproject.toml`'s `[project] name = "torch-choice"` and build an empty wheel from the metadata (since the replication archive has no `torch_choice/` source), installing it over the PyPI version we want.

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

If you ever restrict the benchmark to a single experiment (via `TORCH_EXPERIMENT_NAME=...`) and also restrict generation (via `GENERATE_EXPERIMENTS=...`), the two CLIs use different vocabularies for the same dataset. The generator's `ALL_EXPERIMENTS` set uses these names:

| Benchmark experiment (`TORCH_EXPERIMENT_NAME`) | Generate experiment (`GENERATE_EXPERIMENTS`) |
| ---------------------------------------------- | -------------------------------------------- |
| `num_records_experiment_small`                 | `num_records_experiment_small`               |
| `num_records_experiment_large`                 | **`full_dataset`**                           |
| `num_params_experiment_small`                  | `num_params_experiment_small`                |
| `num_params_experiment_large`                  | **`full_dataset`** (shared)                  |
| `num_items_experiment_small`                   | `num_items_experiment_small`                 |
| `num_items_experiment_large`                   | `num_items_experiment_large`                 |

The default (`GENERATE_EXPERIMENTS=all`) generates everything, so most replicators never hit this. It only bites if you set `GENERATE_EXPERIMENTS` to a benchmark-style name.

**Diagnostic signal:** the generate step finishes without printing any `[Saved] ...` lines, then the benchmark crashes with `FileNotFoundError: ... simulated_choice_data_<name>_seed_42.pt`.

### 3. LBFGS produces NaN on PyTorch ≥ 2.11

Earlier drafts of `paper_demo.py` used `model_optimizer="LBFGS"`, which is numerically unstable on this small dataset starting in PyTorch 2.11 (produces NaN values on both CPU and GPU). The current demo uses `model_optimizer="Adam"` with `learning_rate=0.0005` for 50,000 epochs, which is stable across all PyTorch versions and reaches a marginally tighter training fit than the original LBFGS recipe.

**Diagnostic signal:** if you somehow run the LBFGS path and see a coefficient table full of `nan` and `inf`, switch to Adam.

---

## File layout in the replication archive

After Step 2, the archive contains:

```
torch-choice-replication/
├── LICENSE
├── README.md
├── UV_SETUP.md
├── gpu_memory_limit_test.md             # appendix: detailed GPU memory verification
├── pyproject.toml                        # dependency manifest (used by setup_uv_pypi.sh)
├── replication/
│   ├── car_choice.csv                    # sample dataset used in Section 3
│   ├── paper_demo.py                     # Section 4 demo (CLM + NLM)
│   ├── paper_demo.ipynb                  # same demo as a Jupyter notebook
│   ├── paper_replication_guideline.md    # this file
│   ├── paper_performance_benchmarks/     # Section 5 benchmark pipeline
│   │   ├── run_benchmarking.sh
│   │   ├── paper_performance_benchmark.py
│   │   ├── run_mlogit_experiments.R
│   │   ├── steps/                        # step01 (generate) / step02 (torch-choice) / step03 (visualize)
│   │   ├── benchmark_results_aurora_20250428/   # reference CSVs from the paper run
│   │   └── benchmark_figures_20250428/          # reference PDFs
│   ├── run_paper_demo.sh                 # Section 4 demo wrapper
│   └── run_paper_demo_output.txt         # reference output for cross-checking
└── scripts/
    ├── setup_uv_pypi.sh                  # Step 3 (install torch-choice from PyPI)
    └── monitor_gpu.sh                    # GPU memory monitor (used by gpu_memory_limit_test.md)
```

`torch_choice/` is *not* present. That's by design. The package source comes from PyPI in Step 3.
