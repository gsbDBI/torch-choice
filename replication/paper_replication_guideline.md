# Replication Material

This guide provides a step-by-step walkthrough for reproducing the empirical results from *Torch-Choice: A PyTorch Package for Large-Scale Choice Modeling with Python* on a Linux machine equipped with an NVIDIA GPU. If you do not have a suitable machine, you can easily access a GPU instance through cloud providers such as Google Cloud, AWS, or Azure. We recommend using the terminal feature on Google Colab, which has already set up the Python and R runtime. There is no strict minimum GPU-memory requirement: the replication scripts automatically choose a batch size based on available GPU memory, so the benchmark should run on most CUDA-enabled NVIDIA GPUs.

The procedure has two parts:

1. **Paper demo replication**: an executable Python script that runs all the commands demonstrated in the paper manuscript: dataset management, model estimation (a conditional logit model on ModeCanada and a nested logit model on House Cooling), and reproduction of the coefficient tables.
2. **Performance benchmarks** (Section 5 of the manuscript): runs the full `torch-choice` (Python) vs `mlogit` (R) benchmark comparing computational runtime for different approaches on the grid across sweeps in records, parameters, and items, and regenerates Figures 1–3.

Total wall-clock: **paper demo ≈ 10 seconds**, **full benchmark ≈ 20 hours** (dominated by R/mlogit, we offer a knob to run the PyTorch only part, you can set `SKIP_R=1` to skip the R part, the torch-choice part should only take up to a few hours). The actual runtime depends on your hardware configuration.

A smoke-test configuration of the benchmark runs in a few minutes if you just want to verify the pipeline before committing to the full run.

---

## Prerequisites

- A Linux machine with an NVIDIA GPU and working CUDA drivers (any modern GPU works).
- `nvidia-smi` on `PATH`.
- Internet access for downloading the package and dependencies.
- **R** with the `mlogit`, `tictoc`, and `stringr` packages installed. The R installation is only needed for Step 5 (the Section 5 benchmark). If `R` is already installed, install the packages from the command line with:
  ```bash
  Rscript -e 'install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")'
  ```

---

## Step 1: Install uv

[uv](https://docs.astral.sh/uv/) is the Python package manager we use for the replication environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

---

## Step 2: Get the replication material

There are two ways to get the replication material onto your machine. Pick whichever matches your situation.

### Path A (default): you already have the replication archive

If you received the replication material directly, for example as a submission attachment, a paper's supplementary materials zip, or a download link from the authors, simply extract it and `cd` into it. You can skip ahead to Step 3.

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
# you can change the directory based on your local file system.
bash ./scripts/create_minimal_replication_release.sh /content/torch-choice-replication
cd /content/torch-choice-replication
# you can delete the original torch-choice repo, we will work on the torch-choice-replication folder from now on.
```

**Expected output:** `Cloning into 'torch-choice'...`, then `[DONE] Minimal replication release built at: <replication-dir>`, where `<replication-dir>` is the directory you chose above. After the final `cd`, the listing should match Path A above.

The `create_minimal_replication_release.sh` helper copies the replication scripts, docs, and `pyproject.toml` (for dependency resolution) into the target directory, deliberately excluding the package source (`torch_choice/`), build artifacts, and developer-only material (`tests/`, `docs/`, `tutorials/`, etc.). See the [File layout](#file-layout-in-the-replication-archive) section at the end of this document for the full kept/excluded list.

---

## Step 3: Install torch-choice from PyPI

This is the load-bearing setup step. The `setup_uv_pypi.sh` helper does three things: creates a fresh `.venv`, installs all dependencies via `uv sync` (which reads `pyproject.toml`), and installs `torch-choice==1.0.7` from PyPI on top.

```bash
bash ./scripts/setup_uv_pypi.sh 1.0.7
```

**Wall-clock:** ~5 minutes (most of it is `torch-scatter` building its C++ extension).

**Expected output (success signals):**

- `[SUCCESS] Virtual environment activated (VIRTUAL_ENV=<replication-dir>/.venv)`
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

The reported `location` must be under `<replication-dir>/.venv/lib/python3.12/site-packages/torch_choice/` (not the local repo or `/usr/...`). If the location is wrong, the install used the wrong Python environment.

**Alternative install without uv:** if you prefer plain pip/conda, run `pip install "torch-choice[complete]==1.0.7"` inside any Python 3.10+ environment. All subsequent steps work identically.

---

## Step 4: Reproduce demo code used in the manuscript

This step runs `paper_demo.py`, which walks through the manuscript's main data-setup and model-estimation examples, then fits the Mode Canada conditional logit model and the House Cooling nested logit model used for the Section 4 coefficient tables.

```bash
bash ./replication/run_paper_demo.sh --no-tensorboard
```

The PyTorch Lightning progress bar is enabled by default. In terminal-in-browser environments such as Google Colab, frequent progress-bar updates can make the browser sluggish. To disable it, set `PAPER_DEMO_PROGRESS_BAR=0`:

```bash
PAPER_DEMO_PROGRESS_BAR=0 bash ./replication/run_paper_demo.sh --no-tensorboard
```

If you are running from a browser-backed terminal such as Google Colab and the training output (e.g., the progress bar) is too noisy, redirect both stdout and stderr to a temporary log file:

```bash
bash ./replication/run_paper_demo.sh --no-tensorboard > /tmp/paper_demo_output.txt 2>&1
```

You can inspect the saved output afterward in a text editor.

**Wall-clock:** ~10 seconds, exact time depending on the hardware configuration.

---

## Step 5: Reproduce the Section 5 performance benchmarks

This step runs the full performance-benchmark grid that produced Figures 1–3 of the manuscript. Here, "grid" means the collection of benchmark cases formed by varying the number of records, model parameters, and choice items. The pipeline:

1. Generates synthetic user/item latents and simulates choices (~3M-record full dataset plus smaller datasets for the grid cases).
2. Times `torch-choice` across the benchmark grid.
3. Times R's `mlogit` on equivalent CSV inputs.
4. Renders the side-by-side comparison PDFs.

### 5a. Smoke test (recommended first, ~3 minutes)

Run a reduced-grid version end-to-end to verify the pipeline before committing to the full run,

```bash
SMOKE_TEST=1 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

`SMOKE_TEST=1` swaps in smaller defaults (`NUM_RECORDS=50000` vs `3000000`, `NUM_SEEDS=2` vs `5`, `NUM_EPOCHS=5` vs `50000`), restricts the grid to the `_small` experiment variants, and prefixes the run directory name with `smoke_`.

If you do **not** have R installed and prefer running the Python-only smoke test instead:

```bash
SKIP_R=1 SMOKE_TEST=1 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

This skips the R `mlogit` benchmark and the comparison-figure rendering, but still verifies data generation and the `torch-choice` benchmark path.

**Expected output:** the script prints `[1/4] Generate synthetic data ...`, `[2/4] Benchmark Torch-Choice ...`, `[3/4] Benchmark R (mlogit) ...`, `[4/4] Visualize results ...`, then ends with `Done.` and the three output paths. A `runs/smoke_<timestamp>/` directory should now exist with CSV outputs in `benchmark_results/`.

### 5b. Full run (~20 hours)

The following command launch the full benchmark comparing the run time of mlogit packagei n R and torch-choice in Python.

```bash
SMOKE_TEST=0 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

If you want to run the full Python-only benchmark without R:

```bash
SKIP_R=1 SMOKE_TEST=0 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

This skips the R `mlogit` benchmark and the comparison-figure rendering. It only demonstrates how `torch-choice` scales as the data size changes, without an explicit comparison to R. **Caveat:** because the Section 5 figures are built from the R output, you cannot full regenerate them when `SKIP_R=1`.

To speed up either mode, lower `NUM_SEEDS`, the number of random-seed repetitions for each data point in the paper (in the paper, we ssed `NUM_SEEDS=5`). For a faster exploratory run, which should only take 20% of time.

```bash
NUM_SEEDS=1 SKIP_R=1 SMOKE_TEST=0 bash ./replication/paper_performance_benchmarks/run_benchmarking.sh
```

Using fewer seeds is useful for debugging or checking that the pipeline runs, but the timing curves will be noisier than the manuscript/release results.

**Wall-clock:** ~20 hours on a typical workstation (16 cores, 128 GB RAM, RTX 3090). The runtime is dominated by R/mlogit's step (3), not by `torch-choice`. We measured this on the reference hardware; you will see different absolute numbers on different machines but the *relative* trends (the actual claim of the paper's Section 5) should be the same.

**Output:** a timestamped directory under `./replication/paper_performance_benchmarks/runs/<timestamp>/` containing:

- `synthetic_data/`: `.pt` datasets for `torch-choice` and `.csv` long-format datasets for R/mlogit
- `benchmark_results/`: one CSV per task: `torch_choice_performance_{task}.csv` and `R_performance_{type}.csv`
- `benchmark_figures/`: the PDFs that correspond to Figures 1–3 of the manuscript

For comparison, the reference outputs from the run we conducted while writing the paper are shipped in `replication/paper_performance_benchmarks/benchmark_results_aurora_20250428/` (CSVs) and `benchmark_figures_20250428/` (PDFs). Each reference CSV records the hardware/software metadata of that run (CPU, GPU, Python, PyTorch, R, package versions, date) for transparency.

### 5c. GPU memory & batch size: auto-detection and manual tuning

The benchmark trains hundreds of models. To keep peak GPU memory predictable across heterogeneous hardware, `torch-choice` selects an effective `batch_size` automatically based on the GPU's available VRAM:


| Detected VRAM | Auto `batch_size` | Peak GPU memory (RTX-3090 reference) |
| ------------- | ----------------- | ------------------------------------ |
| < 8 GB        | 8,192             | ~600 MiB                             |
| 8–12 GB       | 16,384            | ~1,100 MiB                           |
| 12–16 GB      | 32,768            | ~2,200 MiB                           |
| 16–22 GB      | 65,536            | ~4,100 MiB                           |
| ≥ 22 GB       | 131,072           | ~8,100 MiB                           |


**Default behavior: no action needed.** The script reads the actual GPU memory via `torch.cuda.get_device_properties(0).total_memory` at startup and picks the appropriate tier. On a standalone GPU you can run the full benchmark with no extra configuration.

**Manual tuning via `GPU_MEM_LIMIT`.** Export `GPU_MEM_LIMIT=<N>` (in GB) to make the auto-tier logic behave as if the GPU had only `<N>` GB. Three common reasons to use this:

- Sharing the GPU with another process. If you have a 24 GB card but another job is using ~14 GB of it, set `GPU_MEM_LIMIT=10` so the benchmark picks the 10 GB tier (`batch_size=16,384`) and stays out of the way.
- Forcing a specific batch size for runtime comparisons. `BATCH_SIZE=<N>` is also accepted as a direct override and bypasses the tier table entirely.

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

---

## Step 6: Save replication outputs

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

## Appendix: File layout in the replication archive

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