#!/usr/bin/env bash
# All-in-one benchmark driver (serves as an executable README).
# Runs the full pipeline in order:
#   1) Generate synthetic data (writes .pt + .csv)
#   2) Benchmark Torch-Choice (writes torch_choice_performance_*.csv)
#   3) Benchmark R / mlogit (writes R_performance_*.csv)
#   4) Visualize (v2) (writes PDFs)
#
# Assumptions:
# - Python env with torch-choice and dependencies available.
# - R env with mlogit/tictoc/stringr installed.
# - You are in `replication/paper_performance_benchmarks/` or paths below resolve.
#
# Notes on uv + R:
# - uv manages the Python environment only, and uv is **required** for the Python steps.
# - The R step still uses your system `Rscript` on PATH.
# - Verify R is installed:
#     Rscript --version
# - This script expects required R packages (mlogit, tictoc, stringr) to be pre-installed.
#   See replication/paper_performance_benchmarks/README.md for the install command:
#   Rscript -e 'install.packages(c("mlogit","tictoc","stringr"), repos="https://cloud.r-project.org")'
#
# Usage:
#   chmod +x run_benchmarking.sh
#   ./run_benchmarking.sh
#
# Tuning knobs (set env vars before running):
#   RUN_PATH (recommended), or DATA_PATH / RESULTS_PATH / FIGURES_PATH
#   NUM_RECORDS, NUM_SEEDS, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE
#     DEVICE can be auto|cpu|cuda
#   SMOKE_TEST
#     SMOKE_TEST=1 runs a small configuration that still produces CSV outputs
#     (few representative values per experiment, not the full grids).
#   SKIP_R
#     SKIP_R=1 skips the R benchmark and visualization steps (PyTorch only).
#     Useful for GPU memory testing when R is not needed.
# Example:
#   RUN_PATH=/tmp/bench_run DEVICE=cuda ./run_benchmarking.sh
#   SKIP_R=1 DEVICE=cuda ./run_benchmarking.sh  # PyTorch only

set -euo pipefail

# Resolve this script's directory (path) to anchor relative paths.
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# uv is required for the Python steps.
if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] 'uv' is required but was not found. Install uv and retry."
  echo "See UV_SETUP.md or: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
# Suppress uv warning for the experimental extra-build-dependencies feature used in pyproject.toml.
PYTHON_CMD="uv run --preview-features extra-build-dependencies python"

# By default, write all outputs into a timestamped run directory under this folder.
# You can override RUN_PATH (recommended), or override DATA_PATH / RESULTS_PATH / FIGURES_PATH individually.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# Smoke test: run a minimal configuration (but still execute and write outputs).
SMOKE_TEST="${SMOKE_TEST:-0}"

# Knobs for generation and training.
if [[ "${SMOKE_TEST}" == "1" ]]; then
  # Make the run ID clearly marked as a smoke run.
  RUN_ID="smoke_${RUN_ID}"
  DEFAULT_NUM_RECORDS=50000
  DEFAULT_NUM_SEEDS=2
  DEFAULT_NUM_EPOCHS=5
  DEFAULT_GENERATE_EXPERIMENTS="num_records_experiment_small num_params_experiment_small num_items_experiment_small"
  DEFAULT_TORCH_EXPERIMENT_NAME="all"
  DEFAULT_R_EXPERIMENT_TYPE="all"
else
  DEFAULT_NUM_RECORDS=3000000
  DEFAULT_NUM_SEEDS=5
  DEFAULT_NUM_EPOCHS=50000
  DEFAULT_GENERATE_EXPERIMENTS="all"
  DEFAULT_TORCH_EXPERIMENT_NAME="all"
  DEFAULT_R_EXPERIMENT_TYPE="all"
fi

RUN_PATH="${RUN_PATH:-${SCRIPT_PATH}/runs/${RUN_ID}}"

# Paths for inputs/outputs.
DATA_PATH="${DATA_PATH:-${RUN_PATH}/synthetic_data}"
RESULTS_PATH="${RESULTS_PATH:-${RUN_PATH}/benchmark_results}"
FIGURES_PATH="${FIGURES_PATH:-${RUN_PATH}/benchmark_figures}"

NUM_RECORDS="${NUM_RECORDS:-${DEFAULT_NUM_RECORDS}}"
NUM_SEEDS="${NUM_SEEDS:-${DEFAULT_NUM_SEEDS}}"
NUM_EPOCHS="${NUM_EPOCHS:-${DEFAULT_NUM_EPOCHS}}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-}"  # Empty string signals auto-detection; set explicit value to override.
DEVICE="${DEVICE:-auto}"  # auto|cpu|cuda

# Reduce CUDA memory fragmentation on smaller GPUs.
# `PYTORCH_CUDA_ALLOC_CONF` is the universally-supported legacy name (honored on
# all PyTorch versions); the newer alias `PYTORCH_ALLOC_CONF` is silently ignored
# on PyTorch < 2.5, so we set the legacy name to match the recommendation embedded
# in the OOM error message itself.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Skip R benchmark (PyTorch only mode).
SKIP_R="${SKIP_R:-0}"

# Optional: restrict experiments for quicker smoke tests.
GENERATE_EXPERIMENTS="${GENERATE_EXPERIMENTS:-${DEFAULT_GENERATE_EXPERIMENTS}}"  # space-separated list or "all"
TORCH_EXPERIMENT_NAME="${TORCH_EXPERIMENT_NAME:-${DEFAULT_TORCH_EXPERIMENT_NAME}}"  # e.g. "num_records_experiment_small"
R_EXPERIMENT_TYPE="${R_EXPERIMENT_TYPE:-${DEFAULT_R_EXPERIMENT_TYPE}}"  # items|records|params|all

check_prereqs() {
  # Fail fast before spending time generating data / training models.
  if ! command -v python >/dev/null 2>&1; then
    echo "[ERROR] 'python' not found on PATH."
    echo "If you are using uv, run: uv run bash ${SCRIPT_PATH}/run_benchmarking.sh"
    exit 1
  fi

  # Only check R if not skipping R benchmark.
  if [[ "${SKIP_R}" != "1" ]]; then
    if ! command -v Rscript >/dev/null 2>&1; then
      echo "[ERROR] 'Rscript' not found on PATH."
      echo "Install R and ensure Rscript is available, then re-run."
      echo "Verify with: Rscript --version"
      exit 1
    fi

    # Check required R packages (mlogit, tictoc, stringr); fail fast if missing.
    missing_pkgs="$(Rscript -e 'pkgs <- c(\"mlogit\",\"tictoc\",\"stringr\"); missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]; cat(missing, sep=" ")' 2>/dev/null || true)"
    if [[ -n "${missing_pkgs}" ]]; then
      echo "[ERROR] Missing required R packages: ${missing_pkgs}"
      echo "Install them manually, e.g.:"
      echo "  Rscript -e 'install.packages(c(\"mlogit\",\"tictoc\",\"stringr\"), repos=\"https://cloud.r-project.org\")'"
      echo "See replication/paper_performance_benchmarks/README.md for details."
      exit 1
    fi
  fi
}

main() {
  # Expand experiment lists safely (space-separated).
  read -r -a GENERATE_EXPERIMENTS_ARR <<< "${GENERATE_EXPERIMENTS}"

  echo "Run directory: ${RUN_PATH}"
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "Mode: SMOKE_TEST=1 (reduced representative values per experiment; produces CSV outputs)"
  fi
  if [[ "${SKIP_R}" == "1" ]]; then
    echo "Mode: SKIP_R=1 (PyTorch only, skipping R benchmarks)"
  fi

  check_prereqs

  GEN_EXTRA_ARGS=()
  TORCH_EXTRA_ARGS=()
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    GEN_EXTRA_ARGS+=(--smoke-test)
    TORCH_EXTRA_ARGS+=(--smoke-test)
    R_ENV_PREFIX=(env "SMOKE_TEST=1")
  else
    R_ENV_PREFIX=()
  fi

  echo "[1/4] Generate synthetic data -> ${DATA_PATH}"
  ${PYTHON_CMD} "${SCRIPT_PATH}/paper_performance_benchmark.py" generate \
    --output-path "${DATA_PATH}" \
    --experiments "${GENERATE_EXPERIMENTS_ARR[@]}" \
    --num-records "${NUM_RECORDS}" \
    --skip-plots \
    ${GEN_EXTRA_ARGS[@]+"${GEN_EXTRA_ARGS[@]}"}

  echo "[2/4] Benchmark Torch-Choice -> ${RESULTS_PATH}"
  ${PYTHON_CMD} "${SCRIPT_PATH}/paper_performance_benchmark.py" torch-choice \
    --data-path "${DATA_PATH}" \
    --output-path "${RESULTS_PATH}" \
    --experiment-name "${TORCH_EXPERIMENT_NAME}" \
    --device "${DEVICE}" \
    --num-seeds "${NUM_SEEDS}" \
    --num-epochs "${NUM_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"} \
    ${TORCH_EXTRA_ARGS[@]+"${TORCH_EXTRA_ARGS[@]}"}

  if [[ "${SKIP_R}" != "1" ]]; then
    echo "[3/4] Benchmark R (mlogit) -> ${RESULTS_PATH}"
    ${R_ENV_PREFIX[@]+"${R_ENV_PREFIX[@]}"} Rscript "${SCRIPT_PATH}/run_mlogit_experiments.R" \
      "${R_EXPERIMENT_TYPE}" \
      "${DATA_PATH}" \
      "${RESULTS_PATH}" \
      "${NUM_SEEDS}"

    echo "[4/4] Visualize results -> ${FIGURES_PATH}"
    ${PYTHON_CMD} "${SCRIPT_PATH}/paper_performance_benchmark.py" visualize \
      --torch-results "${RESULTS_PATH}" \
      --r-results "${RESULTS_PATH}" \
      --output-path "${FIGURES_PATH}"
  else
    echo "[3/4] Skipping R benchmark (SKIP_R=1)"
    echo "[4/4] Skipping visualization (requires R results)"
  fi

  echo "Done."
  echo "  Data   : ${DATA_PATH}"
  echo "  Results: ${RESULTS_PATH}"
  echo "  Figures: ${FIGURES_PATH}"
}

main "$@"