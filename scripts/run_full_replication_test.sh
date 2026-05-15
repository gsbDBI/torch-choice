#!/bin/bash
# run_full_replication_test.sh — End-to-end replication validation.
#
# One script, two verifications:
#   (1) Paper demo replication (Section 4.1.4 CLM on ModeCanada + 4.2.3 NLM
#       on House Cooling). LBFGS optimizer (lr=0.01, 1,000 epochs);
#       log-likelihood ~ -1874.34 expected.
#   (2) GPU memory cap test (paper_performance_benchmarks). With
#       GPU_MEM_LIMIT=10 the auto-batch-size logic selects batch_size=16,384
#       and peak GPU memory should land near ~1,100 MiB on RTX-3090-class
#       hardware (slightly higher on newer architectures; the only failure
#       mode is going over the 10 GB simulated ceiling).
#
# Usage (run from any directory; this script bootstraps from a fresh clone):
#   bash run_full_replication_test.sh
#
# Customize via env vars (with defaults shown):
#   GPU_MEM_LIMIT=10                                      # simulated GPU mem cap
#   TORCH_CHOICE_VERSION=1.0.7                            # torch-choice version to install
#   WORK_DIR=/tmp/torch-choice-full-replication-test      # where everything lives
#   SKIP_DEMO=0                                           # set to 1 to skip phase 5 (paper demo)
#   SKIP_MEMORY_TEST=0                                    # set to 1 to skip phase 6 (GPU memory test)

set -e

GPU_MEM_LIMIT="${GPU_MEM_LIMIT:-10}"
TORCH_CHOICE_VERSION="${TORCH_CHOICE_VERSION:-1.0.7}"
WORK_DIR="${WORK_DIR:-/tmp/torch-choice-full-replication-test}"
SKIP_DEMO="${SKIP_DEMO:-0}"
SKIP_MEMORY_TEST="${SKIP_MEMORY_TEST:-0}"

CLONE_DIR="${WORK_DIR}/torch-choice-source"
ARCHIVE_DIR="${WORK_DIR}/torch-choice-replication"
EVIDENCE_DIR="${WORK_DIR}/evidence"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

echo
echo "=================================================================="
echo "  torch-choice end-to-end replication validation"
echo "=================================================================="
echo "  Version:        ${TORCH_CHOICE_VERSION}"
echo "  GPU mem limit:  ${GPU_MEM_LIMIT} GB (simulated)"
echo "  Work dir:       ${WORK_DIR}"
echo "  Skip demo:      ${SKIP_DEMO}"
echo "  Skip mem test:  ${SKIP_MEMORY_TEST}"
echo "=================================================================="
echo

# Refuse to clobber existing state.
if [[ -e "${WORK_DIR}" ]]; then
    echo -e "${RED}[ERROR]${NC} WORK_DIR already exists: ${WORK_DIR}"
    echo "        Remove it first, or pick a different WORK_DIR."
    exit 1
fi
mkdir -p "${WORK_DIR}" "${EVIDENCE_DIR}"

# --- Prerequisite checks ----------------------------------------------------
echo -e "${BLUE}[CHECK]${NC} uv installed..."
if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} uv not found. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}         ${NC}$(uv --version)"

if [[ "${SKIP_MEMORY_TEST}" != "1" ]]; then
    echo -e "${BLUE}[CHECK]${NC} nvidia-smi available..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}[WARN]${NC}  nvidia-smi not on PATH — phase 6 (GPU mem test) will fail."
    fi
fi

# --- Phase 1: clone --------------------------------------------------------
echo
echo -e "${BLUE}[PHASE 1/7]${NC} Cloning torch-choice..."
git clone https://github.com/gsbDBI/torch-choice.git "${CLONE_DIR}"
cd "${CLONE_DIR}"
echo -e "${GREEN}             ${NC}HEAD $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

# --- Phase 2: build minimal archive ----------------------------------------
echo
echo -e "${BLUE}[PHASE 2/7]${NC} Building minimal replication archive..."
bash ./scripts/create_minimal_replication_release.sh "${ARCHIVE_DIR}"
cd "${ARCHIVE_DIR}"

# --- Phase 3: install from PyPI --------------------------------------------
echo
echo -e "${BLUE}[PHASE 3/7]${NC} Installing torch-choice @ ${TORCH_CHOICE_VERSION} from PyPI..."
bash ./scripts/setup_uv_pypi.sh "${TORCH_CHOICE_VERSION}"

# Capture install-location proof
source .venv/bin/activate
python -c "
import torch_choice, torch
print()
print('=== Install location verification ===')
print(f'  torch-choice version:  {torch_choice.__version__}')
print(f'  torch-choice location: {torch_choice.__file__}')
print(f'  PyTorch version:       {torch.__version__}')
print(f'  CUDA available:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU:                   {p.name}  ({p.total_memory/2**30:.1f} GB)')
" | tee "${EVIDENCE_DIR}/install_verification.txt"
deactivate

# --- Phase 4: tier sanity --------------------------------------------------
if [[ "${SKIP_MEMORY_TEST}" != "1" ]]; then
    echo
    echo -e "${BLUE}[PHASE 4/7]${NC} Tier-selection sanity check (GPU_MEM_LIMIT=${GPU_MEM_LIMIT})..."
    source .venv/bin/activate
    GPU_MEM_LIMIT="${GPU_MEM_LIMIT}" python -c "
import sys; sys.path.insert(0, 'replication/paper_performance_benchmarks/steps')
import step02_torch_choice_benchmark as m
m._set_device('cuda')
print()
print('=== Tier selection ===')
print(f'  selected batch_size: {m._auto_batch_size()}')
" | tee "${EVIDENCE_DIR}/tier_selection.txt"
    deactivate
fi

# --- Phase 5: paper demo replication ---------------------------------------
if [[ "${SKIP_DEMO}" != "1" ]]; then
    echo
    echo -e "${BLUE}[PHASE 5/7]${NC} Reproducing paper demo (Section 4.1.4 CLM + 4.2.3 NLM)..."
    echo "             Wall-clock estimate: ~10 sec (1,000 LBFGS epochs on ModeCanada)."

    DEMO_LOG="${EVIDENCE_DIR}/paper_demo.log"
    bash ./replication/run_paper_demo.sh --no-tensorboard 2>&1 | tee "${DEMO_LOG}"

    # Extract the headline log-likelihood
    DEMO_LL=$(grep -m1 "Training] -" "${DEMO_LOG}" | grep -oE '\-[0-9]+\.[0-9]+' | head -1 || echo "<not found>")
    echo
    echo -e "${GREEN}             ${NC}CLM training log-likelihood: ${DEMO_LL}"
    echo "             (paper Section 4.1.4 expects: ~-1874.38)"
fi

# --- Phase 6: GPU memory test ----------------------------------------------
if [[ "${SKIP_MEMORY_TEST}" != "1" ]]; then
    echo
    echo -e "${BLUE}[PHASE 6/7]${NC} GPU memory cap test (records-large grid, ~20 min)..."

    GPU_METRICS_CSV="${EVIDENCE_DIR}/gpu_metrics_${GPU_MEM_LIMIT}gb.csv"
    BENCH_LOG="${EVIDENCE_DIR}/benchmark.log"

    bash ./scripts/monitor_gpu.sh "${GPU_METRICS_CSV}" &
    MONITOR_PID=$!
    trap "kill ${MONITOR_PID} 2>/dev/null || true" EXIT

    GPU_MEM_LIMIT="${GPU_MEM_LIMIT}" \
    SMOKE_TEST=0 SKIP_R=1 DEVICE=cuda NUM_SEEDS=1 \
    GENERATE_EXPERIMENTS=full_dataset \
    TORCH_EXPERIMENT_NAME=num_records_experiment_large \
      bash ./replication/paper_performance_benchmarks/run_benchmarking.sh 2>&1 \
        | tee "${BENCH_LOG}"

    kill "${MONITOR_PID}" 2>/dev/null || true
    trap - EXIT

    # Extract peak memory
    PEAK_LINE=$(awk -F',' 'NR>1 { gsub(/ MiB/, "", $3); if ($3+0 > max) max=$3+0 } END { print "peak GPU memory used:", max, "MiB" }' "${GPU_METRICS_CSV}")
    AUTO_BATCH_LINE=$(grep -m1 -- "-> batch_size=" "${BENCH_LOG}" || echo "<not found in log>")

    echo
    echo -e "${GREEN}             ${NC}${PEAK_LINE}"
    echo -e "${GREEN}             ${NC}Auto-batch: ${AUTO_BATCH_LINE}"
fi

# --- Phase 7: final summary -------------------------------------------------
echo
echo -e "${BLUE}[PHASE 7/7]${NC} Summary"
echo "=================================================================="
echo "  REPLICATION VALIDATION RESULTS"
echo "=================================================================="

if [[ "${SKIP_DEMO}" != "1" ]]; then
    echo "  Paper demo (CLM):"
    echo "    Training log-likelihood:  ${DEMO_LL:-<skipped>}"
    echo "    Paper expected:           ~-1874.34"
fi

if [[ "${SKIP_MEMORY_TEST}" != "1" ]]; then
    echo
    echo "  GPU memory test (GPU_MEM_LIMIT=${GPU_MEM_LIMIT}):"
    echo "    Auto-batch line:  ${AUTO_BATCH_LINE:-<skipped>}"
    echo "    ${PEAK_LINE:-<skipped>}"
    case "${GPU_MEM_LIMIT}" in
        6)   echo "    Reference (RTX-3090-class): batch_size=8,192   / peak ~600 MiB" ;;
        10)  echo "    Reference (RTX-3090-class): batch_size=16,384  / peak ~1,100 MiB" ;;
        14)  echo "    Reference (RTX-3090-class): batch_size=32,768  / peak ~2,200 MiB" ;;
        18)  echo "    Reference (RTX-3090-class): batch_size=65,536  / peak ~4,100 MiB" ;;
        24)  echo "    Reference (RTX-3090-class): batch_size=131,072 / peak ~8,100 MiB" ;;
        *)   echo "    (no reference for GPU_MEM_LIMIT=${GPU_MEM_LIMIT})" ;;
    esac
fi

echo
echo "  Evidence archived at: ${EVIDENCE_DIR}/"
ls -1 "${EVIDENCE_DIR}/" | sed 's/^/    /'
echo "=================================================================="
