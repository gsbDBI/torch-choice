#!/bin/bash
# create_minimal_replication_release.sh
#
# Builds a minimal replication-release directory from the current torch-choice
# checkout. The output contains only what a replicator needs to reproduce the
# paper's results — the torch-choice package source itself is NOT included.
# Replicators install torch-choice from PyPI via setup_uv_pypi.sh.
#
# Usage:
#   bash ./scripts/create_minimal_replication_release.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: ./torch-choice-replication-release
#
# Run from the repo root.
#
# --- Included in the output -------------------------------------------------
#   replication/                  Paper demo + benchmark scripts
#   scripts/setup_uv_pypi.sh      Install script (uv + PyPI)
#   scripts/monitor_gpu.sh        GPU memory monitor
#   pyproject.toml                Kept so setup_uv_pypi.sh can read the
#                                 dependency list and the
#                                 [tool.uv.extra-build-dependencies] hint
#                                 that lets torch-scatter build correctly
#   gpu_memory_limit_test.md      GPU memory verification guide
#   README.md, UV_SETUP.md        Installation / usage docs
#   LICENSE
#
# --- Excluded from the output -----------------------------------------------
#   torch_choice/, torch_choice.egg-info/    The package source — replicators
#                                            get torch-choice from PyPI instead
#   setup.py, uv.lock                        Build/lock artifacts (not needed
#                                            for the PyPI-install path)
#   scripts/setup_uv.sh                      Editable-source install (would
#                                            fail without torch_choice/)
#   tests/, ai_generated_tests/              Developer tests
#   docs/, docs_src/, mkdocs.yml             Built docs site
#   tutorials/, release_notes/               Non-replication content
#   paper_performance_benchmarks/            Top-level duplicate of reference
#                                            results (the canonical copy lives
#                                            under replication/)
#   replication/paper_performance_benchmarks_legacy/
#                                            Historical earlier-iteration
#                                            scripts superseded by the current
#                                            run_benchmarking.sh; not referenced
#                                            from the replication guideline
#   replication/paper_performance_benchmarks/runs/
#                                            Local working state from prior
#                                            benchmark executions; replicators
#                                            create their own runs/<timestamp>/
#                                            when they run the pipeline.
#                                            (Paper-time canonical references
#                                            live in benchmark_results_aurora_*
#                                            and benchmark_figures_*.)
#   __pycache__/, .ipynb_checkpoints/,
#   lightning_logs/, .DS_Store               Transient artifacts that any local
#                                            run (or macOS Finder) may leave
#                                            behind; pruned defensively even
#                                            when absent today
#   requirements.txt,
#   requirements_complete.txt                Deprecated (superseded by
#                                            pyproject.toml extras)
#   recompile_website.sh,
#   test_all_notebooks.sh                    Dev helpers

set -e

OUTPUT_DIR="${1:-./torch-choice-replication-release}"

# Color codes for output.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Sanity: are we in a torch-choice repo root?
if [[ ! -f pyproject.toml ]] || [[ ! -d replication ]] || [[ ! -d scripts ]]; then
    echo -e "${RED}[ERROR]${NC} Run this from the repo root of torch-choice."
    echo "        Expected pyproject.toml, replication/, scripts/ in CWD."
    exit 1
fi

# Refuse to overwrite — fail loudly rather than risk clobbering work.
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo -e "${RED}[ERROR]${NC} OUTPUT_DIR already exists: ${OUTPUT_DIR}"
    echo "        Remove it first, or pick a different path:"
    echo "          bash ./scripts/create_minimal_replication_release.sh <new_path>"
    exit 1
fi

echo "========================================="
echo "  torch-choice minimal replication build"
echo "========================================="
echo
echo -e "${BLUE}[INFO]${NC} Output directory: ${OUTPUT_DIR}"
echo

mkdir -p "${OUTPUT_DIR}/scripts"

# --- Copy the keepers ------------------------------------------------------
echo -e "${BLUE}[INFO]${NC} Copying replication essentials..."

# Directories
cp -a replication                  "${OUTPUT_DIR}/replication"

# Drop historical-only content that the replication guideline never references.
rm -rf "${OUTPUT_DIR}/replication/paper_performance_benchmarks_legacy"

# Drop local working state from prior benchmark executions; replicators
# generate their own runs/<timestamp>/ when they execute the pipeline.
rm -rf "${OUTPUT_DIR}/replication/paper_performance_benchmarks/runs"

# Defensive sweep: prune transient artifacts that any local execution (or
# macOS Finder) may have left in the source tree. Safe no-ops when absent.
find "${OUTPUT_DIR}" -type d \
    \( -name __pycache__ -o -name .ipynb_checkpoints -o -name lightning_logs \) \
    -prune -exec rm -rf {} + 2>/dev/null || true
find "${OUTPUT_DIR}" -name .DS_Store -delete 2>/dev/null || true

# Setup / install scripts
cp -a scripts/setup_uv_pypi.sh     "${OUTPUT_DIR}/scripts/setup_uv_pypi.sh"
cp -a scripts/monitor_gpu.sh       "${OUTPUT_DIR}/scripts/monitor_gpu.sh"

# Project metadata (used by setup_uv_pypi.sh for the prerequisite install)
cp -a pyproject.toml               "${OUTPUT_DIR}/pyproject.toml"

# Documentation
cp -a gpu_memory_limit_test.md     "${OUTPUT_DIR}/gpu_memory_limit_test.md"
cp -a README.md                    "${OUTPUT_DIR}/README.md"
cp -a UV_SETUP.md                  "${OUTPUT_DIR}/UV_SETUP.md"
cp -a LICENSE                      "${OUTPUT_DIR}/LICENSE"

# Ensure scripts stay executable after copy
chmod +x "${OUTPUT_DIR}/scripts/setup_uv_pypi.sh"
chmod +x "${OUTPUT_DIR}/scripts/monitor_gpu.sh"

echo -e "${GREEN}[SUCCESS]${NC} Files copied."

# --- Verification ----------------------------------------------------------
echo
echo -e "${BLUE}[INFO]${NC} Contents of ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"
echo
echo -e "${BLUE}[INFO]${NC} Size: $(du -sh "${OUTPUT_DIR}" | cut -f1)"

# Cross-check: no package source leaked in
if [[ -d "${OUTPUT_DIR}/torch_choice" ]] || [[ -d "${OUTPUT_DIR}/torch_choice.egg-info" ]]; then
    echo -e "${RED}[ERROR]${NC} torch_choice source ended up in the release — aborting."
    exit 1
fi

echo
echo -e "${GREEN}[DONE]${NC} Minimal replication release built at: ${OUTPUT_DIR}"
echo
echo "Next steps (for a replicator):"
echo "  cd ${OUTPUT_DIR}"
echo "  bash ./scripts/setup_uv_pypi.sh 1.0.7"
echo "  # then follow gpu_memory_limit_test.md or replication/paper_replication_guideline.md"
