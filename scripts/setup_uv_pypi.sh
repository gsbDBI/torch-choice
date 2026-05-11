#!/bin/bash
# setup_uv_pypi.sh — Install torch-choice from PyPI at a pinned version.
#
# Unlike setup_uv.sh (which installs torch-choice in *editable mode* from
# local source — the developer workflow), this script installs torch-choice
# from PyPI at a fixed version. That makes it the right choice for
# replicators or anyone who wants to reproduce a specific archived release.
#
# Install strategy: two-stage.
#   1. Use the local pyproject.toml to resolve and install all prerequisites
#      (torch, numpy, pandas, torch-scatter, tensorboard, etc.) into the
#      .venv. The pyproject.toml's [tool.uv.extra-build-dependencies] hint
#      keeps torch in scope during torch-scatter's PEP 517 build, avoiding
#      the "ModuleNotFoundError: No module named 'torch'" failure that
#      otherwise occurs in a one-shot install.
#   2. Install torch-choice itself from PyPI on top of those prerequisites.
#
# A minimal replication archive can be produced by running
# `create_minimal_replication_release.sh` from the repo root — it preserves
# pyproject.toml (which this script needs in step 1) while excluding the
# package source (which would otherwise shadow the PyPI install).
#
# Usage:
#   bash ./scripts/setup_uv_pypi.sh                       # default: latest from PyPI
#   bash ./scripts/setup_uv_pypi.sh 1.0.7                 # pin a specific version (replicator path)
#   bash ./scripts/setup_uv_pypi.sh latest                # explicitly request latest
#   TORCH_CHOICE_VERSION=1.0.7 bash ./scripts/setup_uv_pypi.sh
#   TORCH_CHOICE_EXTRA=benchmarks bash ./scripts/setup_uv_pypi.sh    # narrower extras
#
# With no arguments, this script installs the latest published torch-choice
# from PyPI. To reproduce the environment archived with a specific paper or
# release, pass the version explicitly — e.g. `1.0.7`.

set -e

# ---------------------------------------------------------------------------
# Configuration (override via positional arg or env var)
# ---------------------------------------------------------------------------

# Default: install whatever is currently latest on PyPI. Replicators
# reproducing a specific paper or archived release should pass the version
# explicitly (e.g. `bash ./scripts/setup_uv_pypi.sh 1.0.7`).
DEFAULT_VERSION="latest"

# Resolution: positional arg > env var > hard-coded default.
VERSION="${1:-${TORCH_CHOICE_VERSION:-${DEFAULT_VERSION}}}"

# Default extras: 'complete' — matches setup_uv.sh and includes everything
# needed for the replication scripts (notebooks, benchmarks, tensorboard).
EXTRA="${TORCH_CHOICE_EXTRA:-complete}"

# Suppress uv's preview-feature warning (extra-build-dependencies is used
# for torch-scatter; see pyproject.toml [tool.uv.extra-build-dependencies]).
export UV_PREVIEW=1

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "  torch-choice PyPI Setup Script (uv)"
echo "========================================="
echo
echo -e "${BLUE}[INFO]${NC} Plan: install torch-choice[${EXTRA}] @ ${VERSION} from PyPI."
echo

# ---------------------------------------------------------------------------
# Step 1: verify uv is installed
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} uv is not installed. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://github.com/astral-sh/uv#installation"
    exit 1
fi
echo -e "${GREEN}[SUCCESS]${NC} uv is installed ($(uv --version))"

# ---------------------------------------------------------------------------
# Step 2: fresh virtual environment with Python 3.12
# ---------------------------------------------------------------------------
if [ -d ".venv" ]; then
    echo -e "${YELLOW}[WARNING]${NC} .venv already exists — removing for a clean install."
    rm -rf .venv
fi
echo -e "${BLUE}[INFO]${NC} Creating .venv with Python 3.12..."
uv venv --python 3.12
echo -e "${GREEN}[SUCCESS]${NC} Virtual environment created at $(pwd)/.venv"

# ---------------------------------------------------------------------------
# Step 3: install prerequisites declared in pyproject.toml
# ---------------------------------------------------------------------------
# We deliberately install in two stages:
#   (a) the dependency tree declared in pyproject.toml's [project.dependencies]
#       and [project.optional-dependencies.${EXTRA}] — this is the
#       "prerequisites" step, and it benefits from the
#       [tool.uv.extra-build-dependencies] hint in pyproject.toml, which makes
#       torch available at build time so torch-scatter's C++ extension
#       compiles without the "ModuleNotFoundError: No module named 'torch'"
#       failure that bites a one-shot install.
#   (b) torch-choice itself, fetched from PyPI at the pinned version
#       (the next step).
#
# `uv sync --no-install-project` installs every resolved dependency but skips
# building the local "torch-choice" project — which is exactly what we want,
# because we're going to install the published torch-choice from PyPI on top.

if [[ ! -f pyproject.toml ]]; then
    echo -e "${RED}[ERROR]${NC} pyproject.toml is required but was not found in CWD."
    echo "        Run this script from a torch-choice repo root or a"
    echo "        replication archive built via create_minimal_replication_release.sh"
    echo "        (which preserves pyproject.toml for this step)."
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Installing prerequisites from pyproject.toml [${EXTRA}] extras..."
uv sync --extra "${EXTRA}" --no-install-project
echo -e "${GREEN}[SUCCESS]${NC} Prerequisites installed."

# ---------------------------------------------------------------------------
# Step 4: install torch-choice itself from PyPI
# ---------------------------------------------------------------------------
if [[ "${VERSION}" == "latest" ]]; then
    SPEC="torch-choice"
else
    SPEC="torch-choice==${VERSION}"
fi

echo -e "${BLUE}[INFO]${NC} Installing ${SPEC} from PyPI..."
# --no-deps because every prerequisite was already resolved in step 3.
uv pip install --no-deps "${SPEC}"
echo -e "${GREEN}[SUCCESS]${NC} ${SPEC} installed from PyPI."

# ---------------------------------------------------------------------------
# Step 5: import-level verification
# ---------------------------------------------------------------------------
echo -e "${BLUE}[INFO]${NC} Verifying imports..."
if ! uv run python -c "import torch_choice; print(f'torch-choice version: {torch_choice.__version__}')"; then
    echo -e "${RED}[ERROR]${NC} Failed to import torch-choice"
    exit 1
fi
if ! uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"; then
    echo -e "${RED}[ERROR]${NC} Failed to import PyTorch"
    exit 1
fi
echo -e "${GREEN}[SUCCESS]${NC} torch-choice and PyTorch import OK."

# ---------------------------------------------------------------------------
# Step 6: end-to-end smoke test — fit a small ConditionalLogitModel
# ---------------------------------------------------------------------------
echo
echo -e "${BLUE}[INFO]${NC} Running a small torch-choice model to validate the install..."
if uv run python - << 'PY'
import torch
import torch_choice

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset = torch_choice.data.load_mode_canada_dataset().to(device)

model = torch_choice.model.ConditionalLogitModel(
    formula='(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)',
    dataset=dataset,
    num_items=4,
).to(device)

model.fit(
    dataset,
    num_epochs=5,
    learning_rate=0.003,
    batch_size=-1,
    model_optimizer="LBFGS",
    device=device,
)
PY
then
    echo -e "${GREEN}[SUCCESS]${NC} Quick model run completed."
else
    echo -e "${RED}[ERROR]${NC} Quick model run failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}[SUCCESS]${NC} Setup complete!"
echo
echo "📦 Installed:"
echo "  • torch-choice[${EXTRA}] @ ${VERSION}  (from PyPI)"
echo "  • Virtual environment: $(pwd)/.venv"
echo
echo "🔧 How to use:"
echo "  uv run python your_script.py        # no activation needed"
echo "  source .venv/bin/activate           # or activate the venv"
echo
echo "📝 Notes:"
if [[ "${VERSION}" == "latest" ]]; then
    echo "  • Installed the most recent torch-choice published on PyPI."
    echo "  • To pin a version:        bash ./scripts/setup_uv_pypi.sh 1.0.7"
else
    echo "  • Pinned to torch-choice @ ${VERSION}."
    echo "  • To upgrade to latest:    bash ./scripts/setup_uv_pypi.sh latest"
fi
echo "  • Replicators reproducing a specific archived release should pass that version explicitly."
echo
echo -e "${GREEN}[SUCCESS]${NC} All done."
