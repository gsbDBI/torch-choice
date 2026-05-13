#!/usr/bin/env bash
# =============================================================================
# run_paper_demo.sh
# Runs the paper_demo.py replication script and launches TensorBoard afterward.
# =============================================================================

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-lightning_logs}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
NUM_EPOCHS="${NUM_EPOCHS:-10000}"
SKIP_TRAINING="${SKIP_TRAINING:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --tensorboard-logdir)
            TENSORBOARD_LOGDIR="$2"
            shift 2
            ;;
        --tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --no-tensorboard)
            NO_TENSORBOARD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-training        Skip model training (quick smoke test)"
            echo "  --num-epochs N         Number of training epochs for the conditional logit Adam fit (default: 10000)"
            echo "  --tensorboard-logdir   Directory for TensorBoard logs (default: lightning_logs)"
            echo "  --tensorboard-port     Port for TensorBoard (default: 6006)"
            echo "  --no-tensorboard       Do not launch TensorBoard after running the demo"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

# Prevent `uv run` from auto-syncing the local project before each command.
# In a stripped replication archive (no torch_choice/ source), an autosync
# would build an empty wheel from pyproject.toml metadata and overwrite the
# PyPI-installed torch-choice in .venv. Safe to set in dev/source checkouts
# too: the editable install survives across uv run calls without needing
# a fresh sync each time.
export UV_NO_SYNC=1

# Silence the experimental-feature warning emitted by `uv run`. Our
# pyproject.toml uses `[tool.uv.extra-build-dependencies]` to make `torch`
# available at build time for `torch-scatter` (a hard build-time requirement
# of that package). The benchmark wrapper already passes this flag; we do
# the same here so the demo run is warning-free.
UV_RUN="uv run --preview-features extra-build-dependencies"

echo "=============================================================================="
echo "Running paper_demo.py"
echo "=============================================================================="

# Build the command
DEMO_CMD="$UV_RUN python replication/paper_demo.py"
DEMO_CMD="$DEMO_CMD --tensorboard-logdir $TENSORBOARD_LOGDIR"
DEMO_CMD="$DEMO_CMD --tensorboard-port $TENSORBOARD_PORT"
DEMO_CMD="$DEMO_CMD --num-epochs $NUM_EPOCHS"

if [ "$SKIP_TRAINING" = true ]; then
    DEMO_CMD="$DEMO_CMD --skip-training"
fi

echo "Command: $DEMO_CMD"
echo ""

# Run the demo script
eval $DEMO_CMD

DEMO_EXIT_CODE=$?

if [ $DEMO_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[Error] paper_demo.py exited with code $DEMO_EXIT_CODE"
    exit $DEMO_EXIT_CODE
fi

echo ""
echo "=============================================================================="
echo "paper_demo.py completed successfully"
echo "=============================================================================="

# Launch TensorBoard if training was performed and --no-tensorboard was not set
if [ "$SKIP_TRAINING" = true ]; then
    echo ""
    echo "[Info] Training was skipped, no TensorBoard logs to visualize."
    exit 0
fi

if [ "$NO_TENSORBOARD" = true ]; then
    echo ""
    echo "[Info] TensorBoard launch skipped (--no-tensorboard flag set)."
    echo "To view logs manually, run:"
    echo "  $UV_RUN tensorboard --logdir $TENSORBOARD_LOGDIR --port $TENSORBOARD_PORT"
    exit 0
fi

echo ""
echo "=============================================================================="
echo "Launching TensorBoard"
echo "=============================================================================="

# Check if tensorboard is available
if ! $UV_RUN python -c "import tensorboard" 2>/dev/null; then
    echo "[Warning] TensorBoard not found in the environment."
    echo "To install TensorBoard, run: uv pip install tensorboard"
    echo "To view logs manually after installation:"
    echo "  $UV_RUN tensorboard --logdir $TENSORBOARD_LOGDIR --port $TENSORBOARD_PORT"
    exit 0
fi

echo "Starting TensorBoard on port $TENSORBOARD_PORT..."
echo "Open http://localhost:$TENSORBOARD_PORT/ in your browser."
echo ""
echo "Press Ctrl+C to stop TensorBoard."
echo ""

# Run TensorBoard in foreground so user can Ctrl+C to stop
$UV_RUN tensorboard --logdir "$TENSORBOARD_LOGDIR" --port "$TENSORBOARD_PORT"

