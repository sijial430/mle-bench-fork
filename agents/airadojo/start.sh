#!/bin/bash
set -x  # Print commands for debugging

# Debug: print environment variables
echo "COMPETITION_ID=${COMPETITION_ID}"
echo "AGENT_DIR=${AGENT_DIR}"

# Activate conda environment (superimage uses 'base', not 'agent')
eval "$(conda shell.bash hook)"
# Note: superimage already has base activated, skip explicit activation

# Set PYTHONPATH to include dojo source from agent directory and mlebench
# mlebench is mounted at /home/mlebench (parent directory so import mlebench works)
export PYTHONPATH="/home/agent/src:/home:${PYTHONPATH}"

# Install minimal mlebench dependencies that might be missing from superimage
# Use --target to install to a specific location we can control
DEPS_DIR=/tmp/pip-deps
mkdir -p ${DEPS_DIR}
echo "Installing dependencies to ${DEPS_DIR}..."
pip install --target ${DEPS_DIR} --no-cache-dir appdirs py7zr python-dotenv google-generativeai 2>&1 || echo "pip install failed, continuing..."
export PYTHONPATH="${DEPS_DIR}:${PYTHONPATH}"

echo "mlebench check..."
python -c "import mlebench; print(f'mlebench loaded from: {mlebench.__file__}')" || echo "mlebench not found"

# Determine hardware
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# Format time limit for display
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time ${TIME_LIMIT_SECS:-21600})

# Create working directory for aira-dojo
WORK_DIR="/tmp/aira-work-${COMPETITION_ID:-unknown}"
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Create symlinks to data
ln -sf /home/data ./data

# Create output directories
mkdir -p ./outputs
mkdir -p ./logs
mkdir -p /home/submission 2>/dev/null || true
mkdir -p /home/logs 2>/dev/null || true
mkdir -p /home/code 2>/dev/null || true

# Set up environment variables for aira-dojo
export STEP_LIMIT=${STEP_LIMIT:-125}

echo "Running aira-dojo for competition: ${COMPETITION_ID}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Determine search policy (greedy, mcts, or evo)
SEARCH_POLICY=${SEARCH_POLICY:-greedy}
echo "Search policy: ${SEARCH_POLICY}"

# Map search policy to solver config
case "${SEARCH_POLICY}" in
  greedy)
    SOLVER_CONFIG="mlebench/greedy"
    ;;
  mcts)
    SOLVER_CONFIG="mlebench/mcts"
    ;;
  evo)
    SOLVER_CONFIG="mlebench/evo"
    ;;
  *)
    echo "WARNING: Unknown search policy '${SEARCH_POLICY}', defaulting to greedy"
    SOLVER_CONFIG="mlebench/greedy"
    ;;
esac

echo "Using solver config: ${SOLVER_CONFIG}"

# Run aira-dojo with Hydra overrides
timeout ${TIME_LIMIT_SECS:-21600} python -m dojo.main_run \
  +_exp=run_example \
  solver=${SOLVER_CONFIG} \
  task.name="${COMPETITION_ID}" \
  task.cache_dir=/home \
  task.public_dir=/home/data \
  task.private_dir=/home/data \
  solver.step_limit=${STEP_LIMIT} \
  interpreter=python \
  hydra.run.dir=${WORK_DIR}/outputs \
  logger.use_wandb=False \
  $@ || true  # Continue even if timeout

# Find the most recent output directory
LATEST_OUTPUT=$(find ${WORK_DIR}/outputs -name "run_*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_OUTPUT" ]; then
  echo "ERROR: No output directory found"
  # Create a dummy submission to avoid complete failure
  echo "Id,Transported" > /home/submission/submission.csv 2>/dev/null || echo "Cannot write submission"
  exit 1
fi

echo "Latest output directory: $LATEST_OUTPUT"

# Copy submission file to expected location
if [ -f "${LATEST_OUTPUT}/submission.csv" ]; then
  cp "${LATEST_OUTPUT}/submission.csv" /home/submission/submission.csv 2>/dev/null || true
  echo "Submission copied successfully"
else
  echo "WARNING: No submission.csv found in output directory"
  # Try to find submission.csv anywhere in the output
  SUBMISSION_FILE=$(find ${LATEST_OUTPUT} -name "submission.csv" -type f | head -1)
  if [ -n "$SUBMISSION_FILE" ]; then
    cp "$SUBMISSION_FILE" /home/submission/submission.csv 2>/dev/null || true
    echo "Found and copied submission from: $SUBMISSION_FILE"
  else
    echo "ERROR: No submission.csv found anywhere in outputs"
  fi
fi

# Copy logs
if [ -d "${LATEST_OUTPUT}" ]; then
  cp -r ${LATEST_OUTPUT}/* /home/logs/ 2>/dev/null || true
  echo "Logs copied"
fi

# Copy any Python files generated as code
find ${LATEST_OUTPUT} -name "*.py" -type f -exec cp {} /home/code/ \; 2>/dev/null || true
find ${LATEST_OUTPUT} -name "solution.py" -type f -exec cp {} /home/code/ \; 2>/dev/null || true

echo "aira-dojo run completed"
