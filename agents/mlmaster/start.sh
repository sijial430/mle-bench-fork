#!/bin/bash
set -x  # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)"  # make conda available to the shell

# Try to activate 'ml-master' environment, fall back to 'base' if it doesn't exist
if conda env list | grep -q "^ml-master "; then
  conda activate ml-master
  echo "Activated conda environment: ml-master"
else
  echo "Conda environment 'ml-master' not found, using base environment"
  conda activate base
fi

# Start ML-Master grading server in background (for check_format feature)
echo "Starting grading server on port 5001..."
nohup python grading_server.py \
  dataset_dir="/home/data/" \
  data_dir="/home/data/" \
  desc_file="none" > /tmp/grading_server.log 2>&1 &
GRADING_SERVER_PID=$!
echo "Grading server started with PID: $GRADING_SERVER_PID"

# Wait for grading server to be ready
for i in {1..10}; do
  if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "Grading server is ready"
    break
  fi
  echo "Waiting for grading server... ($i/10)"
  sleep 1
done

# determine hardware available
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

# check GPU availability
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi

# start a new file to store the full instructions, starting with general instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# Update instructions for agent-specific details: replace `/home/` paths to make paths relative
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt

# move on to agent-specific instructions, with a linebreak in between
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# Create workspace directories
mkdir -p ${AGENT_DIR}/workspaces/exp
mkdir -p ${AGENT_DIR}/logs

# symbolic linking for output extraction
ln -sf ${LOGS_DIR} ${AGENT_DIR}/logs/exp
ln -sf ${CODE_DIR} ${AGENT_DIR}/workspaces/exp/best_solution
ln -sf ${SUBMISSION_DIR} ${AGENT_DIR}/workspaces/exp/best_submission

# Get CPU info for taskset (optional, for parallel execution)
NUM_CPUS=$(nproc)
START_CPU=0
END_CPU=$((NUM_CPUS - 1))

# Set up API keys from environment
if [ -n "$OPENAI_API_KEY" ]; then
  export OPENAI_API_KEY
fi

# Default base URLs (can be overridden by config.yaml kwargs)
# - GPT-5.1: uses OpenAI API
# - Qwen: uses local VLLM (base_url set in config.yaml)
CODE_BASE_URL=${CODE_BASE_URL:-"https://api.openai.com/v1"}
FEEDBACK_BASE_URL=${FEEDBACK_BASE_URL:-"https://api.openai.com/v1"}

# API keys
# - Code model: OpenAI for GPT, dummy for VLLM (Qwen)
# - Feedback model: Always OpenAI (gpt-5.1-mini)
CODE_API_KEY="${OPENAI_API_KEY:-dummy-key}"
FEEDBACK_API_KEY="$OPENAI_API_KEY"

# Run ML-Master with timeout
timeout $TIME_LIMIT_SECS python main_mcts.py \
  data_dir="/home/data/" \
  dataset_dir="/home/data/" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="exp" \
  log_dir="${AGENT_DIR}/logs" \
  workspace_dir="${AGENT_DIR}/workspaces" \
  start_cpu_id="${START_CPU}" \
  cpu_number="${NUM_CPUS}" \
  agent.code.api_key="${CODE_API_KEY}" \
  agent.feedback.api_key="${FEEDBACK_API_KEY}" \
  agent.code.base_url="${CODE_BASE_URL}" \
  agent.feedback.base_url="${FEEDBACK_BASE_URL}" \
  agent.steerable_reasoning=false \
  $@  # forward the bash arguments to main_mcts.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# Copy best submission to the expected location
if [ -f "${AGENT_DIR}/workspaces/exp/best_submission/submission.csv" ]; then
  echo "Found submission.csv in best_submission directory"
elif [ -f "${AGENT_DIR}/workspaces/exp/submission.csv" ]; then
  cp "${AGENT_DIR}/workspaces/exp/submission.csv" "${SUBMISSION_DIR}/submission.csv" 2>/dev/null || true
  echo "Copied submission.csv from workspaces/exp"
else
  # Try to find any submission.csv
  SUBMISSION_FILE=$(find ${AGENT_DIR}/workspaces -name "submission.csv" -type f | head -1)
  if [ -n "$SUBMISSION_FILE" ]; then
    cp "$SUBMISSION_FILE" "${SUBMISSION_DIR}/submission.csv" 2>/dev/null || true
    echo "Found and copied submission from: $SUBMISSION_FILE"
  else
    echo "WARNING: No submission.csv found"
  fi
fi

# Copy best solution code
if [ -f "${AGENT_DIR}/logs/exp/best_solution.py" ]; then
  cp "${AGENT_DIR}/logs/exp/best_solution.py" "${CODE_DIR}/best_solution.py" 2>/dev/null || true
fi

echo "ML-Master run completed with exit code: $EXIT_CODE"
