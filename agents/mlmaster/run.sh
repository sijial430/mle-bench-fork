#!/bin/bash
# NOTE: This script is for LOCAL TESTING ONLY.
# EC2 launches use ec2-startup-mlmaster.sh → run_agent.py → config.yaml
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
EXP_ID=spaceship-titanic
dataset_dir=/home/ubuntu/mle-bench-fork/mlebench
MEMORY_INDEX=0

# Model config - these are passed to main_mcts.py below
code_model=gpt-5.1
code_temp=1
feedback_model=gpt-5-mini-2025-08-07
feedback_temp=1

start_cpu=0
CPUS_PER_TASK=36
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

TIME_LIMIT_SECS=21600

cd ${AGENT_DIR}
export MEMORY_INDEX
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=125

mkdir -p ${AGENT_DIR}/logs

# use the mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  dataset_dir="${dataset_dir}" \
  data_dir="${dataset_dir}/${EXP_ID}/prepared/public" \
  desc_file="./dataset/full_instructions/${EXP_ID}/full_instructions.txt" \
  exp_name="${EXP_ID}_mcts_comp_validcheck_[cpu-${start_cpu}-${end_cpu}]" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.code.model=$code_model \
  agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model \
  agent.feedback.temp=$feedback_temp \
  agent.steerable_reasoning=false

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
