#!/bin/bash
set -x

if [ ! -d logs ]; then
    mkdir -p logs
fi

CONDA_ENV=mleb
CONDA_BIN=/opt/conda/bin/conda

# Verify the docker python package from the correct conda env inside the Apptainer image
apptainer exec mlebench-env.sif "$CONDA_BIN" run -n "$CONDA_ENV" python - << 'EOF'
import sys
import docker
print("python:", sys.executable)
print("docker:", docker.__version__)
EOF

apptainer exec \
  --nv \
  mlebench-env.sif \
  "$CONDA_BIN" run -n "$CONDA_ENV" python run_agent.py "$@"
    # --agent-id aide/gpt-5.1 \
    # --competition-set experiments/splits/low_h5.txt \
    # > logs/run_aide_gpt51_low_h5.log 2>&1

set +x
