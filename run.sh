#!/bin/bash

# Example usage:
# chmod +x run.sh
# ./run.sh --agent-id aide/gpt-5.1 --competition-set experiments/splits/low_h5.txt > logs/run_aide_gpt51_low_h5.log 2>&1

set -x

if [ ! -d logs ]; then
    mkdir -p logs
fi

# Verify the docker python package from the correct conda env inside the Apptainer image
# apptainer exec mlebench-env.sif python - << 'EOF'
# import sys
# import docker
# print("python:", sys.executable)
# EOF

# apptainer exec \
#   --nv \
#   mlebench-env.sif \
#   python run_agent.py "$@"
    # --agent-id aide/gpt-5.1 \
    # --competition-set experiments/splits/low_h5.txt \
    # > logs/run_aide_gpt51_low_h5.log 2>&1

apptainer pull --disable-cache verl.sif docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

apptainer run --nv \
  --bind /lib64/libcuda.so.1:/usr/lib64/libcuda.so.1 \
     mlebench-env.sif python - << 'EOF'
import sys
import docker
print("python:", sys.executable)
EOF

set +x
