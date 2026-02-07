#!/bin/bash

# Redirect ALL output to log file (including errors)
exec > >(tee -a /var/log/mlebench-run.log) 2>&1

echo "=========================================="
echo "User Data script started at $(date)"
echo "=========================================="

# Don't exit on error - we want to see what fails
set -x  # Print each command before executing

# Wait for cloud-init to finish and network to be ready
echo "Waiting for system to be ready..."
sleep 10

# ==========================================
# CONFIGURATION - CUSTOMIZE THESE VALUES
# ==========================================
AWS_REGION="us-east-1"
S3_DATA_BUCKET="mlebench-data"
S3_RESULTS_BUCKET="mlebench-results"
# COMPETITION_ID="${1:-spaceship-titanic}" #tensorflow2-question-answering
# AGENT_ID="${2:-aide/dev}"

# ==========================================
# INSTANCE TAGS (IMDSv2) - COMPETITION_ID / AGENT_ID
# ==========================================
IMDS_BASE="http://169.254.169.254/latest"
IMDS_TOKEN="$(curl -fsS -X PUT "$IMDS_BASE/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"

imds_get() {
  local path="$1"
  curl -fsS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" "$IMDS_BASE/$path"
}

get_tag() {
  local key="$1"
  imds_get "meta-data/tags/instance/${key}" 2>/dev/null || true
}

# Wait for instance tags to be available (metadata can be delayed after boot)
wait_for_tags() {
  local max_attempts=12
  local attempt=1
  while [ "$attempt" -le "$max_attempts" ]; do
    local steps=$(get_tag Steps)
    if [ -n "$steps" ] || [ "$attempt" -eq "$max_attempts" ]; then
      break
    fi
    echo "Instance tags not ready (attempt $attempt/$max_attempts), waiting 10s..."
    sleep 10
    attempt=$((attempt + 1))
  done
}
wait_for_tags

# Defaults if tags are missing / metadata tags not enabled
COMPETITION_ID="$(get_tag Competition)"
AGENT_ID="$(get_tag AgentId)"

: "${COMPETITION_ID:=spaceship-titanic}"
: "${AGENT_ID:=aide/dev}"

# Read agent-specific kwargs from EC2 tags
STEPS="$(get_tag Steps)"
TIME_LIMIT_SECS="$(get_tag TimeLimitSecs)"
DEBUG_DEPTH="$(get_tag DebugDepth)"
DEBUG_PROB="$(get_tag DebugProb)"
EXEC_TIMEOUT="$(get_tag ExecTimeout)"
MAX_PARALLEL_WIDTH="$(get_tag MaxParallelWidth)"
CODE_MODEL="$(get_tag CodeModel)"
CODE_TEMP="$(get_tag CodeTemp)"
FEEDBACK_MODEL="$(get_tag FeedbackModel)"
FEEDBACK_TEMP="$(get_tag FeedbackTemp)"
GIT_BRANCH="$(get_tag GitBranch)"

MLEBENCH_REPO="https://github.com/sijial430/mle-bench-fork.git"  # Change to your repo if forked
API_KEY_SECRET_NAME="sijial_oai_key"

echo "Competition: $COMPETITION_ID"
echo "Agent: $AGENT_ID"
echo "Steps: ${STEPS:-default}"
echo "TimeLimitSecs: ${TIME_LIMIT_SECS:-default}"
echo "DebugDepth: ${DEBUG_DEPTH:-default}"
echo "DebugProb: ${DEBUG_PROB:-default}"
echo "ExecTimeout: ${EXEC_TIMEOUT:-default}"
echo "MaxParallelWidth: ${MAX_PARALLEL_WIDTH:-default}"
echo "GitBranch: ${GIT_BRANCH:-default}"
echo "CodeModel: ${CODE_MODEL:-default}"
echo "CodeTemp: ${CODE_TEMP:-default}"
echo "FeedbackModel: ${FEEDBACK_MODEL:-default}"
echo "FeedbackTemp: ${FEEDBACK_TEMP:-default}"

# ==========================================
# INSTALL DEPENDENCIES
# ==========================================
echo "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Update package lists
sudo apt-get update -y

# Install Docker
echo "Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ubuntu
else
    echo "Docker already installed"
fi

# Install Sysbox (for Docker-in-Docker without privileged mode)
echo "Installing Sysbox..."
if ! command -v sysbox-runc &> /dev/null; then
    # Download and install sysbox
    wget -q https://downloads.nestybox.com/sysbox/releases/v0.6.4/sysbox-ce_0.6.4-0.linux_amd64.deb -O /tmp/sysbox.deb
    sudo apt-get install -y jq
    sudo apt-get install -y /tmp/sysbox.deb
    rm /tmp/sysbox.deb
    
    # Restart Docker to register sysbox runtime
    sudo systemctl restart docker
else
    echo "Sysbox already installed"
fi

# Install AWS CLI v2
echo "Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    sudo apt-get install -y unzip curl
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/awscliv2.zip /tmp/aws
else
    echo "AWS CLI already installed"
fi

# Install Python and pip
echo "Installing Python dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git

# Install Git LFS
echo "Installing Git LFS..."
if ! command -v git-lfs &> /dev/null; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
    git lfs install
else
    echo "Git LFS already installed"
fi

# ==========================================
# GET AWS ACCOUNT INFO
# ==========================================
echo "Getting AWS Account ID..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $AWS_REGION)
SHARED_ACCOUNT_ID="574455268872"
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "ERROR: Could not get AWS Account ID. Check IAM role!"
    exit 1
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "Shared Account ID: $SHARED_ACCOUNT_ID"

# ==========================================
# CLONE AND SETUP MLE-BENCH
# ==========================================
echo "Setting up mle-bench-fork..."
if [ ! -d "/home/ubuntu/mle-bench-fork" ]; then
    echo "Cloning mle-bench-fork repository..."
    cd /home/ubuntu
    sudo -u ubuntu git clone $MLEBENCH_REPO
    cd /home/ubuntu/mle-bench-fork

    # Checkout the specified branch if set
    if [ -n "$GIT_BRANCH" ]; then
        echo "Checking out branch: $GIT_BRANCH"
        git checkout "$GIT_BRANCH"
    fi

    # Pull LFS files (leaderboards, CSVs, top solutions)
    echo "Pulling Git LFS files..."
    git lfs pull
    
    # Create virtual environment and install mle-bench-fork
    echo "Creating virtual environment..."
    python3 -m venv /home/ubuntu/mle-bench-fork/.venv
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    echo "mle-bench-fork already exists"
    cd /home/ubuntu/mle-bench-fork
    # Checkout the specified branch if set
    if [ -n "$GIT_BRANCH" ]; then
        echo "Checking out branch: $GIT_BRANCH"
        git checkout "$GIT_BRANCH"
        git pull origin "$GIT_BRANCH"
    fi
    # Make sure LFS files are up to date
    git lfs pull
    # Activate existing venv
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
fi

# ==========================================
# DOWNLOAD COMPETITION DATA FROM S3
# ==========================================
echo "Downloading competition data from S3..."
sudo mkdir -p /data
sudo chown ubuntu:ubuntu /data
aws s3 sync s3://$S3_DATA_BUCKET/data/$COMPETITION_ID /data/$COMPETITION_ID

# Verify data was downloaded
if [ ! -d "/data/$COMPETITION_ID" ]; then
    echo "ERROR: Competition data not found at /data/$COMPETITION_ID"
    echo "Make sure data exists in S3 bucket: s3://$S3_DATA_BUCKET/data/$COMPETITION_ID"
    exit 1
fi
echo "Competition data downloaded successfully"

# Navigate to mlebench directory
cd /home/ubuntu/mle-bench-fork

# Ensure venv is activated
source /home/ubuntu/mle-bench-fork/.venv/bin/activate

# ==========================================
# SETUP DOCKER
# ==========================================
echo "Setting up Docker..."
sudo systemctl start docker || true
sudo usermod -aG docker ubuntu || true

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull images from ECR
echo "Pulling Docker images..."
# sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
# sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide:latest
sudo docker pull $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide-dev:latest

# Build the base environment image
# sudo docker build --build-arg INSTALL_HEAVY_DEPENDENCIES=false -t mlebench-env -f environment/Dockerfile .

# Tag and push to ECR
# sudo docker tag mlebench-env:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
# sudo docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest

# Build the aide agent
# export SUBMISSION_DIR=/home/submission
# export LOGS_DIR=/home/logs
# export CODE_DIR=/home/code
# export AGENT_DIR=/home/agent

# sudo docker build --platform=linux/amd64 \
#   --build-arg BASE_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest \
#   -t aide agents/aide/ \
#   --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
#   --build-arg LOGS_DIR=$LOGS_DIR \
#   --build-arg CODE_DIR=$CODE_DIR \
#   --build-arg AGENT_DIR=$AGENT_DIR

# Tag and push aide agent
# sudo docker tag aide:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide:latest
# sudo docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide:latest

# Tag with local names (so run_agent.py finds them)
echo "Tagging images..."
# sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest mlebench-env:latest
# sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide:latest aide:latest
sudo docker tag $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest mlebench-env:latest
sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide-dev:latest aide:latest

# ==========================================
# RUN THE AGENT
# ==========================================
# Create competition file
echo "$COMPETITION_ID" > /tmp/competition.txt


NUM_CPUS=$(nproc)
echo "Detected $NUM_CPUS CPUs"

##### Doing this for now as I am pulling direclty from public repo
## Note: I am using t2.micro for test runs
cat > /tmp/container_config.json << EOF
{
    "mem_limit": "64g",
    "shm_size": "64g",
    "nano_cpus": ${NUM_CPUS}e9,
    "runtime": "sysbox-runc"
}
EOF
#####

echo "Container config:"
cat /tmp/container_config.json

# ==========================================
# FETCH SECRETS
# ==========================================
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Fetching secrets from AWS Secrets Manager..."
# Fetch OPENAI_API_KEY. Assumes the secret is stored as a plain string or a JSON with OPENAI_API_KEY field.
SECRET_VALUE=$(aws secretsmanager get-secret-value --secret-id $API_KEY_SECRET_NAME --query SecretString --output text --region $AWS_REGION 2>/dev/null)

if [ $? -eq 0 ]; then
    # Try to parse as JSON, fallback to raw string if not JSON or if jq fails
    if echo "$SECRET_VALUE" | jq -e .OPENAI_API_KEY >/dev/null 2>&1; then
        export OPENAI_API_KEY=$(echo "$SECRET_VALUE" | jq -r .OPENAI_API_KEY)
    else
        export OPENAI_API_KEY="$SECRET_VALUE"
    fi
    echo "Successfully loaded OPENAI_API_KEY"
else
    echo "WARNING: Failed to fetch OPENAI_API_KEY from Secrets Manager"
fi

# Build kwargs array from EC2 tags
KWARGS_ARGS=()
[ -n "$STEPS" ] && KWARGS_ARGS+=("agent.steps=$STEPS")
[ -n "$DEBUG_DEPTH" ] && KWARGS_ARGS+=("agent.search.max_debug_depth=$DEBUG_DEPTH")
[ -n "$DEBUG_PROB" ] && KWARGS_ARGS+=("agent.search.debug_prob=$DEBUG_PROB")
[ -n "$EXEC_TIMEOUT" ] && KWARGS_ARGS+=("exec.timeout=$EXEC_TIMEOUT")
[ -n "$MAX_PARALLEL_WIDTH" ] && KWARGS_ARGS+=("agent.search.max_parallel_width=$MAX_PARALLEL_WIDTH")
[ -n "$CODE_MODEL" ] && KWARGS_ARGS+=("agent.code.model=$CODE_MODEL")
[ -n "$CODE_TEMP" ] && KWARGS_ARGS+=("agent.code.temp=$CODE_TEMP")
[ -n "$FEEDBACK_MODEL" ] && KWARGS_ARGS+=("agent.feedback.model=$FEEDBACK_MODEL")
[ -n "$FEEDBACK_TEMP" ] && KWARGS_ARGS+=("agent.feedback.temp=$FEEDBACK_TEMP")

# Build env-var overrides so container env_vars stay in sync with kwargs.
# config.yaml uses YAML anchors that are resolved at parse time, so env_vars
# like TIME_LIMIT_SECS and STEP_LIMIT won't pick up kwarg overrides otherwise.
ENV_ARGS=()
[ -n "$STEPS" ] && ENV_ARGS+=("STEP_LIMIT=$STEPS")
[ -n "$TIME_LIMIT_SECS" ] && ENV_ARGS+=("TIME_LIMIT_SECS=$TIME_LIMIT_SECS")

echo "Runtime kwargs overrides: ${KWARGS_ARGS[*]:-none}"
echo "Runtime env-var overrides: ${ENV_ARGS[*]:-none}"

echo "Running agent..."
RUN_CMD=(python run_agent.py
    --agent-id $AGENT_ID
    --competition-set /tmp/competition.txt
    --data-dir /data
    --container-config /tmp/container_config.json
)
[ ${#KWARGS_ARGS[@]} -gt 0 ] && RUN_CMD+=(--kwargs "${KWARGS_ARGS[@]}")
[ ${#ENV_ARGS[@]} -gt 0 ] && RUN_CMD+=(--env-vars "${ENV_ARGS[@]}")

"${RUN_CMD[@]}"

# ==========================================
# UPLOAD RESULTS TO S3
# ==========================================
echo "Uploading results to S3..."
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws s3 sync runs/ s3://$S3_RESULTS_BUCKET/results/$INSTANCE_ID/

echo "=========================================="
echo "Run completed at $(date)"
echo "=========================================="

# Optional: shutdown instance when done
sudo shutdown -h now