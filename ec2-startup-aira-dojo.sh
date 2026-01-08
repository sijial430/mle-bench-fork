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

# Defaults if tags are missing / metadata tags not enabled
COMPETITION_ID="$(get_tag Competition)"
AGENT_ID="$(get_tag AgentId)"

: "${COMPETITION_ID:=spaceship-titanic}"
: "${AGENT_ID:=aira-dojo/greedy}"

MLEBENCH_REPO="https://github.com/sijial430/mle-bench-fork.git"
API_KEY_SECRET_NAME="sijial_oai_key"

echo "Competition: $COMPETITION_ID"
echo "Agent: $AGENT_ID"

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
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull base mlebench-env image from ECR
echo "Pulling base mlebench-env image..."
sudo docker pull $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
sudo docker tag $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest mlebench-env:latest

# ==========================================
# BUILD aira-dojo AGENT IMAGE
# ==========================================
echo "Building aira-dojo agent image..."

export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

# Build the aira-dojo agent image
# Note: The Dockerfile is in agents/ and expects aira-dojo/ subdirectory
cd /home/ubuntu/mle-bench-fork/agents
sudo docker build --platform=linux/amd64 \
  --build-arg BASE_IMAGE=mlebench-env:latest \
  -t aira-dojo \
  -f aira-dojo/Dockerfile . \
  --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
  --build-arg LOGS_DIR=$LOGS_DIR \
  --build-arg CODE_DIR=$CODE_DIR \
  --build-arg AGENT_DIR=$AGENT_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build aira-dojo image"
    exit 1
fi

echo "aira-dojo image built successfully"

# Return to mlebench directory
cd /home/ubuntu/mle-bench-fork

# ==========================================
# RUN THE AGENT
# ==========================================
# Create competition file
echo "$COMPETITION_ID" > /tmp/competition.txt

NUM_CPUS=$(nproc)
echo "Detected $NUM_CPUS CPUs"

cat > /tmp/container_config.json << EOF
{
    "mem_limit": "64g",
    "shm_size": "64g",
    "nano_cpus": ${NUM_CPUS}e9,
    "runtime": "sysbox-runc"
}
EOF

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

echo "Running agent..."
python run_agent.py \
    --agent-id $AGENT_ID \
    --competition-set /tmp/competition.txt \
    --data-dir /data \
    --container-config /tmp/container_config.json

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
