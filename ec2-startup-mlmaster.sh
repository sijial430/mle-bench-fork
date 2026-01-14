#!/bin/bash

# Redirect ALL output to log file (including errors)
exec > >(tee -a /var/log/mlebench-run.log) 2>&1

echo "=========================================="
echo "ML-Master User Data script started at $(date)"
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
COMPETITION_ID="spaceship-titanic"
AGENT_ID="ml-master"
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
    wget -q https://downloads.nestybox.com/sysbox/releases/v0.6.4/sysbox-ce_0.6.4-0.linux_amd64.deb -O /tmp/sysbox.deb
    sudo apt-get install -y jq
    sudo apt-get install -y /tmp/sysbox.deb
    rm /tmp/sysbox.deb
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
# SHARED_ACCOUNT_ID="574455268872"
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "ERROR: Could not get AWS Account ID. Check IAM role!"
    exit 1
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"
# echo "Shared Account ID: $SHARED_ACCOUNT_ID"

# ==========================================
# CLONE AND SETUP MLE-BENCH
# ==========================================
echo "Setting up mle-bench-fork..."
if [ ! -d "/home/ubuntu/mle-bench-fork" ]; then
    echo "Cloning mle-bench-fork repository..."
    cd /home/ubuntu
    sudo -u ubuntu git clone $MLEBENCH_REPO
    cd /home/ubuntu/mle-bench-fork

    echo "Pulling Git LFS files..."
    git lfs pull

    echo "Creating virtual environment..."
    python3 -m venv /home/ubuntu/mle-bench-fork/.venv
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    echo "mle-bench-fork already exists"
    cd /home/ubuntu/mle-bench-fork
    git lfs pull
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
fi

# ==========================================
# DOWNLOAD COMPETITION DATA FROM S3
# ==========================================
echo "Downloading competition data from S3..."
sudo mkdir -p /data
sudo chown ubuntu:ubuntu /data
aws s3 sync s3://$S3_DATA_BUCKET/data/$COMPETITION_ID /data/$COMPETITION_ID

if [ ! -d "/data/$COMPETITION_ID" ]; then
    echo "ERROR: Competition data not found at /data/$COMPETITION_ID"
    echo "Make sure data exists in S3 bucket: s3://$S3_DATA_BUCKET/data/$COMPETITION_ID"
    exit 1
fi
echo "Competition data downloaded successfully"

cd /home/ubuntu/mle-bench-fork
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

# Pull Docker images from ECR
echo "Pulling Docker images from ECR..."
sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-mlmaster:latest

# Tag with local names (so run_agent.py finds them)
echo "Tagging images..."
sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-mlmaster:latest mlmaster:latest

# ==========================================
# RUN THE AGENT
# ==========================================
echo "$COMPETITION_ID" > /tmp/competition.txt

NUM_CPUS=$(nproc)
echo "Detected $NUM_CPUS CPUs"

cat > /tmp/container_config.json << EOF
{
    "mem_limit": "50g",
    "shm_size": "50g",
    "nano_cpus": ${NUM_CPUS}e9,
    "runtime": "sysbox-runc"
}
EOF

echo "Container config:"
cat /tmp/container_config.json

cat > /tmp/container_config_nosysbox.json << EOF
{
    "mem_limit": "50g",
    "shm_size": "50g"
}
EOF

echo "Container config (no sysbox):"
cat /tmp/container_config_nosysbox.json

# ==========================================
# FETCH SECRETS
# ==========================================
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Fetching secrets from AWS Secrets Manager..."

# Fetch OpenAI API Key
SECRET_VALUE=$(aws secretsmanager get-secret-value --secret-id $API_KEY_SECRET_NAME --query SecretString --output text --region $AWS_REGION 2>/dev/null)
if [ $? -eq 0 ]; then
    if echo "$SECRET_VALUE" | jq -e .OPENAI_API_KEY >/dev/null 2>&1; then
        export OPENAI_API_KEY=$(echo "$SECRET_VALUE" | jq -r .OPENAI_API_KEY)
    else
        export OPENAI_API_KEY="$SECRET_VALUE"
    fi
    echo "Successfully loaded OPENAI_API_KEY"
else
    echo "WARNING: Failed to fetch OPENAI_API_KEY from Secrets Manager"
fi

# remove cache to avoid permission issues
echo "Cleaning up cache directory..."
sudo rm -rf /home/ubuntu/mle-bench-fork/cache
sudo rm -rf /home/ubuntu/mle-bench-fork/agents/mlmaster/workspaces
sudo rm -rf /home/ubuntu/mle-bench-fork/agents/mlmaster/logs

echo "Running ML-Master agent..."
python run_agent.py \
    --agent-id $AGENT_ID \
    --competition-set /tmp/competition.txt \
    --data-dir /data \
    --container-config /tmp/container_config_nosysbox.json

# ==========================================
# UPLOAD RESULTS TO S3
# ==========================================
echo "Uploading results to S3..."
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws s3 sync runs/ s3://$S3_RESULTS_BUCKET/results/$INSTANCE_ID/

echo "=========================================="
echo "ML-Master run completed at $(date)"
echo "=========================================="

# Optional: shutdown instance when done
# sudo shutdown -h now
