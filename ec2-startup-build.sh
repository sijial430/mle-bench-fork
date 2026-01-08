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
COMPETITION_ID="spaceship-titanic"
AGENT_ID="dummy"
MLEBENCH_REPO="https://github.com/sijial430/mle-agent.git"  # Change to your repo if forked

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
# CLONE AND SETUP MLE-BENCH
# ==========================================
echo "Setting up mle-agent..."
if [ ! -d "/home/ubuntu/mle-agent" ]; then
    echo "Cloning mle-agent repository..."
    cd /home/ubuntu
    sudo -u ubuntu git clone $MLEBENCH_REPO
    cd /home/ubuntu/mle-agent
    
    # Pull LFS files (leaderboards, CSVs, top solutions)
    echo "Pulling Git LFS files..."
    git lfs pull
    
    # Create virtual environment and install mle-agent
    echo "Creating virtual environment..."
    python3 -m venv /home/ubuntu/mle-agent/.venv
    source /home/ubuntu/mle-agent/.venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    echo "mle-agent already exists"
    cd /home/ubuntu/mle-agent
    # Make sure LFS files are up to date
    git lfs pull
    # Activate existing venv
    source /home/ubuntu/mle-agent/.venv/bin/activate
fi

# ==========================================
# GET AWS ACCOUNT INFO
# ==========================================
echo "Getting AWS Account ID..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $AWS_REGION)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "ERROR: Could not get AWS Account ID. Check IAM role!"
    exit 1
fi
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# ==========================================
# SETUP DOCKER
# ==========================================
echo "Setting up Docker..."
sudo systemctl start docker || true
sudo usermod -aG docker ubuntu || true

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull images from ECR
echo "Pulling Docker images..."
sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-dummy:latest

# Build the base environment image
# sudo docker build --build-arg INSTALL_HEAVY_DEPENDENCIES=false -t mlebench-env -f environment/Dockerfile .

# Tag and push to ECR
# sudo docker tag mlebench-env:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
# sudo docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest

# Build the dummy agent
# export SUBMISSION_DIR=/home/submission
# export LOGS_DIR=/home/logs
# export CODE_DIR=/home/code
# export AGENT_DIR=/home/agent

# sudo docker build --platform=linux/amd64 \
#   --build-arg BASE_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest \
#   -t dummy agents/dummy/ \
#   --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
#   --build-arg LOGS_DIR=$LOGS_DIR \
#   --build-arg CODE_DIR=$CODE_DIR \
#   --build-arg AGENT_DIR=$AGENT_DIR

# Tag and push dummy agent
# sudo docker tag dummy:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-dummy:latest
# sudo docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-dummy:latest

# Tag with local names (so run_agent.py finds them)
echo "Tagging images..."
sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest mlebench-env:latest
sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-dummy:latest dummy:latest

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
cd /home/ubuntu/mle-agent

# Ensure venv is activated
source /home/ubuntu/mle-agent/.venv/bin/activate

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
    "mem_limit": "512m",
    "shm_size": "256m",
    "nano_cpus": ${NUM_CPUS}e9,
    "runtime": "sysbox-runc"
}
EOF
#####

echo "Container config:"
cat /tmp/container_config.json

echo "Running agent..."
python run_agent.py \
    --agent-id $AGENT_ID \
    --competition-set /tmp/competition.txt \
    --data-dir /data \
    --container-config /tmp/container_config.json
    # --container-config /home/ubuntu/mle-agent/environment/config/container_configs/small.json

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