#!/bin/bash

# Redirect ALL output to log file (including errors)
exec > >(tee -a /var/log/mlebench-run.log) 2>&1

echo "=== User Data script started at $(date) ==="
set -x
sleep 10

# --- CONFIGURATION ---
AWS_REGION="us-east-1"
S3_DATA_BUCKET="mlebench-data"
S3_RESULTS_BUCKET="mlebench-results"
# COMPETITION_ID="${1:-spaceship-titanic}" #tensorflow2-question-answering
# AGENT_ID="${2:-aide/dev}"

API_KEY_SECRET_NAME="sijial_oai_key"
SLACK_SECRET_NAME="slack_webhook_url"

# --- FETCH SECRETS ---
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

# Fetch Slack webhook URL from Secrets Manager
SLACK_SECRET=$(aws secretsmanager get-secret-value --secret-id slack_webhook_url --query SecretString --output text --region $AWS_REGION 2>/dev/null)
if [ $? -eq 0 ]; then
    if echo "$SLACK_SECRET" | jq -e .slack_webhook_url >/dev/null 2>&1; then
        SLACK_WEBHOOK_URL=$(echo "$SLACK_SECRET" | jq -r .slack_webhook_url)
    else
        SLACK_WEBHOOK_URL="$SLACK_SECRET"
    fi
    echo "Successfully loaded SLACK_WEBHOOK_URL"
else
    echo "WARNING: Failed to fetch SLACK_WEBHOOK_URL from Secrets Manager"
fi

# --- NOTIFICATION ---
send_notification() {
    local status="$1"    # "started", "success", or "error"
    local message="$2"
    
    # Get instance metadata
    local imds_token=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --connect-timeout 2 2>/dev/null || echo "")
    
    local instance_id="unknown"
    local instance_type="unknown"
    if [ -n "$imds_token" ]; then
        instance_id=$(curl -s -H "X-aws-ec2-metadata-token: $imds_token" \
            http://169.254.169.254/latest/meta-data/instance-id --connect-timeout 2 2>/dev/null || echo "unknown")
        instance_type=$(curl -s -H "X-aws-ec2-metadata-token: $imds_token" \
            http://169.254.169.254/latest/meta-data/instance-type --connect-timeout 2 2>/dev/null || echo "unknown")
    fi
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')
    
    local emoji color title
    case "$status" in
        started)
            emoji=":rocket:"
            color="#36a64f"
            title="MLE-Bench Instance Started"
            ;;
        success)
            emoji=":white_check_mark:"
            color="#36a64f"
            title="MLE-Bench Run Completed Successfully"
            ;;
        error)
            emoji=":x:"
            color="#dc3545"
            title="MLE-Bench Run Failed"
            ;;
        *)
            emoji=":information_source:"
            color="#0066cc"
            title="MLE-Bench Notification"
            ;;
    esac
    
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        echo "Sending Slack notification..."
        local preview_text="$emoji $title - $COMPETITION_ID ($AGENT_ID)"
        local slack_payload=$(cat <<EOF
{
    "text": "$preview_text",
    "attachments": [
        {
            "color": "$color",
            "fallback": "$preview_text",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "$emoji $title",
                        "emoji": true
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": "*Instance ID:*\n$instance_id"},
                        {"type": "mrkdwn", "text": "*Instance Type:*\n$instance_type"},
                        {"type": "mrkdwn", "text": "*Competition:*\n$COMPETITION_ID"},
                        {"type": "mrkdwn", "text": "*Agent:*\n$AGENT_ID"},
                        {"type": "mrkdwn", "text": "*Timestamp:*\n$timestamp"}
                    ]
                }
            ]
        }
    ]
}
EOF
)
        if [ -n "$message" ]; then
            slack_payload=$(echo "$slack_payload" | jq --arg msg "$message" '.attachments[0].blocks += [{"type": "section", "text": {"type": "mrkdwn", "text": ("*Details:*\n" + $msg)}}]')
        fi
        
        curl -s -X POST -H 'Content-type: application/json' \
            --data "$slack_payload" \
            "$SLACK_WEBHOOK_URL" || echo "Warning: Failed to send Slack notification"
    fi
}

# --- ERROR TRAP ---
RUN_STATUS="error"
LAST_ERROR=""

trap_handler() {
    local exit_code=$?
    local line_number=$1
    
    if [ "$RUN_STATUS" != "success" ] && [ $exit_code -ne 0 ]; then
        LAST_ERROR="Script failed at line $line_number with exit code $exit_code"
        echo "ERROR: $LAST_ERROR"
        
        # Flush output buffers to disk
        sync
        sleep 1
        
        # Capture last 10 lines of log for error context
        local error_logs=""
        if [ -f /var/log/mlebench-run.log ]; then
            error_logs=$(cat /var/log/mlebench-run.log 2>/dev/null | tail -n 10 | head -c 1500 || echo "Could not read logs")
        fi
        
        # If still empty, note that
        if [ -z "$error_logs" ]; then
            error_logs="(logs not available - check /var/log/mlebench-run.log on instance)"
        fi
        
        send_notification "error" "$LAST_ERROR

\`\`\`
$error_logs
\`\`\`"
        # Shutdown instance on error
        sudo shutdown -h now
        exit $exit_code
    fi
}

trap 'trap_handler $LINENO' ERR

# --- SEND STARTED NOTIFICATION ---
send_notification "started" "The EC2 instance has started and is beginning setup."

# --- INSTANCE TAGS (IMDSv2) ---
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

# --- INSTALL DEPENDENCIES ---
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y

if ! command -v docker &> /dev/null; then
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ubuntu
fi

if ! command -v sysbox-runc &> /dev/null; then
    wget -q https://downloads.nestybox.com/sysbox/releases/v0.6.4/sysbox-ce_0.6.4-0.linux_amd64.deb -O /tmp/sysbox.deb
    sudo apt-get install -y jq /tmp/sysbox.deb
    rm /tmp/sysbox.deb
    sudo systemctl restart docker
fi

if ! command -v aws &> /dev/null; then
    sudo apt-get install -y unzip curl
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi

sudo apt-get install -y python3 python3-pip python3-venv git

if ! command -v git-lfs &> /dev/null; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
    sudo -u ubuntu git lfs install
fi

# --- GET AWS ACCOUNT INFO ---
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region $AWS_REGION)
SHARED_ACCOUNT_ID="574455268872"
[ -z "$AWS_ACCOUNT_ID" ] && { echo "ERROR: Could not get AWS Account ID"; exit 1; }

# --- CLONE AND SETUP MLE-BENCH ---
if [ ! -d "/home/ubuntu/mle-bench-fork" ]; then
    cd /home/ubuntu
    sudo -u ubuntu git clone $MLEBENCH_REPO
    cd /home/ubuntu/mle-bench-fork
    [ -n "$GIT_BRANCH" ] && sudo -u ubuntu git checkout "$GIT_BRANCH"
    sudo -u ubuntu git lfs pull
    python3 -m venv /home/ubuntu/mle-bench-fork/.venv
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    cd /home/ubuntu/mle-bench-fork
    if [ -n "$GIT_BRANCH" ]; then
        sudo -u ubuntu git checkout "$GIT_BRANCH"
        sudo -u ubuntu git pull origin "$GIT_BRANCH"
    fi
    sudo -u ubuntu git lfs pull
    source /home/ubuntu/mle-bench-fork/.venv/bin/activate
fi

# --- DOWNLOAD COMPETITION DATA ---
sudo mkdir -p /data
sudo chown ubuntu:ubuntu /data
aws s3 sync s3://$S3_DATA_BUCKET/data/$COMPETITION_ID /data/$COMPETITION_ID
[ ! -d "/data/$COMPETITION_ID" ] && { echo "ERROR: Competition data not found"; exit 1; }

cd /home/ubuntu/mle-bench-fork
source /home/ubuntu/mle-bench-fork/.venv/bin/activate

# --- SETUP DOCKER ---
sudo systemctl start docker || true
sudo usermod -aG docker ubuntu || true
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
sudo docker pull $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest
sudo docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide-dev:latest

# Tag with local names (so run_agent.py finds them)
sudo docker tag $SHARED_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-env:latest mlebench-env:latest
sudo docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mlebench-aide-dev:latest aide:latest

# --- RUN THE AGENT ---
echo "$COMPETITION_ID" > /tmp/competition.txt
NUM_CPUS=$(nproc)
cat > /tmp/container_config.json << EOF
{"mem_limit":"64g","shm_size":"64g","nano_cpus":${NUM_CPUS}e9,"runtime":"sysbox-runc"}
EOF
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

ENV_ARGS=()
[ -n "$STEPS" ] && ENV_ARGS+=("STEP_LIMIT=$STEPS")
[ -n "$TIME_LIMIT_SECS" ] && ENV_ARGS+=("TIME_LIMIT_SECS=$TIME_LIMIT_SECS")

RUN_CMD=(python run_agent.py
    --agent-id $AGENT_ID
    --competition-set /tmp/competition.txt
    --data-dir /data
    --container-config /tmp/container_config.json
)
[ ${#KWARGS_ARGS[@]} -gt 0 ] && RUN_CMD+=(--kwargs "${KWARGS_ARGS[@]}")
[ ${#ENV_ARGS[@]} -gt 0 ] && RUN_CMD+=(--env-vars "${ENV_ARGS[@]}")

"${RUN_CMD[@]}"

# --- UPLOAD RESULTS ---
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws s3 sync runs/ s3://$S3_RESULTS_BUCKET/results/$INSTANCE_ID/

echo "=== Run completed at $(date) ==="
sudo shutdown -h now