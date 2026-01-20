#!/bin/bash

# Launch EC2 instances for ML-Master agent
# Usage: ./launch-ec2-mlmaster.sh
#
# Configuration is passed via EC2 instance tags and read by ec2-startup-mlmaster.sh

# ==========================================
# CONFIGURATION - MODIFY THESE VALUES
# ==========================================
COMPETITIONS=(
    # "tensorflow2-question-answering"
    "new-york-city-taxi-fare-prediction"
    # "smartphone-decimeter-2022"
    # "dog-breed-identification"
    # "facebook-recruiting-iii-keyword-extraction"
    # "billion-word-imputation"
    # Add more competitions here, one per line
)

AGENT_ID="ml-master"
INSTANCE_TYPE="c5.4xlarge"
VOLUME_SIZE_GB=128
USER_DATA="ec2-startup-mlmaster.sh"

# ML-Master agent kwargs (passed to main_mcts.py)
CODE_MODEL="gpt-5.1"
CODE_TEMP="1"
FEEDBACK_MODEL="gpt-5-mini-2025-08-07"
FEEDBACK_TEMP="1"
STEPS="125"
TIME_LIMIT_SECS="21600"
PARALLEL_SEARCH_NUM="3"
NUM_DRAFTS="5"

# ==========================================
# LAUNCH INSTANCES
# ==========================================
for competition in "${COMPETITIONS[@]}"; do
    echo "Launching instance for competition: $competition"

    aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
        --instance-type "$INSTANCE_TYPE" \
        --key-name 'sijial-key' \
        --block-device-mappings "{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"Encrypted\":false,\"DeleteOnTermination\":true,\"Iops\":3000,\"SnapshotId\":\"snap-01e432caa3e50ae05\",\"VolumeSize\":${VOLUME_SIZE_GB},\"VolumeType\":\"gp3\",\"Throughput\":125}}" \
        --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
        --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
        --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
        --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
        --count '1' \
        --user-data file://${USER_DATA} \
        --tag-specifications "ResourceType=instance,Tags=[
            {Key=Name,Value=${AGENT_ID}-${competition}-${VOLUME_SIZE_GB}gb-${STEPS}steps},
            {Key=Competition,Value=${competition}},
            {Key=AgentId,Value=${AGENT_ID}},
            {Key=CodeModel,Value=${CODE_MODEL}},
            {Key=CodeTemp,Value=${CODE_TEMP}},
            {Key=FeedbackModel,Value=${FEEDBACK_MODEL}},
            {Key=FeedbackTemp,Value=${FEEDBACK_TEMP}},
            {Key=Steps,Value=${STEPS}},
            {Key=TimeLimitSecs,Value=${TIME_LIMIT_SECS}},
            {Key=ParallelSearchNum,Value=${PARALLEL_SEARCH_NUM}},
            {Key=NumDrafts,Value=${NUM_DRAFTS}}
        ]"
done

echo "EC2 instances launched for ML-Master agent"
