#!/bin/bash

# Launch EC2 instances for AIDE agent
# Usage: ./launch-ec2.sh
#
# Configuration is passed via EC2 instance tags and read by ec2-startup-aide-dev.sh

# ==========================================
# CONFIGURATION - MODIFY THESE VALUES
# ==========================================
COMPETITIONS=(
    # "tensorflow2-question-answering"
    # "new-york-city-taxi-fare-prediction"
    # "smartphone-decimeter-2022"
    # "dog-breed-identification"
    "facebook-recruiting-iii-keyword-extraction"
    # "billion-word-imputation"
    # "bms-molecular-translation"  # 128GB volume needed
    # "vinbigdata-chest-xray-abnormalities-detection"  # 200GB volume needed (142GB data)
    # "siim-covid19-detection"  # 200GB volume needed (180GB data)
    # "icecube-neutrinos-in-deep-ice"  # 200GB volume needed (180GB data)
)

AGENT_ID="aide"
INSTANCE_TYPE="c5.4xlarge"
VOLUME_SIZE_GB=64
USER_DATA="ec2-startup-aide-dev.sh"

# AIDE agent kwargs (passed via omegaconf)
STEPS="125"
TIME_LIMIT_SECS="21600"
DEBUG_DEPTH="10"
DEBUG_PROB="0.7"

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
            {Key=Name,Value=${AGENT_ID}-${competition}-${VOLUME_SIZE_GB}gb-${STEPS}steps-debug-depth-${DEBUG_DEPTH}-debug-prob-${DEBUG_PROB}},
            {Key=Competition,Value=${competition}},
            {Key=AgentId,Value=${AGENT_ID}},
            {Key=Steps,Value=${STEPS}},
            {Key=TimeLimitSecs,Value=${TIME_LIMIT_SECS}},
            {Key=DebugDepth,Value=${DEBUG_DEPTH}},
            {Key=DebugProb,Value=${DEBUG_PROB}}
        ]"
done

echo "EC2 instances launched for AIDE agent"