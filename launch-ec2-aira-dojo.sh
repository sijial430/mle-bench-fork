#!/bin/bash

# Launch EC2 instances for aira-dojo-fork agent
# Usage: ./launch-ec2-aira-dojo.sh
#
# Search Policies Available:
#   - aira-dojo/greedy : Simple greedy search with improvement steps (default)
#   - aira-dojo/mcts   : Monte Carlo Tree Search
#   - aira-dojo/evo    : Evolutionary search with islands and crossover
#
# Development variants (fewer steps for testing):
#   - aira-dojo/dev      : Greedy with 5 steps
#   - aira-dojo/mcts-dev : MCTS with 5 steps
#   - aira-dojo/evo-dev  : Evolutionary with 5 steps

# Example competitions (uncomment as needed):
# "dog-breed-identification" \
# "tensorflow2-question-answering" \
# "facebook-recruiting-iii-keyword-extraction" \
# "new-york-city-taxi-fare-prediction" \
# "smartphone-decimeter-2022" \
# "billion-word-imputation" \

# 128GB ebs volume
# "bms-molecular-translation" \

# 200GB ebs volume
# "vinbigdata-chest-xray-abnormalities-detection" \ # 142GB data
# "siim-covid19-detection" \ # 180GB data
# "icecube-neutrinos-in-deep-ice" \ # 180GB data

  # "tensorflow2-question-answering" \
  # "smartphone-decimeter-2022" \

# ==========================================
# MAIN RUNS - Different search policies
# ==========================================
for competition in "spaceship-titanic" \
; do
    # Run with all three search policies
    for agent_id in "aira-dojo/greedy-dev" "aira-dojo/mcts-dev" "aira-dojo/evo-dev"; do
    # for agent_id in "aira-dojo/greedy-dev"; do
        aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
            --instance-type 'c5.4xlarge' \
            --key-name 'sijial-key' \
            --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":64,"VolumeType":"gp3","Throughput":125}}' \
            --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
            --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
            --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
            --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
            --count '1' \
            --user-data file://ec2-startup-aira-dojo.sh \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-64gb-dev-0107},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
    done
done


# ==========================================
# LARGER STORAGE - 128GB EBS
# ==========================================
# For competitions requiring more storage (128GB):
# for competition in "bms-molecular-translation" \
# ; do
#     for agent_id in "aira-dojo/greedy" "aira-dojo/mcts" "aira-dojo/evo"; do
#         aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
#             --instance-type 'c5.4xlarge' \
#             --key-name 'sijial-key' \
#             --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":128,"VolumeType":"gp3","Throughput":125}}' \
#             --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
#             --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
#             --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
#             --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
#             --count '1' \
#             --user-data file://ec2-startup-aira-dojo.sh \
#             --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-128gb-6hrs-125steps-0107},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
#     done
# done

# ==========================================
# LARGER STORAGE - 200GB EBS
# ==========================================
# For competitions requiring even more storage (200GB):
# for competition in "vinbigdata-chest-xray-abnormalities-detection" \
#   "siim-covid19-detection" \
#   "icecube-neutrinos-in-deep-ice" \
# ; do
#     for agent_id in "aira-dojo/greedy" "aira-dojo/mcts" "aira-dojo/evo"; do
#         aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
#             --instance-type 'c5.4xlarge' \
#             --key-name 'sijial-key' \
#             --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":200,"VolumeType":"gp3","Throughput":125}}' \
#             --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
#             --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
#             --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
#             --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
#             --count '1' \
#             --user-data file://ec2-startup-aira-dojo.sh \
#             --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-200gb-6hrs-125steps-0107},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
#     done
# done

# ==========================================
# SINGLE POLICY RUNS (if you want to test one policy at a time)
# ==========================================
# Greedy only:
# for competition in "spaceship-titanic"; do
#     aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
#         --instance-type 'c5.4xlarge' \
#         --key-name 'sijial-key' \
#         --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":64,"VolumeType":"gp3","Throughput":125}}' \
#         --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
#         --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
#         --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
#         --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
#         --count '1' \
#         --user-data file://ec2-startup-aira-dojo.sh \
#         --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=aira-dojo/greedy-$competition-64gb-6hrs-125steps-0107},{Key=Competition,Value=$competition},{Key=AgentId,Value=aira-dojo/greedy}]"
# done

echo "EC2 instances launched for AIRA-Dojo agent"
