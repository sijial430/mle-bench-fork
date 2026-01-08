#!/bin/bash

# Launch EC2 instances for RD-Agent
# Usage: ./launch-ec2-rdagent.sh

for competition in "spaceship-titanic" \
; do
    for agent_id in "rd-agent/dev"; do
        aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
            --instance-type 'c5.4xlarge' \
            --key-name 'sijial-key' \
            --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":64,"VolumeType":"gp3","Throughput":125}}' \
            --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
            --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
            --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
            --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
            --count '1' \
            --user-data file://ec2-startup-rdagent.sh \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-64gb-dev-0107},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
    done
done

echo "EC2 instances launched for RD-Agent"
