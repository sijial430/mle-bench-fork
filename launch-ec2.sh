#!bin/bash

# Launch EC2 instance
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

for competition in "tensorflow2-question-answering" \
  "new-york-city-taxi-fare-prediction" \
  "smartphone-decimeter-2022" \
; do
    for agent_id in "aide"; do
        aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
            --instance-type 'c5.4xlarge' \
            --key-name 'sijial-key' \
            --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":64,"VolumeType":"gp3","Throughput":125}}' \
            --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
            --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
            --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
            --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
            --count '1' \
            --user-data file://ec2-startup-aide-dev.sh \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-64gb-6hrs-125steps-0108},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
    done
done


# for competition in "bms-molecular-translation" \
# ; do
#     for agent_id in "aide"; do
#         aws ec2 run-instances --image-id 'ami-0ecb62995f68bb549' \
#             --instance-type 'c5.4xlarge' \
#             --key-name 'sijial-key' \
#             --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-01e432caa3e50ae05","VolumeSize":128,"VolumeType":"gp3","Throughput":125}}' \
#             --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-02d6553372f58264c"]}' \
#             --iam-instance-profile '{"Arn":"arn:aws:iam::304507008116:instance-profile/ecr-auth"}' \
#             --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required","InstanceMetadataTags":"enabled"}' \
#             --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
#             --count '1' \
#             --user-data file://ec2-startup-aide-dev.sh \
#             --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$agent_id-$competition-128gb-6hrs-125steps-0106},{Key=Competition,Value=$competition},{Key=AgentId,Value=$agent_id}]"
#     done
# done

# tensorflow2-question-answering
# 2025-12-17T20-27-52-GMT_run-group_aide/tensorflow2-question-answering_0ca9f870-addc-490e-b655-cb9818dfe908/run.log