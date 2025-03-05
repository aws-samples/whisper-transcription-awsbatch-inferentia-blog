#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# This script will deploy the infrastructure for this project

project=awsbatch-audio-transcription

# Get the AWS account and region information dynamically, or specify region with the environment variable.
echo "Deploying to ${AWS_REGION:=$(TOKEN=`curl --silent  -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`; \
 curl --silent -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)} region"

##### Gather necessary information and save in a file for later use #####
envFileName=~/envVars-$AWS_REGION

echo "export AWS_REGION=${AWS_REGION}" > $envFileName

AWS_ACCOUNT=$(aws sts get-caller-identity --query "Account" --output text)
echo "export AWS_ACCOUNT=${AWS_ACCOUNT}" >> $envFileName

VPC_ID=`aws ec2 describe-vpcs --output text --query 'Vpcs[*].VpcId' --filters Name=isDefault,Values=true --region ${AWS_REGION}`
echo "export VPC_ID=${VPC_ID}" >> $envFileName

SUBNET_IDS=`aws ec2 describe-subnets --query "Subnets[*].SubnetId" --filters Name=vpc-id,Values=${VPC_ID} --region ${AWS_REGION} --output text | sed 's/\s\+/,/g'`
echo "export SUBNET_IDS=${SUBNET_IDS}" >> $envFileName

SecurityGroup_IDS=`aws ec2 describe-security-groups  --query 'SecurityGroups[*].GroupId' \
                 --filters Name=vpc-id,Values=${VPC_ID}  Name=group-name,Values=default --region ${AWS_REGION} --output text`
echo "export SecurityGroup_IDS=${SecurityGroup_IDS}" >> $envFileName

RouteTable_IDS=`aws ec2 describe-route-tables --query 'RouteTables[*].RouteTableId' \
                 --filters Name=vpc-id,Values=${VPC_ID} --region ${AWS_REGION} --output text | sed 's/\s\+/,/g'`
echo "export RouteTable_IDS=${RouteTable_IDS}" >> $envFileName


##### Provision infrastructure #####
aws cloudformation deploy --stack-name $project --template-file ./deployment.yaml --capabilities CAPABILITY_IAM \
--region ${AWS_REGION} --parameter-overrides VPCId=${VPC_ID} SubnetIds="${SUBNET_IDS}" SGIds="${SecurityGroup_IDS}" RTIds="${RouteTable_IDS}"
