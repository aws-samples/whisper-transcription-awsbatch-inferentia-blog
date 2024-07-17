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

# This script will create an ECR repository for the first time, then update image

project=whisper

# Get the AWS account and region information dynamically
if [[ ! -z $CODEBUILD_BUILD_ARN ]]; then # build with CodeBuild
  export AWS_ACCOUNT=$(echo ${CODEBUILD_BUILD_ARN} | cut -d\: -f5)
else  # build locally on cloud 9
  echo "Updating images in ${AWS_REGION:=$(TOKEN=`curl --silent  -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`; \
       curl --silent -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)} region"
  export AWS_REGION
  export AWS_ACCOUNT=$(aws sts get-caller-identity --output text --query Account)
fi



# build docker image
DOCKER_BUILDKIT=1  docker build -t $project -f Dockerfile ./

# Create an ECR repository if not exist
aws ecr describe-repositories --repository-names $project --region ${AWS_REGION} --no-cli-pager|| \
aws ecr create-repository --repository-name $project --region ${AWS_REGION} --no-cli-pager --image-scanning-configuration scanOnPush=true

# Tag the new image
docker tag $project:latest ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/$project:latest

# Push the new image to the ECR repository
aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/$project:latest
