# Whisper audio transcription powered by AWS Batch and AWS Inferentia

OpenAI's Whisper model is a highly accurate automatic speech recognition (ASR) model that is based on the transformer architecture. It is released under MIT license and available for commercial use. While Amazon SageMaker offers a robust platform for hosting the Whisper model, running a SageMaker inference endpoint continuously can incur unnecessary costs especially if the endpoint isn’t fully utilized and less frequent batch processing is sufficient. If you are looking to host a Whisper model on an Amazon SageMaker inference endpoint, please refer to the blog post on that topic, Host the Whisper Model on Amazon SageMaker: Exploring Inference Options.

For customers seeking to host the Whisper model on AWS but do not need real-time or continuous online inference can optimize for cost by processing audio as a batch workload asynchronously using AWS Batch and by leveraging AWS’s custom silicon for inference, AWS Inferentia. AWS Batch is a fully managed service that enables efficient scheduling and execution of batch computing workloads on the AWS Cloud. AWS Inferentia is a custom-built machine learning inference chip designed to deliver exceptional price per performance.

## Highlights of What’s New in Version v2.0.0

### Performance Improvements

1.1 Preloaded model files into container image to eliminate download time during execution

1.2 Added projection layer optimization resulting in 30% performance improvement

1.3 Reduced Batch job resource requirements and Neuron core number to enable 2 concurrent jobs per inf2 chip on the same instance

### Additional Updates

2.1 Updated to dynamically retrieve the latest ECS Neuron AMI for Batch compute environment provisioning

2.2 Updated Dockerfile with new base image configuration

## Deploy the solution
You can use the AWS CloudFormation template to accelerate the deployment of this solution. In the next section, we describe step-by-step how to implement this solution via the AWS Console.

To deploy the solution using a cloud formation template by proceeding with the following steps:
1.	Choose Launch Stack below to launch the solution in the us-east-1 Region:
   
[<img src="https://github.com/user-attachments/assets/a7897b5a-8722-480d-b385-a9ba82c48cbd" width=auto height=auto />](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=awsbatch-inferentia&templateURL=https://aws-hpc-recipes.s3.amazonaws.com/main/recipes/batch/whisper_transcription_awsbatch_inferentia/assets/deployment.yaml)

2.	For Stack name, enter a unique stack name.
3.	Set the parameters.

Alternatively, you can run the buildArch.sh script to deploy the infrastructure automatically to the default VPC.

## Solution overview

The solution architecture depicted in Figure 1 below shows the event-driven Whisper audio transcription pipeline solution with Amazon EventBridge and AWS Batch. Since AWS Batch creates a containerized compute environment, it requires a container image to be built and staged in Amazon Elastic Container Registry (Amazon ECR). In the AWS Batch job definition, we specify the container image and resource requirements for each job. The Amazon Machine Image (AMI) and Amazon EC2 instance type are configured in AWS Batch compute environment.

![hpc-whisper-batch-inferentia](https://github.com/user-attachments/assets/000e9019-2197-4836-a6a4-b1807c49c8d1)
Figure 1. Event-driven audio transcription pipeline with Amazon EventBridge and AWS Batch

We’ll discuss each numbered architecture feature.

1.	Build docker image and push to Amazon ECR

The docker image is built at design time and pushed to Amazon ECR. The docker image includes all the libraries and python script code to retrieve the model artifacts and the audio file from Amazon S3 and process it using the Whisper model encoder and decoder.

2.	Add audio transcription job to AWS Batch job queue on file upload

When an audio file is uploaded to Amazon S3 in the designated Bucket and folder, the PutEvent is captured by Amazon EventBridge, which triggers AWS Batch to add the file to the job queue.

3.	AWS Batch launches compute resources to process queued jobs

If there’s no active compute resource available, AWS Batch will automatically scale the managed compute environment to process each job in the queue until the queue is empty. Amazon EC2 Inf2 instances with an Amazon Machine Image (AMI) that includes the Neuron SDK libraries will start and pull the container image uploaded to Amazon ECR in the first step.

4.	AWS Batch processes job and writes transcription to Amazon S3 output location

The compute environment processes each audio file and writes a transcription file to an Amazon S3 output location. Once the job queue is empty, AWS Batch scales down the environment.

Next, we will demonstrate how to implement this solution.

## Implementing the solution

### Choose AMI

The AMI we will choose for our AWS Batch compute environment will be an Amazon Linux AMI that is ECS-optimized. This AMI has Neuron drivers pre-installed to support AWS Inferentia2 instances. Remember that AMIs are region-specific, and since we are using the us-east-1 Region, we will use this community AMI published by AWS: ami-07979ad4c774a29ab. We can search and find this AMI in the Application and OS Images (Amazon Machine Image) catalog search in the Amazon EC2 Launch an instance screen (shown below).

<p align="center"><img src="https://github.com/user-attachments/assets/4778d09f-9a0f-49c8-9632-2babe5c0f29f" width=auto height=auto />
</p>

### Build the docker image

After cloning the GitHub repository, locate the Dockerfile. Notice that the referenced base image installs the AWS Neuron SDK for PyTorch 1.13.1 on AWS Inferentia.

```
FROM public.ecr.aws/neuron/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY inference.py inference.py

RUN pip install -U --no-cache-dir -r requirements.txt

# Exit container after the job is done
RUN sed -i '/prevent docker exit/ {n; s/./# &/;}' /usr/local/bin/dockerd-entrypoint.py

# Ensure the model is cached during the image build rather than during runtime
RUN python3 -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    model_id='openai/whisper-large-v3'; \
    WhisperProcessor.from_pretrained(model_id); \
    WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)"

# Stage script to create model artifacts
COPY export-model.py export-model.py

# For inference
CMD ["python3", "inference.py"]

```

We will build the image using the following command.

```
docker build --no-cache -t whisper:latest .
```

Your output should look like this.

<p align="center"><img src="https://github.com/user-attachments/assets/69f86d6f-a5e4-41e9-ba66-cabc26096017" width=auto height=auto/>
</p>

### Push image to AWS ECR

With the image built locally, we need to create a repository in Amazon ECR and push our image to it so that it will be accessible for AWS Batch to use when launching compute environments.

```
aws ecr create-repository --repository-name whisper
```

We then tag the Docker image to match the repository URL.

```
docker tag whisper:latest [your-account-id].dkr.ecr.[your-region].amazonaws.com/whisper:latest
```
Now, get the credential to access the ECR repositories:

```
aws ecr get-login-password | docker login --username AWS --password-stdin [your-account-id].dkr.ecr.[your-region].amazonaws.com
```

Finally, we push the image to the Amazon ECR repository.

```
docker push [your-account-id].dkr.ecr.[your-region].amazonaws.com/whisper:latest
```

### Export the model artifacts

The section describes how to run the export-model.py script to export the model files. The Dockerfile stages the export-model.py script which will create the encoder and decoder files locally.

#### Step 1. Launch an inf2.8xlarge EC2 instance and SSH in.

Ensure that the instance IAM role has permissions to pull images from Amazon ECR.

#### Step 2. Pull image from Amazon ECR.

```
docker pull [your-account-id].dkr.ecr.[your-region].amazonaws.com/whisper:latest
```

#### Step 3. Run container and attach to shell

```
docker run --device /dev/neuron0 -it whisper /bin/bash
```

#### Step 4. Once attached to the container, run the export-model.py script.

```
python3 export-model.py
```

#### Step 5. Exit the container and copy the files to the host

```
sudo docker cp whisper:/whisper_large-v3_1_neuron_encoder.pt .
sudo docker cp whisper:/whisper_large-v3_1_448_neuron_decoder.pt .
sudo docker cp whisper:/whisper_large-v3_1_448_neuron_proj.pt .
```

#### Step 6. Once these files are copied to the host you can then upload them to the S3 location you've designated for your model artifacts.

```
aws s3 cp ./whisper_large-v3_1_neuron_encoder.pt s3://awsbatch-audio-transcription-us-east-1-123456789012-inbucket/model-artifacts/whisper_large-v3_1_neuron_encoder.pt
aws s3 cp ./whisper_large-v3_1_448_neuron_decoder.pt s3://awsbatch-audio-transcription-us-east-1-123456789012-inbucket/model-artifacts/whisper_large-v3_1_448_neuron_decoder.pt
aws s3 cp ./whisper_large-v3_1_448_neuron_proj.pt s3://awsbatch-audio-transcription-us-east-1-123456789012-inbucket/model-artifacts/whisper_large-v3_1_448_neuron_proj.pt
```

### Create AWS Batch job queue -- You can skip the following steps if you already deployed the solution with the AWS CloudFormation template

AWS Batch provides a Wizard (accessible from the menu in the upper left) that steps you through the configuration. You’ll create a compute environment for the inf2.8xlarge instance, a job queue, and a job definition where you will specify the repository URL for the Docker image that we built and uploaded to Amazon ECR.

#### Step 1. Configure job and orchestration type

Choose Amazon Elastic Compute Cloud (Amazon EC2) and click Next.

<p align="center"><img src="https://github.com/user-attachments/assets/b56aca61-e021-4415-9a36-f4e09894847a" width=auto height=auto />
</p>

#### Step 2. Create a compute environment

In the Create a compute environment section, choose a Name and Instance role. Choose the 
minimum and maximum vCPUs. Under Allowed instance types, clear the default and select inf2.8xlarge.

<p align="center"><img src="https://github.com/user-attachments/assets/053e2e91-0cb0-4844-92ae-039007a94603" width=auto height=auto />
</p>


The inf2.8xlarge instance has 32 vCPUs, so setting the maximum vCPU to a multiple of 32 number will determine the throughput capacity of the AWS Batch job queue—or how many EC2 instances AWS Batch will launch to transcribe multiple audio files concurrently.  For our experiment we will choose to process 5 audio files concurrently, so we will set the maximum vCPUs at 160. Once 5 EC2 instances are launched, jobs will simply wait in queue until the EC2 instances become available.

In the Network configuration, choose the Amazon Virtual Private Cloud (Amazon VPC) the compute environment will be deployed in. Select the Subnets and Security groups that will control access to the environment.

#### Step 3. Create a job queue

In the Job queue configuration section, create a Name for the Job queue and set the Priority. Job queues with a higher integer value for priority are given preference for compute environments. Click Next.

<p align="center"><img src="https://github.com/user-attachments/assets/fcbff5d7-89cb-48a5-8ee9-34b91de3d4e0">
</p>


#### Step 4. Create a job definition

Choose a Name for the job definition. If desired, set an Execution timeout greater than 60 seconds to terminate unfinished jobs.

In the Container configuration section, specify the repository URL for the Docker image that we built and uploaded to Amazon ECR in earlier steps.

<p align="center"><img src="https://github.com/user-attachments/assets/fc291f71-e688-4973-9398-a1cc0541221f">
</p>

In the Environment configuration section, add the following environment variables by clicking Add environment variable and specifying the Name and Value for each as shown below.

<p align="center"><img src="https://github.com/user-attachments/assets/a2a58624-8f75-4345-a8f3-726912b664ca">
</p>

Add any additional configurations you need for your use case and click Next.

#### Step 5. Submit job

Next the wizard will ask you to Create a job that will be submitted when the AWS Batch resources are created in the next step.

<p align="center"><img src="https://github.com/user-attachments/assets/539abaff-72b3-4220-a769-7bf80e013997">
</p>


#### Step 6. Review and create

Review all the configurations listed and choose Create resources. Resources such as the job queue and job definition will be created and the test job will be submitted for execution.

Now that we have our AWS Batch workflow configured, the last component we need to configure is the Amazon EventBridge Rule that will add new audio files to our AWS Batch job queue.

### Create Amazon EventBridge Event Rule

Amazon EventBridge Event Bus is a serverless event bus that helps you receive, filter, transform, route, and deliver events. Events generated by AWS services are visible through Amazon EventBridge and can be used to trigger downstream actions on one or more target services.

In the Amazon EventBridge console under the Buses menu, choose Rule and click Create rule. The Create rule wizard will step you through the creation of the event rule.

#### Step 1. Define rule detail
Specify the Name and Description for the event rule. For Event bus choose default and choose Enable the rule on the selected event bus.

<p align="center"><img src="https://github.com/user-attachments/assets/de6edb11-4cf9-4766-81aa-d252fd81d7bf">
</p>

Choose Rule with an event pattern as the Rule type then click Next.

#### Step 2. Build event pattern
To build the event pattern we will choose the Event source as AWS events or EventBridge partner events. Under Sample event we can choose AWS events as the Sample event type and choose Object Created to view the standard JSON event pattern we can customize by replacing example-bucket and example-key.

```
{
  "detail-type": ["Object Created"],
  "source": ["aws.s3"],
  "detail": {
    "bucket": {
      "name": ["awsbatch-audio-transcription-us-east-1-123456789012-inbucket"]
    },
    "object": {
      "key": [{
        "prefix": "audio-input/"
      }]
    }
  },
  "account": ["123456789012"]
}
```

<p align="center"><img src="https://github.com/user-attachments/assets/097ece3c-4d8e-441c-8af8-853ab4f71b4d">
</p>

Click Copy to copy the JSON to the clipboard. Paste it into the Event pattern dialog box in in the next section.

Amazon EventBridge captures all Amazon S3 events and the Rule filters the events down to a specific bucket and folder that we have designated as the start of our pipeline. The Rule’s Event pattern is below.

#### Step 3. Select target(s)
Next, we select the target. Under Target types, choose AWS service. Choose Batch job queue as the target.

<p align="center"><img src="https://github.com/user-attachments/assets/5daf63a2-83ef-4675-a8e9-d8dca688a386" \></p>

After selecting Batch job queue, you’ll be prompted to specify the Amazon Resource Names (ARNs) for the AWS Batch Job queue and the Job definition.

Under "Addiontal settings", choose  "Input transformer", then click "Configure input Transformer" button.

<p align="center"><img src="https://github.com/user-attachments/assets/8931913d-9f60-446c-b850-66fc59f30e9d" \></p>

Set up "Input path" and "Template" like below. This will pass the S3 bucket name and the S3 object key to the triggered Batch job. Click Confirm and go to the next step.

<p align="center"><img src="https://github.com/user-attachments/assets/2153e60d-65a1-42cd-94c6-b884e1a9dc2c" \></p>

#### Step 4. Configure tags – optional

You can configure tags to help you track costs of this event rule in cost explorer.

#### Step 5. Review and create

In the last step, you’ll be able to review the rule details, event pattern, targets, and tags. Once you’ve confirmed they’re correct, choose Create rule.

Now that we’ve fully configured our event-driven audio transcription pipeline, we’re ready to test it.

## Test the solution

You're ready to test the solution. Upload some audio files to your designated S3 input location and validate the transcriptions are being outputted to the S3 output location.

## HPC Recipes

This [solution](https://github.com/aws-samples/aws-hpc-recipes/tree/main/recipes/batch/whisper_transcription_awsbatch_inferentia) can be found on [HPC Recipes](https://github.com/aws-samples/aws-hpc-recipes).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
