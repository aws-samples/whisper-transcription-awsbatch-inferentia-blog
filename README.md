# Whisper audio transcription powered by AWS Batch and AWS Inferentia

OpenAI's Whisper model is a highly accurate automatic speech recognition (ASR) model that is based on the transformer architecture. It is released under MIT license and available for commercial use. While Amazon SageMaker offers a robust platform for hosting the Whisper model, running a SageMaker inference endpoint continuously can incur unnecessary costs especially if the endpoint isn’t fully utilized and less frequent batch processing is sufficient. If you are looking to host a Whisper model on an Amazon SageMaker inference endpoint, please refer to the blog post on that topic, Host the Whisper Model on Amazon SageMaker: Exploring Inference Options.

For customers seeking to host the Whisper model on AWS but do not need real-time or continuous online inference can optimize for cost by processing audio as a batch workload asynchronously using AWS Batch and by leveraging AWS’s custom silicon for inference, AWS Inferentia. AWS Batch is a fully managed service that enables efficient scheduling and execution of batch computing workloads on the AWS Cloud. AWS Inferentia is a custom-built machine learning inference chip designed to deliver exceptional price per performance.

## Deploy the solution
You can use the AWS CloudFormation template to accelerate the deployment of this solution. In the next section, we describe step-by-step how to implement this solution via the AWS Console.

To deploy the solution using a cloud formation template by proceeding with the following steps:
1.	Choose Launch Stack below to launch the solution in the us-east-1 Region:
   
![launchstack](https://github.com/user-attachments/assets/a7897b5a-8722-480d-b385-a9ba82c48cbd)

3.	For Stack name, enter a unique stack name.
4.	Set the parameters.

## Solution overview

The solution architecture depicted in Figure 1 below shows the event-driven Whisper audio transcription pipeline solution with Amazon EventBridge and AWS Batch. Since AWS Batch creates a containerized compute environment, it requires a container image to be built and staged in Amazon Elastic Container Registry (Amazon ECR). In the AWS Batch job definition, we specify the container image, the Amazon Machine Image (AMI), and Amazon EC2 instance type.

![hpc-whisper-batch-inferentia](https://github.com/user-attachments/assets/000e9019-2197-4836-a6a4-b1807c49c8d1)
Figure 1. Event-driven audio transcription pipeline with Amazon EventBridge and AWS Batch

We’ll discuss each numbered architecture feature.

1.	Build docker image and push to Amazon ECR

The docker image is built at design time and pushed to Amazon ECR. The docker image includes all the libraries and python script code to retrieve the model artifacts and the audio file from Amazon S3 and process it using the Whisper model encoder and decoder.

2.	Add audio transcription job to AWS Batch job queue on file upload

When an audio file is uploaded to Amazon S3 in the designated Bucket and folder, the PutEvent is captured by Amazon EventBridge, which triggers AWS Batch to add the file to the job queue.

3.	AWS Batch launches compute environment to process queued jobs

If there’s no compute environment active, AWS Batch will create a managed compute environment to process each job in the queue until the queue is empty. The compute environment is created using an Amazon EC2 Inf2 instance, an Amazon Machine Image (AMI) that includes the Neuron SDK libraries, and the container image uploaded to Amazon ECR in the first step. These configurations are defined in the AWS Batch job definition file.

4.	AWS Batch processes job and writes transcription to Amazon S3 output location

The compute environment processes each audio file and writes a transcription file to an Amazon S3 output location. Once the job queue is empty, AWS Batch shuts down the environment.

Next, we will demonstrate how to implement this solution.

## Implementing the solution

### Choose AMI

Choose AMI

The AMI we will choose for our AWS Batch compute environment will be an Amazon Linux AMI that is ECS-optimized. This AMI has Neuron drivers pre-installed to support AWS Inferentia2 instances. Remember that AMIs are region-specific, and since we are using the us-east-1 Region, we will use this community AMI published by AWS: ami-07979ad4c774a29ab. We can search and find this AMI in the Application and OS Images (Amazon Machine Image) catalog search in the Amazon EC2 Launch an instance screen (shown below).

![AMI](https://github.com/user-attachments/assets/c0685f8c-b394-488e-8051-adc4c496d5fe)

### Build the docker image

After cloning the GitHub repository, locate the Dockerfile. Notice that the referenced base image installs the AWS Neuron SDK for PyTorch 1.13.1 on AWS Inferentia.

```
FROM public.ecr.aws/neuron/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.18.1-ubuntu20.04

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY inference.py inference.py

RUN pip install -U --no-cache-dir -r requirements.txt

# Exit container after the job is done
RUN sed -i '/prevent docker exit/ {n; s/./# &/;}' /usr/local/bin/dockerd-entrypoint.py

CMD ["python3", "inference.py"]
```

We will build the image using the following command.

```
docker build --no-cache -t whisper:latest .
```

Your output should look like this.

![dockerbuild](https://github.com/user-attachments/assets/cf26de60-a058-4361-9db3-5125a5e5c69a)

### Push image to AWS ECR

With the image built locally, we need to create a repository in Amazon ECR and push our image to it so that it will be accessible for AWS Batch to use when launching compute environments.

```
aws ecr create-repository --repository-name whisper
```

We then tag the Docker image to match the repository URL.

```
docker tag whisper:latest [your-account-id].dkr.ecr.[your-region].amazonaws.com/whisper:latest
```

Finally, we push the image to the Amazon ECR repository.

```
docker push [your-account-id].dkr.ecr.[your-region].amazonaws.com/whisper:latest
```

### Create AWS Batch job queue

AWS Batch provides a Wizard (accessible from the menu in the upper left) that steps you through the configuration. You’ll create a compute environment for the inf2.8xlarge instance, a job queue, and a job definition where you will specify the repository URL for the Docker image that we built and uploaded to Amazon ECR.

#### Step 1. Configure job and orchestration type

Choose Amazon Elastic Compute Cloud (Amazon EC2) and click Next.

![awsbatch-1](https://github.com/user-attachments/assets/abc85c42-f158-496e-bbb5-0b953988ff06)

#### Step 2. Create a compute environment

In the Create a compute environment section, choose a Name and Instance role. Choose the 
minimum and maximum vCPUs. Under Allowed instance types, clear the default and select inf2.8xlarge.

![awsbatch-2](https://github.com/user-attachments/assets/3d245655-1266-4bc8-8d11-1405e667ec5b)

The inf2.8xlarge instance has 32 vCPUs, so setting the maximum vCPU to a multiple of 32 number will determine the throughput capacity of the AWS Batch job queue—or how many compute environments AWS Batch will launch to transcribe multiple audio files concurrently. For our experiment we will choose to process 5 audio files concurrently, so we will set the maximum vCPUs at 160. Once 5 compute environments are launched, jobs will simply wait in queue until a compute environment becomes available.

In the Network configuration, choose the Amazon Virtual Private Cloud (Amazon VPC) the compute environment will be deployed in. Select the Subnets and Security groups that will control access to the environment.

#### Step 3. Create a job queue

In the Job queue configuration section, create a Name for the Job queue and set the Priority. Job queues with a higher integer value for priority are given preference for compute environments. Click Next.

#### Step 4. Create a job definition

Choose a Name for the job definition. If desired, set an Execution timeout greater than 60 seconds to terminate unfinished jobs.

In the Container configuration section, specify the repository URL for the Docker image that we built and uploaded to Amazon ECR in earlier steps.

![awsbatch-4-1](https://github.com/user-attachments/assets/33f46683-0210-4160-b624-d9a635839c7d)

In the Environment configuration section, add the following environment variables by clicking Add environment variable and specifying the Name and Value for each as shown below.

![awsbatch-4-2](https://github.com/user-attachments/assets/a2a58624-8f75-4345-a8f3-726912b664ca)

Add any additional configurations you need for your use case and click Next.

#### Step 5. Submit job

Next the wizard will ask you to Create a job that will be submitted when the AWS Batch resources are created in the next step.

![awsbatch-5](https://github.com/user-attachments/assets/a9125cd1-2e7a-4ad1-87a6-e4f7b789cae6)

#### Step 6. Review and create

Review all the configurations listed and choose Create resources. Resources such as the job queue and job definition will be created and the test job will be submitted for execution.

Now that we have our AWS Batch workflow configured, the last component we need to configure is the Amazon EventBridge Rule that will add new audio files to our AWS Batch job queue.

### Create Amazon EventBridge Event Rule

Amazon EventBridge Event Bus is a serverless event bus that helps you receive, filter, transform, route, and deliver events. Events generated by AWS services are visible through Amazon EventBridge and can be used to trigger downstream actions on one or more target services.

In the Amazon EventBridge console under the Buses menu, choose Rule and click Create rule. The Create rule wizard will step you through the creation of the event rule.




### 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

