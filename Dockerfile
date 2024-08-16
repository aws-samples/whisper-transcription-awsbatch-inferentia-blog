FROM public.ecr.aws/neuron/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.18.1-ubuntu20.04

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY inference.py inference.py

RUN pip install -U --no-cache-dir -r requirements.txt
# Exit container after the job is done
RUN sed -i '/prevent docker exit/ {n; s/./# &/;}' /usr/local/bin/dockerd-entrypoint.py

# Stage script to create model artifacts
COPY export-model.py export-model.py

# For inference
CMD ["python3", "inference.py"]
