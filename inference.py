import os
import sys
os.environ['NEURON_RT_NUM_CORES']='1'
import types
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import boto3

input_bucket_name = sys.argv[1]
input_file_key = sys.argv[2]

output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
output_file_prefix =  os.environ['OUTPUT_FILE_PREFIX']

model_artifact_bucket_name = os.environ['MODEL_BUCKET_NAME']
model_artifact_encoder_key = os.environ['MODEL_ENCODER_S3_KEY']
model_artifact_decoder_key = os.environ['MODEL_DECODER_S3_KEY']
model_artifact_proj_key = os.environ['MODEL_PROJ_S3_KEY']

s3_client = boto3.client('s3')

# please, start by selecting the desired model size
#suffix="tiny"
#suffix="small"
#suffix="medium"
suffix="large-v3"
model_id=f"openai/whisper-{suffix}"

# this will load the tokenizer + two copies of the model.
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

# batch size refers to the number of files processed in parallel
# and should not exceed the number of cores on the Neuron device.
batch_size=1

# output_attentions is required if you want to return word timestamps
# if you don't need timestamps, just set this to False and get some better latency
output_attentions=True

# this is the maximum number of tokens the model will be able to decode
# for the sample #3 we selected above, this is enough. If you're planning to 
# process larger samples, you need to adjust it accordinly.
max_dec_len = 448
# num_mel_bins,d_model --> these parameters where copied from model.conf (found on HF repo)
# we need them to correctly generate dummy inputs during compilation
dim_enc=model.config.num_mel_bins
dim_dec=model.config.d_model
print(f'Dim enc: {dim_enc}; Dim dec: {dim_dec}')


import types
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,BaseModelOutput

# Now we need to simplify both encoding & decoding forward methods to make them 
# compilable. Please notice that these methods overwrite the original ones, but
# keeps retro-compatibility. Also, we'll use use a new variable "forward_neuron"
# to invoke the model on inf2
def enc_f(self, input_features, attention_mask, **kwargs):
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(input_features, attention_mask)
    else:
        out = self.forward_(input_features, attention_mask, return_dict=True)
    return BaseModelOutput(**out)

def dec_f(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
    out = None        
    if not attention_mask is None and encoder_hidden_states is None:
        # this is a workaround to align the input parameters for NeuronSDK tracer
        # None values are not allowed during compilation
        encoder_hidden_states, attention_mask = attention_mask,encoder_hidden_states
    inp = [input_ids, encoder_hidden_states]
    
    # pad the input to max_dec_len
    if inp[0].shape[1] > self.max_length:
        raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
    pad_size = torch.as_tensor(self.max_length - inp[0].shape[1])
    inp[0] = F.pad(inp[0], (0, pad_size), "constant", processor.tokenizer.pad_token_id)
    
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(*inp)
    else:
        # output_attentions is required if you want timestamps
        out = self.forward_(input_ids=inp[0], encoder_hidden_states=inp[1], return_dict=True, use_cache=False, output_attentions=output_attentions)
    # unpad the output
    out['last_hidden_state'] = out['last_hidden_state'][:, :input_ids.shape[1], :]
    # neuron compiler doesn't like tuples as values of dicts, so we stack them into tensors
    # also, we need to average axis=2 given we're not using cache (use_cache=False)
    # that way, to avoid an issue with the pipeline we change the shape from:
    #  bs,num selected,num_tokens,1500 --> bs,1,num_tokens,1500
    # I suspect there is a bug in the HF pipeline code that doesn't support use_cache=False for
    # word timestamps, that's why we need that.
    if not out.get('attentions') is None:
        out['attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :input_ids.shape[1]], axis=2, keepdim=True) for o in out['attentions']])
    if not out.get('cross_attentions') is None:
        out['cross_attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :], axis=2, keepdim=True) for o in out['cross_attentions']])
    return BaseModelOutputWithPastAndCrossAttentions(**out)

if not hasattr(model.model.encoder, 'forward_'): 
    model.model.encoder.forward_ = model.model.encoder.forward
if not hasattr(model.model.decoder, 'forward_'): 
    model.model.decoder.forward_ = model.model.decoder.forward
if not hasattr(model.proj_out, 'forward_'):
    model.proj_out.forward_ = model.proj_out.forward

def proj_out_f(self, inp):
    pad_size = torch.as_tensor(self.max_length - inp.shape[1], device=inp.device)
    # pad the input to max_dec_len
    if inp.shape[1] > self.max_length:
        raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
    x = F.pad(inp, (0,0,0,pad_size), "constant", processor.tokenizer.pad_token_id)

    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(x)
    else:
        out = self.forward_(x)
    # unpad the output before returning
    out = out[:, :inp.shape[1], :]
    return out

model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)

model.model.decoder.max_length = max_dec_len
model.proj_out.max_length = max_dec_len

# Trace Encoder
import os
import torch
import torch_neuronx

# download model artifacts from S3
s3_client.download_file(model_artifact_bucket_name, model_artifact_encoder_key, model_artifact_encoder_key.split("/")[-1])
model_encoder_filename=model_artifact_encoder_key.split("/")[-1]
if not os.path.isfile(model_encoder_filename):
    raise Exception("encoder model artifact not found.")
else:
    model.model.encoder.forward_neuron = torch.jit.load(model_encoder_filename)
    
# Trace Decoder
import torch
import torch_neuronx

s3_client.download_file(model_artifact_bucket_name, model_artifact_decoder_key, model_artifact_decoder_key.split("/")[-1])
model_decoder_filename=model_artifact_decoder_key.split("/")[-1]
if not os.path.isfile(model_decoder_filename):
    raise Exception("decoder model artifact not found.")
else:
    model.model.decoder.forward_neuron = torch.jit.load(model_decoder_filename)
    
# Trace Projection Output
import torch
import torch_neuronx

s3_client.download_file(model_artifact_bucket_name, model_artifact_proj_key, model_artifact_proj_key.split("/")[-1])
model_proj_filename=model_artifact_proj_key.split("/")[-1]
if not os.path.isfile(model_proj_filename):
    raise Exception("projection model artifact not found.")
else:
    model.proj_out.forward_neuron = torch.jit.load(model_proj_filename)

# Inference
import torchaudio

# copy from s3
s3_client.download_file(input_bucket_name, input_file_key, input_file_key.split("/")[-1])
audio_path = input_file_key.split("/")[-1]

# Load the audio file
waveform, sample_rate = torchaudio.load(audio_path)

# Ensure the audio is in the correct format (mono, 16kHz)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# chunk the audio
chunk_size = 30*16000 # 30 seconds * 16000 samples / second
chunks = waveform.split(chunk_size, dim=1)

import time
t=time.time()

transcriptions = []
for chunk in chunks:
    inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    transcription = processor.decode(predicted_ids[0])
    print(transcription)
    transcriptions.append(transcription)

print(f"Elapsed inf2: {time.time()-t}")

# Combine the transcriptions
full_transcription = " ".join(transcriptions)
#print("Full Transcription:", full_transcription)

output_filename = audio_path + '.txt'
file = open(output_filename, 'w')
file.write(full_transcription)
file.close()

s3_client.put_object(Body=full_transcription, Bucket=output_bucket_name, Key=output_file_prefix + output_filename)
