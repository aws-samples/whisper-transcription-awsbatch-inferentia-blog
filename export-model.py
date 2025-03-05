import os
os.environ['NEURON_RT_NUM_CORES']='1'
import types
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# please, start by selecting the desired model size
#suffix="tiny"
#suffix="small"
#suffix="medium"
suffix="large-v3"
model_id=f"openai/whisper-{suffix}"

# this will load the tokenizer + two copies of the model. cpu_model will be used later for results comparison
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)
cpu_model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

# Load a sample from the dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample #3 is ~9.9seconds and produces 33 output tokens + pad token
sample = dataset[3]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# batch size refers to the number of files processed in parallel
# and should not exceed the number of cores on the Neuron device.
batch_size=1
# output_attentions is required if you want to return word timestamps
# if you don't need timestamps, just set this to False and get some better latency
output_attentions=False
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

if not hasattr(model.model.encoder, 'forward_'): 
    model.model.encoder.forward_ = model.model.encoder.forward
if not hasattr(model.model.decoder, 'forward_'): 
    model.model.decoder.forward_ = model.model.decoder.forward
if not hasattr(model.proj_out, 'forward_'):
    model.proj_out.forward_ = model.proj_out.forward


model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)

model.model.decoder.max_length = max_dec_len
model.proj_out.max_length = max_dec_len

# warmup model
y1 = model.generate(input_features)

# Trace Encoder
import os
import torch
import torch_neuronx

model_filename=f"whisper_{suffix}_{batch_size}_neuron_encoder.pt"
if not os.path.isfile(model_filename):
    inp = (torch.zeros([1, dim_enc, 3000], dtype=torch.float32), torch.zeros([1, dim_enc], dtype=torch.int64))
    if hasattr(model.model.encoder, 'forward_neuron'): del model.model.encoder.forward_neuron
    neuron_encoder = torch_neuronx.trace(
        model.model.encoder, 
        inp,
        compiler_args='--model-type=transformer --enable-saturate-infinity --auto-cast=all', 
        compiler_workdir='./enc_dir',      
        inline_weights_to_neff=False)
    neuron_encoder.save(model_filename)
    model.model.encoder.forward_neuron = neuron_encoder
else:
    model.model.encoder.forward_neuron = torch.jit.load(model_filename)
    
# Trace Decoder
import torch
import torch_neuronx

model_filename=f"whisper_{suffix}_{batch_size}_{max_dec_len}_neuron_decoder.pt"
if not os.path.isfile(model_filename):
    inp = (torch.zeros([1, max_dec_len], dtype=torch.int64), torch.zeros([1, 1500, dim_dec], dtype=torch.float32))
    if hasattr(model.model.decoder, 'forward_neuron'): del model.model.decoder.forward_neuron
    neuron_decoder = torch_neuronx.trace(
        model.model.decoder, 
        inp,
        compiler_args='--model-type=transformer --enable-saturate-infinity  --auto-cast=all',
        compiler_workdir='./dec_dir',      
        inline_weights_to_neff=True)
    neuron_decoder.save(model_filename)
    model.model.decoder.forward_neuron = neuron_decoder
else:
    model.model.decoder.forward_neuron = torch.jit.load(model_filename)

# Trace Projection Output
import torch
import torch_neuronx

model_filename=f"whisper_{suffix}_{batch_size}_{max_dec_len}_neuron_proj.pt"
if not os.path.isfile(model_filename):
    inp = torch.zeros([1, max_dec_len, dim_dec], dtype=torch.float32)
    if hasattr(model.proj_out, 'forward_neuron'): del model.proj_out.forward_neuron
    neuron_decoder = torch_neuronx.trace(
        model.proj_out,
        inp,
        compiler_args='--model-type=transformer --auto-cast=all --auto-cast-type=bf16',
        compiler_workdir='./proj_out_dir',
        inline_weights_to_neff=True)
    neuron_decoder.save(model_filename)
    model.proj_out.forward_neuron = neuron_decoder
else:
    model.proj_out.forward_neuron = torch.jit.load(model_filename)


# Test

# warmup inf2 model
y1 = model.generate(input_features)

torch.set_num_threads(1)

import time
t=time.time()
y1 = model.generate(input_features)
print(f"Elapsed inf2: {time.time()-t}")
t=time.time()
y2 = cpu_model.generate(input_features)
print(f"Elapsed cpu: {time.time()-t}")
print(f"Tokens inf2: {y1}")
print(f"Tokens cpu: {y2}")
t1 = processor.batch_decode(y1, skip_special_tokens=True)
t2 = processor.batch_decode(y2, skip_special_tokens=True)
print(f"Out inf2: {t1}")
print(f"Out cpu: {t2}")

