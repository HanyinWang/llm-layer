# import necessary pkg
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import sys
sys.path.append('../code')
from utils import *

import argparse

def parse_option():
  parser = argparse.ArgumentParser('argument for answer pair generation')
  parser.add_argument('--sample_note', type = str, default = 'sample_note.txt')
  parser.add_argument('--sample_condition', type = str, default = 'diabetes')

  opt = parser.parse_args()

  return opt

opt = parse_option()

with open(opt.sample_note, 'r') as f:
    sample_note = f.read()

# tokenizer arguments
tokenizer_kwargs = {
  "padding": "max_length",
  "truncation": True,
  "return_tensors": "pt",
  "padding_side": "left"
}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("hanyinwang/layer-project-diagnostic-mistral", **tokenizer_kwargs)
tokenizer.pad_token = tokenizer.eos_token

# generation arguments
generation_kwargs = {
  "min_length": -1,
  "top_k": 40,
  "top_p": 0.95,
  "do_sample": True,
  "pad_token_id": tokenizer.eos_token_id,
  "max_new_tokens":11,
  "temperature":0.1,
  "repetition_penalty":1.2
}

# load fine-tuned model
model = AutoModelForCausalLMWithValueHead.from_pretrained("hanyinwang/layer-project-diagnostic-mistral").cuda()

# query the model
query_tensors = tokenizer.encode(format_prompt_mistral(sample_note, opt.sample_condition), return_tensors="pt")
prompt_length = query_tensors.shape[1]

# response
outputs = model.generate(query_tensors.cuda(), **generation_kwargs)
response = tokenizer.decode(outputs[0][prompt_length:])

print('Reponse from diagnostic-mistral: %s'%(response))