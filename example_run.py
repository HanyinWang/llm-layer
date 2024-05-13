# import necessary pkg
import sys
sys.path.append('./code')
from utils import *
from MedRAG.src.medrag import MedRAG
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import json
import numpy as np


import argparse

def parse_option():
  parser = argparse.ArgumentParser('argument for answer pair generation')
  parser.add_argument('--sample_note', type = str, default = 'sample_note.txt')
  parser.add_argument('--sample_condition', type = str, default = 'diabetes',
    help = 'choose from `cancer` or `diabetes`')
  parser.add_argument('--use_rag', action='store_true',
    help = 'result with or without rag')

  opt = parser.parse_args()

  return opt

opt = parse_option()

with open(opt.sample_note, 'r') as f:
    sample_note = f.read()


# if with rag
if opt.use_rag:
  import os
  os.chdir('./code')
  print('loading RAG content')
  medrag = MedRAG(llm_name="hanyinwang/layer-project-diagnostic-mistral", rag=True, retriever_name="MedCPT", corpus_name="StatPearls", 
    condition = opt.sample_condition)

  response, snippets, scores = medrag.answer(question=sample_note, options={}, k=8)

else:
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

w_wo = ['with' if opt.use_rag else "w/o"]
print('Reponse from diagnostic-mistral %s RAG: %s'%(w_wo[0], response))