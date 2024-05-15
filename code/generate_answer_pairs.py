"""
Code to generate answer pairs for reward model training. 
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from utils import *

import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for answer pair generation')

    parser.add_argument('--model_name', type = str, default = 'openai-community/gpt2', 
    	help = 'please use a model huggingface available for AutoModelForCausalLMWithValueHead')
    parser.add_argument('--data_path', type = str, default = None, 
        help = 'should contain a `text` column, a `condition` column and a `label` column')
    parser.add_argument('--gen_top_p', type = float, default = 1.0, 
        help = 'top_p for generation')
    parser.add_argument('--gen_top_k', type = int, default = 40, 
        help = 'top_k for generation')
    parser.add_argument('--gen_max_new_tokens', type = int, default = 11, 
        help = 'max_new_tokens for generation')
    parser.add_argument('--gen_temperature', type = float, default = 1.2, 
        help = 'temperature for generation')
    parser.add_argument('--gen_repetition_penalty', type = float, default = 1.0, 
        help = 'top_p for generation')
    parser.add_argument('--num_pairs', type = int, default = None, 
        help = 'number of answer pairs to generate, None: use each sample once, int: number of final answer pairs')
    parser.add_argument('--save_path', type = str, default = '../data/answer-pairs.csv',
        help = 'path to save the generated answer pairs')

    opt = parser.parse_args()

    return opt
opt = parse_option()

# data
data_partially_labeled = pd.read_csv(opt.data_path)
data_partially_labeled = repeat_df(data_partially_labeled, total_rows = opt.num_pairs)

# use PEFT 
peft_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

# model
random_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    opt.model_name, 
    peft_config=peft_config, 
    load_in_4bit=True   
).cuda()

# tokenizer
tokenizer_kwargs = { 
    "padding": "max_length",
    'max_length': 256,
    "truncation": True,
    "return_tensors": "pt"
              }
random_tokenizer = AutoTokenizer.from_pretrained(opt.model_name, padding_side='left')
random_tokenizer.pad_token = random_tokenizer.eos_token


# generation
generation_kwargs = {
    "min_length": -1,
    'top_k': opt.gen_top_k,
    'top_p': opt.gen_top_p,
    "do_sample": True,
    "pad_token_id": random_tokenizer.eos_token_id,
    'max_new_tokens':opt.gen_max_new_tokens, # limit response length
    'temperature':opt.gen_temperature, # high randomness
    'repetition_penalty':opt.gen_repetition_penalty # no penalty
    
}
prompt_lst = [format_prompt_ap(data_partially_labeled['text'].iloc[i], data_partially_labeled['condition'].iloc[i]) for i in range(len(data_partially_labeled))]
# list of unwanted answers
rejected_lst = []
for text in tqdm(prompt_lst):
    query_tensors = random_tokenizer.encode(text, **tokenizer_kwargs).cuda()
    response_tensors = random_model.generate(query_tensors, **generation_kwargs)
    prompt_length = query_tensors.shape[1]
    rejected_lst.append(random_tokenizer.decode(response_tensors.squeeze()[prompt_length:]))
data_partially_labeled['prompt'] = prompt_lst
data_partially_labeled['rejected'] = rejected_lst

# add GOOD output, generate from textual label
data_partially_labeled['chosen'] = ['{\"%s\": "%s"}'%(data_partially_labeled['condition'].iloc[i], 
                                                      data_partially_labeled['label'].iloc[i]) for i in range(len(data_partially_labeled))]

# save
output_data = data_partially_labeled[['prompt', 'rejected', 'chosen']]
output_data.to_csv(opt.save_path)
