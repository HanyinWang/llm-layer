import pandas as pd
from tqdm import tqdm
import json
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from utils import *

import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for answer pair generation')

    parser.add_argument('--data_path', type = str, default = '../data/layer-sample-data-unlabeled.csv', 
    	help = 'unlabeled data for ppo training, should contain `text` and `condition`')
    parser.add_argument('--num_samples', type = int, default = 200, 
        help = 'number of samples to use for PPO training')
    parser.add_argument('--trained_reward_model', type = str, default = 'hanyinwang/layer-project-reward-model', 
        help = 'trained reward model from huggingface')
    parser.add_argument('--base_policy_model', type = str, default = 'mistralai/Mistral-7B-Instruct-v0.2', 
        help = 'sft trained policy model')
    parser.add_argument('--push_to_hub_repo', type = str, default = 'hanyinwang/layer-project-diagnostic-mistral', 
        help = 'set to the repo name if pushing the ppo trained model to hub')
    parser.add_argument('--learning_rate', type = float, default = 5e-5, 
        help = 'the initial learning rate')
    parser.add_argument('--batch_size', type = int, default = 1, 
        help = 'batch size for PPO training')
    parser.add_argument('--mini_batch_size', type = int, default = 1, 
        help = 'mini batch size for PPO training')
    parser.add_argument('--gen_min_length', type = int, default = -1,
        help = 'set to -1 if do not ignore eos token')
    parser.add_argument('--gen_top_k', type = int, default = 100, 
        help = 'top k for generation')
    parser.add_argument('--gen_top_p', type = float, default = 1.0,
        help = 'top p for generation')
    parser.add_argument('--gen_max_new_tokens', type = int, default = 11,
        help = 'max_new_token for generation')
    parser.add_argument('--gen_temperature', type = float, default = 0.5,
    	help = 'temperature for generation')
    parser.add_argument('--gen_repetition_penalty', type = float, default = 1.2,
    	help = 'repetition_penalty for generation')
    parser.add_argument('--gen_return_prompt', type = bool, default = False,
    	help = 'if return prompt with generation')
    parser.add_argument('--num_epoch', type = int, default = 1,
    	help = 'ppo training epochs')
    parser.add_argument('--save_path', type = str, default = 'layer-project-diagnostic-mistral',
        help = 'folder to save trained model')

    opt = parser.parse_args()

    return opt
opt = parse_option()

# load training data
data_unlabeled = pd.read_csv(opt.data_path)

# format data for PPO training
mistral_query = [format_prompt_mistral(data_unlabeled['text'].iloc[i], 
                                   data_unlabeled['condition'].iloc[i]) for i in range(len(data_unlabeled))]
data_ppo = pd.DataFrame.from_dict({'query': mistral_query})
dataset_ppo = Dataset.from_pandas(data_ppo.sample(n = opt.num_samples, random_state=49)) 

# Load trained reward model
rw_model = AutoModelForSequenceClassification.from_pretrained(opt.trained_reward_model, num_labels = 1).cuda()
rw_tokenizer = AutoTokenizer.from_pretrained(opt.trained_reward_model, padding_side='left')

def rw_tokenize(sample):
    sample['input_ids'] = rw_tokenizer.encode(sample['query'],
                                                padding="max_length",
                                                return_tensors="pt",
                                                truncation=True,
                                              
                                             )[0]
    return sample


# PPO training
# push to hub
push_kwargs = push_kwargs = {
    'repo_id': opt.push_to_hub_repo,
}

# PPO training configuration
config = PPOConfig(
    model_name=opt.base_policy_model,
    learning_rate=opt.learning_rate,
    batch_size=opt.batch_size,
    mini_batch_size=opt.mini_batch_size,
    # use the following argument and provide `push_to_hub_repo` if pushing to hub
    # push_to_hub_if_best_kwargs = push_kwargs
)

# PEFT settings
peft_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

# tokenizer settings
tokenizer_kwargs = {
    "padding": "max_length",
     "truncation": True,
    "return_tensors": "pt",
    "padding_side": 'left'
              }

# 
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    device_map="auto", 
    peft_config=peft_config, 
)

policy_tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)
policy_tokenizer.pad_token = policy_tokenizer.eos_token

# tokenize prompts
def policy_tokenize(sample):
    sample['input_ids'] = policy_tokenizer.encode(sample['query'])
    return sample
tokenized_ppo = dataset_ppo.map(policy_tokenize)

# PPO trainer
ppo_trainer = PPOTrainer(
    model=policy_model,
    config=config,
    tokenizer=policy_tokenizer,
    dataset=tokenized_ppo,
    data_collator=lambda x: x
)

# arguments for generation
generation_kwargs = {
    "min_length": opt.gen_min_length, # do not ignore eos token
    "top_k": opt.gen_top_k, # number of highest probability vocabulary tokens to keep
    "top_p": opt.gen_top_p, # only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
    "pad_token_id": policy_tokenizer.eos_token_id,
    'max_new_tokens':opt.gen_max_new_tokens, # len(policy_tokenizer.encode('\nDiabetes: MAYBE\n')) = 11, longest expected output
    "do_sample": True, # allow temperature & sampling parameter effect
    'temperature':opt.gen_temperature, # small randomness
    'repetition_penalty':opt.gen_repetition_penalty, # penalty on repetition
    'return_prompt': opt.gen_return_prompt,
}

# train
for epoch in tqdm(range(opt.num_epoch), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader): 
        batch = batch[0]
        query_tensors = torch.tensor(batch["input_ids"], dtype=torch.int64).cuda()
    
        #### Get response from original model
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = policy_tokenizer.decode(response_tensors.squeeze())
    
        #### Compute reward score
        rewards = get_score(rw_model, rw_tokenizer, batch["query"], batch["response"])
    
        #### Run PPO step
        stats = ppo_trainer.step([query_tensors], [response_tensors[0]], [rewards[0]])
        ppo_trainer.log_stats(stats, batch, rewards)


#### Save model
ppo_trainer.save_pretrained(opt.save_path)


















