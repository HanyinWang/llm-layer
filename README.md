# layer-project-IMO

## Usage
Install packages
```bash
pip install torch transformers trl peft ipykernel ipywidgets
```
Inference on one sample, providing a *note* and a *condition* of interest.
```python
# import necessary pkg
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from utils import *

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
query_tensors = tokenizer.encode(format_prompt_mistral(<note>, <condition>), return_tensors="pt")
# <note>: clinical note
# <condition>: "cancer" or "diabetes"
prompt_length = query_tensors.shape[1]

# response
outputs = model.generate(query_tensors.cuda(), **generation_kwargs)
response = tokenizer.decode(outputs[0][prompt_length:])
```


## Training details
### Generate answer pairs
The answer pairs used for reward model training is available at [hanyinwang/layer-project-reward-training](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training).

To generate additional answer pairs:
```bash
cd code;
python generate_answer_pairs.py \
	--model_name openai-community/gpt2 \
	--data_path ../data/layer-sample-data-partially-labeled.csv \
	--gen_top_p 1.0 \
	--gen_top_k 40 \
	--gen_max_new_tokens 11 \
	--gen_temperature 1.2 \
	--gen_repetition_penalty 1.0 \
	--num_pairs 150 \ # to generate 150 pairs
	--save_path "../data/answer-pairs.csv"
```

### To train reward model
Trainer uses wandb, [setup](https://docs.wandb.ai/tutorials/huggingface) before running.
```bash
cd code;
python train_reward_model.py \
	--model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
	--reward_training_data hanyinwang/layer-project-reward-training \ # data on huggingface
	--output_dir reward_model_checkpoint/ \
	--num_train_epochs 20 \
	--gradient_accumulation_steps 1 \
	--save_strategy epoch \
	--evaluation_strategy epoch \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 1 \
	--eval_accumulation_steps 1 \
	--eval_steps 1 \
	--save_steps 1 \
	--warmup_steps 10 \
	--learning_rate 1e-5 \
	--save_total_limit 1 \
	--no_cuda False \
	--remove_unused_columns True
```

### To train policy model using PPO
```bash
cd code;
python ppo_train_policy_model.py \
	--data_path '../data/layer-sample-data-unlabeled.csv' \ # unlabeled data for PPO training
	--num_samples 200 \
	--trained_reward_model "hanyinwang/layer-project-reward-model" \ # trained reward model
	--base_policy_model 'mistralai/Mistral-7B-Instruct-v0.2' \
	--push_to_hub_repo "hanyinwang/layer-project-diagnostic-mistral" \
	--learning_rate 5e-5 \
	--batch_size 1 \
	--mini_batch_size 1 \
	--gen_min_length -1 \
	--gen_top_k 100 \
	--gen_top_p 0.95 \
	--gen_max_new_tokens 11 \
	--gen_temperature 0.5 \
	--gen_repetition_penalty 1.2 \
	--num_epoch 1
```
