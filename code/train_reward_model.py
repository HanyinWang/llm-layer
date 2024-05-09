from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset
from trl import RewardTrainer

from utils import *

import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for answer pair generation')

    parser.add_argument('--model_name', type = str, default = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 
    	help = 'please use a model huggingface available for AutoModelForSequenceClassification')
    parser.add_argument('--reward_training_data', type = str, default = 'hanyinwang/layer-project-reward-training', 
        help = 'huggingface dataset')
    parser.add_argument('--output_dir', type = str, default = 'reward_model_checkpoint/', 
        help = 'directory to locally save checkpoint if not pushing to hub')
    parser.add_argument('--num_train_epochs', type = int, default = 20, 
        help = 'number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1, 
        help = 'number of updates steps to accumulate the gradients for, before performing a backward/update pass.')
    parser.add_argument('--save_strategy', type = str, default = 'epoch', 
        help = '`no`, `epoch`, or `steps`')
    parser.add_argument('--save_steps', type = int, default = 1, 
        help = 'number of updates steps before two checkpoint saves if save_strategy="steps"')
    parser.add_argument('--evaluation_strategy', type = str, default = 'epoch', 
        help = '`no`, `epoch`, or `steps`')
    parser.add_argument('--eval_steps', type = int, default = 1,
        help = 'number of update steps between two evaluations if evaluation_strategy="steps"')
    parser.add_argument('--per_device_train_batch_size', type = int, default = 2, 
        help = 'batch size per GPU/CPU for training')
    parser.add_argument('--per_device_eval_batch_size', type = int, default = 1,
        help = 'batch size per GPU/CPU for testing')
    parser.add_argument('--eval_accumulation_steps', type = int, default = 1,
        help = 'number of predictions steps to accumulate the output tensors for')
    parser.add_argument('--warmup_steps', type = int, default = 0,
    	help = 'number of steps used for a linear warmup from 0 to learning_rate.')
    parser.add_argument('--learning_rate', type = float, default = 5e-5,
    	help = 'the initial learning rate')
    parser.add_argument('--save_total_limit', type = int, default = None,
    	help = 'if a value is passed, will limit the total amount of checkpoints')
    parser.add_argument('--no_cuda', type = bool, default = False,
    	help = 'no cuda')
    parser.add_argument('--remove_unused_columns', type = bool, default = True,
    	help = 'whether or not to automatically remove the columns unused by the model forward method')
    parser.add_argument('--layers_to_train', type = str, default = '21',
    	help = 'common seperated string, train only the mentioned layers, freeze weights for the rw_dataset_train_test')

    opt = parser.parse_args()

    return opt
opt = parse_option()


# defininig the reward tokenizer and model
rw_tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
rw_tokenizer.pad_token = rw_tokenizer.eos_token

rw_model = AutoModelForSequenceClassification.from_pretrained(opt.model_name, num_labels = 1)
rw_model.config.pad_token_id = rw_model.config.eos_token_id

# free layer except for the last, since we don't have enough to data 
train_layers = ['model.layers.%s'%(l) for l in opt.layers_to_train.split(',')]
for name, param in rw_model.named_parameters():
     if not name.startswith(tuple(train_layers)): 
        param.requires_grad = False

# define PEFT settings
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

# load data from huggingface
rw_dataset_train_test = load_dataset(opt.reward_training_data)

# generate necessary tokens and masks for reward model training
formatted_rw_dataset = rw_dataset_train_test.map(format_prompt_rw, fn_kwargs={"tokenizer": rw_tokenizer})

### Loading the TRL reward trainer and training the trainer
training_args = TrainingArguments(
        output_dir=opt.output_dir,
        num_train_epochs=opt.num_train_epochs,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        save_strategy=opt.save_strategy,
        evaluation_strategy=opt.evaluation_strategy,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        eval_accumulation_steps=opt.eval_accumulation_steps,
        eval_steps=opt.eval_steps,
        save_steps=opt.save_steps,
        warmup_steps=opt.warmup_steps,
        learning_rate=opt.learning_rate,
        save_total_limit=opt.save_total_limit,
        no_cuda=opt.no_cuda,
        remove_unused_columns=opt.remove_unused_columns,
        ## use the following arguments when pushing to hub
        # hub_strategy="every_save",
        # push_to_hub=True,
        # hub_model_id="hanyinwang/layer-project-reward-model",
        # hub_private_repo=True,
    )

trainer = RewardTrainer(model=rw_model,
                        tokenizer=rw_tokenizer,
                        train_dataset=formatted_rw_dataset['train'],
                        eval_dataset=formatted_rw_dataset['test'],
                        args= training_args,
                        peft_config=peft_config
                        )
trainer.train()

## if push to hub
# trainer.push_to_hub()

















