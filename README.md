# layer-project-IMO



### Generate answer pairs
The answer pairs used for reward model training is available at [hanyinwang/layer-project-reward-training](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training).

To generate additional answer pairs:
```
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
```
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
