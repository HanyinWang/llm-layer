# <ins>L</ins>LM <ins>A</ins>ugmented s<ins>Y</ins>mptom <ins>E</ins>xtraction & <ins>R</ins>ecognition (LAYER)

##
Slides: [project-summary.pdf](https://github.com/HanyinWang/llm-layer/blob/main/project-summary.pdf)
## Scope

1. [Zero-shot GPT-4](https://github.com/HanyinWang/llm-layer/blob/main/code/0-GPT4.ipynb)
2. Fine-tuning using Reimforcemnent Learning with Human Feedback (RLHF)
   - Base (policy) model: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
   - [Reward model fine-tuning](https://github.com/HanyinWang/llm-layer/blob/main/code/train_reward_model.py)
     - Base model: [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
     - Training data: [answer pairs](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training) derived from the sample dataset.
     - Resulting reward model: [hanyinwang/layer-project-reward-model](https://huggingface.co/hanyinwang/layer-project-reward-model)
   - [Update policy model with PPO](https://github.com/HanyinWang/llm-layer/blob/main/code/ppo_train_policy_model.py)
     - Training data: unlabeled data from provided dataset
     - Resulting model: [hanyinwang/layer-project-diagnostic-mistral](https://huggingface.co/hanyinwang/layer-project-diagnostic-mistral)


## Usage
Install packages
```bash
pip install torch transformers trl peft bitsandbytes ipykernel ipywidgets faiss-cpu==1.7.4 python-liquid==1.10.2 sentence_transformers==2.2.2 openai==0.28.0 tiktoken==0.6.0
```
Inference on one sample, providing a `note` and a `condition` of interest ("cancer" or "diabetes" for now).
- With RAG
```bash
python example_run.py --sample_note 'sample_note.txt' --sample_condition diabetes --use_rag

## Reponse from diagnostic-mistral with RAG:  {"diabetes": "NO"}
```
- Without RAG
```bash
python example_run.py --sample_note 'sample_note.txt' --sample_condition diabetes

## Reponse from diagnostic-mistral without RAG: 

## {"diabetes": "NO"}</s>
```

> [!NOTE]
> RAG is implemented from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG/tree/main) with [MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder) retriever on [StatPearls](https://www.statpearls.com/) corpus


## Training details
### Generate answer pairs
The answer pairs used for reward model training is available at [hanyinwang/layer-project-reward-training](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training).

To generate additional answer pairs:
```bash
cd code
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
cd code
python train_reward_model.py \
	--model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
	--reward_training_data hanyinwang/layer-project-reward-training \ # data on huggingface
	--output_dir reward_model_checkpoint/ \
	--num_train_epochs 20 \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--eval_steps 1 \
	--save_steps 1 \
	--remove_unused_columns
```

### To train policy model using PPO
```bash
cd code
python ppo_train_policy_model.py \
	--data_path '../data/layer-sample-data-unlabeled.csv' \ # unlabeled data for PPO training
	--num_samples 200 \
	--trained_reward_model "hanyinwang/layer-project-reward-model" \ # trained reward model
	--base_policy_model 'mistralai/Mistral-7B-Instruct-v0.2' \
	--push_to_hub_repo "hanyinwang/layer-project-diagnostic-mistral" \
	--learning_rate 5e-5 \
	--batch_size 1 \
	--gen_min_length -1 \
	--gen_top_k 100 \
	--gen_top_p 0.95 \
	--gen_max_new_tokens 11 \
	--gen_temperature 0.5 \
	--gen_repetition_penalty 1.2 \
	--num_epoch 1
```
