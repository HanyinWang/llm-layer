# layer-project-IMO



### Generate answer pairs
The answer pairs used for reward model training is available at [hanyinwang/layer-project-reward-training](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training)
To generate answer pairs:
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
