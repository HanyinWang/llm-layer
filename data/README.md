## Source data
- The entire sample dataset: [layer-sample-data.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/layer-sample-data.csv)
    - Subset of binary labeled data (n=50): [layer-sample-data-labeled.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/layer-sample-data-labeled.csv)
      - For evaluation  
    - Subset of textual labeled data (n=101): [layer-sample-data-partially-labeled.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/layer-sample-data-partially-labeled.csv)
      - For reward model
    - Subset of data without binary label (n=1950): [layer-sample-data-unlabeled.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/layer-sample-data-unlabeled.csv)
      - For PPO training 
 
 ## Generated data
 - Model output
     - GPT-4 output on evaluation set: [gpt4_response_update.pkl](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/gpt4_response_update.pkl)
     - All outputs for *cancer* on evaluation set: [outputs_cancer.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/outputs_cancer.csv)
     - All outputs for *diabetes* on evaluation set: [outputs_diabetes.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/outputs_diabetes.csv)
     - Evaluation results: [evaluation.ipynb](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/evaluation.ipynb)
 - Answer pairs for reward model training
     - Example answer pairs generated from [layer-sample-data-partially-labeled.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/layer-sample-data-partially-labeled.csv): [answer-pairs.csv](https://github.com/HanyinWang/layer-project-IMO/blob/main/data/answer-pairs.csv)
     - Answer pairs (fixed) used for reward model training available at: [hanyinwang/layer-project-reward-training](https://huggingface.co/datasets/hanyinwang/layer-project-reward-training)
