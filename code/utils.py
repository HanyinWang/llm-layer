import numpy as np
import torch

def repeat_df(df, total_rows):
    """
    repead dataset to designated number of rows
    """
    if total_rows is not None:        
        # Calculate repeat factor
        repeat_factor = -(-total_rows // len(df))  # Ceiling division
        
        # Repeat the DataFrame
        repeated_indices = np.tile(df.index, repeat_factor)[:total_rows]
        repeated_df = df.loc[repeated_indices].reset_index(drop=True)
    else:
        repeated_df = df

    return repeated_df

def format_prompt_ap(text, condition):
    """
    process text to prompt for generating answer pairs
    """
    prompt = """You are a medical doctor specialized in %s diagnosis. 
From the provided document, assert if the patient historically or currently has %s.
For each condition, only pick from "YES", "NO", or "MAYBE". And you must follow format without anything further. The results have to be directly parseable with python json.loads(). 
Sample output: {"%s": 'MAYBE'}.
Never output anything beyond the format.
Provided document: \n%s
"""%(condition, condition, condition, text)
    return prompt


def format_prompt_rw(examples, tokenizer):
    """
    process text to prompt for reward model (tiny llama) training
    """
    kwargs = {"padding": "max_length",
              "truncation": True,
              "return_tensors": "pt"
              }

    # Prepend the prompt and a line break to the responses.
    prompt_plus_chosen_response = examples["prompt"] + "\n" + examples["chosen"]
    prompt_plus_rejected_response = examples["prompt"] + "\n" + examples["rejected"]

    # Tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

    # Return according to the required format for reward trainer 
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }


# format prompt for mistral
def format_prompt_mistral(text, condition):
    """
    process text to prompt for policy model (mistral) 
    """
    prompt = """<s>[INST]You are a medical doctor specialized in %s diagnosis. 
From the provided document, assert if the patient historically and currently has %s.
For each condition, only pick from "YES", "NO", or "MAYBE". And you must follow format without anything further. The results have to be directly parseable with python json.loads(). 
Sample output: {"%s": "MAYBE"}
Never output anything beyond the format.[/INST]
Provided document: %s"""%(condition, condition, condition, text)
    return prompt


def get_score(model, tokenizer, prompt, response):
    """
    get reward from reward model
    """

    input_ids = tokenizer.encode(prompt, response, return_tensors = "pt").cuda()
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits

    return logits