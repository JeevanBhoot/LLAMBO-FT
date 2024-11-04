import torch
import pandas as pd
import numpy as np
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_squared_error
from peft import PeftModel


def generate_response(prompt, model, tokenizer, max_new_tokens=40):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that helps people find information."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

def extract_performance(response):
    # Remove hashtags if they exist and extract the performance value
    response = response.replace("#", "")
    match = re.search(r"-?\d+\.\d+", response)
    if match:
        return float(match.group(0))
    return None

def main():
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    validation_data_dir = "valid_data_csv/"

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, "finetuned_llama/exp04/checkpoint-450")
    model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    predicted_values = []
    true_values = []
    for filename in os.listdir(validation_data_dir):
        if 'discriminative' in filename and filename.endswith('.csv'):
            validation_data_path = os.path.join(validation_data_dir, filename)
            validation_data = pd.read_csv(validation_data_path)

            for idx, row in validation_data.iterrows():
                prompt = row['prompt']
                true_response = row['response']
                
                generated_response = generate_response(prompt, model, tokenizer)
                predicted_value = extract_performance(generated_response)

                true_value = extract_performance(true_response)
                if predicted_value is not None and true_value is not None:
                    predicted_values.append(predicted_value)
                    true_values.append(true_value)

    mse = mean_squared_error(true_values, predicted_values)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()