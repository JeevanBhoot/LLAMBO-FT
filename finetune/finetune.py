import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Script for finetuning Llama model using LoRA and QLoRA.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="training_data/csv"
        help="Directory containing CSV files with training data."
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        help="HuggingFace model path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to store the finetuned model."
    )
    return parser.parse_args()


def load_data_from_csv(data_dir):
    """
    Load prompts and responses from CSV files in the given directory.

    Parameters:
    - data_dir (str): Directory containing CSV files.

    Returns:
    - pd.DataFrame: Dataframe containing prompts and responses.
    """
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            data.append(df)
    df = pd.concat(data, ignore_index=True)
    df.rename(columns={"response": "completion"}, inplace=True)
    return Dataset.from_pandas(df)


def main(args):
    args = parse_args()
    data_dir = args.data_dir
    model_id = args.model_id
    output_dir = args.output_dir

    dataset = load_data_from_csv(data_dir)

    # QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=quantization_config, 
    )
    
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        num_train_epochs=5,
        fp16=False,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        packing=False,
        max_seq_length=512,
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

if __name__ == "__main__":
    args = parse_args()
    main(args)
