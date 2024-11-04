#!/bin/bash

set -Eeuo pipefail

python finetuning_dataset/generate_data.py
python finetuning_dataset/dataset_info.py
python finetuning_dataset/generate_prompts.py
python finetuning_dataset/generate_warmstarting_prompts.py