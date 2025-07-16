#!/bin/bash

# task list
tasks=("Analyse_de_sentiment" "Identification_de_paraphrases" "Natural_language_inference")

# name/path for all models
models=(
    "camembert-base"
    "flaubert/flaubert_base_cased"
    "flaubert/flaubert_base_uncased"
    "/home/sgu/models/Text_Base_fr_4GB_v0/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_v1/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step500K/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step1M/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step2M/HuggingFace"
    "/home/sgu/models/Text_Base_fr_OSCAR_camtok/HuggingFace"
)

# name/path for all tokenizers(maybe different to the model name/path)
tokenizers=(
    "camembert-base"
    "flaubert/flaubert_base_cased"
    "flaubert/flaubert_base_uncased"
    "/home/sgu/models/Text_Base_fr_4GB_v0/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_v1/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step500K/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step1M/HuggingFace"
    "/home/sgu/models/Text_Base_fr_4GB_camtok_step2M/HuggingFace"
    "/home/sgu/models/Text_Base_fr_OSCAR_camtok/HuggingFace"
)

# run each model on each task
for task in "${tasks[@]}"; do
  for i in "${!models[@]}"; do
    model_name="${models[$i]}"
    tokenizer_name="${tokenizers[$i]}"
    
    echo "Running task: $task with model: $model_name"

    # modify corresponding task's config.yaml
    CONFIG_FILE="$task/config.yaml"
    sed -i "s|^model_name:.*|model_name: \"$model_name\"|" "$CONFIG_FILE"
    sed -i "s|^tokenizer_name:.*|tokenizer_name: \"$tokenizer_name\"|" "$CONFIG_FILE"

    # start running
    python3 main.py --task "$task"
  done
done
