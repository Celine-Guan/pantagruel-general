#!/bin/bash

# task list (run a single task)
tasks=("Identification_de_paraphrases")

# name/path for all models
models=(
    # "camembert-base"
    # "flaubert/flaubert_base_cased"
    "PantagrueLLM/camembert-base-wikipedia-4gb-weight-fix"
    #"PantagrueLLM/Text_Base_fr_4GB_camtok_step500K"
    # "PantagrueLLM/Text_Base_fr_OSCAR_camtok"
    # "PantagrueLLM/Text_Base_fr_4GB_step500K_tok_customized"
    # "PantagrueLLM/Text_Base_fr_croissant_data2vec_and_mlm_avg"
)
  
# name/path for all tokenizers(maybe different to the model name/path)
tokenizers=(
    # "camembert-base"
    # "flaubert/flaubert_base_cased"
    "PantagrueLLM/camembert-base-wikipedia-4gb-weight-fix"
    # "PantagrueLLM/Text_Base_fr_4GB_camtok_step500K"
    # "PantagrueLLM/Text_Base_fr_OSCAR_camtok"
    # "PantagrueLLM/Text_Base_fr_4GB_step500K_tok_customized"
    # "PantagrueLLM/Text_Base_fr_croissant_data2vec_and_mlm_avg"
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

    # set 5 seeds
    SEEDS_LINE='seeds: [1, 1234]'
    if grep -q '^seeds:' "$CONFIG_FILE"; then
      sed -i "s|^seeds:.*|$SEEDS_LINE|" "$CONFIG_FILE"
    elif grep -q '^seed:' "$CONFIG_FILE"; then
      sed -i "s|^seed:.*|$SEEDS_LINE|" "$CONFIG_FILE"
    else
      echo "$SEEDS_LINE" >> "$CONFIG_FILE"
    fi

    # set per-model learning rates
    case "$model_name" in
      "camembert-base")
        LR_LIST='learning_rates: [1e-5]'
        ;;
      "flaubert/flaubert_base_cased")
        LR_LIST='learning_rates: [5e-6]'
        ;;
      "PantagrueLLM/camembert-base-wikipedia-4gb-weight-fix")
        LR_LIST='learning_rates: [1e-5]'
        ;;
      "PantagrueLLM/Text_Base_fr_4GB_camtok_step500K")
        LR_LIST='learning_rates: [5e-6]'
        ;;
      "PantagrueLLM/Text_Base_fr_OSCAR_camtok")
        LR_LIST='learning_rates: [1e-7]'
        ;;
      "PantagrueLLM/Text_Base_fr_4GB_step500K_tok_customized")
        LR_LIST='learning_rates: [1e-5]'
        ;;
      "PantagrueLLM/Text_Base_fr_croissant_data2vec_and_mlm_avg")
        LR_LIST='learning_rates: [5e-6]'
        ;;

      *)
        LR_LIST='learning_rates: [5e-5, 1e-5, 5e-6, 1e-6]'
        ;;
    esac

    if grep -q '^learning_rates:' "$CONFIG_FILE"; then
      sed -i "s|^learning_rates:.*|$LR_LIST|" "$CONFIG_FILE"
    else
      echo "$LR_LIST" >> "$CONFIG_FILE"
    fi

    # start running
    python3 main.py --task "$task"
  done
done
