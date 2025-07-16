import os
import argparse
import logging
from Analyse_de_sentiment import run_training as run_sentiment
from Identification_de_paraphrases import run_training as run_paraphrase
from Natural_language_inference import run_training as run_nli
from common import load_yaml
from common import init_logger

TASK_CONFIGS = {
    "Analyse_de_sentiment": "Analyse_de_sentiment/config.yaml",
    "Identification_de_paraphrases": "Identification_de_paraphrases/config.yaml",
    "Natural_language_inference": "Natural_language_inference/config.yaml"
}

def main():
    parser = argparse.ArgumentParser(description="Sequence Classification Task Runner")
    parser.add_argument(
        "--task", choices=TASK_CONFIGS.keys(), required=True,
        help="Which task to run: Analyse_de_sentiment, Identification_de_paraphrases, or Natural_language_inference"
    )
    args = parser.parse_args()
    task_name = args.task

    config_path = TASK_CONFIGS[task_name]
    config = load_yaml(config_path)

    # Setup logging
    model_name = config.get("model_name", "model")
    short_model_id = model_name.split("/models/")[-1].replace("/", "_")
    log_file = f"training_{task_name}_{short_model_id}.log"

    init_logger(log_file)
    logging.info(f"Starting task: {task_name} with model: {short_model_id}")
    
    if task_name == "Analyse_de_sentiment":
        run_sentiment(config)
    elif task_name == "Identification_de_paraphrases":
        run_paraphrase(config)
    elif task_name == "Natural_language_inference": 
        run_nli(config)
        
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
