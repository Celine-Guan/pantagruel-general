import os
import argparse
import logging
from Analyse_de_sentiment.train import run_training as run_sentiment
from Identification_de_paraphrases.train import run_training as run_paraphrase
from common import load_yaml

TASK_CONFIGS = {
    "analyse_de_sentiment": "analyse_de_sentiment/config.yaml",
    "identification_de_paraphrases": "identification_de_paraphrases/config.yaml"
}

def main():
    parser = argparse.ArgumentParser(description="Sequence Classification Task Runner")
    parser.add_argument(
        "--task", choices=TASK_CONFIGS.keys(), required=True,
        help="Which task to run: analyse_de_sentiment or identification_de_paraphrases"
    )
    args = parser.parse_args()
    task_name = args.task

    config_path = TASK_CONFIGS[task_name]
    config = load_yaml(config_path)

    # Setup logging
    os.makedirs(config["log_dir"], exist_ok=True)
    log_path = os.path.join(config["log_dir"], f"training_{task_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting task: {task_name}")
    if task_name == "analyse_de_sentiment":
        run_sentiment(config)
    elif task_name == "identification_de_paraphrases":
        run_paraphrase(config)
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
