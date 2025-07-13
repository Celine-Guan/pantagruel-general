# Sequence Classification in French

This project provides a modular framework for training and evaluating transformer-based models on French sequence classification tasks, including:

- **Sentiment Analysis (CLS)**
- **Paraphrase Identification (PAWS-X)**

## 🗂️ Project Structure

## 🚀 Quick Start
### 1. Clone the repository

```bash
git clone git@github.com:GUShuyue/Pantagruel-eval.git
cd ~/Pantagruel-eval/Sequence-classification
```
### 2. Create and activate the conda environment
If you're using Conda:
```bash
bash setup.sh
```
Or manually:
```bash
conda env create -f environment.yml
conda activate sequence-classification
```
**RB: To run the Pantagruel models we need to install the pantagruel branch of transformer, the installation code is in the environment.yml**
### 3. Run training
#### ✨ If you want to test each time one model on one task, modify the model and tokenizer name in corresponding config file, then run the following code in the terminal.
For sentiment analysis task:
```bash
python3 main.py --task Analyse_de_sentiment
```
For paraphrasing task
```bash
python3 main.py --task Identification_de_paraphrases
```
#### ✨ If you want to run all models on all tasks, you can run the run_all_models bash file in terminal:
```bash
bash run_all_models.sh
```
⏳ Since testing all models on all tasks really takes time, you could try to run it in the background:
```bash
nohup bash run_all_models.sh > all_runs.log 2>&1 &
```
Then you can run the following code to monitor the real-time output，press Ctrl+C to stop following:
```bash
tail -f all_runs.log
```
## ⚙️ Configuration
Each task (e.g., sentiment analysis or paraphrasing) has its own config.yaml, specifying:

- data_dir: Dataset location

- batch_size, max_seq_length, num_epochs

- learning_rates: List of LR values for model tuning

- early_stopping_patience: Patience value for early stopping

- model_name, tokenizer_name: The name/path of evaluated model and tokenizer, if the model is a standard HuggingFace model which can be pulled directly from transformer, the model name and tokenizer name are the same, like "camembert-base", otherwise use the path of model and tokenizer on your server, when changing model, you can simply modify the corresponding config.yaml
## 🧩 Supported Models
This project supports any Hugging Face-compatible transformer model. Custom models like pantagruel are also supported by installing a forked version of transformers.
## 📦 Dependencies
See requirements.txt and environment.yml for full dependency details.
## 📁 Datasets
- Sentiment Analysis: CLS (French subset) [The Cross Lingual Sentiment CLS dataset](https://zenodo.org/records/3251672?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0OTA2NjQzOCwiZXhwIjoxNzUxNTg3MTk5fQ.eyJpZCI6IjYyN2MyNmJjLWFkNmItNGE5NS04Yzk4LWYxOGMxMGZlYTUzZSIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZjA5MDRjODNjM2M1OTMxYTAxYmI3MjFjYzI2Nzg1MyJ9.6yPUvs6pGeeow7GlzLereqoj93BNUFz_B2hO2IU-ulmjI703Gc3QKEtTRNREoeMFQz2ljVJIDYd9V4z07AdvIQ) 

- Paraphrasing: [PAWS-X (French subset)](https://github.com/google-research-datasets/paws/tree/master/pawsx).

Make sure the datasets are downloaded and placed in the appropriate paths, as defined in config.yaml. Your can use your own path and write it in the corresponding config.yaml file.
## 📝 Reference
Sentiment analysis and Paraphrasing tasks in this project are part of the FLUE benchmark, to know more about FLUE, you can read the [reference paper](https://aclanthology.org/2020.lrec-1.302/).
