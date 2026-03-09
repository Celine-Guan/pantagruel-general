# Sequence Classification

Assign a label to a given text sequence. Three tasks are supported: sentiment analysis, paraphrase identification, and natural language inference.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Tasks](#tasks)
  - [1. Sentiment Analysis](#1-sentiment-analysis)
  - [2. Paraphrase Identification](#2-paraphrase-identification)
  - [3. Natural Language Inference](#3-natural-language-inference)
- [Running a Single Task](#running-a-single-task)
- [Running All Models](#running-all-models)

---

## Installation

```bash
cd classification
pip install -r requirements.txt
```

Create a `.env` file in the `classification/` directory with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

---

## Configuration

Each task has its own `config.yaml` file:

```
classification/
├── Analyse_de_sentiment/config.yaml
├── Identification_de_paraphrases/config.yaml
└── Natural_language_inference/config.yaml
```

Before running a task, edit the corresponding `config.yaml` to set the model, data paths, and training parameters. The key fields are:

| Field | Description |
|-------|-------------|
| `model_name` | HuggingFace model name or local path |
| `tokenizer_name` | HuggingFace tokenizer name or local path (usually same as `model_name`) |
| `data_dir` | Path to the dataset directory |
| `seeds` | Single seed or list of seeds for reproducibility |
| `device` | `"cuda"` or `"cpu"` |
| `batch_size` | Training batch size |
| `num_epochs` | Number of training epochs |
| `learning_rates` | List of learning rates to search over (best is selected on validation accuracy) |
| `max_seq_length` | Maximum input sequence length |
| `num_classes` | Number of output classes |
| `dropout` | Dropout rate |

---

## Tasks

### 1. Sentiment Analysis

**Dataset:** [CLS](https://zenodo.org/records/3251672)

Binary classification — determine whether an input review is positive (1) or negative (0).

The dataset contains three domains: `books`, `dvd`, and `music`. All three are evaluated by default.

Example `config.yaml`:

```yaml
seeds: [1, 42, 1234, 999]
device: "cuda"
data_dir: "/path/to/CLS/raw/cls-acl10-unprocessed/fr"
model_name: "PantagrueLLM/text-base-oscar-mlm"
tokenizer_name: "PantagrueLLM/text-base-oscar-mlm"
max_seq_length: 512
num_classes: 2
dropout: 0.1
add_pooling_layer: False
use_pooler: False
batch_size: 16
num_epochs: 30
learning_rates: [5e-6]
valid_ratio: 0.2
early_stopping_patience: 5
datasets: ["books", "dvd", "music"]
```

---

### 2. Paraphrase Identification

**Dataset:** [PAWS-X](https://github.com/google-research-datasets/paws)

Binary classification — determine whether two input sentences convey the same meaning.

Example:
- **Input A:** "A man is playing a guitar."
- **Input B:** "Someone is playing an instrument."
- **Output:** Paraphrase (1)

Example `config.yaml`:

```yaml
seeds: [1, 999, 1234, 2025]
device: "cuda"
data_dir: "/path/to/PAWS-X"
model_name: "PantagrueLLM/text-base-oscar-mlm"
tokenizer_name: "PantagrueLLM/text-base-oscar-mlm"
max_seq_length: 512
num_classes: 2
dropout: 0.1
add_pooling_layer: False
use_pooler: False
batch_size: 16
num_epochs: 30
learning_rates: [1e-05]
early_stopping_patience: 5
datasets: ["train", "dev", "test"]
```

---

### 3. Natural Language Inference

**Dataset:** [XNLI](https://github.com/facebookresearch/XNLI)

Three-class classification — determine the logical relationship between a premise and a hypothesis: entailment (0), contradiction (1), or neutral (2).

Example:
- **Input A (Premise):** "A woman is walking her dog in the park."
- **Input B (Hypothesis):** "A woman is outside with her pet."
- **Output:** Entailment (0)

Example `config.yaml`:

```yaml
seeds: [1, 42, 1234, 2025]
device: "cuda"
data_dir: "/path/to/XNLI/fr"
model_name: "PantagrueLLM/text-base-oscar-mlm"
tokenizer_name: "PantagrueLLM/text-base-oscar-mlm"
max_seq_length: 512
num_classes: 3
dropout: 0.1
batch_size: 16
num_epochs: 30
learning_rates: [1e-5, 5e-5, 1e-6, 5e-6]
early_stopping_patience: 3
datasets: ["train", "dev", "test"]
```

---

## Running a Single Task

From the `classification/` directory:

```bash
python main.py --task Analyse_de_sentiment
python main.py --task Identification_de_paraphrases
python main.py --task Natural_language_inference
```

Logs are saved to `logs/<model_name>/training_<task>.log`.

---

## Running All Models

To evaluate multiple models in batch, edit and run the provided shell script:

```bash
bash run_all_models.sh
```

The script iterates over a list of models, updates each task's `config.yaml` automatically (model name, tokenizer, seeds, learning rates), and runs `main.py` for each combination. Edit the `models`, `tokenizers`, and `tasks` arrays in the script to configure which models and tasks to run.
