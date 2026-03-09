# Word Sense Disambiguation

The code is adapted from the [WSD module in the FlauBERT repository](https://github.com/getalp/Flaubert/tree/master/flue/wsd), with modifications in the `forward` function of `wsd_encoder.py` to support Pantagruel multimodal models.

---

## Table of Contents

- [Installation](#installation)
- [Verb Sense Disambiguation](#verb-sense-disambiguation)
  - [Dataset](#dataset)
  - [Prepare the Data](#prepare-the-data)
  - [Run the Model](#run-the-model)
  - [Running All Models](#running-all-models)
- [Use Other Vectors](#use-other-vectors)
- [References](#references)

---

## Installation

```bash
cd Word-sense-disambiguation/verbs
pip install -r requirements_wsd_verb
```

Create a `.env` file in `Word-sense-disambiguation/` with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

---

## Verb Sense Disambiguation

Verb Sense Disambiguation (VSD) is a subpart of WSD where only verbs are the target of disambiguation. We use the [FrenchSemEval (FSE)](http://www.llf.cnrs.fr/dataset/fse/) dataset.

The disambiguation process:
1. Run the model on training / evaluation data to obtain contextual representations for target verb occurrences.
2. Compute sense representations by averaging the vector representations of their instances.
3. Use a KNN classifier based on cosine similarity to predict labels by comparing evaluation instances to sense representations.

### Dataset

The FSE dataset includes:
- **Training data:** examples extracted from sense entries in Wiktionary (dump of 04-20-2018).
- **Evaluation data:** manual annotations of verbs with senses from Wiktionary, extracted from French Wikipedia, the French Treebank, and the Sequoia corpus.

Both splits use the format from Raganato's WSD evaluation framework:
- A `.data.xml` file containing sentences in XML format.
- A `.gold.key.txt` file containing instance labels.

### Prepare the Data

Download the [FSE dataset](http://www.llf.cnrs.fr/dataset/fse/), then run:

```bash
cd verbs
python prepare_data.py --data $FSE_DIR --output_dir $DATA_DIR
```

Use `--train $OTHER_TRAIN_DIR` to substitute alternative training data (must follow the same format).

### Run the Model

```bash
cd verbs
python flue_vsd.py \
  --exp_name myexp \
  --model PantagrueLLM/text-base-oscar-mlm \
  --data $DATA_DIR \
  --padding 80 \
  --batchsize 32 \
  --device 0 \
  --output $OUTPUT_DIR \
  --output_score $SCORE_FILE
```

| Argument | Description |
|----------|-------------|
| `--exp_name` | Name of the experiment |
| `--model` | HuggingFace model name or path to a checkpoint |
| `--data` | Path to directory containing `train/` and `test/` subdirectories |
| `--padding` | Pad sentences to this length |
| `--batchsize` | Batch size |
| `--device` | GPU device index (`-1` for CPU) |
| `--output` | Directory for output vectors |
| `--output_score` | Path to output score CSV |
| `--output_logs` | (optional) Path to output logs CSV |
| `--output_pred` | (optional) Path to output predictions |

### Running All Models

To evaluate multiple models in batch, edit the model list and paths in `run_all_models.sh`, then run from the `Word-sense-disambiguation/` directory:

```bash
bash run_all_models.sh
```

The script loops over a list of models and runs `flue_vsd.py` for each, saving vectors and scores to the configured output directories.

---

## Use Other Vectors

It is possible to evaluate vectors from any model (not just transformers):

1. Dump vectors into `$TRAIN_VECS` and `$TEST_VECS` files (one `instance_id \t vector` per line).
2. Run the evaluation:

```bash
cd verbs
python wsd_evaluation.py \
  --exp_name myexp \
  --train_data $TRAIN_DIR \
  --train_vecs $TRAIN_VECS \
  --test_data $TEST_DIR \
  --test_vecs $TEST_VECS \
  --average \
  --target_pos V
```

See `python wsd_evaluation.py --help` for further details and options.

---

## References

- Segonne, V., Candito, M., and Crabbe, B. (2019). *Using Wiktionary as a Resource for WSD: the Case of French Verbs.* In Proceedings of the 13th International Conference on Computational Semantics - Long Papers, pages 259-270.
- Abeille, A. and N. Barrier (2004). *Enriching a French Treebank.* In Proceedings of LREC 2004, Lisbon, Portugal.
- Candito, M. and D. Seddah (2012). *Le corpus Sequoia: annotation syntaxique et exploitation pour l'adaptation d'analyseur par pont lexical.* In TALN 2012.
- Raganato, A., J. Camacho-Collados, and R. Navigli (2017). *Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison.* In Proceedings of the 15th Conference of the European Chapter of the ACL, Volume 1, pp. 99-110.
