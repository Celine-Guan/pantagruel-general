# Pantagruel-FLUE

The Pantagruel project (ANR 23-IAS1-0001) aims to develop and evaluate inclusive multimodal linguistic models (written, oral, pictograms) for French.

Pantagruel-eval evaluates the performance of a series of Pantagruel French models using a variety of NLP tasks, most of which are from the [FLUE benchmark](https://github.com/getalp/Flaubert) designed to evaluate [FlauBERT](https://aclanthology.org/2020.lrec-1.302/). For tasks involved in FLUE, we use the same datasets, model structure and hyperparameter settings as described in the reference paper.

**NB:** In each evaluation task, the script is general: we only need to change the model path/name to evaluate different models.

---

## Table of Contents

| Task | Description | Link |
|------|-------------|------|
| **Sequence Classification** | Sentiment analysis, paraphrase identification, natural language inference | [classification/](classification/) |
| **Word Sense Disambiguation** | Verb sense disambiguation using contextual embeddings and KNN | [Word-sense-disambiguation/](Word-sense-disambiguation/) |

---

## 1. Sequence Classification

Assign a label to a given text sequence. Three sub-tasks are supported:

| Sub-task | Dataset | Classes |
|----------|---------|---------|
| Sentiment Analysis | [CLS](https://zenodo.org/records/3251672) | 2 (positive / negative) |
| Paraphrase Identification | [PAWS-X](https://github.com/google-research-datasets/paws) | 2 (paraphrase / not) |
| Natural Language Inference | [XNLI](https://github.com/facebookresearch/XNLI) | 3 (entailment / contradiction / neutral) |

See the full documentation: **[classification/README.md](classification/README.md)**

---

## 2. Word Sense Disambiguation

Verb Sense Disambiguation (VSD) using the [FrenchSemEval (FSE)](http://www.llf.cnrs.fr/dataset/fse/) dataset. Contextual representations from transformer models are used with a KNN classifier to predict verb senses.

The code is adapted from the [WSD module in the FlauBERT repository](https://github.com/getalp/Flaubert/tree/master/flue/wsd), with modifications to support Pantagruel multimodal models.

See the full documentation: **[Word-sense-disambiguation/README.md](Word-sense-disambiguation/README.md)**

---

## Project Structure

```
pantagruel-general/
├── README.md
├── classification/
│   ├── README.md
│   ├── main.py
│   ├── requirements.txt
│   ├── run_all_models.sh
│   ├── common/
│   ├── Analyse_de_sentiment/
│   ├── Identification_de_paraphrases/
│   └── Natural_language_inference/
└── Word-sense-disambiguation/
    ├── README.md
    ├── run_all_models.sh
    └── verbs/
        ├── flue_vsd.py
        ├── run_model.py
        ├── wsd_evaluation.py
        ├── prepare_data.py
        └── requirements_wsd_verb
```

---

## References

- Le, H., et al. (2020). *FlauBERT: Unsupervised Language Model Pre-training for French.* In Proceedings of LREC 2020.
- Segonne, V., Candito, M., and Crabbe, B. (2019). *Using Wiktionary as a Resource for WSD: the Case of French Verbs.* In Proceedings of the 13th International Conference on Computational Semantics - Long Papers, pages 259-270.
- Raganato, A., J. Camacho-Collados, and R. Navigli (2017). *Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison.* In Proceedings of the 15th Conference of the European Chapter of the ACL, Volume 1, pp. 99-110.
