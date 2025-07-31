# Pantagruel-eval
The Pantagruel project (ANR 23-IAS1-0001) aims to develop and evaluate inclusive multimodal linguistic models (written, oral, pictograms) for French.
The Pantagruel-eval project evaluates the performance of a series of Pantagruel French models using a variety of NLP tasks, most of which are from the FLUE benchmark designed to evaluate the performance of FlauBERT, a French language model. In this Pantagruel-eval project, for tasks involved in FLUE, we use the same datasets, follow the same model structure and the same hyperparameter settings. All are in the reference paper: [FLauBERT](https://aclanthology.org/2020.lrec-1.302/) and the github [Flaubert github](https://github.com/getalp/Flaubert).

**NB:** In each evaluation task, the script is general: we only need to change model path/name to evaluate different models, others codes remain the same. 
## 1. Sequence Classification
Assign a label to a given text sequence
## 1.1 Sentiment analysis
Dataset: [CLS](https://zenodo.org/records/3251672?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0OTA2NjQzOCwiZXhwIjoxNzUxNTg3MTk5fQ.eyJpZCI6IjYyN2MyNmJjLWFkNmItNGE5NS04Yzk4LWYxOGMxMGZlYTUzZSIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZjA5MDRjODNjM2M1OTMxYTAxYmI3MjFjYzI2Nzg1MyJ9.6yPUvs6pGeeow7GlzLereqoj93BNUFz_B2hO2IU-ulmjI703Gc3QKEtTRNREoeMFQz2ljVJIDYd9V4z07AdvIQ)

Task description: this task is binary classification, we need to determine whether a input review is positive(1) or negative(0)
## 1.2 Paraphrase Identification 
Dataset: [PAWS-X](https://github.com/google-research-datasets/paws)

Task description: it's also a binary classification on determining whether two input sentences convey the same meaning.
Example:
- Input A: "A man is playing a guitar."
- Input B: "Someone is playing an instrument."
- Output: Paraphrase (1)
## 1.3 Natural language inference 
Dataset: [XNLI](https://github.com/facebookresearch/XNLI)

Task description: It's a three-class classification task that determines the logical relationship between a premise sentence and a hypothesis sentence. The output label indicates whether the hypothesis is entailed by, contradicts, or is neutral with respect to the premise.
Example:
- Input A (Premise): "A woman is walking her dog in the park."
- Input B (Hypothesis): "A woman is outside with her pet."
- Output: Entailment (0)
## 2. Word Sense Disambiguation
The code and documents in Word-sense-disambiguation are directly copied from [directory wsd in flue github](https://github.com/getalp/Flaubert/tree/master/flue/wsd)
## 2.1 Verb sense disambiguation
There is a little modification in "forward" function in wsd_encoder.py to make the code compatible for Pantagruel models which are multi-modal
## 2.2 Noun sense disambiguation
There is still a problem "Version hell", since the original code was written 5 years ago, the version of packages used in requirements_wsd_noun.txt is not compatible to the version of Pantagruel branch of transformer, so the code need to be modified to evaluate Pantagruel models.
