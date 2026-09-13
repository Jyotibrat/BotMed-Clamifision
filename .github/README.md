# BotMed-Clamifision

Training and inference code for Clamifision, BotMed's binary medical/non-medical text classifier, fine-tuned from ModernBERT-base.

- Model: [BJyotibrat/Clamifision-v1](https://huggingface.co/BJyotibrat/Clamifision-v1)
- Dataset: [BJyotibrat/Clamifision-v1-Data](https://huggingface.co/datasets/BJyotibrat/Clamifision-v1-Data)
- Data pipeline: [BotMed-Clamifision-Data-Pipeline](https://github.com/Jyotibrat/BotMed-Clamifision-Data-Pipeline)
- BotMed collection: [huggingface.co/collections/BJyotibrat/botmed](https://huggingface.co/collections/BJyotibrat/botmed)
- Live demo: [botmed.netlify.app](https://botmed.netlify.app/)

## Table of contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Related resources](#related-resources)
- [License](#license)

## Overview

Clamifision-v1 classifies input text as medical/health-related or not, acting as a routing gate for BotMed's specialized medical models. This repo contains:

- A ready-to-run inference example (script and notebook)
- The training notebook used to fine-tune the model
- Example batch-classification output

Training data was curated separately in the [data pipeline repo](https://github.com/Jyotibrat/BotMed-Clamifision-Data-Pipeline) and is published as [BJyotibrat/Clamifision-v1-Data](https://huggingface.co/datasets/BJyotibrat/Clamifision-v1-Data).

## Installation

```bash
pip install -r inference/requirements.txt
```

## Usage

### Quick inference

Use `inference/inference.py` directly, or run `inference/clamifision_v1_inference_example.ipynb` in Google Colab (loads the model straight from Hugging Face, no local files needed).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("BJyotibrat/Clamifision-v1")
model = AutoModelForSequenceClassification.from_pretrained("BJyotibrat/Clamifision-v1")

inputs = tokenizer("I've had a headache for 3 days", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()

print(model.config.id2label[pred])
```

### Training

`notebooks/Clamifision_v1.ipynb` contains the full training pipeline: loading the dataset, tokenization, fine-tuning ModernBERT-base, validation and test evaluation, and exporting the model and model card. Designed to run on Google Colab.

## Results

`results/classification_results.csv` is example output from a batch-classification run. For full evaluation metrics (accuracy, precision, recall, F1, confusion matrix), see the [model card](https://huggingface.co/BJyotibrat/Clamifision-v1).

## Related resources

| Resource | Link |
|---|---|
| Model | [BJyotibrat/Clamifision-v1](https://huggingface.co/BJyotibrat/Clamifision-v1) |
| Dataset | [BJyotibrat/Clamifision-v1-Data](https://huggingface.co/datasets/BJyotibrat/Clamifision-v1-Data) |
| Data pipeline | [BotMed-Clamifision-Data-Pipeline](https://github.com/Jyotibrat/BotMed-Clamifision-Data-Pipeline) |
| BotMed collection | [huggingface.co/collections/BJyotibrat/botmed](https://huggingface.co/collections/BJyotibrat/botmed) |
| Live demo | [botmed.netlify.app](https://botmed.netlify.app/) |

## License

GPL-3.0 — see [LICENSE](https://github.com/Jyotibrat/BotMed-Clamifision/blob/main/LICENSE).