"""
Minimal inference example for the Clamifision medical/non-medical text classifier.
Loads the model from the Hugging Face Hub -- run `pip install -r requirements.txt` first.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "BJyotibrat/Clamifision-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


def classify(text: str) -> dict:
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
    return {
        "label": model.config.id2label[pred],
        "confidence": probs[pred].item(),
    }


if __name__ == "__main__":
    examples = [
        "I've had a headache for 3 days and my vision is blurry",
        "whats the best pizza topping combo",
    ]
    for text in examples:
        result = classify(text)
        print(f"[{result['label']:>11} | {result['confidence']:.2f}]  {text}")
