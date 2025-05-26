import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

# Construct path relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to 'lawpavillion/'
MODEL_PATH = os.path.join(BASE_DIR, "models")

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(torch.device("cpu"))
    return model, tokenizer

def load_label_encoder():
    return joblib.load(f"{MODEL_PATH}/saved_model.pkl")
