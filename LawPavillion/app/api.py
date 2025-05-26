from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from app.model_utils import load_model_and_tokenizer, load_label_encoder

router = APIRouter()

# Load once
model, tokenizer = load_model_and_tokenizer()
label_encoder = load_label_encoder()

# Request model
class ReportInput(BaseModel):
    full_report: str

@router.post("/predict")
def predict_area_of_law(input: ReportInput):
    try:
        inputs = tokenizer(input.full_report, truncation=True, padding=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_id])[0]

        return {"predicted_area_of_law": pred_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
